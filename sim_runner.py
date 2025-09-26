import argparse, asyncio, threading, json, queue, websockets, time, sys
import base64
import os
import time
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import cv2
import numpy as np
from loguru import logger
import time, queue as pyqueue
import genesis as gs
import gymnasium as gym
from maniladder.envs import ManiLadderBaseEnv
from maniladder.utils.wrappers.RecordEpisodeWrapper import RecordEpisodeWrapper
from genesis.utils.geom import quat_to_R, _np_euler_to_R, R_to_quat, quat_to_xyz
from maniladder import ASSETS_DIR

from utils.utils import next_name_by_count

async def frame_sender(ws_url: str, img_queue: queue.Queue, stop_evt: threading.Event):
    try:
        async with websockets.connect(ws_url, max_size=2**22) as ws:
            print(f"[frame_sender] connected to hub for frame upload: {ws_url}")
            while not stop_evt.is_set():
                try:
                    frame_b64 = img_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                print(f"[frame_sender] sending frame, len={len(frame_b64)}")
                payload = {"frame": frame_b64}
                await ws.send(json.dumps(payload, separators=(",", ":")))
    except Exception as e:
        print(f"[frame_sender] frame sender error: {e}")

def _to_rgb_u8(img):
    """把可能的 float/整型、通道顺序不一的输入规范成 RGB uint8 (H,W,3)。"""
    arr = np.asarray(img)
    # 如果是 CHW (C,H,W) -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1,2,0))
    # 单通道 -> 3 通道
    if arr.ndim == 2:
        arr = np.repeat(arr[...,None], 3, axis=2)
    # float -> uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1) if arr.dtype.kind == 'f' and arr.max()<=1.0 else np.clip(arr, 0, 255)
        arr = (arr*255.0).astype(np.uint8) if arr.dtype.kind == 'f' else arr.astype(np.uint8)
    return arr  # 约定输入本来就是 RGB

def make_composite(front_img, left_img, top_img, target_h=720, gap=6, bg_color=(0,0,0)):
    """
    目标尺寸：高 target_h，左列为 front（等高缩放），右列上下各半：left / top。
    返回 RGB uint8 的拼接图。
    """
    f = _to_rgb_u8(front_img)
    l = _to_rgb_u8(left_img)
    t = _to_rgb_u8(top_img)

    right_each_h = (target_h - gap) // 2

    # 缩放 front 到 target_h 高度
    fh, fw = f.shape[:2]
    front_scale = target_h / float(fh)
    front_w = max(1, int(round(fw * front_scale)))
    f_resized = cv2.resize(f, (front_w, target_h), interpolation=cv2.INTER_AREA if front_scale < 1 else cv2.INTER_LINEAR)

    # 缩放 left、top 到 right_each_h 高度（等高缩放）
    def resize_to_h(img, out_h):
        ih, iw = img.shape[:2]
        s = out_h / float(ih)
        out_w = max(1, int(round(iw * s)))
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

    l_resized = resize_to_h(l, right_each_h)
    t_resized = resize_to_h(t, right_each_h)

    # 右列宽度取二者最大，较窄的在右侧用空白 padding 补齐
    right_w = max(l_resized.shape[1], t_resized.shape[1])
    def pad_to_w(img, out_w):
        h, w = img.shape[:2]
        if w == out_w:
            return img
        pad = np.full((h, out_w - w, 3), bg_color, dtype=np.uint8)
        return np.concatenate([img, pad], axis=1)  # 右侧补齐

    l_pad = pad_to_w(l_resized, right_w)
    t_pad = pad_to_w(t_resized, right_w)

    # 右列上下堆叠，中间插入 gap
    gap_v = np.full((gap, right_w, 3), bg_color, dtype=np.uint8)
    right_col = np.concatenate([l_pad, gap_v, t_pad], axis=0)  # (target_h, right_w, 3)

    # 左右列之间加一个 gap
    gap_h = np.full((target_h, gap, 3), bg_color, dtype=np.uint8)

    # 最终拼接
    composite = np.concatenate([f_resized, gap_h, right_col], axis=1)  # RGB uint8
    return composite

def send_frame_to_queue(front_img, img_queue):
    try:
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
        frame_b64 = base64.b64encode(buf).decode("ascii")
        try:
            img_queue.put_nowait(frame_b64)
        except queue.Full:
            try: img_queue.get_nowait()
            except queue.Empty: pass
            try: img_queue.put_nowait(frame_b64)
            except queue.Full: pass
    except Exception as e:
        print(f"[sim_runner] failed to encode frame: {e}")

def wrap_euler_angles(euler_angles):
    wrapped_angles = np.mod(euler_angles + np.pi, 2 * np.pi) - np.pi
    return wrapped_angles

def empty_action_generator(env_type: str = "single_arm_gripper", hand_type: str = "right"):
    if env_type == "single_arm_gripper":
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

def receiver_thread(ws_url: str, q: queue.Queue, stop_evt: threading.Event):
    async def go():
        ws = None
        try:
            async with websockets.connect(ws_url, max_size=2**22) as ws:
                # 带超时的循环：定期检查 stop_evt
                while not stop_evt.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue  # 周期性醒来检查 stop_evt
                    except websockets.ConnectionClosed:
                        break
                    # 尝试解析 JSON
                    try:
                        frame = json.loads(msg)
                    except Exception:
                        continue
                    # 丢旧保新
                    try:
                        q.put_nowait(frame)
                    except queue.Full:
                        try: q.get_nowait()
                        except queue.Empty: pass
                        try: q.put_nowait(frame)
                        except queue.Full: pass
        finally:
            # 退出前尽量关闭连接
            try:
                if ws and not ws.closed:
                    await ws.close()
            except Exception:
                pass

    # 在线程里跑事件循环
    try:
        asyncio.run(go())
    except Exception as e:
        # 避免异常把线程卡住
        # print(f"[receiver] error: {e}")
        pass

# =========================
# 新增：倒计时 + 预热 + 初始化对齐
# =========================
def run_countdown_preheat(env, q, img_queue, args, duration_s=3.0):
    """
    在倒计时期间完成 prepare 预热 & 初始化对齐。
    结束时返回：i 设置为 args.prepare_frames（让主循环直接进入“可控态”），
    以及 position_correction / init_quest_R / init_ee_R / last_quest_pose。
    """
    tick = 1.0 / float(args.tick_hz)
    target_frames = max(1, int(round(duration_s * float(args.tick_hz))))

    i_count = 0
    last_quest_pose = None
    position_correction = None
    init_quest_R = None
    init_ee_R = None

    next_t = time.perf_counter()

    while i_count < target_frames:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(min(0.001, next_t - now))
            continue
        next_t += tick

        # 拿最新一包手数据
        pkt = None
        try:
            while True:
                pkt = q.get_nowait()
        except queue.Empty:
            pass

        # 空动作步进
        action = empty_action_generator()

        obs, reward, terminated, done, info = env.step(action)
        front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
        left_img  = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
        top_img   = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()

        # 倒计时叠字
        secs_left = int(np.ceil((target_frames - i_count) / float(args.tick_hz)))
        combo = make_composite(front_img, left_img, top_img, target_h=720, gap=6)
        cv2.rectangle(combo, (12, 12), (260, 70), (0, 0, 0), thickness=-1)
        cv2.putText(combo, f"Starting in {secs_left}s",
                    (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # 本地显示
        cv2.imshow("front_view", cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("left_view",  cv2.cvtColor(left_img,  cv2.COLOR_RGB2BGR))
        cv2.imshow("top_view",   cv2.cvtColor(top_img,   cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        # 上传
        send_frame_to_queue(combo, img_queue)

        # 记录最后一帧手姿态（用于对齐）
        if pkt is not None:
            palm = pkt.get("palm")
            if palm:
                origin = palm.get("origin")
                quat   = palm.get("quat")
                if origin is not None and quat is not None:
                    last_quest_pose = np.array([*origin, *quat], dtype=np.float32)

        i_count += 1

    # 倒计时结束：若拿到过手数据，则完成初始化对齐，并开始录制（若支持）
    if last_quest_pose is not None:
        quest_position = last_quest_pose[:3] * args.position_scale
        quest_quat     = last_quest_pose[3:]
        quest_R        = quat_to_R(quest_quat)

        ee_pos  = env.env.env.agent.tcp_pos.cpu().numpy()
        ee_quat = env.env.env.agent.tcp_quat.cpu().numpy()
        ee_R    = quat_to_R(ee_quat)

        position_correction = quest_position - ee_pos
        init_quest_R = quest_R
        init_ee_R    = ee_R

        try:
            env.start_record("front_camera")
        except Exception:
            pass

    return {
        "i_after": max(args.prepare_frames, 0),
        "position_correction": position_correction,
        "init_quest_R": init_quest_R,
        "init_ee_R": init_ee_R,
        "last_quest_pose": last_quest_pose,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/sub")
    ap.add_argument("--env_name", default="StackBlock-v1")
    ap.add_argument("--env_type", default="single_arm_gripper")
    ap.add_argument("--tick_hz", default="10")
    ap.add_argument("--hand_type", default="right")
    ap.add_argument("--video_save_dir", default="tele_videos")
    ap.add_argument("--max_accident_steps", default=4, type=int)
    ap.add_argument("--prepare_frames", default=50, type=int)
    ap.add_argument("--max_steps", default=600, type=int)
    ap.add_argument("--position_scale", default=1.5, type=float)
    ap.add_argument("--screen_resolution", default=(2560, 1440), type=tuple)
    ap.add_argument("--RECORDING", default=False, type=bool)

    args = ap.parse_args()

    q = queue.Queue(maxsize=1024)
    stop_evt = threading.Event()
    t = threading.Thread(target=receiver_thread, args=(args.url, q, stop_evt), daemon=True)
    t.start()

    img_queue = queue.Queue(maxsize=8)
    # 新开一个线程专门跑 frame_sender
    frame_thread = threading.Thread(
        target=lambda: asyncio.run(frame_sender(args.url.replace("/sub", ""), img_queue, stop_evt)),
        daemon=True
    )
    frame_thread.start()
    
    print(f"sim-worker started: tick={args.tick_hz}Hz")
    tick = 1.0 / float(args.tick_hz)
    next_t = time.perf_counter()
    last_tick_t = next_t
    
    # env building
    print(f"Start building env {args.env_name}")
    gs.init(
        seed=None,
        precision='32',
        debug=False,
        eps=1e-12,
        logging_level=None,
        backend=gs.gpu,
        theme='dark',
        logger_verbose_time=False,
    )
    
    env = gym.make(
        args.env_name,
        num_envs=1,
        obs_mode="rgbd",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        enable_shadow=True,
        # robot_uids="xarm7_leaphand",
        show_viewer=False,
        show_world_frame=False,
        device=gs.device,
    )

    if args.RECORDING:
        env = RecordEpisodeWrapper(env, output_dir="trajectories", trajectory_name="teleoperation_trajectory")
    env.reset()

    # ========= 新增调用：首次开始前做 3s 倒计时 + 预热 + 初始化对齐 =========
    state = run_countdown_preheat(env, q, img_queue, args, duration_s=3.0)
    i = state["i_after"]
    position_correction = state["position_correction"]
    init_quest_R = state["init_quest_R"]
    init_ee_R = state["init_ee_R"]
    last_quest_pose = state["last_quest_pose"]

    front_view_win_name = "front_view"
    cv2.namedWindow(front_view_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(front_view_win_name, 2 * args.screen_resolution[1] // 3, args.screen_resolution[1] // 2)
    cv2.moveWindow(front_view_win_name, args.screen_resolution[0] // 3 - args.screen_resolution[1] // 3, 0)  # position it
    
    left_view_win_name = "left_view"
    cv2.namedWindow(left_view_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(left_view_win_name, 2 * args.screen_resolution[1] // 3, args.screen_resolution[1] // 2)
    cv2.moveWindow(left_view_win_name, 0, args.screen_resolution[1] // 2)  # position it
    
    top_view_win_name = "top_view"
    cv2.namedWindow(top_view_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(top_view_win_name, 2 * args.screen_resolution[1] // 3, args.screen_resolution[1] // 2)
    cv2.moveWindow(top_view_win_name, args.screen_resolution[0] // 3, args.screen_resolution[1] // 2)  # position it
    
    if args.video_save_dir is not None:
        assert isinstance(args.video_save_dir, str), "video_save_dir must be a string"
        os.makedirs(os.path.join(args.video_save_dir, args.env_name), exist_ok=True)
    
    accident_steps = 0
    init_gripper_dis = None
    done = False
    
    try:
        while True:
            now = time.perf_counter()
            
            # 简单的定时器：保持大致 tick_hz 的循环频率
            if now < next_t:
                # 轻量 sleep，避免空转吃满CPU
                time.sleep(min(0.001, next_t - now))
                continue
            dt = now - last_tick_t
            last_tick_t = now
            next_t = now + tick
            
            # 非阻塞拿到“最新一帧”——把旧帧全丢，只留最后
            pkt = None            
            try:
                while True:
                    pkt = q.get_nowait()
            except queue.Empty:
                pass

            if pkt is None:
                print(f"{args.hand_type} hand is not detected.")
                # if first detection has not come yet
                if last_quest_pose is None:
                    action = empty_action_generator()

                    obs, reward, terminated, done, info = env.step(action)
                    front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
                    left_img = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
                    top_img = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()
                    cv2.imshow(front_view_win_name, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
                    cv2.imshow(left_view_win_name, cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
                    cv2.imshow(top_view_win_name, cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR))
                    
                    combo = make_composite(front_img, left_img, top_img, target_h=720, gap=6)
                    send_frame_to_queue(combo, img_queue)

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                    # here i not increase
                    i = 0   # reset i
                    print(f"reset i to {i}")
                    
                # if already moved, but there is some empty action
                else:
                    if accident_steps < args.max_accident_steps:
                        quest_pose = last_quest_pose
                        accident_steps += 1
                    else:
                        i = 0
                        print(f"no hand detected for a while. reset i to {i}")
                        
                    action = empty_action_generator()

                    obs, reward, terminated, done, info = env.step(action)
                    front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
                    left_img = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
                    top_img = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()
                    cv2.imshow(front_view_win_name, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
                    cv2.imshow(left_view_win_name, cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
                    cv2.imshow(top_view_win_name, cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR))
                    
                    combo = make_composite(front_img, left_img, top_img, target_h=720, gap=6)

                    send_frame_to_queue(combo, img_queue)
                    
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
            else:
                accident_steps = 0
                hand = pkt.get("hand")
                palm = pkt.get("palm")

                if palm is not None:
                    origin = palm.get("origin")
                    quat = palm.get("quat")
                else:
                    print(f"Warning: Palm data is missing in the current packet.")
                    continue  # Skip this iteration and process the next packet if palm is missing

                origin = palm.get("origin")
                quat = palm.get("quat")
                
                quest_pose = np.array([*origin, *quat], dtype=np.float32)  # (7,)
                
                i += 1
                last_quest_pose = quest_pose
                if i < args.prepare_frames:
                    print(f"accumulated i: {i}/{args.prepare_frames}")
                    action = empty_action_generator()

                    obs, reward, terminated, done, info = env.step(action)
                    front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
                    left_img = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
                    top_img = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()
                    cv2.imshow(front_view_win_name, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
                    cv2.imshow(left_view_win_name, cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
                    cv2.imshow(top_view_win_name, cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR))
                    
                    combo = make_composite(front_img, left_img, top_img, target_h=720, gap=6)
                    send_frame_to_queue(combo, img_queue)

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                else:
                    # if i > args.max_steps + args.prepare_frames or done:
                    if i > args.max_steps + args.prepare_frames:
                        save_to_filename=next_name_by_count(f"{args.video_save_dir}/{args.env_name}_fail", ".mp4")
                        env.stop_record("front_camera", save_to_filename=save_to_filename, fps=30)
                        env.reset()

                        # ========= 新增调用：回合重置后也做 3s 倒计时 + 预热 + 初始化对齐 =========
                        state = run_countdown_preheat(env, q, img_queue, args, duration_s=3.0)
                        i = state["i_after"]
                        position_correction = state["position_correction"]
                        init_quest_R = state["init_quest_R"]
                        init_ee_R = state["init_ee_R"]
                        last_quest_pose = state["last_quest_pose"]

                        accident_steps = 0
                        done = False
                        print("resetting, exiting ...")
                        continue
            
                    quest_position = quest_pose[:3] * args.position_scale
                    quest_quat = quest_pose[3:]
                    quest_R = quat_to_R(quest_quat)

                    ee_pos = env.env.env.agent.tcp_pos.cpu().numpy()
                    ee_quat = env.env.env.agent.tcp_quat.cpu().numpy()
                    ee_R = quat_to_R(ee_quat)
                    
                    if i == args.prepare_frames:
                        position_correction = quest_position - ee_pos
                        init_quest_R = quest_R
                        init_ee_R = ee_R
                        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                        print(f"teleoperating init done, start moving ...")
                        env.start_record("front_camera")
                        
                    else:
                        position_action = quest_position - ee_pos - position_correction
                        R_action = init_ee_R @ init_quest_R.T @ quest_R @ ee_R.T
                        euler_action = quat_to_xyz(R_to_quat(R_action.T), degrees=False)
                        
                        gripper_action = 1.0
                        
                        action = np.concatenate([position_action, euler_action, [gripper_action]], axis=0).astype(np.float32)
                        print(f"teleoperating... {i-args.prepare_frames+1}/{args.max_steps}, action: {action}")
                        
                        obs, reward, terminated, done, info = env.step(action)
                        front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
                        left_img = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
                        top_img = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()
                        cv2.imshow(front_view_win_name, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
                        cv2.imshow(left_view_win_name, cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
                        cv2.imshow(top_view_win_name, cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR))

                        combo = make_composite(front_img, left_img, top_img, target_h=720, gap=6)
                        send_frame_to_queue(combo, img_queue)

                        k = cv2.waitKey(1) & 0xFF
                        if k == ord('q'):
                            break
                        
    except KeyboardInterrupt:
        print("exiting ...")
        cv2.destroyWindow(front_view_win_name)
        cv2.destroyWindow(left_view_win_name)
        cv2.destroyWindow(top_view_win_name)        
        env.close()        
        # 通知接收线程收尾
        stop_evt.set()
        # 等待一小会儿给它机会退出
        t.join(timeout=3.0)
        # 若仍未退出，直接结束进程（可选）
        sys.exit(0)
