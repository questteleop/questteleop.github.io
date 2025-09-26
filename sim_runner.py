import argparse, asyncio, threading, json, queue, websockets, sys
import base64
import os
import cv2
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R

import genesis as gs
import gymnasium as gym
from maniladder.envs import ManiLadderBaseEnv
from maniladder.utils.wrappers.RecordEpisodeWrapper import RecordEpisodeWrapper
from genesis.utils.geom import quat_to_R, R_to_quat, quat_to_xyz
from maniladder import ASSETS_DIR
from utils.utils import next_name_by_count
import time  # 保留一次

# ========== 参数：可按需微调 ==========
POS_WIN = 8                  # 位置滑窗长度
ROT_WIN = 8                  # 姿态滑窗长度
EMA_POS_ALPHA = 0.35         # 位置 EMA 系数（0~1，越大越跟手）
EMA_ROT_ALPHA = 0.35         # 姿态 EMA 系数
POS_JUMP_REJECT = 0.12       # 单帧跳变 > 12cm 视为离群，忽略
ANG_JUMP_REJECT = 0.7        # 单帧转角 > 0.7rad 视为离群，忽略
POS_DEADBAND = 0.002         # 2mm 以内忽略
ANG_DEADBAND = 0.02          # 约1.1度以内忽略
POS_STEP_MAX = 0.03          # 每tick最大位移 3cm
ANG_STEP_MAX = 0.15          # 每tick最大转角 0.15rad

# ========== 滑动窗口 ==========
POS_WINDOW = deque(maxlen=POS_WIN)     # 保存 wxyz 对应的同一帧位置
QUAT_WINDOW_WXYZ = deque(maxlen=ROT_WIN)

# ========== 工具 ==========
def quat_wxyz_to_xyzw(q_wxyz):
    # 输入 [w,x,y,z] -> 输出 [x,y,z,w]
    q = np.asarray(q_wxyz, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)

def quat_xyzw_to_wxyz(q_xyzw):
    q = np.asarray(q_xyzw, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)

def average_quaternion_wxyz(quats_wxyz):
    """
    对一组 wxyz 四元数做球面平均，返回 wxyz。
    """
    if len(quats_wxyz) == 1:
        return np.asarray(quats_wxyz[0], dtype=float)
    q_xyzw = np.stack([quat_wxyz_to_xyzw(q) for q in quats_wxyz], axis=0)
    R_list = R.from_quat(q_xyzw)  # xyzw
    R_mean = R_list.mean()
    q_mean_xyzw = R_mean.as_quat()
    return quat_xyzw_to_wxyz(q_mean_xyzw)

def exp_smooth(prev, new, alpha):
    if prev is None: return new
    return (1.0 - alpha) * np.asarray(prev) + alpha * np.asarray(new)

def clamp_vec(v, max_norm):
    n = np.linalg.norm(v)
    if n <= max_norm or n == 0: return v
    return v * (max_norm / n)

def axis_angle_between_q(q1_wxyz, q2_wxyz):
    # 返回从 q1 -> q2 的相对转角（弧度）
    r1 = R.from_quat(quat_wxyz_to_xyzw(q1_wxyz))
    r2 = R.from_quat(quat_wxyz_to_xyzw(q2_wxyz))
    r = r2 * r1.inv()
    ang = np.linalg.norm(r.as_rotvec())
    return ang

# ========== WS 图像帧上传 ==========
async def frame_sender(ws_url: str, img_queue: queue.Queue, stop_evt: threading.Event):
    try:
        async with websockets.connect(ws_url, max_size=2**22) as ws:
            # print(f"[frame_sender] connected: {ws_url}")
            while not stop_evt.is_set():
                try:
                    frame_b64 = img_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                await ws.send(json.dumps({"frame": frame_b64}, separators=(",", ":")))
    except Exception as e:
        print(f"[frame_sender] error: {e}")

def _to_rgb_u8(img):
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1,2,0))
    if arr.ndim == 2:
        arr = np.repeat(arr[...,None], 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1) if arr.dtype.kind == 'f' and arr.max()<=1.0 else np.clip(arr, 0, 255)
        arr = (arr*255.0).astype(np.uint8) if arr.dtype.kind == 'f' else arr.astype(np.uint8)
    return arr

def make_composite(front_img, left_img, top_img, target_h=720, gap=6, bg_color=(0,0,0)):
    f = _to_rgb_u8(front_img); l = _to_rgb_u8(left_img); t = _to_rgb_u8(top_img)
    right_each_h = (target_h - gap) // 2
    fh, fw = f.shape[:2]; front_scale = target_h / float(fh)
    front_w = max(1, int(round(fw * front_scale)))
    f_resized = cv2.resize(f, (front_w, target_h), interpolation=cv2.INTER_AREA if front_scale < 1 else cv2.INTER_LINEAR)
    def resize_to_h(img, out_h):
        ih, iw = img.shape[:2]; s = out_h / float(ih)
        out_w = max(1, int(round(iw * s)))
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    l_resized = resize_to_h(l, right_each_h); t_resized = resize_to_h(t, right_each_h)
    right_w = max(l_resized.shape[1], t_resized.shape[1])
    def pad_to_w(img, out_w):
        h, w = img.shape[:2]
        if w == out_w: return img
        pad = np.full((h, out_w - w, 3), bg_color, dtype=np.uint8)
        return np.concatenate([img, pad], axis=1)
    l_pad = pad_to_w(l_resized, right_w); t_pad = pad_to_w(t_resized, right_w)
    gap_v = np.full((gap, right_w, 3), bg_color, dtype=np.uint8)
    right_col = np.concatenate([l_pad, gap_v, t_pad], axis=0)
    gap_h = np.full((target_h, gap, 3), bg_color, dtype=np.uint8)
    composite = np.concatenate([f_resized, gap_h, right_col], axis=1)
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
        print(f"[sim_runner] encode frame error: {e}")

def empty_action_generator():
    return np.array([0,0,0, 0,0,0, 1.0], dtype=np.float32)

def receiver_thread(ws_url: str, q: queue.Queue, stop_evt: threading.Event):
    async def go():
        ws = None
        try:
            async with websockets.connect(ws_url, max_size=2**22) as ws:
                while not stop_evt.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.ConnectionClosed:
                        break
                    try:
                        frame = json.loads(msg)
                    except Exception:
                        continue
                    try:
                        q.put_nowait(frame)
                    except queue.Full:
                        try: q.get_nowait()
                        except queue.Empty: pass
                        try: q.put_nowait(frame)
                        except queue.Full: pass
        finally:
            try:
                if ws and not ws.closed:
                    await ws.close()
            except Exception:
                pass
    try:
        asyncio.run(go())
    except Exception:
        pass

# =========================
# 倒计时 + 预热 + 初始化对齐
# =========================
def run_countdown_preheat(env, q, img_queue, args, duration_s=3.0, avg_frames=30):
    tick = 1.0 / float(args.tick_hz)
    target_frames = max(1, int(round(duration_s * float(args.tick_hz))))

    i_count = 0
    quest_positions = []
    quest_quats_wxyz = []
    next_t = time.perf_counter()

    while i_count < target_frames:
        now = time.perf_counter()
        if now < next_t:
            time.sleep(min(0.001, next_t - now)); continue
        next_t += tick

        pkt = None
        try:
            while True:
                pkt = q.get_nowait()
        except queue.Empty:
            pass

        action = empty_action_generator()
        obs, _, _, _, _ = env.step(action)
        front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
        left_img  = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
        top_img   = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()

        secs_left = int(np.ceil((target_frames - i_count) / float(args.tick_hz)))
        combo = make_composite(front_img, left_img, top_img, target_h=720, gap=6)
        cv2.rectangle(combo, (12, 12), (260, 70), (0, 0, 0), thickness=-1)
        cv2.putText(combo, f"Starting in {secs_left}s", (18,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("front_view", cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("left_view",  cv2.cvtColor(left_img,  cv2.COLOR_RGB2BGR))
        cv2.imshow("top_view",   cv2.cvtColor(top_img,   cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        send_frame_to_queue(combo, img_queue)

        if pkt is not None and pkt.get("palm"):
            origin = np.array(pkt["palm"].get("origin"), dtype=float)
            quat_wxyz = np.array(pkt["palm"].get("quat"), dtype=float)  # hub 保证 wxyz
            quest_positions.append(origin * args.position_scale)
            quest_quats_wxyz.append(quat_wxyz)

        i_count += 1

    position_correction = None
    init_quest_R = None
    init_ee_R = None
    last_quest_pose = None

    if len(quest_positions) >= max(3, min(avg_frames, len(quest_positions))):
        quest_positions = np.stack(quest_positions[-avg_frames:], axis=0)
        quest_quats_wxyz = quest_quats_wxyz[-avg_frames:]
        avg_pos = quest_positions.mean(axis=0)
        avg_q_wxyz = average_quaternion_wxyz(quest_quats_wxyz)
        # to matrix
        quest_R = R.from_quat(quat_wxyz_to_xyzw(avg_q_wxyz)).as_matrix()

        ee_pos  = env.env.env.agent.tcp_pos.cpu().numpy()
        ee_quat = env.env.env.agent.tcp_quat.cpu().numpy()
        ee_R    = quat_to_R(ee_quat)  # 下游内置，已匹配 wxyz

        position_correction = avg_pos - ee_pos
        init_quest_R = quest_R
        init_ee_R    = ee_R
        last_quest_pose = np.concatenate([avg_pos, avg_q_wxyz[[1,2,3,0]]])  # 存 xyzw 不再使用，这里保持兼容

        try:
            env.start_record("front_camera")
        except Exception:
            pass
    else:
        print(f"[preheat] Warning: only {len(quest_positions)} frames collected, skip re-alignment")

    return {
        "i_after": max(args.prepare_frames, 0),
        "position_correction": position_correction,
        "init_quest_R": init_quest_R,
        "init_ee_R": init_ee_R,
        "last_quest_pose": last_quest_pose,
    }

# =========================
# 主程序
# =========================
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
    frame_thread = threading.Thread(
        target=lambda: asyncio.run(frame_sender(args.url.replace("/sub", ""), img_queue, stop_evt)),
        daemon=True
    )
    frame_thread.start()

    print(f"sim-worker started: tick={args.tick_hz}Hz")
    tick = 1.0 / float(args.tick_hz)
    next_t = time.perf_counter()
    last_tick_t = next_t

    # env
    print(f"Start building env {args.env_name}")
    gs.init(seed=None, precision='32', debug=False, eps=1e-12, logging_level=None,
            backend=gs.gpu, theme='dark', logger_verbose_time=False)

    env = gym.make(
        args.env_name,
        num_envs=1,
        obs_mode="rgbd",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        enable_shadow=True,
        show_viewer=False,
        show_world_frame=False,
        device=gs.device,
    )

    if args.RECORDING:
        env = RecordEpisodeWrapper(env, output_dir="trajectories", trajectory_name="teleoperation_trajectory")
    env.reset()

    # 预热对齐
    state = run_countdown_preheat(env, q, img_queue, args, duration_s=3.0)
    i = state["i_after"]
    position_correction = state["position_correction"]
    init_quest_R = state["init_quest_R"]
    init_ee_R = state["init_ee_R"]
    last_quest_pose = state["last_quest_pose"]

    # 显示窗口
    front_view_win_name = "front_view"
    left_view_win_name = "left_view"
    top_view_win_name = "top_view"
    cv2.namedWindow(front_view_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(front_view_win_name, 2 * args.screen_resolution[1] // 3, args.screen_resolution[1] // 2)
    cv2.moveWindow(front_view_win_name, args.screen_resolution[0] // 3 - args.screen_resolution[1] // 3, 0)
    cv2.namedWindow(left_view_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(left_view_win_name, 2 * args.screen_resolution[1] // 3, args.screen_resolution[1] // 2)
    cv2.moveWindow(left_view_win_name, 0, args.screen_resolution[1] // 2)
    cv2.namedWindow(top_view_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(top_view_win_name, 2 * args.screen_resolution[1] // 3, args.screen_resolution[1] // 2)
    cv2.moveWindow(top_view_win_name, args.screen_resolution[0] // 3, args.screen_resolution[1] // 2)

    if args.video_save_dir is not None:
        os.makedirs(os.path.join(args.video_save_dir, args.env_name), exist_ok=True)

    accident_steps = 0
    done = False

    # EMA 状态
    ema_pos = None
    ema_q_wxyz = None

    try:
        while True:
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(0.001, next_t - now)); continue
            dt = now - last_tick_t
            last_tick_t = now
            next_t = now + tick

            # 取最新一包
            pkt = None
            try:
                while True:
                    pkt = q.get_nowait()
            except queue.Empty:
                pass

            if pkt is None:
                # no packet
                action = empty_action_generator()
                obs, _, _, _, _ = env.step(action)
                front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
                left_img  = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
                top_img   = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()
                cv2.imshow(front_view_win_name, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
                cv2.imshow(left_view_win_name,  cv2.cvtColor(left_img,  cv2.COLOR_RGB2BGR))
                cv2.imshow(top_view_win_name,   cv2.cvtColor(top_img,   cv2.COLOR_RGB2BGR))
                send_frame_to_queue(make_composite(front_img,left_img,top_img,720,6), img_queue)
                if last_quest_pose is None:
                    i = 0
                continue

            # 有数据
            palm = pkt.get("palm")
            if palm is None or palm.get("origin") is None or palm.get("quat") is None:
                continue

            origin = np.array(palm["origin"], dtype=float) * float(args.position_scale)
            q_wxyz = np.array(palm["quat"], dtype=float)  # wxyz（hub已保证）

            # 离群点剔除
            if len(POS_WINDOW) > 0:
                if np.linalg.norm(origin - POS_WINDOW[-1]) > POS_JUMP_REJECT:
                    # 丢弃这帧位置
                    origin = POS_WINDOW[-1].copy()
                if axis_angle_between_q(q_wxyz, QUAT_WINDOW_WXYZ[-1]) > ANG_JUMP_REJECT:
                    q_wxyz = QUAT_WINDOW_WXYZ[-1].copy()

            POS_WINDOW.append(origin)
            QUAT_WINDOW_WXYZ.append(q_wxyz)

            # 滑窗均值
            quest_position = np.mean(np.stack(POS_WINDOW, axis=0), axis=0) if len(POS_WINDOW)>1 else origin
            quest_q_wxyz   = average_quaternion_wxyz(list(QUAT_WINDOW_WXYZ)) if len(QUAT_WINDOW_WXYZ)>1 else q_wxyz

            # EMA
            ema_pos = exp_smooth(ema_pos, quest_position, EMA_POS_ALPHA)
            ema_q_wxyz = average_quaternion_wxyz([ema_q_wxyz, quest_q_wxyz]) if ema_q_wxyz is not None else quest_q_wxyz
            quest_position = ema_pos
            quest_q_wxyz   = ema_q_wxyz

            quest_R = R.from_quat(quat_wxyz_to_xyzw(quest_q_wxyz)).as_matrix()

            ee_pos  = env.env.env.agent.tcp_pos.cpu().numpy()
            ee_quat = env.env.env.agent.tcp_quat.cpu().numpy()
            ee_R    = quat_to_R(ee_quat)

            i += 1
            if i == args.prepare_frames:
                # 对齐
                position_correction = quest_position - ee_pos
                init_quest_R = quest_R
                init_ee_R    = ee_R
                action = empty_action_generator()
                try: env.start_record("front_camera")
                except Exception: pass
                # 下一帧进入控制
                obs, _, _, _, _ = env.step(action)
                continue

            if i < args.prepare_frames:
                # 预热
                action = empty_action_generator()
                obs, _, _, _, _ = env.step(action)
            else:
                # 控制
                # 位置增量（带 offset）、死区+限幅
                position_action = quest_position - ee_pos - position_correction
                if np.linalg.norm(position_action) < POS_DEADBAND:
                    position_action[:] = 0
                position_action = clamp_vec(position_action, POS_STEP_MAX)

                # 姿态增量：与原逻辑相同（但下游四元数统一）
                R_action = init_ee_R @ init_quest_R.T @ quest_R @ ee_R.T
                euler_action = quat_to_xyz(R_to_quat(R_action.T), degrees=False)

                # 死区 + 限幅
                euler_action = np.where(np.abs(euler_action) < ANG_DEADBAND, 0.0, euler_action)
                e = np.asarray(euler_action).astype(np.float32)
                e = np.clip(e, -ANG_STEP_MAX, ANG_STEP_MAX)

                gripper_action = 1.0
                action = np.concatenate([position_action, e, [gripper_action]], axis=0).astype(np.float32)

                obs, _, _, _, _ = env.step(action)

            # 画面
            front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
            left_img  = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
            top_img   = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()
            cv2.imshow(front_view_win_name, cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR))
            cv2.imshow(left_view_win_name,  cv2.cvtColor(left_img,  cv2.COLOR_RGB2BGR))
            cv2.imshow(top_view_win_name,   cv2.cvtColor(top_img,   cv2.COLOR_RGB2BGR))
            send_frame_to_queue(make_composite(front_img,left_img,top_img,720,6), img_queue)

            if i > args.max_steps + args.prepare_frames:
                save_to_filename = next_name_by_count(f"{args.video_save_dir}/{args.env_name}_fail", ".mp4")
                try: env.stop_record("front_camera", save_to_filename=save_to_filename, fps=30)
                except Exception: pass
                env.reset()
                # 重新对齐
                state = run_countdown_preheat(env, q, img_queue, args, duration_s=3.0)
                i = state["i_after"]
                position_correction = state["position_correction"]
                init_quest_R = state["init_quest_R"]
                init_ee_R = state["init_ee_R"]
                last_quest_pose = state["last_quest_pose"]
                accident_steps = 0
                done = False
                continue

            # 键盘退出
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try: env.close()
        except Exception: pass
        stop_evt.set()
        t.join(timeout=3.0)
        sys.exit(0)
