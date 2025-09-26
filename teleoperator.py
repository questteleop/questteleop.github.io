import multiprocessing
import os
import time
from queue import Empty, Full
from typing import Optional

import cv2
import numpy as np
from loguru import logger
import pyrealsense2 as rs

from utils.dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from utils.dex_retargeting.retargeting_config import RetargetingConfig
from utils.hand_detector import HandDetector

import genesis as gs
import gymnasium as gym
from maniladder.envs import ManiLadderBaseEnv
from maniladder.utils.wrappers.RecordEpisodeWrapper import RecordEpisodeWrapper

from utils.filter import Filter

# from maniladder.utils.transform_utils import quaternion_to_euler
from genesis.utils.geom import quat_to_R, _np_euler_to_R, R_to_quat, quat_to_xyz
from utils.utils import next_name_by_count
from maniladder import ASSETS_DIR

def wrap_euler_angles(euler_angles):
    # Ensure each Euler angle is wrapped into the range (-pi, pi]
    wrapped_angles = np.mod(euler_angles + np.pi, 2 * np.pi) - np.pi
    
    return wrapped_angles

class Teleoperator:
    def __init__(
        self,
        env_name,
        env_type,
        camera_path=None,
        hand_type: str = "right",
        video_save_dir = "tele_videos",
        X_SCALE: float = 0.04,
        Y_SCALE: float = 2.0,
        Z_SCALE: float = 2.0,
        rotation_threshold: float = 0.08,
        rotation_scale: float = 0.5,
        prepare_frames: int = 50,
        robot_name: RobotName = RobotName.human_shadow, 
        retargeting_type: RetargetingType = RetargetingType.position,
        screen_resolution: tuple = (2560, 1440)
    ):
        hand_type_dict = {
            "right": HandType.right,
            "left": HandType.left,
            "dual_arm": HandType.right,  # use right hand for some functions
        }

        self.env_name = env_name
        self.env_type = env_type
        assert self.env_type in ["single_arm_gripper", "single_arm_dex", "dual_arm_gripper", "dual_arm_dex"], f"env_type {self.env_type} not supported."
        
        self.camera_path = camera_path
        self.config_path = get_default_config_path(robot_name, retargeting_type, hand_type_dict[hand_type])
        self.robot_dir = ASSETS_DIR / "hands"

        self.X_SCALE = X_SCALE
        self.Y_SCALE = Y_SCALE
        self.Z_SCALE = Z_SCALE
        self.rotation_threshold = rotation_threshold
        self.rotation_scale = rotation_scale
        self.prepare_frames = prepare_frames
        self.hand_type = hand_type
        self.screen_resolution = screen_resolution
        self.video_save_dir = video_save_dir

    def start(self):
        ctx = multiprocessing.get_context("spawn")
        start_evt = ctx.Event()  # gate start
        stop_evt = ctx.Event()   # signal a clean shutdown
        queue = ctx.Queue(maxsize=5)

        producer = ctx.Process(
            target=self.produce_frame,
            name="FrameProducer",
            args=(start_evt, stop_evt, queue, self.camera_path),
        )
        consumer = ctx.Process(
            target=self.teleoperate,
            name="Teleoperate",
            args=(start_evt, stop_evt, queue, str(self.robot_dir), str(self.config_path), self.env_name),
        )
        producer.start()
        consumer.start()

        # Let children proceed
        start_evt.set()
        
        try:
            # Wait for consumer to finish work naturally
            consumer.join()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping processes.")
        finally:
            # Ensure we request shutdown, and unblock any queue.get()
            stop_evt.set()
            try:
                queue.put_nowait(None)  # sentinel
            except Full:
                pass

            # Join with timeouts; if stuck, terminate as last resort
            for p in (producer, consumer):
                p.join(timeout=5)
            for p in (producer, consumer):
                if p.is_alive():
                    logger.warning(f"{p.name} still alive after timeout; terminating.")
                    p.terminate()
            for p in (producer, consumer):
                p.join(timeout=5)

            # Clean queue feeder thread in the parent
            try:
                queue.close()
                queue.join_thread()
            except Exception as e:
                logger.warning(f"Queue cleanup warning: {e}")

        print("done")

    def teleoperate(
        self,
        start_evt,
        stop_evt,
        queue: multiprocessing.Queue,
        robot_dir: str, 
        config_path: str, 
        env_name: str, 
        RECORDING: bool = False
    ):
        # Wait until the parent signals everything is ready
        start_evt.wait()

        # build retargeting

        if self.hand_type == "dual_arm":
            RetargetingConfig.set_default_urdf_dir(str(robot_dir))
            logger.info(f"Start retargeting with config {config_path}")
            override = dict(add_dummy_free_joint=False)
            left_config = RetargetingConfig.load_from_file(config_path,override=override)
            left_retargting = left_config.build()
            left_detector = HandDetector(hand_type="Left")

            RetargetingConfig.set_default_urdf_dir(str(robot_dir))
            logger.info(f"Start retargeting with config {config_path}")
            override = dict(add_dummy_free_joint=False)
            right_config = RetargetingConfig.load_from_file(config_path,override=override)
            right_retargting = right_config.build()
            right_detector = HandDetector(hand_type="Right")
        else:
            RetargetingConfig.set_default_urdf_dir(str(robot_dir))
            logger.info(f"Start retargeting with config {config_path}")
            override = dict(add_dummy_free_joint=False)
            config = RetargetingConfig.load_from_file(config_path,override=override)
            retargeting = config.build()            
            hand_type = "Right" if "right" in config_path.lower() else "Left"
            detector = HandDetector(hand_type=hand_type)

        # filter
        if self.hand_type == "dual_arm":
            left_point_filter = Filter()
            left_point_filter.reset()

            right_point_filter = Filter()
            right_point_filter.reset()
        else:
            point_filter = Filter()
            point_filter.reset()

        # env building
        logger.info(f"Start building env {env_name}")
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
            env_name,
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

        if RECORDING:
            env = RecordEpisodeWrapper(env, output_dir="trajectories", trajectory_name="teleoperation_trajectory")
        
        # try:
        #     viewer = env.unwrapped.scene.viewer._pyrender_viewer
        #     viewer.wait_until_initialized()
        # except AttributeError:
        #     logger.error("Error Encountered")
        #     env.close()
        #     return
        
        env.reset()

        # window showing
        win_name = "realtime_retargeting_demo"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 2 * self.screen_resolution[1] // 3, self.screen_resolution[1] // 2)
        cv2.moveWindow(win_name, 2 * self.screen_resolution[0] // 3, 0)  # position it
        
        front_view_win_name = "front_view"
        cv2.namedWindow(front_view_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(front_view_win_name, 2 * self.screen_resolution[1] // 3, self.screen_resolution[1] // 2)
        cv2.moveWindow(front_view_win_name, self.screen_resolution[0] // 3 - self.screen_resolution[1] // 3, 0)  # position it
        
        left_view_win_name = "left_view"
        cv2.namedWindow(left_view_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(left_view_win_name, 2 * self.screen_resolution[1] // 3, self.screen_resolution[1] // 2)
        cv2.moveWindow(left_view_win_name, 0, self.screen_resolution[1] // 2)  # position it
        
        top_view_win_name = "top_view"
        cv2.namedWindow(top_view_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(top_view_win_name, 2 * self.screen_resolution[1] // 3, self.screen_resolution[1] // 2)
        cv2.moveWindow(top_view_win_name, self.screen_resolution[0] // 3, self.screen_resolution[1] // 2)  # position it
        
        if self.video_save_dir is not None:
            assert isinstance(self.video_save_dir, str), "video_save_dir must be a string"
            os.makedirs(os.path.join(self.video_save_dir, env_name), exist_ok=True)
            env.start_record("front_camera")
        
        i = 0

        # the correction for the first frame
        if self.hand_type == "dual_arm":
            left_position_correction = None
            # left_init_hand_euler = None
            # left_init_ee_euler = None
            left_init_hand_R = None
            left_init_ee_R = None
            left_init_gripper_dis = None

            right_position_correction = None
            # right_init_hand_euler = None
            # right_init_ee_euler = None
            right_init_hand_R = None
            right_init_ee_R = None
            right_init_gripper_dis = None
        else:
            position_correction = None
            init_hand_R = None
            init_ee_R = None
            init_gripper_dis = None
            

        joint_pos = None
        left_joint_pos = None
        right_joint_pos = None
        
        try:
            while not stop_evt.is_set():
                # 0. Get a frame (or a sentinel)
                try:
                    item = queue.get(timeout=0.5)
                except Empty:
                    # Periodically check for stop; continue otherwise
                    continue

                if item is None:
                    # Sentinel received: exit cleanly
                    break

                bgr = item
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # (480, 640, 3)

                # 1. Hand Detection and 3D Pose Estimation
                if self.hand_type == "dual_arm":
                    _, left_joint_pos, left_keypoint_2d = left_detector.detect(rgb)
                    _, right_joint_pos, right_keypoint_2d = right_detector.detect(rgb)
                else:
                    _, joint_pos, keypoint_2d = detector.detect(rgb)

                # 2. Drawing skeleton
                if self.hand_type == "dual_arm":
                    detect_img = left_detector.draw_skeleton_on_image(bgr, left_keypoint_2d, style="default")
                    detect_img = right_detector.draw_skeleton_on_image(detect_img, right_keypoint_2d, style="white")
                else:
                    detect_img = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
                cv2.imshow(win_name, detect_img)

                # Allow user to quit
                if (cv2.waitKey(1) & 0xFF == ord("q")):
                    stop_evt.set()
                    # Try to unblock producer quickly
                    try:
                        queue.put_nowait(None)
                    except Full:
                        pass
                    break
                
                # 3. Retargeting
                if (self.hand_type != "dual_arm" and joint_pos is None) or (self.hand_type == "dual_arm" and (left_joint_pos is None or right_joint_pos is None)):
                    logger.warning(f"{self.hand_type} hand is not detected.")
                    if (self.hand_type == "dual_arm" and (left_point_filter.last_points is None or right_point_filter.last_points is None)) or (self.hand_type != "dual_arm" and point_filter.last_points is None):
                        # Safe idle action
                        if self.hand_type == "dual_arm":
                            left_gripper_action = 1.0
                            right_gripper_action = 1.0
                            action = {
                                "panda-0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, left_gripper_action]),
                                "panda-1": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, right_gripper_action]),
                            }
                            
                            if "dex" in self.env_type:
                                action = {
                                    "panda_leaphand-0": np.zeros(22, dtype=np.float32),
                                    "panda_leaphand-1": np.zeros(22, dtype=np.float32)
                                }
                                action["panda_leaphand-0"][6:22] = 0.0
                                action["panda_leaphand-1"][6:22] = 0.0
                        else:
                            gripper_action = 1.0
                            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action], dtype=np.float32)
                            if "dex" in self.env_type:
                                action = np.zeros(22, dtype=np.float32)
                                action[6:22] = 0.0

                        obs, reward, terminated, done, info = env.step(action)
                        front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
                        left_img = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
                        top_img = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()
                        cv2.imshow(front_view_win_name, cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
                        cv2.imshow(left_view_win_name, cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
                        cv2.imshow(top_view_win_name, cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB))

                        # here i not increase
                        continue
                    else:
                        # if it is a sudden vanish, we will use the last points
                        if self.hand_type == "dual_arm":
                            left_joint_pos = left_point_filter.last_points
                            right_joint_pos = right_point_filter.last_points
                        else:
                            joint_pos = point_filter.last_points

                if (self.hand_type == "dual_arm" and (left_joint_pos is not None and right_joint_pos is not None)) or (self.hand_type != "dual_arm" and joint_pos is not None):
                    # Filter and retarget
                    if self.hand_type == "dual_arm":
                        _, left_filtered = left_point_filter.filter(left_joint_pos)
                        _, right_filtered = right_point_filter.filter(right_joint_pos)
                        left_indices = left_retargting.optimizer.target_link_human_indices
                        right_indices = right_retargting.optimizer.target_link_human_indices
                        left_ref_value = left_filtered[left_indices, :]
                        right_ref_value = right_filtered[right_indices, :]
                        left_qpos = left_retargting.retarget(left_ref_value)
                        right_qpos = right_retargting.retarget(right_ref_value)

                        # get position
                        left_hand_position = left_ref_value[0].copy()
                        left_hand_position[0] *= self.X_SCALE
                        left_hand_position[1] *= self.Y_SCALE
                        left_hand_position[2] *= self.Z_SCALE

                        right_hand_position = right_ref_value[0].copy()
                        right_hand_position[0] *= self.X_SCALE
                        right_hand_position[1] *= self.Y_SCALE
                        right_hand_position[2] *= self.Z_SCALE

                        # get orientation
                        left_hand_euler = wrap_euler_angles(left_qpos[3:6])
                        right_hand_euler = wrap_euler_angles(right_qpos[3:6])
                        left_hand_R = _np_euler_to_R(left_hand_euler)
                        right_hand_R = _np_euler_to_R(right_hand_euler)

                        # get gripper
                        left_index_tip, left_thumb_tip = left_joint_pos[8], left_joint_pos[4]
                        left_finger_dist = float(np.linalg.norm(left_index_tip - left_thumb_tip))

                        right_index_tip, right_thumb_tip = right_joint_pos[8], right_joint_pos[4]
                        right_finger_dist = float(np.linalg.norm(right_index_tip - right_thumb_tip))

                        # get ee pose
                        ee_pos = env.env.env.agent.tcp_pos
                        if "dex" in self.env_type:
                            left_ee_pos = ee_pos['panda_leaphand-0'].cpu().numpy()
                            right_ee_pos = ee_pos['panda_leaphand-1'].cpu().numpy()
                        else:
                            left_ee_pos = ee_pos['panda-0'].cpu().numpy()
                            right_ee_pos = ee_pos['panda-1'].cpu().numpy()
                            
                        ee_quat = env.env.env.agent.tcp_quat
                        if "dex" in self.env_type:
                            # left_ee_euler = wrap_euler_angles(quaternion_to_euler(ee_quat['panda_leaphand-0']).cpu().numpy())
                            # right_ee_euler = wrap_euler_angles(quaternion_to_euler(ee_quat['panda_leaphand-1']).cpu().numpy())
                            left_ee_R = quat_to_R(ee_quat['panda_leaphand-0'].cpu().numpy())
                            right_ee_R = quat_to_R(ee_quat['panda_leaphand-1'].cpu().numpy())
                        else:
                            # left_ee_euler = wrap_euler_angles(quaternion_to_euler(ee_quat['panda-0']).cpu().numpy())
                            # right_ee_euler = wrap_euler_angles(quaternion_to_euler(ee_quat['panda-1']).cpu().numpy())
                            left_ee_R = quat_to_R(ee_quat['panda-0'].cpu().numpy())
                            right_ee_R = quat_to_R(ee_quat['panda-1'].cpu().numpy())

                        # 4. get finger joint pos
                        if "dex" in self.env_type:
                            left_hand_qpos = np.concatenate([left_qpos[6+2:6+14], left_qpos[6+19:6+21], left_qpos[6+22:6+24]])  # without global pos
                            right_hand_qpos = np.concatenate([right_qpos[6+2:6+14], right_qpos[6+19:6+21], right_qpos[6+22:6+24]])  # without global pos

                        if i <= self.prepare_frames:
                            left_position_correction = left_hand_position - left_ee_pos
                            # left_init_hand_euler = left_hand_euler
                            # left_init_ee_euler = left_ee_euler
                            left_init_hand_R = left_hand_R
                            left_init_ee_R = left_ee_R
                            left_init_gripper_dis = max(left_finger_dist, 1e-6)

                            right_position_correction = right_hand_position - right_ee_pos
                            # right_init_hand_euler = right_hand_euler
                            # right_init_ee_euler = right_ee_euler
                            right_init_hand_R = right_hand_R
                            right_init_ee_R = right_ee_R
                            right_init_gripper_dis = max(right_finger_dist, 1e-6)

                            action = {
                                "panda-0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, left_gripper_action]),
                                "panda-1": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, right_gripper_action]),
                            }
                            if "dex" in self.env_type:
                                action = {
                                    "panda_leaphand-0": np.zeros(22, dtype=np.float32),
                                    "panda_leaphand-1": np.zeros(22, dtype=np.float32)
                                }
                                action["panda_leaphand-0"][6:22] = left_hand_qpos
                                action["panda_leaphand-1"][6:22] = right_hand_qpos
                        else:
                            left_position_action = left_hand_position - left_ee_pos - left_position_correction
                            # left_euler_action = -wrap_euler_angles(left_hand_euler - left_init_hand_euler) + wrap_euler_angles(left_ee_euler - left_init_ee_euler)
                            left_R_action = left_init_ee_R @ left_init_hand_R.T @ left_hand_R @ left_ee_R.T
                            left_euler_action = quat_to_xyz(R_to_quat(left_R_action.T), degrees=False)
                            
                            left_gripper_ratio = left_finger_dist / (left_init_gripper_dis + 1e-6)
                            left_gripper_action = 1.0 if left_gripper_ratio > 0.8 else -1.0

                            right_position_action = right_hand_position - right_ee_pos - right_position_correction
                            # right_euler_action = -wrap_euler_angles(right_hand_euler - right_init_hand_euler) + wrap_euler_angles(right_ee_euler - right_init_ee_euler)
                            right_R_action = right_init_ee_R @ right_init_hand_R.T @ right_hand_R @ right_ee_R.T
                            right_euler_action = quat_to_xyz(R_to_quat(right_R_action.T), degrees=False)
                            right_gripper_ratio = right_finger_dist / (right_init_gripper_dis + 1e-6)
                            right_gripper_action = 1.0 if right_gripper_ratio > 0.8 else -1.0
                            
                            action = {
                                "panda-0": np.concatenate([left_position_action, left_euler_action, [left_gripper_action]]),
                                "panda-1": np.concatenate([right_position_action, right_euler_action, [right_gripper_action]]),
                            }
                            if "dex" in self.env_type:
                                action = {
                                    "panda_leaphand-0": np.zeros(22, dtype=np.float32),
                                    "panda_leaphand-1": np.zeros(22, dtype=np.float32)
                                }
                                action["panda_leaphand-0"][0:3] = left_position_action
                                action["panda_leaphand-0"][3:6] = left_euler_action
                                action["panda_leaphand-0"][6:22] = left_hand_qpos

                                action["panda_leaphand-1"][0:3] = right_position_action
                                action["panda_leaphand-1"][3:6] = right_euler_action
                                action["panda_leaphand-1"][6:22] = right_hand_qpos
                            
                            # to filter the small values of rotation
                            for k in action.keys():
                                small = np.abs(action[k][3:6]) < self.rotation_threshold
                                action[k][3:6][small] = 0.0

                    else:
                        _, filtered = point_filter.filter(joint_pos)
                        indices = retargeting.optimizer.target_link_human_indices
                        ref_value = filtered[indices, :]
                        qpos = retargeting.retarget(ref_value)

                        # get position
                        hand_position = ref_value[0].copy()
                        hand_position[0] *= self.X_SCALE
                        hand_position[1] *= self.Y_SCALE
                        hand_position[2] *= self.Z_SCALE

                        # get orientation
                        hand_euler = wrap_euler_angles(qpos[3:6])
                        R_hand = _np_euler_to_R(hand_euler)

                        # 3. get gripper
                        index_tip, thumb_tip = joint_pos[8], joint_pos[4]
                        finger_dist = float(np.linalg.norm(index_tip - thumb_tip))

                        # 4. get finger joint pos
                        if "dex" in self.env_type:
                            hand_qpos = np.concatenate([qpos[6+2:6+14], qpos[6+19:6+21], qpos[6+22:6+24]])  # without global pos
                            # TODO: add some offset to make this more suitable for leaphand

                        # get ee pose
                        ee_pos = env.env.env.agent.tcp_pos.cpu().numpy()
                        ee_quat = env.env.env.agent.tcp_quat.cpu().numpy()
                        ee_R = quat_to_R(ee_quat)
                        # ee_euler = wrap_euler_angles(quaternion_to_euler(ee_quat).cpu().numpy())
                        
                        if i <= self.prepare_frames:
                            position_correction = hand_position - ee_pos
                            # init_hand_euler = hand_euler
                            # init_ee_euler = ee_euler
                            init_hand_R = R_hand
                            init_ee_R = ee_R
                            init_gripper_dis = max(finger_dist, 1e-6)
                            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # stay unmoved before prepare_frames
                            
                            if "dex" in self.env_type:
                                action = np.zeros(22, dtype=np.float32)
                                action[6:22] = hand_qpos
                                
                        else:
                            position_action = hand_position - ee_pos - position_correction
                            R_action = init_ee_R @ init_hand_R.T @ R_hand @ ee_R.T
                            # in genesis, the ee frame is an inversion of the action given
                            euler_action = quat_to_xyz(R_to_quat(R_action.T), degrees=False)
                            
                            # euler_action = -wrap_euler_angles(hand_euler - init_hand_euler) + wrap_euler_angles(ee_euler - init_ee_euler)
                            gripper_ratio = finger_dist / (init_gripper_dis + 1e-6)
                            gripper_action = 1.0 if gripper_ratio > 0.8 else -1.0

                            action = np.concatenate([position_action, euler_action, [gripper_action]], axis=0).astype(np.float32)
                            if "dex" in self.env_type:
                                action = np.zeros(22, dtype=np.float32)
                                action[0:3] = position_action
                                action[3:6] = euler_action
                                action[6:22] = hand_qpos
                            # to filter the small values of rotation
                            small = np.abs(action[3:6]) < self.rotation_threshold
                            action[3:6][small] = 0.0

                if i <= self.prepare_frames:
                    print(f"Preparing for teleoperation... {i+1}/{self.prepare_frames}")
                else:
                    print(f"i: {i}, action: {action}")

                obs, reward, terminated, done, info = env.step(action)
                front_img = obs['sensor_data']['front_camera']['rgb'].cpu().numpy()
                left_img = obs['sensor_data']['left_camera']['rgb'].cpu().numpy()
                top_img = obs['sensor_data']['top_camera']['rgb'].cpu().numpy()
                cv2.imshow(front_view_win_name, cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
                cv2.imshow(left_view_win_name, cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
                cv2.imshow(top_view_win_name, cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB))
                i += 1

        finally:
            try:
                cv2.destroyWindow(win_name)
                cv2.destroyWindow(front_view_win_name)
                cv2.destroyWindow(left_view_win_name)
                cv2.destroyWindow(top_view_win_name)
            except Exception:
                pass
            try:
                env.stop_record("front_camera", save_to_filename=next_name_by_count(f"{self.video_save_dir}/{env_name}", ".mp4"), fps=30)
                env.close()
            except Exception:
                pass
            # Ensure shutdown is requested (idempotent)
            stop_evt.set()

    def produce_frame(self, start_evt, stop_evt, queue: multiprocessing.Queue, camera_path: Optional[str] = None):
        start_evt.wait()
        pipe = None
        cap = None
        using_rs = (camera_path == "rs")

        try:
            if using_rs:
                # Initialize RealSense pipeline; use non-blocking poll to avoid hang on stop
                pipe = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
                pipe.start(config)
            else:
                if camera_path is None:
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(camera_path)

            while not stop_evt.is_set():
                if using_rs:
                    frames = pipe.poll_for_frames()
                    if not frames:
                        # no frame available right now
                        time.sleep(0.001)
                        continue
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    image = np.asanyarray(color_frame.get_data())
                else:
                    if not cap.isOpened():
                        break
                    success, image = cap.read()
                    if not success:
                        time.sleep(0.005)
                        continue

                # Try to enqueue without blocking indefinitely; drop frames if full
                try:
                    queue.put(image, timeout=0.05)
                except Full:
                    # consumer is slow or gone; drop this frame
                    pass

                # keep a modest pace without hard sleep dependence
                time.sleep(1 / 120.0)

        finally:
            # Always attempt to unblock the consumer and signal stop
            stop_evt.set()
            try:
                queue.put_nowait(None)  # sentinel to unblock consumer's get()
            except Full:
                pass
            except Exception:
                pass

            if using_rs and pipe is not None:
                try:
                    pipe.stop()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

