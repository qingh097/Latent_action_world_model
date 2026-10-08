import logging
import os
from contextlib import redirect_stderr
from typing import Any, Sequence

import av
import numpy as np
import torch
from robot_wm.datasets.utils import random_wrist_mask, pad_to_num_views
import torchvision.transforms.v2 as transforms

try:
    from torchcodec.decoders import VideoDecoder
    _TORCHCODEC_AVAILABLE = True
except Exception:
    _TORCHCODEC_AVAILABLE = False

import robot_wm.utils.distributed as dist
from robot_wm.datasets.base import Transform
from robot_wm.datasets.utils import sparse_history_inds
from robot_wm.datasets.utils import _get_T_delta, quat_to_rot_mat, rot_mat_to_rot_6d

logger = logging.getLogger(__name__)

def scale_gripper_state_0_to_1(
    gripper_state, min_state: float = 0.0, max_state: float = 1.0
):
    """Convert the gripper_state to values between 0 and 1."""
    if gripper_state.shape[1] == 2:
        gripper_type = "PARALLEL_GRIPPER"
        gripper_state_binary = np.clip(gripper_state, min_state, max_state)
        gripper_state_binary = (gripper_state_binary - min_state) / (
            max_state - min_state
        )
    elif gripper_state.shape[1] == 12:
        gripper_type = "DEXHAND_GRIPPER"
        gripper_state_binary = None
    else:
        gripper_type = "UNKNOWN_GRIPPER"
        gripper_state_binary = None

    return gripper_state_binary, gripper_type


def _get_state(episode: dict[str, Any]) -> np.ndarray:
    proprio_dict = episode["proprio_stats"]

    # get position and orientation state for both arms
    left_xyz, right_xyz = proprio_dict["state_end_position"].transpose(1, 0, 2)
    left_quat, right_quat = proprio_dict["state_end_orientation"].transpose(1, 0, 2)
    # get gripper state
    gripper_state_binary, flag_imp = scale_gripper_state_0_to_1(
        gripper_state=proprio_dict["state_effector_position"],
        min_state=34.0,
        max_state=125.0,
    )
    if flag_imp == "PARALLEL_GRIPPER":
        assert gripper_state_binary.shape[1] == 2
        left_gripper = gripper_state_binary[:, 0]
        right_gripper = gripper_state_binary[:, 1]
    else:
        raise ValueError("gripper not implemented")

    left_rot = quat_to_rot_mat(left_quat)  # xyzw
    right_rot = quat_to_rot_mat(right_quat)

    left_T = np.tile(np.eye(4)[None, :, :], (left_xyz.shape[0], 1, 1))
    right_T = np.tile(np.eye(4)[None, :, :], (right_xyz.shape[0], 1, 1))
    left_T[:, :3, 3] = left_xyz
    right_T[:, :3, 3] = right_xyz
    left_T[:, :3, :3] = left_rot
    right_T[:, :3, :3] = right_rot

    base_xyz = proprio_dict["state_robot_position"]
    base_quat = proprio_dict["state_robot_orientation"]
    zero_norm_indices = np.where(np.linalg.norm(base_quat, axis=1) == 0)[0]
    if len(zero_norm_indices) > 0:
        base_quat[zero_norm_indices] = np.array([0, 0, 0, 1])
    base_rot = quat_to_rot_mat(base_quat)
    base_T = np.tile(np.eye(4)[None, :, :], (base_xyz.shape[0], 1, 1))
    base_T[:, :3, 3] = base_xyz
    base_T[:, :3, :3] = base_rot

    left_wT = base_T @ left_T
    right_wT = base_T @ right_T
    left_xyz = left_wT[:, :3, 3]
    right_xyz = right_wT[:, :3, 3]
    left_rot_6d = rot_mat_to_rot_6d(left_wT[:, :3, :3])
    right_rot_6d = rot_mat_to_rot_6d(right_wT[:, :3, :3])

    # stack state
    left_10d = np.hstack((left_xyz, left_rot_6d, left_gripper[:, np.newaxis]))
    right_10d = np.hstack((right_xyz, right_rot_6d, right_gripper[:, np.newaxis]))

    combined_20d = np.hstack((left_10d, right_10d)).astype(np.float32)
    return combined_20d


def _get_gt_action(episode: dict[str, Any]) -> np.ndarray:
    """Returns the robot action."""
    proprio_dict = episode["proprio_stats"]

    # get position and orientation actions for both arms
    left_xyz, right_xyz = proprio_dict["action_end_positon"].transpose(1, 0, 2)
    left_quat, right_quat = proprio_dict["action_end_orientation"].transpose(1, 0, 2)

    gripper_state_binary, flag_imp = scale_gripper_state_0_to_1(
        gripper_state=proprio_dict["action_effector_position"],
        min_state=0.0,
        max_state=1.0,
    )
    if flag_imp == "PARALLEL_GRIPPER":
        assert gripper_state_binary.shape[1] == 2
        left_gripper = gripper_state_binary[:, 0]
        right_gripper = gripper_state_binary[:, 1]
    else:
        raise ValueError("gripper not implemented")

    left_rot = quat_to_rot_mat(left_quat)  # xyzw
    right_rot = quat_to_rot_mat(right_quat)

    left_T = np.tile(np.eye(4)[None, :, :], (left_xyz.shape[0], 1, 1))
    right_T = np.tile(np.eye(4)[None, :, :], (right_xyz.shape[0], 1, 1))
    left_T[:, :3, 3] = left_xyz
    right_T[:, :3, 3] = right_xyz
    left_T[:, :3, :3] = left_rot
    right_T[:, :3, :3] = right_rot
    left_rot_6d = rot_mat_to_rot_6d(left_T[:, :3, :3])
    right_rot_6d = rot_mat_to_rot_6d(right_T[:, :3, :3])

    # stack actions
    left_10d = np.hstack((left_xyz, left_rot_6d, left_gripper[:, np.newaxis]))
    right_10d = np.hstack((right_xyz, right_rot_6d, right_gripper[:, np.newaxis]))
    combined_20d = np.hstack((left_10d, right_10d)).astype(np.float32)
    return combined_20d


def _get_delta_action_from_10d(ee_10d: np.ndarray, abs_finger) -> np.ndarray:
    vec_9d = ee_10d[:, 0:9]
    delta_9d = _get_T_delta(vec_9d)
    gripper = ee_10d[:, 9:10]
    delta_gripper = np.zeros_like(gripper)
    if abs_finger:
        delta_gripper[:-1] = gripper[1:]
    else:
        delta_gripper[:-1] = gripper[1:] - gripper[:-1]

    delta_action_10d = np.hstack((delta_9d, delta_gripper))
    return delta_action_10d


def _get_delta_action(episode: dict[str, Any], abs_finger) -> np.ndarray:
    ee_state_20d = _get_state(episode)  # (T, 20)
    left_10d = ee_state_20d[:, 0:10]
    right_10d = ee_state_20d[:, 10:20]
    left_delta_10d = _get_delta_action_from_10d(left_10d, abs_finger)
    right_delta_10d = _get_delta_action_from_10d(right_10d, abs_finger)
    return np.hstack((left_delta_10d, right_delta_10d))  # (T, 20)


def extrinsic_to_matrix(extrinsic):
    R = np.array(extrinsic["rotation_matrix"])  # shape (3,3)
    t = np.array(extrinsic["translation_vector"])  # shape (3,)

    # Create a 4x4 identity matrix
    T = np.eye(4)
    # Fill in rotation matrix
    T[:3, :3] = R
    # Fill in translation vector
    T[:3, 3] = t
    return T


def _load_camera_extrinsics(episode, camera):
    if "head" in camera:
        camera_params_file = "head_extrinsic_params_aligned"
    else:
        raise NotImplementedError
    camera_params = episode["camera_params"][camera_params_file]
    matrices = [extrinsic_to_matrix(item["extrinsic"]) for item in camera_params]
    # Stack into a single numpy array of shape (T, 4, 4)
    matrices_array = np.stack(matrices)
    # convert to 9d
    rot_6d = rot_mat_to_rot_6d(matrices_array[:, :3, :3])
    camera_motion_9d = np.hstack((matrices_array[:, :3, 3], rot_6d))
    return camera_motion_9d


def _get_delta_joint(episode: dict[str, Any]) -> np.ndarray:
    """Delta of the 14d (7/arm) joint positions; matches the delta-state convention."""
    j = np.asarray(episode["proprio_stats"]["state_joint_positon"], dtype=np.float32)
    dj = np.zeros_like(j)
    dj[:-1] = j[1:] - j[:-1]
    return dj


def _get_actions(episode: dict[str, Any], action_type: str = "action") -> np.ndarray:
    if "action" in action_type:
        action = _get_gt_action(episode)
    elif "delta-state" in action_type:
        action = _get_delta_action(episode, "abs_finger" in action_type)
    else:
        raise ValueError(f"invalid {action_type = }")

    # agibot records joint states -> decode delta joint pose via the 6_joint_states head.
    # Component order MUST match action_type_split: [ee/wrist, joint_states, camera].
    action = np.concatenate([action, _get_delta_joint(episode)], axis=-1)

    if "camera" in action_type:
        camera_motion = _load_camera_extrinsics(episode, "head_color")
        action = np.concatenate([action, camera_motion], axis=-1)
    return action


class AgibotTransform(Transform):
    def __init__(
        self,
        cameras: Sequence[str],
        output_keys: Sequence[str],
        sample_size: int,
        chunk_size: int = 1,
        action_type: str = "delta-state",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        resize_to=None,
        seed=0,
        multiview: bool = False,
        wrist_mask_prob: float = 0.0,
        num_views=None,
        history_frames: int = 0,
        history_stride_mult: int = 1,
    ):
        self._cameras = list(cameras)
        self._multiview = multiview
        self._wrist_mask_prob = wrist_mask_prob
        # Ctrl-World style sparse history (see sparse_history_inds); defaults are a no-op.
        self._history_frames = history_frames
        self._history_stride_mult = max(1, int(history_stride_mult))
        self._num_views = num_views
        self._output_keys = output_keys
        self._sample_size = sample_size
        self._chunk_size = chunk_size
        self._action_type = action_type
        self._mean = mean
        self._std = std
        self._resize_to = resize_to
        self._rgb_transform = transforms.Compose(
            [
                (
                    transforms.Resize(tuple(int(x) for x in resize_to))
                    if resize_to is not None
                    else transforms.Lambda(lambda x: x)
                ),  # resize
                (
                    transforms.Normalize(mean=mean, std=std, inplace=True)
                    if mean is not None
                    else transforms.Lambda(lambda x: x)
                ),  # normalize
            ]
        )
        self._gen = torch.Generator()
        self._seed = seed
        if dist.is_initialized():
            self._seed = dist.get_global_rank() + seed
        self._gen.manual_seed(self._seed)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).float()

    def sample_frame_inds(self, total_frames):
        return sparse_history_inds(
            self._sample_size, self._chunk_size, self._history_frames,
            self._history_stride_mult, total_frames, self._gen,
        )

    def _decode_camera(self, video_path, inds_valid, all_inds_valid, N):
        """Decode frames at inds_valid from one camera mp4 (PyAV). Returns [T,C,H,W] float in [0,1]."""
        if _TORCHCODEC_AVAILABLE:
            with open(os.devnull, "w") as f, redirect_stderr(f):
                decoder = VideoDecoder(video_path, device="cpu")
            return decoder.get_frames_at(inds_valid).data.float() / 255
        start_frame = int(all_inds_valid[0])
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        time_base = float(stream.time_base)
        seek_pts = int(max(0.0, (start_frame - 1) / fps) / time_base)
        container.seek(seek_pts, stream=stream)
        # Keep only the frames actually asked for, keyed by absolute index. With sparse
        # history the requested frames span far more than sample_size*chunk_size positions,
        # so collecting a fixed-length run of frames and indexing into it ran off the end;
        # a dict also tolerates the repeated indices that clipping can produce.
        wanted = {int(i) for i in inds_valid}
        last_wanted = max(wanted)
        frames, current_pos = {}, None
        for frame in container.decode(stream):
            current_pos = round(frame.pts * time_base * fps) if current_pos is None else current_pos + 1
            if current_pos in wanted:
                frames[current_pos] = frame.to_ndarray(format="rgb24")
            if current_pos >= last_wanted or len(frames) == len(wanted):
                break
        container.close()
        if len(frames) < len(wanted):  # short/truncated video: fall back to the last frame
            last = frames[max(frames)] if frames else np.zeros((1, 1, 3), dtype=np.uint8)
            for i in wanted:
                frames.setdefault(i, last)
        selected = np.stack([frames[int(i)] for i in inds_valid])
        return torch.from_numpy(np.ascontiguousarray(selected)).permute(0, 3, 1, 2).float() / 255

    def __call__(self, episode: dict[str, Any]) -> dict[str, Any]:
        N = len(episode["proprio_stats"]["timestamp"])

        # sample indices
        inds, all_inds = self.sample_frame_inds(N)

        # clip indices
        inds_valid = inds.clip(min=0, max=N - 1)
        all_inds_valid = all_inds.clip(min=0, max=N - 1)

        # rgb (multiview-aware: ego + wrists, optional wrist masking)
        rgbs, names = [], []
        for cam in self._cameras:
            r = self._decode_camera(episode["video_path"][cam], inds_valid, all_inds_valid, N)
            rgbs.append(self._rgb_transform(r))
            names.append(cam)
        if self._multiview:
            rgbs, names = pad_to_num_views(rgbs, names, self._num_views)
            rgbs = random_wrist_mask(rgbs, names, self._wrist_mask_prob, self._gen)
            rgb = torch.concatenate(rgbs, dim=-1)  # T, C, H, n*W
        else:
            rgb = rgbs[0]

        # mask
        mask = torch.tensor(inds < N, dtype=bool)

        # state
        state_20d = _get_state(episode)
        state_20d = state_20d[all_inds_valid]  # (T, D)
        state_20d = self._to_tensor(state_20d)
        state_20d = state_20d.view(self._sample_size, self._chunk_size, -1)

        # actions
        actions_20d = _get_actions(episode, action_type=self._action_type)
        actions_20d = actions_20d[all_inds_valid]  # (T, D)
        actions_20d = self._to_tensor(actions_20d)
        actions_20d = actions_20d.view(self._sample_size, self._chunk_size, -1)

        output = {
            "rgb": rgb,
            "mask": mask,
            "state": state_20d,
            "actions": actions_20d,
        }
        return {k: output[k] for k in self._output_keys}

    def state_dict(self) -> dict[str, Any]:
        return {"_gen": self._gen.get_state()}

    def load_state_dict(self, state_dict):
        self._gen.set_state(state_dict["_gen"])
