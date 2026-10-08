import logging
from typing import Any, Sequence

import numpy as np
import torch
import torchvision.transforms.v2 as transforms

import robot_wm.utils.distributed as dist
from robot_wm.datasets.base import Transform
from robot_wm.datasets.utils import (
    random_wrist_mask,
    _get_T_delta,
    axisangle_to_rot_6d,
    sparse_history_inds,
)

logger = logging.getLogger(__name__)


def _get_state(episode: dict[str, Any]) -> np.ndarray:
    proprio_dict = episode["obs"]
    ee_states = proprio_dict["ee_states"]  # pos + axis_angle
    gripper_states = proprio_dict["gripper_states"]
    ee_states_rot6d = axisangle_to_rot_6d(ee_states[:, 3:6])
    ee_states_10d = np.concatenate(
        [ee_states[:, :3], ee_states_rot6d, gripper_states[:, None]], axis=-1
    ).astype(np.float32)
    return ee_states_10d


def _get_gt_action(episode: dict[str, Any]) -> np.ndarray:
    """Returns the robot action."""
    proprio_dict = episode["obs"]
    actions = proprio_dict["action"]
    return actions


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
    ee_state_10d = _get_state(episode)  # (T, 10)
    delta_10d = _get_delta_action_from_10d(ee_state_10d, abs_finger)
    return delta_10d  # (T, 10)


def _get_actions(episode: dict[str, Any], action_type: str = "action") -> np.ndarray:
    if "action" in action_type:
        action = _get_gt_action(episode)
    elif "delta-state" in action_type:
        action = _get_delta_action(episode, "abs_finger" in action_type)
    else:
        raise ValueError(f"invalid {action_type = }")

    if "camera" in action_type:
        camera_motion = np.zeros((action.shape[0], 9))
        action = np.concatenate([action, camera_motion], axis=-1)
    return action


class LiberoTransform(Transform):
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
        multiview=False,
        wrist_mask_prob: float = 0.0,
        # widen the HISTORY span: the gaps between the first `history_frames` sampled
        # frames are multiplied by this factor, so the model sees a longer stretch of
        # scene context (Ctrl-World spaces its history 1-2 s apart) while the FUTURE
        # frames keep the uniform chunk stride the latent packing/action pooling requires.
        history_frames: int = 0,
        history_stride_mult: int = 1,
    ):
        self._cameras = cameras
        self._multiview = multiview
        self._history_frames = history_frames
        self._history_stride_mult = max(1, int(history_stride_mult))
        self._wrist_mask_prob = wrist_mask_prob
        self._output_keys = output_keys
        self._sample_size = sample_size
        self._chunk_size = chunk_size
        self._action_type = action_type
        self._mean = mean
        self._std = std
        self._resize_to = resize_to
        self._rgb_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (T, C, H, W)
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
        # Same sampler as every other dataset, so short episodes are handled identically
        # (history padded at the start, future always real).
        return sparse_history_inds(
            self._sample_size, self._chunk_size, self._history_frames,
            self._history_stride_mult, total_frames, self._gen,
        )

    def __call__(self, episode: dict[str, Any]) -> dict[str, Any]:
        N = episode["obs"]["timestamp"]

        # sample indices
        inds, all_inds = self.sample_frame_inds(N)

        # clip indices
        inds_valid = inds.clip(min=0, max=N - 1)
        all_inds_valid = all_inds.clip(min=0, max=N - 1)

        # rgb
        if not self._multiview:
            camera_idx = torch.randint(
                0, len(self._cameras), (1,), generator=self._gen
            ).item()
            camera = self._cameras[camera_idx]
            rgb = episode["obs"][camera][inds_valid]  # H,W,C
            rgb = self._rgb_transform(self._to_tensor(rgb))
        else:
            all_rgbs = []
            for camera in self._cameras:
                rgb = episode["obs"][camera][:][inds_valid]  # H,W,C
                rgb = self._rgb_transform(self._to_tensor(rgb))
                all_rgbs.append(rgb)
            all_rgbs = random_wrist_mask(all_rgbs, self._cameras, self._wrist_mask_prob, self._gen)
            # views are stacked along width (the model encodes each view separately)
            rgb = torch.concatenate(all_rgbs, axis=-1)

        # mask
        mask = torch.tensor(inds < N, dtype=bool)

        # state
        state_10d = _get_state(episode)
        state_10d = state_10d[all_inds_valid]  # (T, D)
        state_10d = self._to_tensor(state_10d)
        state_10d = state_10d.view(self._sample_size, self._chunk_size, -1)

        # actions
        actions_7d = _get_actions(episode, action_type=self._action_type)
        actions_7d = actions_7d[all_inds_valid]  # (T, D)
        actions_7d = self._to_tensor(actions_7d)
        actions_7d = actions_7d.view(self._sample_size, self._chunk_size, -1)

        if "joint_states" in self._action_type:
            # joint states
            joint_states = episode["obs"]["joint_states"][all_inds_valid]
            delta_joint_states = np.zeros_like(joint_states)
            delta_joint_states[:-1] = joint_states[1:] - joint_states[:-1]
            delta_joint_states = self._to_tensor(delta_joint_states)
            delta_joint_states = delta_joint_states.view(
                self._sample_size, self._chunk_size, -1
            )
            actions = torch.concatenate([actions_7d, delta_joint_states], axis=-1)
        else:
            actions = actions_7d

        output = {
            "rgb": rgb,
            "mask": mask,
            "state": state_10d,
            "actions": actions,
        }
        return {k: output[k] for k in self._output_keys}

    def state_dict(self) -> dict[str, Any]:
        return {"_gen": self._gen.get_state()}

    def load_state_dict(self, state_dict):
        self._gen.set_state(state_dict["_gen"])
