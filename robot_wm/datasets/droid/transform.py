"""Transform for DROID: samples a clip, builds the action vector, and width-stacks the views.

Both DROID loaders (TFDS and LeRobot) hand this transform the same ``episode_data`` dict.

References:
    https://droid-dataset.github.io/droid/the-droid-dataset.html#-dataset-schema
"""

import logging
from typing import Any, Sequence

import numpy as np
import torch
import torchvision.transforms.v2 as transforms

import robot_wm.utils.distributed as dist
from robot_wm.datasets.base import Transform
from robot_wm.datasets.utils import (
    _get_T_delta,
    euler_to_rot_6d,
    pad_to_num_views,
    random_wrist_mask,
    sparse_history_inds,
)

logger = logging.getLogger(__name__)

DROID_RGB_CAMERAS = [
    "wrist_image_left",
    "exterior_image_1_left",
    "exterior_image_2_left",
]

DROID_ACTION_TYPES = [
    "action",
    "action+camera",
    "delta-action",
    "delta-action+camera",
    "state",
    "state+camera",
    "delta-state",
    "delta-state+camera",
    "delta-state+camera+abs_finger",
]


def _get_action(episode: dict[str, Any]) -> np.ndarray:
    """The commanded action as xyz + rot6d + gripper (10-d)."""
    action = episode["episode_data"]["action"]  # xyz, euler angles, gripper position
    rot_6d = euler_to_rot_6d(action[:, 3:6])
    return np.concatenate([action[:, :3], rot_6d, action[:, 6:]], axis=1)


def _get_delta(values: np.ndarray, abs_finger=False, values_type="state") -> np.ndarray:
    """Per-step end-effector delta of a (T, 10) xyz + rot6d + gripper sequence. With
    ``abs_finger`` the gripper channel holds the next absolute value instead of a delta."""
    gripper_pos = values[:, 9:]
    ee_delta = _get_T_delta(values[:, :9])
    gripper_delta = np.zeros_like(gripper_pos)
    if abs_finger:
        if values_type == "state":
            gripper_delta[:-1] = gripper_pos[1:]
        else:
            gripper_delta = gripper_pos
    else:
        gripper_delta[:-1] = gripper_pos[1:] - gripper_pos[:-1]
    return np.concatenate([ee_delta, gripper_delta], axis=1)


class DroidTransform(Transform):
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
        history_frames: int = 0,
        history_stride_mult: int = 1,
        num_views=None,
    ):
        assert action_type in DROID_ACTION_TYPES, action_type
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

        self._rgb_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (T, C, H, W)
                transforms.Resize(tuple(int(x) for x in resize_to))
                if resize_to is not None
                else transforms.Lambda(lambda x: x),  # resize
                transforms.Normalize(mean=mean, std=std, inplace=True)
                if mean is not None
                else transforms.Lambda(lambda x: x.clamp(0, 1)),  # normalize
            ]
        )

        self._gen = torch.Generator()
        self._seed = dist.get_global_rank() + seed if dist.is_initialized() else seed
        self._gen.manual_seed(self._seed)

    @staticmethod
    def _to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x)).float()

    @staticmethod
    def _get_state(episode: dict[str, Any]) -> np.ndarray:
        """The robot state as xyz + rot6d + gripper (10-d)."""
        obs = episode["episode_data"]["observation"]
        cartesian_pos = obs["cartesian_position"]  # xyz + euler angles
        rot_6d = euler_to_rot_6d(cartesian_pos[:, 3:6])
        gripper_pos = obs["gripper_position"]
        if gripper_pos.ndim == 1:
            gripper_pos = gripper_pos[:].reshape(cartesian_pos.shape[0], -1)
        return np.concatenate([cartesian_pos[:, :3], rot_6d, gripper_pos], axis=1)

    def _get_actions(self, episode: dict[str, Any]) -> np.ndarray:
        action_type = self._action_type
        abs_finger = "abs_finger" in action_type
        if "action" in action_type:
            action = _get_action(episode)
            if "delta" in action_type:
                action = _get_delta(action, abs_finger=abs_finger, values_type="action")
        else:
            action = self._get_state(episode)
            if "delta" in action_type:
                action = _get_delta(action, abs_finger=abs_finger)

        if "camera" in action_type:
            # The camera slot carries the EGO / THIRD-PERSON camera's motion, nothing else.
            # DROID's exterior camera is fixed for the duration of an episode, so this is
            # zero -- deliberately NOT the end-effector pose, which belongs to the first
            # half of the latent action and is decoded by the "ee" head.
            action = np.concatenate([action, np.zeros((action.shape[0], 9))], axis=-1)
        return action

    def __call__(self, episode: dict[str, Any]) -> dict[str, Any]:
        state = self._get_state(episode)
        actions = self._get_actions(episode)
        if self._multiview:
            # every configured camera, in config order (ego/exterior first, then wrist), so
            # the stacked grid stays consistent with rgb_pos_embed
            cameras = self._cameras
        else:
            idx = torch.randint(0, len(self._cameras), (1,), generator=self._gen).item()
            cameras = [self._cameras[idx]]
        videos = []
        for camera in cameras:
            assert camera in DROID_RGB_CAMERAS, camera
            video = episode["episode_data"]["observation"][camera]
            assert video.shape[3] == 3, f"Invalid {camera = }, {video.shape = }"
            assert len(video) == len(state)
            videos.append(video)
        assert len(state) == len(actions)

        T = len(state)
        inds, all_inds = sparse_history_inds(
            self._sample_size, self._chunk_size, self._history_frames,
            self._history_stride_mult, T, self._gen,
        )
        # Indices outside the episode are clipped to its ends.
        inds = np.clip(inds, 0, T - 1)
        all_inds = np.clip(all_inds, 0, T - 1)

        rgbs = [self._rgb_transform(self._to_tensor(video[inds])) for video in videos]  # (T, C, H, W)
        state = self._to_tensor(state[all_inds]).reshape(self._sample_size, self._chunk_size, -1)
        actions = self._to_tensor(actions[all_inds]).reshape(self._sample_size, self._chunk_size, -1)
        mask = torch.ones(self._sample_size, dtype=bool)

        rgbs, names = pad_to_num_views(rgbs, list(cameras), self._num_views)
        rgbs = random_wrist_mask(rgbs, names, self._wrist_mask_prob, self._gen)
        rgb = torch.concatenate(rgbs, dim=-1)  # (T, C, H, n*W)

        output = {"rgb": rgb, "mask": mask, "state": state, "actions": actions}
        return {k: v for k, v in output.items() if k in self._output_keys}

    def state_dict(self) -> dict[str, Any]:
        return {"_gen": self._gen.get_state()}

    def load_state_dict(self, state_dict):
        self._gen.set_state(state_dict["_gen"])
