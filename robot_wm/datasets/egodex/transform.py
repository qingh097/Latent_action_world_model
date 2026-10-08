"""Transform for EgoDex: samples a clip of the egocentric video and builds the hand action.

Action layout (157-d with the camera slot), per raw step:
    [  0: 69)  left hand : wrist pose delta (9 = pos3 + rot6d) + 20 finger joints (60)
    [ 69:138)  right hand: same
    [138:148)  reserved, always zero (keeps the layout of the released decoder head)
    [148:157)  camera slot: ABSOLUTE head-camera pose (pos3 + rot6d); MultiDataset
               re-references it to the last history frame
With ``abs_finger`` the finger channels hold the next step's joint positions relative to
the wrist; otherwise per-step deltas.
"""

import logging
from typing import Any, Sequence

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F

import robot_wm.utils.distributed as dist
from robot_wm.datasets.base import Transform
from robot_wm.datasets.utils import _get_T_delta, sparse_history_inds

logger = logging.getLogger(__name__)

EGODEX_ACTION_TYPES = [
    "delta-state",
    "delta-state+camera",
    "delta-state+camera+abs_finger",
]
HAND_STATE_DIM = 9 + 20 * 3  # wrist pose + 20 finger joint positions
RESERVED_DIM = 10


def _hand_delta(state: np.ndarray, abs_finger=False) -> np.ndarray:
    """(T, 69) wrist pose + finger joints -> (T, 69) wrist delta + finger action."""
    finger_pose = state[:, 9:]
    finger_delta = np.zeros_like(finger_pose)
    if abs_finger:
        rel = (finger_pose.reshape(-1, 20, 3) - state[:, None, :3]).reshape(-1, 20 * 3)
        finger_delta[:-1] = rel[1:]  # finger pose relative to the wrist, next step
    else:
        finger_delta[:-1] = finger_pose[1:] - finger_pose[:-1]
    return np.concatenate([_get_T_delta(state[:, :9]), finger_delta], axis=-1)


class ResizeThenCenterCrop:
    """Resize preserving aspect ratio until both sides cover the target, then center crop."""

    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, img):
        orig_height, orig_width = img.shape[-2], img.shape[-1]
        scale = max(self.target_height / orig_height, self.target_width / orig_width)
        new_size = (int(round(orig_height * scale)), int(round(orig_width * scale)))
        img = F.resize(img, new_size, max_size=None,
                       interpolation=transforms.InterpolationMode.BICUBIC, antialias="warn")
        return F.center_crop(img, (self.target_height, self.target_width))


class EgoDexTransform(Transform):
    def __init__(
        self,
        cameras: Sequence[str],
        output_keys: Sequence[str],
        sample_size: int,
        chunk_size: int = 1,
        action_type: str = "delta-state+camera+abs_finger",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        resize_to=None,
        num_views=None,
        history_frames: int = 0,
        history_stride_mult: int = 1,
        seed: int = 0,
    ):
        assert action_type in EGODEX_ACTION_TYPES, action_type
        assert list(cameras) == ["ego_centric_image"], cameras
        self._cameras = list(cameras)
        self._output_keys = output_keys
        self._sample_size = sample_size
        self._chunk_size = chunk_size
        self._action_type = action_type
        self._num_views = num_views
        # Ctrl-World style sparse history (see sparse_history_inds); defaults are a no-op.
        self._history_frames = history_frames
        self._history_stride_mult = max(1, int(history_stride_mult))
        self._rgb_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # (T, C, H, W)
                ResizeThenCenterCrop(*resize_to) if resize_to is not None
                else transforms.Lambda(lambda x: x),
                transforms.Normalize(mean=mean, std=std, inplace=True) if mean is not None
                else transforms.Lambda(lambda x: x.clamp(0, 1)),
            ]
        )
        self._gen = torch.Generator()
        self._gen.manual_seed(dist.get_global_rank() + seed if dist.is_initialized() else seed)

    @staticmethod
    def _get_state(obs: dict[str, Any]) -> np.ndarray:
        """(T, 148): left wrist + fingers, right wrist + fingers, reserved zeros."""
        state = np.concatenate(
            [
                obs["ekf_hand_pose_left/WristState"],
                obs["ekf_hand_pose_left/JointAngles"],
                obs["ekf_hand_pose_right/WristState"],
                obs["ekf_hand_pose_right/JointAngles"],
            ],
            axis=-1,
        )
        return np.concatenate([state, np.zeros((state.shape[0], RESERVED_DIM))], axis=-1)

    def _get_actions(self, obs: dict[str, Any], state: np.ndarray) -> np.ndarray:
        abs_finger = "abs_finger" in self._action_type
        left = _hand_delta(state[:, :HAND_STATE_DIM], abs_finger)
        right = _hand_delta(state[:, HAND_STATE_DIM:2 * HAND_STATE_DIM], abs_finger)
        action = np.concatenate([left, right, np.zeros((len(state), RESERVED_DIM))], axis=1)
        if "camera" in self._action_type:
            # absolute head-camera pose; MultiDataset re-references it
            action = np.concatenate([action, obs["camera"]], axis=-1)
        return action

    def __call__(self, episode: dict[str, Any]) -> dict[str, Any]:
        obs = episode["episode_data"]["observation"]
        state = self._get_state(obs)
        actions = self._get_actions(obs, state)
        video = obs["ego_centric_image"]

        T = len(state)
        inds, all_inds = sparse_history_inds(
            self._sample_size, self._chunk_size, self._history_frames,
            self._history_stride_mult, T, self._gen,
        )
        # Indices outside the episode are clipped to its ends.
        inds = np.clip(inds, 0, T - 1)
        all_inds = np.clip(all_inds, 0, T - 1)

        rgb = self._rgb_transform(torch.from_numpy(np.ascontiguousarray(video[np.clip(inds, 0, len(video) - 1)])).float())
        if self._num_views is not None and self._num_views > 1:
            # ego view + zero-filled slots for the wrist views EgoDex does not have
            rgb = torch.concatenate([rgb] + [torch.zeros_like(rgb)] * (self._num_views - 1), dim=-1)
        state = torch.from_numpy(state[all_inds]).float().reshape(self._sample_size, self._chunk_size, -1)
        actions = torch.from_numpy(actions[all_inds]).float().reshape(self._sample_size, self._chunk_size, -1)
        mask = torch.ones(self._sample_size, dtype=bool)

        output = {"rgb": rgb, "mask": mask, "state": state, "actions": actions}
        return {k: v for k, v in output.items() if k in self._output_keys}

    def state_dict(self) -> dict[str, Any]:
        return {"_gen": self._gen.get_state()}

    def load_state_dict(self, state_dict):
        self._gen.set_state(state_dict["_gen"])
