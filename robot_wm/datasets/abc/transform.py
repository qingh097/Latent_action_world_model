"""Transform for ABC-130k: samples frames, builds bimanual actions, multiview
width-stack of [top(ego), left_wrist, right_wrist] with optional wrist masking.

The camera slot of the action vector is zero: the top camera is fixed.
"""
import logging
from typing import Any, Sequence

import numpy as np
import torch
import torchvision.transforms.v2 as transforms

import robot_wm.utils.distributed as dist
from robot_wm.datasets.base import Transform
from robot_wm.datasets.utils import pad_to_num_views, random_wrist_mask, sparse_history_inds

logger = logging.getLogger(__name__)

class ABCTransform(Transform):
    def __init__(
        self,
        cameras: Sequence[str],
        output_keys: Sequence[str],
        sample_size: int,
        chunk_size: int = 1,
        action_type: str = "action",
        mean=None,
        std=None,
        resize_to=None,
        multiview: bool = False,
        wrist_mask_prob: float = 0.0,
        num_views=None,
        history_frames: int = 0,
        history_stride_mult: int = 1,
        seed: int = 0,
        **kwargs,
    ):
        self._cameras = list(cameras)
        self._output_keys = output_keys
        self._sample_size = sample_size
        self._chunk_size = chunk_size
        self._action_type = action_type
        self._multiview = multiview
        self._wrist_mask_prob = wrist_mask_prob
        self._num_views = num_views
        # Ctrl-World style sparse history (see sparse_history_inds); defaults are a no-op.
        self._history_frames = history_frames
        self._history_stride_mult = max(1, int(history_stride_mult))
        tfs = [
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # T,H,W,C -> T,C,H,W
            transforms.Lambda(lambda x: x / 255.0),
        ]
        if resize_to is not None:
            tfs.append(transforms.Resize(resize_to))
        if mean is not None and std is not None:
            tfs.append(transforms.Normalize(mean, std))
        self._rgb_transform = transforms.Compose(tfs)
        self._gen = torch.Generator()
        s = dist.get_global_rank() + seed if dist.is_initialized() else seed
        self._gen.manual_seed(s)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).float()

    def _get_action_seq(self, episode: dict[str, Any]) -> np.ndarray:
        o = episode["obs"]
        if "state" in self._action_type:
            js, gs = o["joint_states"], o["gripper_states"]
            dj = np.zeros_like(js)
            dj[:-1] = js[1:] - js[:-1]  # delta joint state
            st = np.concatenate([dj, gs], axis=1)  # [T,14]
            if "camera" in self._action_type:
                st = np.concatenate([st, np.zeros((len(st), 9), dtype=st.dtype)], axis=1)
            return st
        act = np.concatenate([o["joint_actions"], o["gripper_actions"]], axis=1)  # [T,14]
        if "camera" in self._action_type:
            act = np.concatenate([act, np.zeros((len(act), 9), dtype=act.dtype)], axis=1)  # +camera (static)
        return act

    def __call__(self, episode: dict[str, Any]) -> dict[str, Any]:
        obs = episode["obs"]
        action_seq = self._get_action_seq(episode)
        T = len(action_seq)
        inds, all_inds = sparse_history_inds(
            self._sample_size, self._chunk_size, self._history_frames,
            self._history_stride_mult, T, self._gen,
        )
        # Indices outside the episode are clipped to its ends; frames past the end are
        # flagged in `mask` so the flow loss skips them.
        inds_valid = inds.clip(min=0, max=T - 1)
        all_inds_valid = all_inds.clip(min=0, max=T - 1)

        rgbs, names = [], []
        for cam in self._cameras:
            rgbs.append(self._rgb_transform(self._to_tensor(obs[cam][inds_valid])))
            names.append(cam)

        actions = self._to_tensor(action_seq[all_inds_valid]).reshape(
            self._sample_size, self._chunk_size, -1)
        mask = torch.tensor(inds < T, dtype=bool)

        if self._multiview:
            rgbs, names = pad_to_num_views(rgbs, names, self._num_views)
            rgbs = random_wrist_mask(rgbs, names, self._wrist_mask_prob, self._gen)
            rgb = torch.concatenate(rgbs, dim=-1)  # T, C, H, n*W
        else:
            rgb = rgbs[0]

        out = {"rgb": rgb, "actions": actions, "mask": mask}
        return {k: out[k] for k in self._output_keys if k in out}

    def state_dict(self) -> dict[str, Any]:
        return {"_gen": self._gen.get_state()}

    def load_state_dict(self, state_dict):
        self._gen.set_state(state_dict["_gen"])
