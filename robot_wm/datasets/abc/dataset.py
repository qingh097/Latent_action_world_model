"""ABC-130k dataset loader (reads preprocessed mp4 + states.npz).

Preprocessing (abc_preprocess.py) produced, per episode dir:
    top.mp4, left_wrist.mp4, right_wrist.mp4  (remuxed H.265, decord-readable)
    states.npz: joint_states[T,12], joint_actions[T,12],
                gripper_states[T,2], gripper_actions[T,2], frame_ts, instruction

Manifest: one preprocessed episode dir per line.
"""
import logging
import os
from typing import Any, Optional

import decord
import numpy as np

from robot_wm.datasets.abc.transform import ABCTransform
from robot_wm.datasets.base import Dataset

logger = logging.getLogger(__name__)

ABC_CAMERAS = ["top", "left_wrist", "right_wrist"]


class _DecordFrames:
    """Array-like lazy view over a video; slicing/indexing decodes only those frames."""

    def __init__(self, path: str):
        self._vr = decord.VideoReader(path)
        self._len = len(self._vr)
        h, w, c = self._vr[0].shape
        self.shape = (self._len, h, w, c)

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        if isinstance(key, slice):
            idx = list(range(*key.indices(self._len)))
        elif isinstance(key, (int, np.integer)):
            idx = [int(key)]
        else:
            idx = [int(i) for i in key]
        idx = [min(max(i, 0), self._len - 1) for i in idx]  # clamp
        return self._vr.get_batch(idx).asnumpy()


class ABCDataset(Dataset):
    def __init__(
        self,
        manifest: Optional[str] = None,
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[ABCTransform] = None,
        subsample_traj: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self.manifest = manifest
        with open(manifest) as f:
            self.episode_dirs = [ln.strip() for ln in f if ln.strip()]
        if subsample_traj is not None:
            self.episode_dirs = self.episode_dirs[:subsample_traj]
        self.ee_action_dim = 14  # 12 joints + 2 grippers (+9 camera)
        logger.info(f"ABCDataset: {len(self.episode_dirs)} episodes from {manifest}")

    @property
    def name(self) -> str:
        return "ABCDataset"

    def _get_length(self) -> int:
        return len(self.episode_dirs)

    def __len__(self) -> int:
        return self._get_length()

    def _load_episode(self, d: str) -> dict[str, Any]:
        st = np.load(os.path.join(d, "states.npz"), allow_pickle=True)
        obs = {
            "joint_states": st["joint_states"].astype(np.float32),      # [T,12]
            "joint_actions": st["joint_actions"].astype(np.float32),    # [T,12]
            "gripper_states": st["gripper_states"].astype(np.float32),  # [T,2]
            "gripper_actions": st["gripper_actions"].astype(np.float32),  # [T,2]
        }
        for cam in ABC_CAMERAS:
            obs[cam] = _DecordFrames(os.path.join(d, f"{cam}.mp4"))
        return {"obs": obs, "instruction": str(st["instruction"])}

    def __get_sample(self, index: int) -> dict[str, Any]:
        episode = self._load_episode(self.episode_dirs[index])
        if self._transform is not None:
            episode = self._transform(episode)
        return episode

    def _get_sample(self, index: int) -> dict[str, Any]:
        for _ in range(8):
            try:
                return self.__get_sample(index)
            except Exception as e:
                logger.warning(f"ABCDataset sample {index} failed ({e}); retrying another")
                index = int(np.random.randint(0, self._get_length()))
        return self.__get_sample(index)

    def __getitem__(self, index):
        return self._get_sample(index)
