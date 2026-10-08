"""DROID loader for the LeRobot v2.1 format (e.g. ``cadene/droid``).

Reads per-episode parquet (state/action) + per-camera mp4 and presents an
``episode_data`` structure compatible with :class:`DroidTransform`.

LeRobot ``cadene/droid`` layout::

    data/chunk-{chunk:03d}/episode_{idx:06d}.parquet
    videos/chunk-{chunk:03d}/observation.images.{cam}/episode_{idx:06d}.mp4

state[8] = [xyz(3), euler(3), gripper(1), unused(1)] ; action[7] = [xyz, euler, gripper]
"""
import glob
import logging
import os
from typing import Any, Optional

import av
import numpy as np
import pandas as pd

from robot_wm.datasets.base import Dataset
from robot_wm.datasets.droid.transform import DroidTransform

logger = logging.getLogger(__name__)

DROID_LEROBOT_CAMERAS = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
]


class DroidLeRobotDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[DroidTransform] = None,
        subsample_traj: Optional[int] = None,
        chunks_size: int = 1000,
        **kwargs,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self.data_dir = data_dir
        self.chunks_size = chunks_size
        parquets = sorted(
            glob.glob(os.path.join(data_dir, "data", "chunk-*", "episode_*.parquet"))
        )
        self.episodes = [
            int(os.path.basename(p).split("_")[1].split(".")[0]) for p in parquets
        ]

        def _complete(ep):
            chunk = ep // self.chunks_size
            return all(
                os.path.exists(
                    os.path.join(
                        data_dir,
                        "videos",
                        f"chunk-{chunk:03d}",
                        f"observation.images.{cam}",
                        f"episode_{ep:06d}.mp4",
                    )
                )
                for cam in DROID_LEROBOT_CAMERAS
            )

        n_before = len(self.episodes)
        self.episodes = [e for e in self.episodes if _complete(e)]
        if len(self.episodes) < n_before:
            logger.info(
                f"DroidLeRobot: dropped {n_before - len(self.episodes)} episodes "
                f"with missing mp4 (download in progress); {len(self.episodes)} usable"
            )
        if subsample_traj is not None:
            self.episodes = self.episodes[:subsample_traj]
        self.ee_action_dim = 10  # 10 ee + 9 camera = 19
        logger.info(f"DroidLeRobot: {len(self.episodes)} episodes from {data_dir}")

    @property
    def name(self) -> str:
        return "DroidLeRobotDataset"

    def _get_length(self) -> int:
        return len(self.episodes)

    def __len__(self) -> int:
        return self._get_length()

    def _paths(self, ep: int):
        chunk = ep // self.chunks_size
        dp = os.path.join(
            self.data_dir, "data", f"chunk-{chunk:03d}", f"episode_{ep:06d}.parquet"
        )
        vids = {
            cam: os.path.join(
                self.data_dir,
                "videos",
                f"chunk-{chunk:03d}",
                f"observation.images.{cam}",
                f"episode_{ep:06d}.mp4",
            )
            for cam in DROID_LEROBOT_CAMERAS
        }
        return dp, vids

    @staticmethod
    def _read_video(path: str) -> np.ndarray:
        container = av.open(path)
        vs = container.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(vs)]
        container.close()
        return np.stack(frames)

    def _load_episode(self, ep: int) -> dict[str, Any]:
        dp, vids = self._paths(ep)
        df = pd.read_parquet(dp)
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # [T,8]
        action = np.stack(df["action"].to_numpy()).astype(np.float32)  # [T,7]
        observation = {
            "cartesian_position": state[:, :6],  # xyz + euler
            "gripper_position": state[:, 6:7],
        }
        for cam in DROID_LEROBOT_CAMERAS:
            observation[cam] = self._read_video(vids[cam])  # [T,H,W,3] uint8
        return {
            "episode_data": {"action": action, "observation": observation},
            "episode_metadata": {"trajectory_length": len(action)},
        }

    def __get_sample(self, index: int) -> dict[str, Any]:
        ep = self.episodes[index]
        episode = self._load_episode(ep)
        if self._transform is not None:
            episode = self._transform(episode)
        return episode

    def _get_sample(self, index: int) -> dict[str, Any]:
        for _ in range(8):
            try:
                return self.__get_sample(index)
            except Exception as e:
                logger.warning(f"DroidLeRobotDataset sample {index} failed ({e}); retrying another")
                index = int(np.random.randint(0, self._get_length()))
        return self.__get_sample(index)

    def __getitem__(self, index):
        return self._get_sample(index)
