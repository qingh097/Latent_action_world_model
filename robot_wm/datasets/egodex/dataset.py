"""EgoDex loader: one hdf5 (hand/camera transforms) + mp4 (egocentric video) per episode.

Manifest: one hdf5 path per line; the video is the sibling .mp4.
"""
import csv
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from robot_wm.datasets.base import Dataset
from robot_wm.datasets.egodex.transform import EgoDexTransform
from robot_wm.datasets.utils import euler_to_rot_6d

logger = logging.getLogger(__name__)


LEFT_HAND_JOINTS = [
    "leftThumbKnuckle",
    "leftThumbIntermediateBase",
    "leftThumbIntermediateTip",
    "leftThumbTip",
    "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase",
    "leftIndexFingerIntermediateTip",
    "leftIndexFingerTip",
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase",
    "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerTip",
    "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase",
    "leftRingFingerIntermediateTip",
    "leftRingFingerTip",
    "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase",
    "leftLittleFingerIntermediateTip",
    "leftLittleFingerTip",
]
RIGHT_HAND_JOINTS = [
    "rightThumbKnuckle",  # 22
    "rightThumbIntermediateBase",  # 20
    "rightThumbIntermediateTip",  # 21
    "rightThumbTip",  # 23
    "rightIndexFingerKnuckle",  # 2
    "rightIndexFingerIntermediateBase",  # 0
    "rightIndexFingerIntermediateTip",  # 1
    "rightIndexFingerTip",  # 4
    "rightMiddleFingerKnuckle",  # 12
    "rightMiddleFingerIntermediateBase",  # 10
    "rightMiddleFingerIntermediateTip",  # 11
    "rightMiddleFingerTip",  # 14
    "rightRingFingerKnuckle",  # 17
    "rightRingFingerIntermediateBase",  # 15
    "rightRingFingerIntermediateTip",  # 16
    "rightRingFingerTip",  # 19
    "rightLittleFingerKnuckle",  # 7
    "rightLittleFingerIntermediateBase",  # 5
    "rightLittleFingerIntermediateTip",  # 6
    "rightLittleFingerTip",  # 9
]


STATE_ROT_JOINTS = ["camera", "leftHand", "rightHand"]


def _pose_9d(pose_matrix: np.ndarray) -> np.ndarray:
    """(T, 4, 4) transforms -> (T, 9) position + rot6d."""
    euler = Rotation.from_matrix(pose_matrix[:, :3, :3]).as_euler("xyz", degrees=False)
    return np.concatenate([pose_matrix[:, :3, 3], euler_to_rot_6d(euler)], axis=-1)


class EgoDexDataset(Dataset):
    def __init__(
        self,
        manifest: Union[str, Path],
        seed: int = 0,
        infinite: bool = True,
        transform: Optional[EgoDexTransform] = None,
        subsample_traj: Optional[int] = None,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self._manifest = manifest
        self.ee_action_dim = 148  # two hands (2 x 69) + 10 reserved; +9 camera -> 157
        self._paths = self._load_manifest(manifest)
        if subsample_traj is not None:
            self._paths = self._paths[:subsample_traj]
        logger.info(f"full dataset size ({manifest}): {len(self._paths):,}")

    @staticmethod
    def _load_manifest(manifest: Union[str, Path]) -> Sequence[Path]:
        with Path(manifest).open() as f:
            return [Path(row[0]) for row in csv.reader(f, delimiter=" ") if row]

    @property
    def name(self) -> str:
        return "EgoDexDataset"

    def _get_length(self) -> int:
        return len(self._paths)

    def __len__(self) -> int:
        return self._get_length()

    def _get_sample(self, index: int) -> dict[str, Any]:
        h5_path = self._paths[index]
        frames = []
        cap = cv2.VideoCapture(str(h5_path.with_suffix(".mp4")))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (640, 360))[..., ::-1])  # BGR -> RGB
        if len(frames) == 0:
            logger.warning(f"Failed to get sample at index {index}")
            return self._get_sample(np.random.randint(0, len(self)))

        with h5py.File(h5_path) as f:
            transforms = f["transforms"]
            poses = {joint: _pose_9d(transforms[joint][:]) for joint in STATE_ROT_JOINTS}
            left = np.concatenate([transforms[j][:, :3, 3] for j in LEFT_HAND_JOINTS], axis=-1)
            right = np.concatenate([transforms[j][:, :3, 3] for j in RIGHT_HAND_JOINTS], axis=-1)

        episode = {
            "episode_data": {
                "observation": {
                    "ego_centric_image": np.stack(frames),
                    "ekf_hand_pose_left/WristState": poses["leftHand"],
                    "ekf_hand_pose_right/WristState": poses["rightHand"],
                    "ekf_hand_pose_left/JointAngles": left,
                    "ekf_hand_pose_right/JointAngles": right,
                    "camera": poses["camera"],
                }
            },
        }
        if self._transform is not None:
            episode = self._transform(episode)
            assert episode["actions"].shape[-1] == self.ee_action_dim + 9, (
                f"Expected action dim {self.ee_action_dim + 9}, got {episode['actions'].shape[-1]}"
            )
        return episode

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._get_sample(index)
