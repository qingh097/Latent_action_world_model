# Mixes several sub-datasets (one per embodiment) into one stream; see MultiDataset.
import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn.functional as F

from robot_wm.datasets.base import Dataset

# Morphology id per dataset class. The ids select the morphology embedding and the motion
# decoder head (configs key the heads by these numbers), so they must stay stable across
# pretraining and finetuning checkpoints.
Morphology_Mapping = {
    "DroidDataset": 0,  # DroidTFDSDataset
    "DroidLeRobotDataset": 0,
    "EgoDexDataset": 2,
    "LiberoDataset": 5,
    "AgibotDataset": 6,
    "ABCDataset": 9,
}


def random_crop_resize_same(batch: torch.Tensor, ratio_range=(0.6, 1.0)):
    """
    Apply the same random crop (based on a sampled ratio) to all images in the batch,
    then resize them back to original size.

    Args:
        batch: (B, C, H, W)
        ratio_range: tuple of (min_ratio, max_ratio)

    Returns:
        Transformed batch: (B, C, H, W)
    """
    B, C, H, W = batch.shape

    # Sample crop ratio
    ratio = torch.empty(1).uniform_(*ratio_range).item()
    crop_h = int(H * ratio)
    crop_w = int(W * ratio)

    # Sample top-left crop corner (same for all images)
    top = torch.randint(0, H - crop_h + 1, (1,)).item()
    left = torch.randint(0, W - crop_w + 1, (1,)).item()

    # Crop and resize each image
    cropped = batch[:, :, top : top + crop_h, left : left + crop_w]
    resized = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)

    return resized


class MultiDataset(Dataset):
    """
    Concatenation of several per-embodiment datasets (``datasets``: name -> dataset).
    Each sample is tagged with its morphology id, has its camera slot re-referenced to the
    last history frame, its action vector zero-padded to ``padding_dim``, and, with
    ``img_augment``, gets the photometric augmentation plus an independently augmented
    copy (``crop_rgb``) for the inverse model.
    """

    def __init__(
        self,
        datasets: dict,
        transform: Optional[dict] = None,
        seed: int = 0,
        infinite: bool = True,
        padding_dim: int = 0,
        img_augment: bool = False,
        fill_relative_camera_pose: bool = False,
        camera_pose_anchor: int = 4,
        strong_augment_datasets: Optional[Sequence[str]] = None,
    ):
        super().__init__(seed=seed, infinite=infinite, transform=transform)
        self.datasets = datasets

        self.dataset_lengths = [len(dataset) for dataset in self.datasets.values()]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)
        for name, dataset in self.datasets.items():
            logger.info(f"MultiDataset: {name} has {len(dataset):,} episodes")
        logger.info(f"MultiDataset: total {self.cumulative_lengths[-1]:,} episodes")
        self.padding_dim = padding_dim
        self.img_augment = img_augment
        # Camera motion: every sub-dataset writes the ABSOLUTE 9-d pose (pos3 + rot6d) of
        # its ego / third-person camera into a 9-d slot of its action vector; here we
        # re-reference it to the last history frame so the camera-motion target means
        # exactly the same thing for every morphology in the mix.
        self.fill_relative_camera_pose = fill_relative_camera_pose
        self.camera_pose_anchor = camera_pose_anchor
        self.augment = torch.nn.Sequential(
            K.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=1.0,
                same_on_batch=True,
            ),
            K.RandomGamma(
                gamma=(0.7, 1.5), p=0.5, same_on_batch=True
            ),  # simulates lighting change
        )
        self.rot_augment = K.RandomRotation(degrees=20, same_on_batch=True, p=1.0)

        # Datasets that get the heavier photometric pipeline below -- in this mix that is
        # the human data (EgoDex), whose appearance statistics are the furthest from the
        # robot sources, so the model should not be able to lean on them.
        self.strong_augment_datasets = set(strong_augment_datasets or ())
        # Photometric only, deliberately: a geometric augmentation (rotation, shear) would
        # move the image plane while the camera-motion targets still describe the original
        # motion, i.e. it would train the model on pose labels that contradict the pixels.
        # same_on_batch=True keeps every frame of a clip under one sampled transform, so
        # the augmentation cannot invent apparent motion between frames either.
        self.strong_augment = torch.nn.Sequential(
            K.ColorJitter(
                brightness=0.6,
                contrast=0.6,
                saturation=0.6,
                hue=0.2,
                p=1.0,
                same_on_batch=True,
            ),
            K.RandomGamma(gamma=(0.6, 1.8), p=0.7, same_on_batch=True),
            K.RandomGrayscale(p=0.2, same_on_batch=True),
            K.RandomGaussianBlur(
                kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.3, same_on_batch=True
            ),
            K.RandomGaussianNoise(mean=0.0, std=0.03, p=0.3, same_on_batch=True),
        )

    @property
    def name(self) -> str:
        return "MultiDataset"

    @property
    def full_name(self) -> str:
        all_dataset_names = list(self.datasets.keys())
        return "MultiDataset" + " - ".join(all_dataset_names)

    def _get_length(self):
        return self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self._get_length()

    def _get_sample(self, index):
        # locate the sub-dataset that holds this index
        dataset_index = np.searchsorted(self.cumulative_lengths, index, "right")
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumulative_lengths[dataset_index - 1]

        dataset_name = list(self.datasets.keys())[dataset_index]
        dataset = self.datasets[dataset_name]
        episode = dataset._get_sample(sample_index)

        # Start index of the 9-d camera slot. Most datasets append it right after their
        # ee_action_dim action dims; LIBERO puts joint states after it and declares its own.
        cam_lo, _ = getattr(
            dataset, "camera_slice", (dataset.ee_action_dim, dataset.ee_action_dim + 9)
        )
        episode["pose_lo"] = torch.tensor(cam_lo, dtype=torch.long)

        if self.fill_relative_camera_pose:
            self._relativize_camera_pose(episode)

        if dataset.ee_action_dim < self.padding_dim:
            # episode["actions"] is a tensor of shape (N, T, D), padding at last dim
            # create a tensor of shape (N, T, self.padding_dim - action_dim)
            padding = torch.zeros(
                *(episode["actions"].shape[:-1]),
                self.padding_dim - episode["actions"].shape[-1],
            )
            episode["actions"] = torch.cat(
                [
                    episode["actions"],
                    padding,
                ],
                dim=-1,
            )

        assert dataset.name in Morphology_Mapping, (
            f"Dataset {dataset.name} not found in Morphology_Mapping. "
            f"Available datasets: {list(Morphology_Mapping.keys())}"
        )
        morphology_index = Morphology_Mapping[dataset.name]

        episode["morphology_index"] = torch.tensor(
            morphology_index, dtype=torch.long
        )  # Add morphology as a tensor

        if self.img_augment and "rgb" in episode:
            augment = (
                self.strong_augment
                if dataset_name in self.strong_augment_datasets
                else self.augment
            )
            original_rgb = episode["rgb"].clone()
            # rgb is normalized to [-1,1] (mean=std=0.5). kornia ColorJitter/RandomGamma assume
            # [0,1] input (they clamp to [0,1]), so augment in [0,1] -- i.e. BEFORE normalization --
            # then re-normalize back to [-1,1]. (Augmenting the normalized [-1,1] tensor directly
            # collapses the range to [0,1] and feeds the VAE out-of-distribution data.)
            rgb01 = ((original_rgb + 1.0) * 0.5).clamp(0.0, 1.0)
            # Clamp before re-normalizing: ColorJitter/RandomGamma clamp themselves, but
            # RandomGaussianNoise does not, and anything outside [0,1] here lands outside
            # the [-1,1] the VAE was trained on.
            episode["rgb"] = augment(rgb01).clamp(0.0, 1.0) * 2.0 - 1.0
            episode["crop_rgb"] = random_crop_resize_same(
                self.rot_augment(augment(rgb01).clamp(0.0, 1.0)) * 2.0 - 1.0
            )

        return episode


    def _relativize_camera_pose(self, episode: dict[str, Any]) -> None:
        """Rewrite the trailing 9-d camera-pose slot of ``episode["actions"]`` in place,
        from absolute pose to pose relative to the last history frame.

        Must run BEFORE the zero-padding to ``padding_dim``, while the trailing 9 dims are
        still the camera slot.  Datasets that have no pose (all-zero slot) are left alone.
        """
        from robot_wm.datasets.utils import relative_pose_9d

        actions = episode["actions"]
        if actions.shape[-1] < 9 or actions.ndim != 3:
            return
        is_torch = isinstance(actions, torch.Tensor)
        arr = actions.numpy() if is_torch else np.asarray(actions)
        T, C, D = arr.shape
        flat = arr.reshape(T * C, D)
        lo = int(episode["pose_lo"]) if "pose_lo" in episode else D - 9
        cam = flat[:, lo : lo + 9]
        # No pose available for this dataset (slot left at zeros) -> nothing to do; a
        # zero rot6d has no valid rotation matrix and would poison the tokens with NaNs.
        if np.abs(cam[:, 3:]).max() < 1e-8:
            return
        anchor = min(self.camera_pose_anchor * C, T * C - 1)
        flat[:, lo : lo + 9] = relative_pose_9d(cam, anchor)
        out = flat.reshape(T, C, D)
        episode["actions"] = torch.from_numpy(out) if is_torch else out

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self._get_sample(index)
        return episode

    def state_dict(self) -> dict[str, Any]:
        output = {
            "_start_idx": self._start_idx,
            "_gen": self._gen.get_state(),
            "_transform": {},
        }
        for dataset_name, dataset in self.datasets.items():
            transform = dataset._transform
            if transform is not None:
                output["_transform"][dataset_name] = transform.state_dict()
        return output

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._start_idx = state_dict["_start_idx"]
        self._gen.set_state(state_dict["_gen"])
        for dataset_name, dataset in self.datasets.items():
            transform = dataset._transform
            if transform is not None:
                transform.load_state_dict(state_dict["_transform"][dataset_name])


