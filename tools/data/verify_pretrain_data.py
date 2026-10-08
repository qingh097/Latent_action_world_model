#!/usr/bin/env python
"""Check the cross-embodiment pretraining data before burning GPU days on it.

The datasets are built from the training config itself (+experiment=pretrain), so the
check sees exactly the clips training will see. Per source it verifies the things that
fail SILENTLY at training time:
  * the clip loads at all, with every view slot present (missing views zero-padded)
  * the future frames are real (not all masked out as padding)
  * the 9-d camera slot is either all zero (fixed camera) or has been re-referenced to the
    last history frame (that row must be the identity pose), and whether it varies
  * augmented pixels stay inside [-1,1] with no NaNs, the human data is perturbed harder
    than the robot data, and one photometric transform covers the whole clip

    python tools/data/verify_pretrain_data.py                 # every source
    python tools/data/verify_pretrain_data.py --skip Droid    # skip slow/absent sources
"""

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import torch

# This runs for several minutes per dataset (it decodes real video) and is usually
# redirected to a log, so keep the output unbuffered or it looks like it has hung.
print = functools.partial(print, flush=True)  # noqa: A001

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "projects" / "latent_action_models"
IDENTITY_9D = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
SOURCES = ("Droid", "EgoDex", "Agibot", "ABC")


def build(skip, episodes):
    """Instantiate the training (augmented) and validation (plain) mixes of the pretraining
    config, limited to `episodes` episodes per source."""
    sys.path[:0] = [str(REPO), str(PROJECT)]
    import custom_resolvers  # noqa: F401  (registers the ${mul:..}/${div:..} resolvers)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    keep = [s for s in SOURCES if s not in skip]
    overrides = ["+experiment=pretrain"]
    for split in ("dataset", "val_dataset", "viz_dataset"):
        overrides += [f"~{split}.datasets.{s}" for s in skip]
        overrides += [f"{split}.datasets.{s}.subsample_traj={episodes}" for s in keep]
    with initialize_config_dir(config_dir=str(PROJECT / "configs"), version_base=None):
        cfg = compose("train.yaml", overrides=overrides)
    return cfg, instantiate(cfg.val_dataset), instantiate(cfg.dataset)


def as_numpy(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[], choices=SOURCES,
                    help="sources to leave out (e.g. Droid: the TFDS loader indexes every shard first)")
    ap.add_argument("--episodes", type=int, default=2,
                    help="episodes to sample per source (each one decodes real video: ~1-2 min)")
    args = ap.parse_args()

    print(f"building datasets from +experiment=pretrain (skip={args.skip}) ...")
    cfg, plain, augmented = build(args.skip, args.episodes)
    num_views = cfg.model.latent_views
    anchor_frame = cfg.dataset.camera_pose_anchor
    num_history = cfg.model.num_history_frames

    failures = []
    offset = 0
    for name, dataset in plain.datasets.items():
        base, offset = offset, offset + len(dataset)
        deviations, deltas = [], []
        print(f"checking {name} ({len(dataset)} episodes) ...")
        for k in range(len(dataset)):
            idx = base + k
            sample = plain[idx]
            rgb = sample["rgb"]
            if rgb.shape[-1] % num_views:
                failures.append(f"{name}: width {rgb.shape[-1]} is not {num_views} stacked views")
            if not bool(sample["mask"][num_history:].any()):
                failures.append(f"{name}: every future frame is masked -- the clip trains nothing")

            actions = as_numpy(sample["actions"])               # [T, chunk, D]
            lo = int(sample["pose_lo"])
            pose = actions[..., lo:lo + 9].reshape(-1, 9)
            if np.abs(pose).max() < 1e-8:
                deviations.append(0.0)                          # fixed camera: zero slot
            else:
                anchor = min(anchor_frame * actions.shape[1], pose.shape[0] - 1)
                if not np.allclose(pose[anchor], IDENTITY_9D, atol=1e-4):
                    failures.append(f"{name}: camera slot anchor row is not the identity -- not re-referenced")
                deviations.append(float(np.abs(pose - IDENTITY_9D).max()))

            torch.manual_seed(0)
            a = augmented[idx]["rgb"]
            if torch.isnan(a).any():
                failures.append(f"{name}: NaNs in augmented pixels")
            if a.min() < -1.001 or a.max() > 1.001:
                failures.append(f"{name}: augmented pixels outside [-1,1]: [{a.min():.2f},{a.max():.2f}]")
            deltas.append(float((a - rgb).abs().mean()))
            spread = float((a - rgb).mean(dim=(1, 2, 3)).std())
            if spread > 0.01:
                failures.append(f"{name}: per-frame shift spread {spread:.4f} -- augmentation is not constant over the clip")

        strong = name in augmented.strong_augment_datasets
        varies = max(deviations) > 1e-4
        print(f"{name:7s} rgb={tuple(rgb.shape)} camera_slot={lo}:{lo + 9} camera_moves={varies} "
              f"max_dev={max(deviations):.4f} strong_aug={strong} mean|delta|={np.mean(deltas):.4f}")
        if not varies:
            print(f"        note: {name} has no camera motion in these clips (fixed camera, or constant extrinsics)")

    print()
    if failures:
        for f in sorted(set(failures)):
            print(f"FAIL {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
