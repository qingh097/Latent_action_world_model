#!/usr/bin/env python
"""Build the EgoDex, AgiBot and ABC manifests for the cross-embodiment pretraining.

Nothing is downloaded or converted: the loaders only need manifests of the episodes that
are already on disk. setup_training.sh produces the standard layout and calls this script,
in which case the defaults are correct. DROID needs no manifest (its loader reads
$DROID_DATA_DIR directly).

Manifests hold absolute paths, so build them on each machine rather than copying them.
EgoDex and ABC manifests are shuffled with a fixed seed: the loaders' `subsample_traj`
keeps a PREFIX of the manifest, so a task-ordered file would silently train on only the
first few tasks.

If the AgiBot observations, camera parameters and proprio live under three SEPARATE roots
(some clusters stage them that way), pass them individually; this then writes a symlink
tree joining them into the single root the loader expects. Point AGIBOT_ALPHA_ROOT at it.

Idempotent: safe to re-run. Use --dry-run to see what it would write.

    python tools/data/build_manifests.py --out ./data
    python tools/data/build_manifests.py --out ./data --abc-limit 20000
    python tools/data/build_manifests.py --out ./data \\
        --egodex-root /raw/egodex --agibot-obs /raw/obs \\
        --agibot-params /raw/params --agibot-state /raw/state
"""

import argparse
import os
import random
from pathlib import Path

AGIBOT_CAMERAS = ("head_color", "hand_left_color", "hand_right_color")
ABC_FILES = ("top.mp4", "left_wrist.mp4", "right_wrist.mp4", "states.npz")


def _limit(rows, limit):
    return rows if limit is None else rows[:limit]


def _write(dst: Path, lines, dry_run: bool):
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(f"{line}\n" for line in lines))


def build_egodex(src: str, out_dir: Path, seed: int, limit, dry_run: bool) -> int:
    """One hdf5 path per line, for every episode that also has its video."""
    rows = []
    if os.path.isdir(src):
        for hdf5 in sorted(Path(src).rglob("*.hdf5")):
            if hdf5.with_suffix(".mp4").exists():
                rows.append(str(hdf5))
    random.Random(seed).shuffle(rows)
    rows = _limit(rows, limit)
    dst = out_dir / "egodex" / "manifest.csv"
    _write(dst, rows, dry_run)
    print(f"egodex: {len(rows)} episodes -> {dst}")
    return len(rows)


def build_agibot(obs: str, params: str, state: str, out_dir: Path, limit, dry_run: bool) -> int:
    """`task_id,episode_id,dataset` rows after a header line. Only episodes that have all
    three pieces AND every configured camera are kept."""
    rows = []
    for task in sorted(os.listdir(obs)) if os.path.isdir(obs) else []:
        obs_t, par, sta = (Path(r) / task for r in (obs, params, state))
        if not (par.is_dir() and sta.is_dir()):
            continue
        episodes = sorted(
            set(os.listdir(obs_t)) & set(os.listdir(par)) & set(os.listdir(sta)), key=int
        )
        for ep in episodes:
            videos = obs_t / ep / "videos"
            if (
                all((videos / f"{c}.mp4").exists() for c in AGIBOT_CAMERAS)
                and (sta / ep / "proprio_stats.h5").exists()
                and (par / ep / "parameters" / "camera").is_dir()
            ):
                rows.append(f"{task},{ep},alpha")
    rows = _limit(rows, limit)
    dst = out_dir / "agibot" / "manifest.csv"
    _write(dst, ["task_episode_dataset", *rows], dry_run)
    print(f"agibot: {len(rows)} episodes -> {dst}")
    return len(rows)


def build_agibot_root(obs: str, params: str, state: str, out_dir: Path, dry_run: bool) -> Path:
    """Symlink tree giving the loader the single root it expects. Point the loader at it
    with AGIBOT_ALPHA_ROOT."""
    root = out_dir / "agibot_root"
    links = {"observations": obs, "parameters": params, "proprio_stats": state}
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        for name, target in links.items():
            link = root / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
    print(f"agibot root: {root} -> {', '.join(links.values())}")
    return root


def build_abc(src: str, out_dir: Path, seed: int, limit, dry_run: bool) -> int:
    """One preprocessed episode dir (<task>/episode_*) per line, for every episode that the
    preprocessing finished (all three videos and states.npz present)."""
    rows = []
    if os.path.isdir(src):
        for ep in sorted(Path(src).glob("*/episode_*")):
            if all((ep / f).exists() for f in ABC_FILES):
                rows.append(str(ep))
    random.Random(seed).shuffle(rows)
    rows = _limit(rows, limit)
    dst = out_dir / "abc_pp" / "manifest.txt"
    _write(dst, rows, dry_run)
    print(f"abc: {len(rows)} episodes -> {dst}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("LACWM_DATA", "./data"),
                    help="dataset root the manifests are written under (default: $LACWM_DATA or ./data)")
    ap.add_argument("--egodex-root", default=None,
                    help="EgoDex task dirs, searched recursively (default: <out>/egodex_cdn)")
    ap.add_argument("--agibot-obs", default=None, help="default: <out>/agibot/observations")
    ap.add_argument("--agibot-params", default=None, help="default: <out>/agibot/parameters")
    ap.add_argument("--agibot-state", default=None, help="default: <out>/agibot/proprio_stats")
    ap.add_argument("--abc-root", default=None,
                    help="preprocessed ABC episodes <task>/episode_* (default: <out>/abc_pp)")
    ap.add_argument("--only", nargs="*", default=["egodex", "agibot", "abc"],
                    choices=["egodex", "agibot", "abc"], help="which manifests to build")
    for name in ("egodex", "agibot", "abc"):
        ap.add_argument(f"--{name}-limit", type=int, default=None, help=f"cap on {name} episodes")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    print(f"writing under {out}{' (dry run)' if args.dry_run else ''}")
    if "egodex" in args.only:
        build_egodex(args.egodex_root or str(out / "egodex_cdn"), out, args.seed,
                     args.egodex_limit, args.dry_run)
    if "agibot" in args.only:
        obs = args.agibot_obs or str(out / "agibot" / "observations")
        params = args.agibot_params or str(out / "agibot" / "parameters")
        state = args.agibot_state or str(out / "agibot" / "proprio_stats")
        build_agibot(obs, params, state, out, args.agibot_limit, args.dry_run)
        # Only needed when the three AgiBot trees are separate.
        if str(Path(obs).parent) != str(out / "agibot"):
            build_agibot_root(obs, params, state, out, args.dry_run)
    if "abc" in args.only:
        build_abc(args.abc_root or str(out / "abc_pp"), out, args.seed, args.abc_limit, args.dry_run)
    print("\nDROID needs no manifest: its loader reads $DROID_DATA_DIR directly")


if __name__ == "__main__":
    main()
