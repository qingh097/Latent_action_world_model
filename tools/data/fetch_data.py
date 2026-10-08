#!/usr/bin/env python
"""Download the parts of ABC-130k and AgiBot World Alpha that pretraining reads.

Both datasets are gated on Hugging Face: accept their terms on the dataset pages and log in
(`huggingface-cli login`, or export HF_TOKEN) first. setup_training.sh calls this script.

  abc     downloads `--limit` raw episodes (episode.mcap, ~0.5 GB each), spread evenly over
          the tasks. Preprocess them afterwards with
          robot_wm/datasets/abc/preprocessing/abc_batch_preprocess.py.
  agibot  picks tasks (seeded order) until they hold `--limit` episodes, then streams the
          release's tar archives and extracts only what the loader reads: the three camera
          videos, proprio_stats.h5 and the aligned camera extrinsics. Archives are read
          straight from the Hub, so no 48 GB tar is ever staged on disk.

Idempotent: finished episodes / archives are skipped on re-run.

    python tools/data/fetch_data.py abc --out $LACWM_DATA/abc_raw --limit 10000
    python tools/data/fetch_data.py agibot --out $LACWM_DATA/agibot --limit 10000
"""

import argparse
import json
import random
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

ABC_REPO = "XDOF/ABC-130k"
AGIBOT_REPO = "agibot-world/AgiBotWorld-Alpha"
AGIBOT_VIDEOS = ("head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4")
AGIBOT_PARAMS = tuple(f"{c}_extrinsic_params_aligned.json" for c in ("head", "hand_left", "hand_right"))


def _dirs(api, repo, path):
    return sorted(x.path for x in api.list_repo_tree(repo, path_in_repo=path, repo_type="dataset"))


# ----------------------------------------------------------------------------------- ABC
def fetch_abc(out: Path, limit: int, seed: int, workers: int) -> None:
    api = HfApi()
    tasks = _dirs(api, ABC_REPO, "data/train")
    rng = random.Random(seed)
    per_task = {}
    for t in tasks:
        eps = _dirs(api, ABC_REPO, t)
        rng.shuffle(eps)
        per_task[t] = eps
    # round-robin over tasks so a capped download still covers every task
    chosen, depth = [], 0
    while len(chosen) < limit and any(depth < len(e) for e in per_task.values()):
        for t in tasks:
            if depth < len(per_task[t]) and len(chosen) < limit:
                chosen.append(per_task[t][depth])
        depth += 1
    print(f"abc: {len(chosen)} episodes from {len(tasks)} tasks -> {out}", flush=True)

    def get(ep):
        if (out / ep / "episode.mcap").exists():
            return
        for attempt in range(5):
            try:
                hf_hub_download(ABC_REPO, f"{ep}/episode.mcap", repo_type="dataset", local_dir=out)
                return
            except Exception as e:  # transient network errors
                print(f"  retry {attempt + 1} {ep}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"failed to download {ep}")

    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(get, chosen), 1):
            if i % 100 == 0 or i == len(chosen):
                print(f"  {i}/{len(chosen)}", flush=True)


# -------------------------------------------------------------------------------- AgiBot
def _extract_stream(fs, repo_path: str, dest: Path, keep) -> int:
    """Stream one tar from the Hub and extract the members `keep(name)` accepts."""
    n = 0
    with fs.open(f"datasets/{AGIBOT_REPO}/{repo_path}", "rb", block_size=64 << 20) as f:
        with tarfile.open(fileobj=f, mode="r|") as tar:
            for m in tar:
                if m.isfile() and keep(m.name):
                    tar.extract(m, dest)
                    n += 1
    return n


def _extract_once(fs, repo_path: str, dest: Path, keep) -> None:
    done = dest / ".fetched" / (repo_path.replace("/", "__") + ".done")
    if done.exists():
        print(f"  skip {repo_path} (done)", flush=True)
        return
    for attempt in range(5):
        try:
            n = _extract_stream(fs, repo_path, dest, keep)
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(str(n))
            print(f"  {repo_path}: {n} files", flush=True)
            return
        except Exception as e:  # a dropped stream restarts the archive
            print(f"  retry {attempt + 1} {repo_path}: {e}", flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"failed to extract {repo_path}")


def fetch_agibot(out: Path, limit: int, seed: int) -> None:
    api, fs = HfApi(), HfFileSystem()
    tasks = [p.split("/")[-1] for p in _dirs(api, AGIBOT_REPO, "observations")]
    random.Random(seed).shuffle(tasks)
    chosen, total = [], 0
    for t in tasks:
        if total >= limit:
            break
        info = hf_hub_download(AGIBOT_REPO, f"task_info/task_{t}.json", repo_type="dataset")
        total += len(json.load(open(info)))
        chosen.append(t)
    keep_tasks = set(chosen)
    print(f"agibot: tasks {sorted(chosen)} (~{total} episodes) -> {out}", flush=True)

    # observations/<task>/<range>.tar holds <episode>/videos/*.mp4 (+ depth, fisheye: skipped)
    for t in sorted(chosen):
        for tar in _dirs(api, AGIBOT_REPO, f"observations/{t}"):
            _extract_once(fs, tar, out / "observations" / t,
                          lambda n: n.split("/")[-2:-1] == ["videos"] and n.split("/")[-1] in AGIBOT_VIDEOS)
    # proprio_stats/*.tar and parameters/*.tar hold <task>/<episode>/... for every task
    for tar in _dirs(api, AGIBOT_REPO, "proprio_stats"):
        _extract_once(fs, tar, out / "proprio_stats",
                      lambda n: n.split("/")[0] in keep_tasks and n.endswith("/proprio_stats.h5"))
    for tar in _dirs(api, AGIBOT_REPO, "parameters"):
        _extract_once(fs, tar, out / "parameters",
                      lambda n: n.split("/")[0] in keep_tasks and n.split("/")[-1] in AGIBOT_PARAMS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["abc", "agibot"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=10000, help="episodes to fetch (default 10000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="parallel ABC downloads")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "abc":
        fetch_abc(args.out, args.limit, args.seed, args.workers)
    else:
        fetch_agibot(args.out, args.limit, args.seed)


if __name__ == "__main__":
    main()
