"""Batch-preprocess all downloaded ABC episodes: mcap -> per-cam mp4 + states.npz.

Parallel over episodes. Idempotent: skips episodes whose outputs already exist.
The training manifest is built afterwards by tools/data/build_manifests.py.
"""
import glob
import os
import traceback
from multiprocessing import Pool

from robot_wm.datasets.abc.preprocessing.abc_preprocess import preprocess

# Paths are env-overridable (defaults are relative to the working directory):
#   ABC_RAW  raw episodes root, globbed as <ABC_RAW>/<task>/<episode>/episode.mcap
#   ABC_PP   output root: abc_pp/<task>/<episode>/{top,left_wrist,right_wrist}.mp4 + states.npz
_LACWM_DATA = os.environ.get("LACWM_DATA", "./data")
ABC_RAW = os.environ.get("ABC_RAW", os.path.join(_LACWM_DATA, "abc/data/train"))
ABC_PP = os.environ.get("ABC_PP", os.path.join(_LACWM_DATA, "abc_pp"))
NPROC = int(os.environ.get("ABC_NPROC", "16"))


def out_dir_for(mcap_path):
    # .../abc/data/train/<task>/<episode_xxx>/episode.mcap -> abc_pp/<task>/<episode_xxx>
    rel = os.path.relpath(os.path.dirname(mcap_path), ABC_RAW)
    return os.path.join(ABC_PP, rel)


def work(mcap_path):
    out = out_dir_for(mcap_path)
    try:
        if os.path.exists(os.path.join(out, "states.npz")) and \
           os.path.exists(os.path.join(out, "top.mp4")):
            return ("skip", out)
        preprocess(mcap_path, out)
        return ("ok", out)
    except Exception as e:
        return ("err", f"{out}: {e}\n{traceback.format_exc()[:300]}")


def main():
    mcaps = sorted(glob.glob(os.path.join(ABC_RAW, "*", "*", "episode.mcap")))
    print(f"found {len(mcaps)} episodes to preprocess (nproc={NPROC})", flush=True)
    ok, skip, err = [], [], []
    with Pool(NPROC) as p:
        for i, (status, info) in enumerate(p.imap_unordered(work, mcaps, chunksize=4)):
            if status == "ok":
                ok.append(info)
            elif status == "skip":
                skip.append(info)
            else:
                err.append(info)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(mcaps)} | ok={len(ok)} skip={len(skip)} err={len(err)}", flush=True)
    print(f"DONE preprocess: ok={len(ok)} skip={len(skip)} err={len(err)} -> {ABC_PP}", flush=True)
    for e in err[:10]:
        print("  ERR", e, flush=True)


if __name__ == "__main__":
    main()
