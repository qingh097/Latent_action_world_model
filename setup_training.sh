#!/usr/bin/env bash
#
# setup_training.sh — provision a new server to train the Wan-DiT latent-action
# world model: env + Wan weights + VideoX-Fun, the 4 datasets, their manifests,
# path repointing, and validation.
#
# Pretraining (+experiment=pretrain) trains on ABC + Agibot + DROID + EgoDex
# (dataset config transformed_multi_abc_agibot_droid_egodex).
#
# Data (FETCH below):
#   download  — pull data from the public sources (default). Each dataset has a toggle
#               and an episode cap; only what the loaders read is kept on disk.
#   skip      — data already in place; only (re)generate manifests.
#
# Sources:
#   DROID   https://huggingface.co/datasets/cadene/droid   (LeRobot v2.1, matches the loader;
#           NOTE lerobot/droid_1.0.1 is v3.0 file-packed parquet and is NOT loadable as-is)
#   Agibot  https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha   (GATED)
#   ABC     https://huggingface.co/datasets/XDOF/ABC-130k                    (GATED)
#   EgoDex  https://github.com/apple/ml-egodex  (zips on Apple's CDN)
# The two gated datasets need their terms accepted on the dataset page and a Hugging Face
# login (`huggingface-cli login`) before running this.
#
# Usage:
#   ./setup_training.sh                 # all: env + fetch + manifests + repoint + validate
#   ./setup_training.sh datasets        # just fetch the enabled datasets
#   ./setup_training.sh manifests       # (re)generate manifests for present data
#   ./setup_training.sh validate
#
# Manifests store ABSOLUTE paths, so they are regenerated per-server here (by
# tools/data/build_manifests.py) — never copy them between machines.

set -euo pipefail

# ─────────────────────────────── CONFIG ──────────────────────────────────────
BASE="${BASE:-$PWD}"                  # holds data + weights + VideoX-Fun
# These four are EXPORTED so the configs + module defaults (which read them via
# ${oc.env:VAR,default} / os.environ.get) resolve to this server. Defaults are relative to $PWD.
export LACWM_DATA="${LACWM_DATA:-$BASE/lacwm_data}"       # datasets
export LACWM_RUNS="${LACWM_RUNS:-$BASE/lacwm_runs}"       # training outputs (run dirs)
export WAN_DIR="${WAN_DIR:-$BASE/wan_fun_1.3b_control}"   # Wan2.1-Fun weights + null prompt
export VIDEOX_HOME="${VIDEOX_HOME:-$BASE/VideoX-Fun}"     # VideoX-Fun library
DATA_ROOT="$LACWM_DATA"; VIDEOX_DIR="$VIDEOX_HOME"        # internal aliases
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_ENV="${CONDA_ENV:-lacwm-dit}"

FETCH="${FETCH:-download}"                    # download | skip

# Per-dataset: enable (1/0) and a cap on episodes. The pretraining config reads at most
# 10000 episodes per source (subsample_traj), so that is the default cap; "all" = no cap.
# The cap bounds the download for DROID (1000-episode chunks), ABC (exact) and AgiBot
# (whole tasks until the cap is covered); EgoDex downloads whole parts (EGODEX_PARTS).
DROID_ENABLE="${DROID_ENABLE:-1}";   DROID_LIMIT="${DROID_LIMIT:-10000}"
AGIBOT_ENABLE="${AGIBOT_ENABLE:-1}"; AGIBOT_LIMIT="${AGIBOT_LIMIT:-10000}"
EGODEX_ENABLE="${EGODEX_ENABLE:-1}"; EGODEX_LIMIT="${EGODEX_LIMIT:-10000}"
ABC_ENABLE="${ABC_ENABLE:-1}";       ABC_LIMIT="${ABC_LIMIT:-10000}"

EGODEX_PARTS="${EGODEX_PARTS:-part2}"         # space-separated subset of: part1..part5 test extra (~320 GB each)
ABC_KEEP_RAW="${ABC_KEEP_RAW:-0}"             # 1 = keep the raw episode.mcap files after preprocessing
# ──────────────────────────────────────────────────────────────────────────────

log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*"; }
prep() { printf '\033[1;36m[PREP]\033[0m %s\n'  "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n'  "$*" >&2; exit 1; }
cap()  { if [ "$2" = all ]; then cat; else head -n "$2"; fi; }   # truncate stdin to a limit

# ─── 1. Environment ───────────────────────────────────────────────────────────
setup_env() {
  log "conda env '$CONDA_ENV' + package install"
  command -v conda >/dev/null || die "conda not found"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda env list | grep -qE "^\s*$CONDA_ENV\s" || conda create -y -n "$CONDA_ENV" python=3.10
  conda activate "$CONDA_ENV"
  pip install --upgrade pip
  pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 || \
    warn "torch install failed — install a build matching your CUDA, then re-run"
  pip install -r "$REPO_DIR/requirements.txt"
  pip install -e "$REPO_DIR"
  [ -d "$VIDEOX_DIR/.git" ] || git clone https://github.com/aigc-apps/VideoX-Fun.git "$VIDEOX_DIR"
}

# ─── 2. Weights ───────────────────────────────────────────────────────────────
fetch_weights() {
  [ "$FETCH" = download ] || { log "weights: FETCH=$FETCH (using existing $WAN_DIR)"; return; }
  mkdir -p "$WAN_DIR"
  log "hf download alibaba-pai/Wan2.1-Fun-1.3B-Control"
  huggingface-cli download alibaba-pai/Wan2.1-Fun-1.3B-Control --local-dir "$WAN_DIR"
  [ -f "$WAN_DIR/Wan2.1_VAE.pth" ] || warn "missing $WAN_DIR/Wan2.1_VAE.pth"
  # null_prompt_umt5.pt (the cached empty-prompt umT5 embedding) is not in the official
  # release; build it from the downloaded text encoder.
  if [ ! -f "$WAN_DIR/null_prompt_umt5.pt" ]; then
    log "building $WAN_DIR/null_prompt_umt5.pt"
    PYTHONPATH="$VIDEOX_DIR:${PYTHONPATH:-}" python "$REPO_DIR/tools/launch/gen_null_prompt.py" \
      || warn "null prompt generation failed — run tools/launch/gen_null_prompt.py by hand"
  fi
}

# ─── 3. Datasets (download from public sources) ───────────────────────────────
fetch_droid() {                                  # -> $DATA_ROOT/droid_lerobot {data,meta,videos}
  [ "$DROID_ENABLE" = 1 ] || { log "droid: disabled"; return; }
  [ "$FETCH" = download ] || { log "droid: FETCH=$FETCH (using existing data)"; return; }
  mkdir -p "$DATA_ROOT/droid_lerobot"
  # cadene/droid is LeRobot v2.1 (data/chunk-*/episode_*.parquet + per-episode mp4 with the
  # exact camera names the loader uses) -> loads directly, no conversion. ~1000 episodes/chunk.
  local nchunks
  if [ "$DROID_LIMIT" = all ]; then nchunks=93; else nchunks=$(( (DROID_LIMIT + 999) / 1000 )); fi
  log "hf download cadene/droid (v2.1): meta + $nchunks chunk(s) (~$((nchunks * 1000)) episodes)"
  local inc=(--include "meta/*") i c cam
  for i in $(seq 0 $((nchunks - 1))); do
    c=$(printf %03d "$i"); inc+=(--include "data/chunk-$c/*")
    for cam in exterior_image_1_left wrist_image_left; do   # the two views the loader reads
      inc+=(--include "videos/chunk-$c/observation.images.$cam/*")
    done
  done
  huggingface-cli download cadene/droid --repo-type dataset --local-dir "$DATA_ROOT/droid_lerobot" "${inc[@]}"
}

fetch_agibot() {                                 # -> $DATA_ROOT/agibot {observations,proprio_stats,parameters}
  [ "$AGIBOT_ENABLE" = 1 ] || { log "agibot: disabled"; return; }
  [ "$FETCH" = download ] || { log "agibot: FETCH=$FETCH (using existing data)"; return; }
  # The release ships ~48 GB tar archives; this streams them and keeps only the three camera
  # videos, proprio_stats.h5 and the aligned extrinsics of whole tasks covering the cap.
  local lim="$AGIBOT_LIMIT"; [ "$lim" = all ] && lim=1000000000
  log "agibot: streaming AgiBotWorld-Alpha (cap $AGIBOT_LIMIT episodes) -> $DATA_ROOT/agibot"
  python "$REPO_DIR/tools/data/fetch_data.py" agibot --out "$DATA_ROOT/agibot" --limit "$lim"
}

fetch_egodex() {                                 # -> $DATA_ROOT/egodex_cdn/<part>/<task>/<n>.hdf5
  [ "$EGODEX_ENABLE" = 1 ] || { log "egodex: disabled"; return; }
  [ "$FETCH" = download ] || { log "egodex: FETCH=$FETCH (using existing data)"; return; }
  mkdir -p "$DATA_ROOT/egodex_cdn"
  for p in $EGODEX_PARTS; do
    log "egodex: download + unzip $p (Apple CDN, ~300GB each for part1-5)"
    curl -L "https://ml-site.cdn-apple.com/datasets/egodex/${p}.zip" -o "$DATA_ROOT/egodex_cdn/${p}.zip"
    unzip -q -o "$DATA_ROOT/egodex_cdn/${p}.zip" -d "$DATA_ROOT/egodex_cdn/" && rm -f "$DATA_ROOT/egodex_cdn/${p}.zip"
  done
}

fetch_abc() {                                    # -> $DATA_ROOT/abc_pp/<task>/episode_*/{*.mp4,states.npz}
  [ "$ABC_ENABLE" = 1 ] || { log "abc: disabled"; return; }
  [ "$FETCH" = download ] || { log "abc: FETCH=$FETCH (using existing data)"; return; }
  # XDOF/ABC-130k ships one raw episode.mcap (~0.5 GB) per episode; fetch only the cap,
  # spread over the tasks, then remux to per-camera mp4 + states.npz.
  local lim="$ABC_LIMIT"; [ "$lim" = all ] && lim=1000000000
  log "abc: downloading up to $ABC_LIMIT episodes -> $DATA_ROOT/abc_raw"
  python "$REPO_DIR/tools/data/fetch_data.py" abc --out "$DATA_ROOT/abc_raw" --limit "$lim"
  log "abc: preprocessing mcap -> abc_pp (parallel, idempotent; set ABC_NPROC to tune)"
  ABC_RAW="$DATA_ROOT/abc_raw/data/train" ABC_PP="$DATA_ROOT/abc_pp" \
    python -m robot_wm.datasets.abc.preprocessing.abc_batch_preprocess
  if [ "$ABC_KEEP_RAW" != 1 ]; then
    log "abc: removing raw mcap files (ABC_KEEP_RAW=1 keeps them)"
    rm -rf "$DATA_ROOT/abc_raw"
  fi
}

fetch_datasets() { fetch_droid; fetch_agibot; fetch_egodex; fetch_abc; }

# ─── 4. Manifests (regenerated for THIS server; capped by *_LIMIT) ────────────
make_manifests() {
  log "Regenerating manifests under $DATA_ROOT"
  local args=(--out "$DATA_ROOT" --only) n lim
  [ "$EGODEX_ENABLE" = 1 ] && args+=(egodex)
  [ "$AGIBOT_ENABLE" = 1 ] && args+=(agibot)
  [ "$ABC_ENABLE" = 1 ]    && args+=(abc)
  for n in EGODEX AGIBOT ABC; do
    lim="${n}_LIMIT"
    [ "${!lim}" = all ] || args+=("--$(echo "$n" | tr '[:upper:]' '[:lower:]')-limit" "${!lim}")
  done
  python "$REPO_DIR/tools/data/build_manifests.py" "${args[@]}"
  if [ "$DROID_ENABLE" = 1 ] && [ -d "$DATA_ROOT/droid_lerobot/data" ]; then
    log "  droid  : $(find "$DATA_ROOT/droid_lerobot/data" -name 'episode_*.parquet' 2>/dev/null | wc -l) episodes (no manifest; the loader reads the directory)"
  fi
}

# ─── 5. Write a sourceable env file (configs read these vars at train time) ───
write_env() {
  local f="$REPO_DIR/.lacwm_env"
  cat > "$f" <<EOF
# Written by setup_training.sh; tools/launch/env.sh sources it, so every launcher
# resolves to this server's paths.
export LACWM_DATA="$LACWM_DATA"
export LACWM_RUNS="$LACWM_RUNS"
export WAN_DIR="$WAN_DIR"
export VIDEOX_HOME="$VIDEOX_HOME"
export CONDA_ENV="$CONDA_ENV"
# setup_training.sh downloads DROID in LeRobot form
export DROID_LOADER=robot_wm.datasets.droid.lerobot_dataset.DroidLeRobotDataset
export DROID_DATA_DIR="$LACWM_DATA/droid_lerobot"
EOF
  log "wrote $f  (tools/launch/env.sh sources it)"
}

# ─── 6. Validate + launch hint ────────────────────────────────────────────────
validate() {
  log "Validating"
  local ok=1
  for f in "$WAN_DIR/Wan2.1_VAE.pth" "$WAN_DIR/diffusion_pytorch_model.safetensors" \
           "$WAN_DIR/null_prompt_umt5.pt" "$VIDEOX_DIR/config/wan2.1/wan_civitai.yaml"; do
    [ -e "$f" ] && echo "  ok  $f" || { echo "  MISSING  $f"; ok=0; }
  done
  [ "$EGODEX_ENABLE" = 1 ] && { [ -f "$DATA_ROOT/egodex/manifest.csv" ]     && echo "  ok  egodex manifest" || { echo "  MISSING egodex manifest"; ok=0; }; }
  [ "$ABC_ENABLE" = 1 ]    && { [ -f "$DATA_ROOT/abc_pp/manifest.txt" ]     && echo "  ok  abc manifest"    || { echo "  MISSING abc manifest"; ok=0; }; }
  [ "$AGIBOT_ENABLE" = 1 ] && { [ -f "$DATA_ROOT/agibot/manifest.csv" ]     && echo "  ok  agibot manifest" || { echo "  MISSING agibot manifest"; ok=0; }; }
  [ "$DROID_ENABLE" = 1 ]  && { [ -d "$DATA_ROOT/droid_lerobot/data" ]      && echo "  ok  droid data"      || { echo "  MISSING droid data"; ok=0; }; }
  [ "$ok" = 1 ] && log "Validation passed." || warn "Validation found gaps (see above + any [PREP] notes)."
  cat <<EOF

Next (from $REPO_DIR):
  mkdir -p slurm_logs
  sbatch --account=<acct> --partition=<part> tools/launch/verify_data.sbatch
  sbatch --account=<acct> --partition=<part> --gres=gpu:1 tools/launch/train.sbatch pretrain_smoke 1
  sbatch --account=<acct> --partition=<part> --gres=gpu:8 tools/launch/train.sbatch pretrain 8
EOF
}

case "${1:-all}" in
  all)       setup_env; fetch_weights; fetch_datasets; make_manifests; write_env; validate ;;
  env)       setup_env ;;
  weights)   fetch_weights ;;
  datasets)  fetch_datasets ;;
  manifests) make_manifests ;;
  paths)     write_env ;;
  validate)  validate ;;
  *) die "usage: $0 [all|env|weights|datasets|manifests|paths|validate]" ;;
esac
