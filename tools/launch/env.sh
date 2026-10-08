# Shared environment for every launcher in this directory. Sourced, not executed.
#
# Everything is overridable by exporting the variable before sbatch (or through the
# .lacwm_env file setup_training.sh writes). Defaults are relative to the repo root you
# submit from:
#   read-only : WAN_DIR (Wan2.1-Fun weights), VIDEOX_HOME (VideoX-Fun), LACWM_DATA (datasets)
#   written   : LACWM_BASE (runs, tmp, wandb, caches)
# REQUIRED_DIRS lists which read-only paths must exist; a launcher that does not need the
# model (e.g. the data check) narrows it before sourcing this file.

REPO=${LACWM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
# Paths recorded by setup_training.sh on this machine, if it was used.
# shellcheck disable=SC1091
[ -f "$REPO/.lacwm_env" ] && source "$REPO/.lacwm_env"
BASE=${LACWM_BASE:-$REPO/lacwm_out}

# conda is not on PATH inside a batch job until its profile is sourced, so it cannot be
# located with `conda info --base` here.
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
CONDA_SH=${CONDA_SH:-$CONDA_BASE/etc/profile.d/conda.sh}
[ -f "$CONDA_SH" ] || { echo "conda profile not found at $CONDA_SH; set CONDA_SH or CONDA_BASE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "${CONDA_ENV:-lacwm-dit}"

export WAN_DIR=${WAN_DIR:-$REPO/checkpoints/wan_fun_1.3b_control}
export VIDEOX_HOME=${VIDEOX_HOME:-$REPO/VideoX-Fun}
export LACWM_DATA=${LACWM_DATA:-$REPO/data}
for v in ${REQUIRED_DIRS:-WAN_DIR VIDEOX_HOME LACWM_DATA}; do
  [ -d "${!v}" ] || { echo "$v does not exist: ${!v} -- set it for this machine" >&2; exit 1; }
done
export AGIBOT_ALPHA_ROOT=${AGIBOT_ALPHA_ROOT:-$LACWM_DATA/agibot}

export LACWM_RUNS=${LACWM_RUNS:-$BASE/runs}
export TMPDIR=${TMPDIR:-$BASE/tmp}
export WANDB_DIR=${WANDB_DIR:-$BASE/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$BASE/wandb_cache}
export WANDB_CONFIG_DIR=$WANDB_CACHE_DIR
export WANDB_DATA_DIR=$WANDB_CACHE_DIR
export DROID_SHARD_CACHE=${DROID_SHARD_CACHE:-$BASE/.droid_shard_offsets}
mkdir -p "$TMPDIR" "$LACWM_RUNS" "$WANDB_DIR" "$WANDB_CACHE_DIR"

# The repo root comes first so this checkout's robot_wm wins over any other copy installed
# in the environment (an editable install of a different checkout, for example).
export PYTHONPATH=$REPO:$REPO/projects/latent_action_models:$VIDEOX_HOME:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Per-job bytecode cache. On a shared filesystem a job started within ~60 s of a code edit
# can see stale file attributes, accept the old .pyc as valid and run the pre-edit code.
export PYTHONPYCACHEPREFIX=$TMPDIR/pycache_${SLURM_JOB_ID:-$$}
