#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x_%j.log
# Training launcher for pretraining and every adaptation stage. Submit from the repo root:
#   mkdir -p slurm_logs
#   sbatch --account=<acct> --partition=<part> --gres=gpu:<n> \
#     tools/launch/train.sbatch <experiment> <nproc> [hydra overrides ...]
# <experiment> is a file name under projects/latent_action_models/configs/experiment/
# (with or without .yaml). Account and partition are site-specific, so they are not set
# here: pass them as flags or export SBATCH_ACCOUNT / SBATCH_PARTITION. Paths are
# configured in tools/launch/env.sh.
set -e
EXPERIMENT=${1:?experiment name under configs/experiment/, e.g. pretrain}
EXPERIMENT=${EXPERIMENT%.yaml}
NPROC=${2:-1}
shift $(( $# < 2 ? $# : 2 ))
source "${LACWM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/tools/launch/env.sh"

cd "$REPO/projects/latent_action_models"
echo "[launch] host=$(hostname) experiment=$EXPERIMENT nproc=$NPROC runs=$LACWM_RUNS overrides=$*"
nvidia-smi -L || true
torchrun --standalone --nproc_per_node="$NPROC" train.py "+experiment=$EXPERIMENT" "$@"
