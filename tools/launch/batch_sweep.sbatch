#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.log
# Find the largest per-GPU batch that fits: runs three real training iterations of the
# pretraining recipe at each candidate size. Pick the GPU type with --gres. Submit from the
# repo root:
#   sbatch --account=<acct> --partition=<part> --gres=gpu:1 --cpus-per-task=12 --mem=150G \
#     tools/launch/batch_sweep.sbatch 4 8 16
# Extra hydra overrides can be passed in SWEEP_OVERRIDES (e.g. the no-ABC deletions).
set -uo pipefail
source "${LACWM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/tools/launch/env.sh"
export WANDB_MODE=disabled
cd "$REPO/projects/latent_action_models"
nvidia-smi -L || true
for B in "$@"; do
  echo "###### BATCH $B ######"
  # shellcheck disable=SC2086
  timeout 900 torchrun --standalone --nproc_per_node=1 train.py \
    "+experiment=pretrain_sweep" "data_loader.batch_size=$B" ${SWEEP_OVERRIDES:-} \
    2>&1 | grep -aiE "loss:|out of memory|ERROR __main__" | head -6
  echo "###### BATCH $B done ######"
done
echo SWEEP_DONE
