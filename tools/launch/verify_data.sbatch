#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=verify_data
#SBATCH --output=slurm_logs/%x_%j.log
# Pre-flight check on the pretraining data. CPU only. Submit from the repo root:
#   sbatch --account=<acct> --partition=<part> tools/launch/verify_data.sbatch [--skip Droid ...]
# Run it as a batch job rather than on a login node: it decodes real video for minutes.
set -eo pipefail
REQUIRED_DIRS=LACWM_DATA
source "${LACWM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/tools/launch/env.sh"
cd "$REPO"
python tools/data/verify_pretrain_data.py "$@"
