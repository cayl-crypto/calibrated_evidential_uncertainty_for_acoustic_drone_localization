#!/bin/bash
# Base Model + Conformal: train -> clean calibration -> clean test (scripts/run_base.py).
#SBATCH --job-name=ddl-base
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/base.%j.out
#SBATCH -e slurm_logs/base.%j.err

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p slurm_logs
# source /path/to/venv/bin/activate
python -u scripts/run_base.py
