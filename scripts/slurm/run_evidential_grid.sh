#!/bin/bash
# SLURM array over the final grid: 16 settings (coeff x kl_weight), seed 12321, one GPU each.
# Each task: train -> clean calibration -> clean + AWGN/clutter test -> validation (scripts/run_evidential.py).
# Edit the partition / environment lines for your cluster; set DDL_DATA_ROOT (and optionally
# DDL_RUNS_ROOT, DDL_RESULTS_ROOT) before submitting:  sbatch scripts/slurm/run_evidential_grid.sh
#SBATCH --job-name=ddl-evidential
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH -o slurm_logs/evidential.%A_%a.out
#SBATCH -e slurm_logs/evidential.%A_%a.err

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p slurm_logs
# source /path/to/venv/bin/activate
python -u scripts/run_evidential.py --job_index "${SLURM_ARRAY_TASK_ID}"
