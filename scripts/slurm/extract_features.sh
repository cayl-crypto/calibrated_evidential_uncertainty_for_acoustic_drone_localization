#!/bin/bash
# CPU job: log-mel features of every condition; add --save_wav to also write the shifted WAVs.
#SBATCH --job-name=ddl-features
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o slurm_logs/features.%j.out
#SBATCH -e slurm_logs/features.%j.err

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p slurm_logs
# source /path/to/venv/bin/activate
for c in plain awgn clutter random; do
  python -u scripts/extract_features.py --condition "$c"
done
