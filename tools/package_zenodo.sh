#!/bin/bash
# Builds the upload folders of the three Zenodo records (tars + README + SHA256SUMS).
# Tars are written straight from the source folders, nothing is copied first.
#
#   DATA_ROOT     folder with annotations/, WavFiles/, plain/features/logmel/
#   SHIFTED_ROOT  output of scripts/extract_features.py --save_wav for awgn, clutter, random
#   OUT_ROOT      where <record>/upload/ folders are created
#
# Usage: bash tools/package_zenodo.sh DATA_ROOT SHIFTED_ROOT OUT_ROOT [clean|shifted|code|all]
set -euo pipefail
DATA_ROOT=$1; SHIFTED_ROOT=$2; OUT_ROOT=$3; WHAT=${4:-all}
RELEASE=$(cd "$(dirname "$0")/.." && pwd)

checksums() { (cd "$1" && sha256sum $(ls | grep -v SHA256SUMS) > SHA256SUMS); }

if [[ $WHAT == clean || $WHAT == all ]]; then
  U=$OUT_ROOT/DDL_clean/upload; mkdir -p "$U"
  tar -czf "$U/annotations.tar.gz" -C "$DATA_ROOT" annotations
  tar -cf  "$U/WavFiles.tar"       -C "$DATA_ROOT" WavFiles
  tar -cf  "$U/plain_logmel.tar"   -C "$DATA_ROOT" plain/features/logmel
  cp "$RELEASE/zenodo/DDL_clean_README.md" "$U/README.md"
  checksums "$U"; echo "[OK] $U"
fi

if [[ $WHAT == shifted || $WHAT == all ]]; then
  U=$OUT_ROOT/DDL_shifted/upload; mkdir -p "$U"
  for c in awgn_15dB awgn_5dB awgn_-5dB clutter_15dB clutter_5dB clutter_-5dB random_noise; do
    tar -cf "$U/$c.tar" -C "$SHIFTED_ROOT" "$c"
  done
  tar -czf "$U/verification.tar.gz" -C "$SHIFTED_ROOT" $(cd "$SHIFTED_ROOT" && ls verify_*.csv)
  cp "$RELEASE/zenodo/DDL_shifted_README.md" "$U/README.md"
  checksums "$U"; echo "[OK] $U"
fi

if [[ $WHAT == code || $WHAT == all ]]; then
  U=$OUT_ROOT/code_weights/upload; mkdir -p "$U"
  tar -cf  "$U/weights.tar"           -C "$RELEASE" weights
  tar -czf "$U/per_sample.tar.gz"     -C "$RELEASE" results/final_seed12321/per_sample
  cp "$RELEASE/zenodo/code_weights_README.md" "$U/README.md"
  checksums "$U"; echo "[OK] $U"
fi
