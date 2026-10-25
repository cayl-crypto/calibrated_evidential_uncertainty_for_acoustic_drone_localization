"""
config.py

The only place paths are defined. Every script takes --data_root / --runs_root / --results_root
flags; when a flag is not given, the environment variable is used, then the default below.

  DDL_DATA_ROOT     unpacked DDL_clean + DDL_shifted records (one folder):
                      annotations/, WavFiles/, plain/features/logmel/,
                      awgn_{15,5,-5}dB/..., clutter_{15,5,-5}dB/..., random_noise/...
  DDL_RUNS_ROOT     model weights and conformal q-hat files, one folder per run
                      (default: <release>/weights)
  DDL_RESULTS_ROOT  per-sample CSVs, tables and figures (default: <release>/results/final_seed12321)
"""
from __future__ import annotations

import os
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("DDL_DATA_ROOT", RELEASE_ROOT / "data"))
RUNS_ROOT = Path(os.environ.get("DDL_RUNS_ROOT", RELEASE_ROOT / "weights"))
RESULTS_ROOT = Path(os.environ.get("DDL_RESULTS_ROOT", RELEASE_ROOT / "results" / "final_seed12321"))

# Final grid of the paper: evidence-regularizer coefficient c x external-prior KL weight eta.
# eta is called kl_weight / "kl" in code and file names (c0p7_kl0p01 = c 0.7, eta 0.01).
FINAL_COEFFS = [0.05, 0.1, 0.3, 0.7]
FINAL_KL_WEIGHTS = [0.0, 0.01, 0.05, 0.2]
FINAL_SEED = 12321

TEST_CONDITIONS = ["plain", "awgn_15dB", "awgn_5dB", "awgn_-5dB", "clutter_15dB", "clutter_5dB", "clutter_-5dB"]
SNR_LEVELS_DB = [15, 5, -5]

MAX_RANGE = 250.0      # metres; range targets are normalised by this
SAMPLE_RATE = 96000
AUDIO_LENGTH = 9600    # samples per recording (0.1 s)
NUM_CHANNELS = 8

EXPECTED_TEST_ROWS = 11628
EXPECTED_VAL_ROWS = 2904    # validation evaluation split (conformal_role == "evaluation")
EXPECTED_CALIB_ROWS = 2912  # validation calibration split (conformal_role == "calibration")


def annotations_dir(data_root: Path) -> Path:
    return Path(data_root) / "annotations"


def logmel_dir(data_root: Path, condition: str) -> Path:
    return Path(data_root) / condition / "features" / "logmel"


def wav_dir(data_root: Path, condition: str) -> Path:
    """Clean recordings are in WavFiles/; each shifted condition keeps its own wav/ folder."""
    return Path(data_root) / "WavFiles" if condition == "plain" else Path(data_root) / condition / "wav"


def hp_tag(coeff: float, kl_weight: float) -> str:
    return f"c{coeff:g}_kl{kl_weight:g}".replace(".", "p")
