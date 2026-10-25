"""
run_base.py

Base Model + Conformal, end to end (from train_localization.py,
calibrate_base_localization_v2_tim_coverage_levels.py and document_base_localization_v5_tim_coverage_levels.py):
  1. train on clean audio (MSE)                       -> <runs_root>/base_model/model.pt
  2. q-hat on the clean calibration split, absolute errors per output
     (sin, cos, range [m], bearing [deg], euclidean [m]) -> <runs_root>/base_model/conformal_q_{type}_{tag}.pt
  3. per-sample results on the clean test set          -> <results_root>/per_sample/base_model/plain.csv

Usage:  python scripts/run_base.py [--seed 0]
The released Base Model was trained once without a fixed seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddl_conformal import config  # noqa: E402
from ddl_conformal.conformal import (ALPHA_TAGS, BASE_SCORE_TYPES, BASE_UNITS, collect_base_scores,  # noqa: E402
                                     load_qhats, save_qhats)
from ddl_conformal.data import get_localization_loaders  # noqa: E402
from ddl_conformal.evaluation import evaluate_base_per_sample  # noqa: E402
from ddl_conformal.models import load_checkpoint  # noqa: E402
from ddl_conformal.training import train_base  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evidential import set_seed, write_rows_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data_root", default=str(config.DATA_ROOT))
    ap.add_argument("--runs_root", default=str(config.RUNS_ROOT))
    ap.add_argument("--results_root", default=str(config.RESULTS_ROOT))
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    run_dir = Path(args.runs_root) / "base_model"
    ps_dir = Path(args.results_root) / "per_sample" / "base_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (run_dir / "model.pt").exists():
        print("[SKIP] training already done")
    else:
        if args.seed is not None:
            set_seed(args.seed)
        summary = train_base(args.data_root, run_dir, device, num_workers=args.num_workers)
        summary["seed"] = args.seed
        write_rows_csv(run_dir / "run_summary.csv", [summary])
    model = load_checkpoint(run_dir / "model.pt", map_location=device).to(device)

    _, _, val_calib_loader, test_loader = get_localization_loaders(
        args.data_root, "plain", batch_size=64, num_workers=args.num_workers, pin_memory=False)
    if all((run_dir / f"conformal_q_{st}_{t}.pt").exists() for st in BASE_SCORE_TYPES for t in ALPHA_TAGS):
        print("[SKIP] calibration already done")
    else:
        save_qhats(collect_base_scores(model, val_calib_loader, device), run_dir, BASE_UNITS, run_name="base_model")
    q = load_qhats(run_dir, BASE_SCORE_TYPES)                       # {tag: {type: q}}
    qhats = {st: {t: q[t][st] for t in ALPHA_TAGS} for st in BASE_SCORE_TYPES}

    ps_dir.mkdir(parents=True, exist_ok=True)
    df, summary = evaluate_base_per_sample(model, test_loader, qhats, device, run_name="base_model")
    df.to_csv(ps_dir / "plain.csv", index=False)
    summary.to_csv(ps_dir / "plain_summary.csv", index=False)
    print(f"[OK] {len(df)} rows -> {ps_dir / 'plain.csv'}")


if __name__ == "__main__":
    main()
