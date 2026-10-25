"""
run_evidential.py

One (coeff, kl_weight) setting of the final grid, end to end (from run_seed_repeat.py):
  1. train on clean audio                         -> <runs_root>/evidential/<hp>/model.pt
  2. conformal q-hat on the clean calibration split -> <runs_root>/evidential/<hp>/conformal_q_{head}_{tag}.pt
  3. per-sample results with the clean q-hat on the clean test set and the six AWGN / clutter
     test sets                                      -> <results_root>/per_sample/evidential/<hp>/<condition>.csv
  4. per-sample results on the clean validation evaluation split (used for HP selection)
                                                    -> <results_root>/per_sample/evidential/<hp>/plain_val.csv
Each stage is skipped when its outputs exist and are complete, so a task can be resubmitted.

Grid: coeff {0.05, 0.1, 0.3, 0.7} x kl_weight {0, 0.01, 0.05, 0.2}, seed 12321; --job_index 0..15
selects the setting (coeff-major order).

Usage:  python scripts/run_evidential.py --job_index 13            (c 0.7, eta 0.01: the selected model)
        python scripts/run_evidential.py --coeff 0.7 --kl_weight 0.01
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddl_conformal import config  # noqa: E402
from ddl_conformal.conformal import HEADS, collect_evidential_scores, load_qhats, save_qhats  # noqa: E402
from ddl_conformal.data import get_localization_loaders  # noqa: E402
from ddl_conformal.evaluation import evaluate_one_cached  # noqa: E402
from ddl_conformal.models import load_checkpoint  # noqa: E402
from ddl_conformal.training import train_evidential  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_complete_csv(path: Path, n_rows: int) -> bool:
    if not path.exists():
        return False
    with open(path, "rb") as f:
        return sum(1 for _ in f) == n_rows + 1


def write_rows_csv(path: Path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_index", type=int, default=None, help="0..15 over the final grid")
    ap.add_argument("--coeff", type=float, default=None)
    ap.add_argument("--kl_weight", type=float, default=None)
    ap.add_argument("--seed", type=int, default=config.FINAL_SEED)
    ap.add_argument("--data_root", default=str(config.DATA_ROOT))
    ap.add_argument("--runs_root", default=str(config.RUNS_ROOT))
    ap.add_argument("--results_root", default=str(config.RESULTS_ROOT))
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    if args.job_index is not None:
        coeff, kl_weight = [(c, k) for c in config.FINAL_COEFFS for k in config.FINAL_KL_WEIGHTS][args.job_index]
    elif args.coeff is not None and args.kl_weight is not None:
        coeff, kl_weight = args.coeff, args.kl_weight
    else:
        ap.error("give --job_index or both --coeff and --kl_weight")

    hp = config.hp_tag(coeff, kl_weight)
    run_dir = Path(args.runs_root) / "evidential" / hp
    ps_dir = Path(args.results_root) / "per_sample" / "evidential" / hp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] coeff={coeff} kl_weight={kl_weight} seed={args.seed} device={device}\n"
          f"       weights -> {run_dir}\n       per-sample -> {ps_dir}")

    def loaders(condition):
        return get_localization_loaders(args.data_root, condition, batch_size=64,
                                        num_workers=args.num_workers, pin_memory=False)

    # 1. train
    if (run_dir / "model.pt").exists() and (run_dir / "run_summary.csv").exists():
        print("[SKIP] training already done")
    else:
        set_seed(args.seed)
        summary = train_evidential(args.data_root, run_dir, device, coeff, kl_weight,
                                   num_workers=args.num_workers, pin_memory=True)
        summary["seed"] = args.seed
        write_rows_csv(run_dir / "run_summary.csv", [summary])
    model = load_checkpoint(run_dir / "model.pt", map_location=device).to(device)

    # 2. calibrate on the clean calibration split
    if all((run_dir / f"conformal_q_{h}_{t}.pt").exists() for h in HEADS for t in ["6827", "9545", "9973"]):
        print("[SKIP] calibration already done")
    else:
        _, _, val_calib_loader, _ = loaders("plain")
        scores = collect_evidential_scores(model, val_calib_loader, device)
        save_qhats(scores, run_dir, {h: "ratio |y-mu|/sigma" for h in HEADS}, run_name=hp)
        print(f"[OK] calibration written (n_calib={scores['sin'].numel()})")
    q_values = load_qhats(run_dir)

    # 3. clean test + 6 shifted test sets, clean q-hat
    ps_dir.mkdir(parents=True, exist_ok=True)
    for condition in config.TEST_CONDITIONS:
        out_path = ps_dir / f"{condition}.csv"
        if is_complete_csv(out_path, config.EXPECTED_TEST_ROWS):
            print(f"[SKIP] {condition} already done")
            continue
        _, _, _, test_loader = loaders(condition)
        df = evaluate_one_cached(model, test_loader, q_values, device, hp, desc=f"{hp} @ {condition}")
        df["test_setting"] = condition
        df["seed"] = args.seed
        df.to_csv(out_path, index=False)
        print(f"[OK] {condition}: {len(df)} rows -> {out_path}")

    # 4. clean validation evaluation split (HP selection)
    out_path = ps_dir / "plain_val.csv"
    if is_complete_csv(out_path, config.EXPECTED_VAL_ROWS):
        print("[SKIP] validation already done")
    else:
        _, val_eval_loader, _, _ = loaders("plain")
        df = evaluate_one_cached(model, val_eval_loader, q_values, device, hp, desc=f"{hp} @ val")
        df["test_setting"] = "plain_val"
        df["seed"] = args.seed
        df.to_csv(out_path, index=False)
        print(f"[OK] validation: {len(df)} rows -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
