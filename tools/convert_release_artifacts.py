"""
convert_release_artifacts.py  (one-time, run by the authors; kept for provenance)

Copies the outputs of the original final run into the release layout:
  * the original checkpoints pickled the whole module (torch.save(model)); each is loaded with the
    original code, its state_dict is put into the release model class, the outputs of both on real
    test features are compared, and only then saved as model.pt (state_dict + config);
  * conformal q-hat files, run_summary.csv and val_epoch_csv/ are copied unchanged;
  * per-sample CSVs are copied and renamed  run=plain__<hp>__seed12321__test=<cond>.csv -> <cond>.csv;
  * Base Model checkpoint, q-hat files (clean calibration) and clean-test per-sample CSV likewise;
  * results/final_seed12321/qhat_table.csv lists every q-hat.

Usage: python tools/convert_release_artifacts.py --orig_repo <original repo> --final_runs <final run dir>
       --data_root <DDLDataset>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

RELEASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RELEASE))
from ddl_conformal import config  # noqa: E402
from ddl_conformal.conformal import ALPHA_TAGS, BASE_SCORE_TYPES, HEADS  # noqa: E402
from ddl_conformal.models import build_base, build_evidential, save_checkpoint  # noqa: E402


def test_batch(data_root: Path, n: int = 64) -> torch.Tensor:
    files = sorted(config.logmel_dir(data_root, "plain").glob("*.pt"))[:n]
    return torch.stack([torch.load(f, weights_only=True).float() for f in files])


@torch.no_grad()
def max_diff_evidential(old, new, x) -> float:
    (m0, p0), (m1, p1) = old.forward_all(x), new.forward_all(x)
    return max(float((a - b).abs().max()) for a, b in zip(list(m0) + list(p0), list(m1) + list(p1)))


@torch.no_grad()
def max_diff_base(old, new, x) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(old(x), new(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_repo", required=True)
    ap.add_argument("--final_runs", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--runs_root", default=str(config.RUNS_ROOT))
    ap.add_argument("--results_root", default=str(config.RESULTS_ROOT))
    args = ap.parse_args()

    repo, final_runs = Path(args.orig_repo), Path(args.final_runs)
    runs_root, results_root = Path(args.runs_root), Path(args.results_root)
    for p in [repo, repo / "evidential_external_prior_ablation_codes", repo / "evidential_loss_review"]:
        sys.path.append(str(p))    # original modules, needed only to unpickle the original checkpoints
    x = test_batch(Path(args.data_root))
    q_rows, log = [], []

    for c in config.FINAL_COEFFS:
        for k in config.FINAL_KL_WEIGHTS:
            hp = config.hp_tag(c, k)
            src = final_runs / f"plain__{hp}__seed{config.FINAL_SEED}"
            dst = runs_root / "evidential" / hp
            old = torch.load(src / "checkpoints" / "best_evidential_external_prior_model.pt",
                             map_location="cpu", weights_only=False).eval()
            new = build_evidential(c, k)
            new.load_state_dict(old.state_dict())
            new.eval()
            diff = max_diff_evidential(old, new, x)
            assert diff == 0.0, f"{hp}: outputs differ by {diff}"
            save_checkpoint(new, dst / "model.pt", coeff=c, kl_weight=k)
            for f in (src / "conformal_calibration_tim_coverage_levels").glob("conformal_q_*.pt"):
                shutil.copy2(f, dst / f.name)
                p = torch.load(f, weights_only=False)
                q_rows.append(dict(model="evidential", hp=hp, coeff=c, kl_weight=k, head=p["score_type"],
                                   alpha_tag=p["alpha_tag"], alpha=p["alpha"], qhat=p["qhat"], n_calib=p["n_calib"]))
            shutil.copy2(src / "run_summary.csv", dst / "run_summary.csv")
            if (src / "val_epoch_csv").exists():
                shutil.copytree(src / "val_epoch_csv", dst / "val_epoch_csv", dirs_exist_ok=True)
            ps = results_root / "per_sample" / "evidential" / hp
            ps.mkdir(parents=True, exist_ok=True)
            for f in (src / "per_sample").glob("*.csv"):
                shutil.copy2(f, ps / (f.stem.split("__test=")[1] + ".csv"))
            log.append(dict(model=hp, max_abs_output_diff=diff))
            print(f"[OK] {hp}: outputs identical, {len(list(ps.glob('*.csv')))} per-sample files")

    # Base Model
    base_run = repo / "base_localization_runs_ablation" / "plain_results_v3_ablation" / "plain__un0_wr1"
    base_q = (repo / "base_localization_runs_ablation" / "conformal_calibration_tim_coverage_levels"
              / "model_source=plain" / "calib_source=plain" / "plain__un0_wr1")
    base_ps = (repo / "base_localization_runs_ablation" / "ood_per_sample_results_grid_mismatch_tim_coverage_levels"
               / "model_source=plain" / "calib_source=plain" / "run=plain__un0_wr1__test=plain.csv")
    dst = runs_root / "base_model"
    old = torch.load(base_run / "checkpoints" / "best_plain_model.pt", map_location="cpu", weights_only=False).eval()
    new = build_base()
    new.load_state_dict(old.state_dict())
    new.eval()
    diff = max_diff_base(old, new, x)
    assert diff == 0.0, f"base: outputs differ by {diff}"
    save_checkpoint(new, dst / "model.pt")
    for st in BASE_SCORE_TYPES:
        for t in ALPHA_TAGS:
            f = base_q / f"conformal_q_{st}_{t}.pt"
            shutil.copy2(f, dst / f.name)
            p = torch.load(f, weights_only=False)
            q_rows.append(dict(model="base_model", hp="", coeff="", kl_weight="", head=st, alpha_tag=t,
                               alpha=p["alpha"], qhat=p["qhat"], n_calib=p["n_calib"]))
    if (base_run / "val_epoch_csv").exists():
        shutil.copytree(base_run / "val_epoch_csv", dst / "val_epoch_csv", dirs_exist_ok=True)
    (results_root / "per_sample" / "base_model").mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_ps, results_root / "per_sample" / "base_model" / "plain.csv")
    log.append(dict(model="base_model", max_abs_output_diff=diff))
    print("[OK] base model: outputs identical")

    q = pd.DataFrame(q_rows).sort_values(["model", "hp", "head", "alpha_tag"])
    q.to_csv(results_root / "qhat_table.csv", index=False)
    pd.DataFrame(log).to_csv(runs_root / "conversion_check.csv", index=False)
    print(f"[OK] {len(q)} q-hat values -> {results_root / 'qhat_table.csv'}")


if __name__ == "__main__":
    main()
