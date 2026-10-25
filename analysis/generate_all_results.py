"""
generate_all_results.py  --  tables and figures of the paper from the released per-sample results

Inputs (release layout, see ddl_conformal/config.py):
  <results_root>/per_sample/evidential/<hp>/{plain,plain_val,awgn_*,clutter_*}.csv   (scripts/run_evidential.py)
  <results_root>/per_sample/base_model/plain.csv                                    (scripts/run_base.py)
  <results_root>/qhat_table.csv     q-hat of every model, head and level (built from the weights if missing)
  <results_root>/test_sample_class.csv  drone class per test recording (for the per-class figure)

Method:
 1. HP selection on the VALIDATION evaluation split (2,904 clean recordings, disjoint from calibration
    and test): a setting passes if its sin, cos AND range conformal coverage at 95.45% are each
    >= 95.45%; passing settings are ranked by validation Euclidean MAE; the top 3 are reported
    (star = rank 1).
 2. Every interval uses its TRUE clipped bounds:
      Evid+Conf  L/U = clip(mu +/- qhat*sigma)
      NIG Base   stored L/U_*_base  (raw +/- 1 sigma)
      Plain+Conf L/U = clip(pred +/- q_{head}), q_range stored in metres -> / R_MAX
    Used for MWIS, interval area, coverage-vs-area and the polar figures. Coverage flags are
    recomputed from these bounds and checked against the stored ones.
 3. Robustness: the selected models (trained and calibrated on clean audio only) evaluated on AWGN /
    clutter at 15, 5, -5 dB with the clean q-hat.

Outputs -> <results_root>/
  tables/   hp_selection, accuracy_table, coverage_table, spatial_coverage_table,
            noise_coverage_table (.csv + .tex), noise_full_metrics.csv
  figures/  coverage_vs_area.png, range_bearing/, compact_comparison/, polar_qualitative/, noise_robustness.png
  all_settings/ (with --all_settings) every (coeff, kl) setting, clean and shifted

Usage: python analysis/generate_all_results.py [--results_root R] [--all_settings]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddl_conformal import config  # noqa: E402

RESULTS_ROOT = config.RESULTS_ROOT          # set by --results_root in main()
RUNS_ROOT = config.RUNS_ROOT
OUT_ROOT = RESULTS_ROOT.parent              # outputs go to OUT_ROOT / <exp> = RESULTS_ROOT
FINAL_GRID = (config.FINAL_COEFFS, config.FINAL_KL_WEIGHTS)
EXPERIMENTS = [RESULTS_ROOT.name]
CONDITIONS = ["plain", "awgn_15dB", "awgn_5dB", "awgn_-5dB", "clutter_15dB", "clutter_5dB", "clutter_-5dB"]
COND_LABEL = {"plain": "Clean", "awgn_15dB": "AWGN 15", "awgn_5dB": "AWGN 5", "awgn_-5dB": "AWGN $-$5",
              "clutter_15dB": "Clutter 15", "clutter_5dB": "Clutter 5", "clutter_-5dB": "Clutter $-$5"}

R_MAX        = 250.0
FULL_DISK_M2 = np.pi * R_MAX ** 2
TARGET_COV   = 0.9545
LEVELS    = [("q6827", 68.27), ("q9545", 95.45), ("q9973", 99.73)]
ALPHA_MAP = {"q6827": 1 - 0.6827, "q9545": 1 - 0.9545, "q9973": 1 - 0.9973}
HEADS = {"sin": (-1.0, 1.0), "cos": (-1.0, 1.0), "range": (0.0, 1.0)}

RANGE_EDGES   = np.array([0, 50, 100, 150, 200, 250], dtype=float)
BEARING_EDGES = np.linspace(0, 360, 9)
N_RANGE       = len(RANGE_EDGES) - 1
N_BEARING     = len(BEARING_EDGES) - 1
_TH_EDGES     = np.deg2rad(BEARING_EDGES)
_T_CENTERS    = np.deg2rad((BEARING_EDGES[:-1] + BEARING_EDGES[1:]) / 2)
_R_CENTERS    = (RANGE_EDGES[:-1] + RANGE_EDGES[1:]) / 2
_T_GRID, _R_GRID = np.meshgrid(_TH_EDGES, RANGE_EDGES)

_N   = 720
_TH  = np.linspace(0, 2 * np.pi, _N, endpoint=False)
_SIN = np.sin(_TH)
_COS = np.cos(_TH)

_COV_CMAP = matplotlib.colormaps["RdYlGn"].copy()
_COV_CMAP.set_bad("#d0d0d0")

EVID_COLS = [
    "sample_id", "y_sin", "y_cos", "y_range",
    "mu_sin", "mu_cos", "mu_range", "sigma_sin", "sigma_cos", "sigma_range", "theta_hat_rad",
    "L_sin_base", "U_sin_base", "L_cos_base", "U_cos_base", "L_range_base", "U_range_base",
    "inside_sin_base", "inside_cos_base", "inside_range_base",
] + [f"inside_{h}_{tag}" for tag, _ in LEVELS for h in HEADS]

BASE_COLS = [
    "sample_id",
    "err_sin", "err_cos", "err_range_m", "err_bearing_deg", "err_euclidean_m",
    "pred_sin", "pred_cos", "pred_range_norm", "gt_sin", "gt_cos", "gt_range_norm",
] + [f"cov_{h}_{tag}" for tag, _ in LEVELS for h in HEADS] \
  + [f"q_{h}_{tag[1:]}" for tag, _ in LEVELS for h in HEADS]

# ══════════════════════════════════════════════════════════════════════════════
# Experiment locators
# ══════════════════════════════════════════════════════════════════════════════

def evid_file(exp, hp, cond):
    """cond: a test condition, or 'plain_val' for the clean validation evaluation split."""
    return RESULTS_ROOT / "per_sample" / "evidential" / hp / f"{cond}.csv"


def exp_hps(exp):
    return [hp_tag(c, k) for c in FINAL_GRID[0] for k in FINAL_GRID[1]]


def val_metrics(exp):
    """Validation coverage at 95.45% per output, Euclidean MAE and MWIS for every setting."""
    rows = []
    for c in FINAL_GRID[0]:
        for k in FINAL_GRID[1]:
            hp = hp_tag(c, k)
            df = load_evid(exp, hp, "plain_val")
            rows.append(dict(coeff=c, kl_weight=k, run=exp, n_val=len(df),
                             **{f"cov_{h}_9545": float(df[f"inside_{h}_q9545"].mean()) for h in HEADS},
                             euclidean_mae_m=float(_euclidean_err(df).mean()),
                             mwis_9545=_mwis_evid_conf(df, "q9545", ALPHA_MAP["q9545"])))
    return pd.DataFrame(rows)


def _qhat_table():
    """q-hat of every model from qhat_table.csv; written from <runs_root>/evidential/<hp>/ if missing."""
    path = RESULTS_ROOT / "qhat_table.csv"
    if not path.exists():
        rows = []
        for hp in exp_hps(None):
            for f in (RUNS_ROOT / "evidential" / hp).glob("conformal_q_*.pt"):
                p = torch.load(f, weights_only=False)
                rows.append(dict(model="evidential", hp=hp, head=p["score_type"], alpha_tag=p["alpha_tag"], qhat=p["qhat"]))
        pd.DataFrame(rows).to_csv(path, index=False)
    q = pd.read_csv(path, dtype={"alpha_tag": str})
    return q[q.model == "evidential"]


def load_qhat(exp, hp):
    q = _qhat_table()
    q = q[q.hp == hp].set_index(["alpha_tag", "head"]).qhat
    return {tag: {h: float(q[(tag[1:], h)]) for h in HEADS} for tag, _ in LEVELS}


def hp_tag(c, k):
    return f"c{c:g}_kl{k:g}".replace(".", "p")

# ══════════════════════════════════════════════════════════════════════════════
# Interval helpers (true bounds)
# ══════════════════════════════════════════════════════════════════════════════

def _arc_rad_from_LU(L_sin, U_sin, L_cos, U_cos):
    Ls = np.clip(L_sin, -1.0, 1.0)[:, None]
    Us = np.clip(U_sin, -1.0, 1.0)[:, None]
    Lc = np.clip(L_cos, -1.0, 1.0)[:, None]
    Uc = np.clip(U_cos, -1.0, 1.0)[:, None]
    return ((_SIN >= Ls) & (_SIN <= Us) & (_COS >= Lc) & (_COS <= Uc)).mean(axis=1) * 2 * np.pi


def _annular_area_per_sample(L_r_norm, U_r_norm, arc_rad):
    L_r = np.clip(L_r_norm, 0.0, 1.0) * R_MAX
    U_r = np.clip(U_r_norm, 0.0, 1.0) * R_MAX
    return 0.5 * np.maximum(0.0, U_r ** 2 - L_r ** 2) * arc_rad


def _annular_area(L_r_norm, U_r_norm, arc_rad):
    return float(np.mean(_annular_area_per_sample(L_r_norm, U_r_norm, arc_rad)))


def _winkler(L, U, y, alpha):
    return (U - L) + (2 / alpha) * np.maximum(0, L - y) + (2 / alpha) * np.maximum(0, y - U)


def add_evid_bounds(df_e, q):
    """Evid+Conf bounds clip(mu +/- qhat*sigma) as columns L_{h}_{tag}/U_{h}_{tag}; checks stored flags."""
    mismatch = 0
    for tag, _ in LEVELS:
        for h, (lo, hi) in HEADS.items():
            L = np.clip(df_e[f"mu_{h}"] - q[tag][h] * df_e[f"sigma_{h}"], lo, hi)
            U = np.clip(df_e[f"mu_{h}"] + q[tag][h] * df_e[f"sigma_{h}"], lo, hi)
            df_e[f"L_{h}_{tag}"], df_e[f"U_{h}_{tag}"] = L, U
            ins = (df_e[f"y_{h}"] >= L) & (df_e[f"y_{h}"] <= U)
            if f"inside_{h}_{tag}" in df_e:
                mismatch += int((ins != df_e[f"inside_{h}_{tag}"].astype(bool)).sum())
            df_e[f"inside_{h}_{tag}"] = ins.astype(int)
    return mismatch


def add_plain_bounds(df_b):
    for tag, _ in LEVELS:
        for h, (lo, hi) in HEADS.items():
            pred = df_b["pred_range_norm"] if h == "range" else df_b[f"pred_{h}"]
            q = df_b[f"q_{h}_{tag[1:]}"] / (R_MAX if h == "range" else 1.0)
            df_b[f"Lp_{h}_{tag}"] = np.clip(pred - q, lo, hi)
            df_b[f"Up_{h}_{tag}"] = np.clip(pred + q, lo, hi)


def _y_plain(df_b, h):
    return df_b["gt_range_norm"] if h == "range" else df_b[f"gt_{h}"]


def _mwis_evid_base(df_e, alpha):
    return float(sum(_winkler(df_e[f"L_{h}_base"].values, df_e[f"U_{h}_base"].values, df_e[f"y_{h}"].values, alpha)
                     for h in HEADS).mean())


def _mwis_evid_conf(df_e, tag, alpha):
    return float(sum(_winkler(df_e[f"L_{h}_{tag}"].values, df_e[f"U_{h}_{tag}"].values, df_e[f"y_{h}"].values, alpha)
                     for h in HEADS).mean())


def _mwis_plain_conf(df_b, tag, alpha):
    return float(sum(_winkler(df_b[f"Lp_{h}_{tag}"].values, df_b[f"Up_{h}_{tag}"].values, _y_plain(df_b, h).values, alpha)
                     for h in HEADS).mean())


def _euclidean_err(df_e):
    gt_b = np.arctan2(df_e["y_sin"].values, df_e["y_cos"].values)
    gt_r = df_e["y_range"].values * R_MAX
    pr_b = df_e["theta_hat_rad"].values
    pr_r = df_e["mu_range"].values * R_MAX
    return np.sqrt((pr_r * np.cos(pr_b) - gt_r * np.cos(gt_b)) ** 2 + (pr_r * np.sin(pr_b) - gt_r * np.sin(gt_b)) ** 2)

# ══════════════════════════════════════════════════════════════════════════════
# HP selection (validation)
# ══════════════════════════════════════════════════════════════════════════════

def select_top3_hps(exp, tbl_dir):
    v = val_metrics(exp)
    cc = [f"cov_{h}_9545" for h in HEADS]
    v["passes"] = (v[cc] >= TARGET_COV).all(axis=1)
    v["hp"] = [hp_tag(c, k) for c, k in zip(v.coeff, v.kl_weight)]
    pool = v[v.passes] if v.passes.sum() >= 1 else v
    if not v.passes.any():
        print("  [WARN] no setting passes the validation coverage gate -- ranking all by validation MAE")
    pool = pool.sort_values("euclidean_mae_m")
    print(f"  {int(v.passes.sum())}/{len(v)} settings pass (sin, cos, range each >= {TARGET_COV} on validation)")
    top = pool.head(3)
    for i, r in enumerate(top.itertuples()):
        print(f"  HP{i+1}{' ★' if i == 0 else ''}: {r.hp}  val cov {r.cov_sin_9545:.4f}/{r.cov_cos_9545:.4f}/"
              f"{r.cov_range_9545:.4f}  val MAE {r.euclidean_mae_m:.3f} m")

    ranked = pool.reset_index(drop=True)
    out = ranked[["hp", "coeff", "kl_weight"] + cc + ["euclidean_mae_m", "mwis_9545"]].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.to_csv(tbl_dir / "hp_selection.csv", index=False)
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Validation-based HP selection ({exp}): settings whose sin, cos and range conformal "
             rf"coverage at 95.45\% each reach the target on the validation split ({int(v.passes.sum())} of {len(v)}), "
             r"ranked by validation Euclidean MAE. $\star$~=~selected.}",
             rf"\label{{tab:hp_selection_{exp}}}", r"\begin{tabular}{rlrrrrr}", r"\toprule",
             r"Rank & HP & sin & cos & range & Euclid. MAE (m) & MWIS \\", r"\midrule"]
    for r in out.itertuples():
        star = r" $\star$" if r.rank == 1 else ""
        lines.append(f"{r.rank}{star} & {r.hp.replace('_', ' ')} & "
                     f"{r.cov_sin_9545:.4f} & {r.cov_cos_9545:.4f} & {r.cov_range_9545:.4f} & "
                     f"{r.euclidean_mae_m:.2f} & {r.mwis_9545:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tbl_dir / "hp_selection.tex").write_text("\n".join(lines))
    return list(top.hp)

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_evid(exp, hp, cond="plain"):
    fp = evid_file(exp, hp, cond)
    avail = set(pd.read_csv(fp, nrows=0).columns)
    df = pd.read_csv(fp, usecols=[c for c in EVID_COLS if c in avail])
    mism = add_evid_bounds(df, load_qhat(exp, hp))
    if mism:
        print(f"  [CHECK] {hp} {cond}: {mism} coverage flags differ from the stored ones")
    return df


def load_base():
    fp = RESULTS_ROOT / "per_sample" / "base_model" / "plain.csv"
    avail = set(pd.read_csv(fp, nrows=0).columns)
    df = pd.read_csv(fp, usecols=[c for c in BASE_COLS if c in avail])
    add_plain_bounds(df)
    return df


def load_meta():
    meta = pd.read_csv(RESULTS_ROOT / "test_sample_class.csv", usecols=["ID", "Sample Class"])
    meta.columns = ["sample_id", "drone_class"]
    return meta.dropna(subset=["drone_class"])


def align_all(df_evids, df_b):
    common = df_evids[0]["sample_id"].values
    for df_e in df_evids[1:]:
        common = np.intersect1d(common, df_e["sample_id"].values)
    common = np.intersect1d(common, df_b["sample_id"].values)
    df_b = df_b[df_b["sample_id"].isin(common)].sort_values("sample_id").reset_index(drop=True)
    df_evids = [df_e[df_e["sample_id"].isin(common)].sort_values("sample_id").reset_index(drop=True) for df_e in df_evids]
    return df_evids, df_b

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Accuracy
# ══════════════════════════════════════════════════════════════════════════════

def compute_accuracy_rows(hp_suffixes, df_evids, df_b):
    rows = [{
        "method": "Plain+Conf", "hp": "—", "star": "",
        "mae_sin": float(df_b["err_sin"].abs().mean()), "mae_cos": float(df_b["err_cos"].abs().mean()),
        "mae_bearing_deg": float(df_b["err_bearing_deg"].abs().mean()),
        "mae_range_m": float(df_b["err_range_m"].abs().mean()), "mae_euclidean_m": float(df_b["err_euclidean_m"].mean()),
    }]
    for i, (hp, df_e) in enumerate(zip(hp_suffixes, df_evids)):
        gt_b = np.arctan2(df_e["y_sin"].values, df_e["y_cos"].values)
        pr_b = df_e["theta_hat_rad"].values
        rows.append({
            "method": "Evid+Conf", "hp": hp, "star": " ★" if i == 0 else "",
            "mae_sin": float(np.abs(df_e["mu_sin"] - df_e["y_sin"]).mean()),
            "mae_cos": float(np.abs(df_e["mu_cos"] - df_e["y_cos"]).mean()),
            "mae_bearing_deg": float(np.abs(np.rad2deg(np.arctan2(np.sin(pr_b - gt_b), np.cos(pr_b - gt_b)))).mean()),
            "mae_range_m": float(np.abs(df_e["mu_range"] - df_e["y_range"]).mean() * R_MAX),
            "mae_euclidean_m": float(_euclidean_err(df_e).mean()),
        })
    return rows


def save_accuracy_table(rows, tbl_dir, exp):
    cols = ["Method", "HP", "sin MAE", "cos MAE", "Bearing MAE (deg)", "Range MAE (m)", "Euclidean MAE (m)"]
    data = [[r["method"] + r["star"], r["hp"], f"{r['mae_sin']:.4f}", f"{r['mae_cos']:.4f}",
             f"{r['mae_bearing_deg']:.2f}", f"{r['mae_range_m']:.2f}", f"{r['mae_euclidean_m']:.2f}"] for r in rows]
    pd.DataFrame(data, columns=cols).to_csv(tbl_dir / "accuracy_table.csv", index=False)
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Point prediction accuracy on the matched clean condition (external prior, {exp}). "
             r"$\star$~=~selected HP. $\downarrow$ lower is better.}",
             rf"\label{{tab:accuracy_{exp}}}", r"\begin{tabular}{llccccc}", r"\toprule",
             r"Method & HP & $|\hat{s}-s|$ & $|\hat{c}-c|$ & Bearing$\downarrow$ (°) & Range$\downarrow$ (m) "
             r"& Euclidean$\downarrow$ (m) \\", r"\midrule"]
    for r in data:
        lines.append(" & ".join(x.replace("_", " ").replace("★", r"$\star$") for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tbl_dir / "accuracy_table.tex").write_text("\n".join(lines))
    print("  Saved accuracy_table.{csv,tex}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Coverage (component + MWIS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_coverage_rows(hp_suffixes, df_evids, df_b):
    rows = []
    r = {"method": "Plain+Conf", "hp": "—", "star": ""}
    for tag, _ in LEVELS:
        for h in HEADS:
            r[f"{h}_{tag}"] = float(df_b[f"cov_{h}_{tag}"].mean())
        r[f"mwis_{tag}"] = _mwis_plain_conf(df_b, tag, ALPHA_MAP[tag])
    rows.append(r)
    for i, (hp, df_e) in enumerate(zip(hp_suffixes, df_evids)):
        r = {"method": "NIG Base", "hp": hp, "star": " ★" if i == 0 else ""}
        for tag, _ in LEVELS:
            for h in HEADS:
                r[f"{h}_{tag}"] = float(df_e[f"inside_{h}_base"].mean())
            r[f"mwis_{tag}"] = _mwis_evid_base(df_e, ALPHA_MAP[tag])
        rows.append(r)
    for i, (hp, df_e) in enumerate(zip(hp_suffixes, df_evids)):
        r = {"method": "Evid+Conf", "hp": hp, "star": " ★" if i == 0 else ""}
        for tag, _ in LEVELS:
            for h in HEADS:
                r[f"{h}_{tag}"] = float(df_e[f"inside_{h}_{tag}"].mean())
            r[f"mwis_{tag}"] = _mwis_evid_conf(df_e, tag, ALPHA_MAP[tag])
        rows.append(r)
    return rows


def _tex_rows(data, lines):
    prev = None
    for r in data:
        base = r[0].rstrip(" ★")
        if prev is not None and base != prev:
            lines.append(r"\midrule")
        prev = base
        lines.append(" & ".join(x.replace("_", " ").replace("★", r"$\star$") for x in r) + r" \\")


def save_coverage_table(rows, tbl_dir, exp):
    cols = ["Method", "HP"]
    for _, t in LEVELS:
        cols += [f"sin ({t:.2f}%)", f"cos ({t:.2f}%)", f"range ({t:.2f}%)", f"MWIS ({t:.2f}%)"]
    data = []
    for r in rows:
        row = [r["method"] + r["star"], r["hp"]]
        for tag, _ in LEVELS:
            row += [f"{r[f'sin_{tag}']:.4f}", f"{r[f'cos_{tag}']:.4f}", f"{r[f'range_{tag}']:.4f}", f"{r[f'mwis_{tag}']:.4f}"]
        data.append(row)
    pd.DataFrame(data, columns=cols).to_csv(tbl_dir / "coverage_table.csv", index=False)
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Component coverage and total MWIS at all three nominal levels (external prior, {exp}; "
             r"true clipped interval bounds). $\star$~=~selected HP.}",
             rf"\label{{tab:coverage_{exp}}}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll" + "rrrr" * 3 + "}", r"\toprule",
             r"& & \multicolumn{4}{c}{68.27\%} & \multicolumn{4}{c}{95.45\%} & \multicolumn{4}{c}{99.73\%} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
             r"Method & HP & sin & cos & range & MWIS & sin & cos & range & MWIS & sin & cos & range & MWIS \\",
             r"\midrule"]
    _tex_rows(data, lines)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (tbl_dir / "coverage_table.tex").write_text("\n".join(lines))
    print("  Saved coverage_table.{csv,tex}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Spatial coverage
# ══════════════════════════════════════════════════════════════════════════════

def _spatial(in_s, in_c, in_r, L_s, U_s, L_c, U_c, L_r, U_r):
    arc = _arc_rad_from_LU(L_s, U_s, L_c, U_c)
    area = _annular_area(L_r, U_r, arc)
    return {"full": float((in_s & in_c & in_r).mean()), "s_not_c": float((in_s & in_r & ~in_c).mean()),
            "c_not_s": float((in_c & in_r & ~in_s).mean()), "area_m2": area, "area_pct": area / FULL_DISK_M2 * 100.0,
            "empty": float((arc == 0).mean())}


def _spatial_metrics_evid_base(df_e, tag=None):
    g = lambda c: df_e[c].values
    return _spatial(df_e["inside_sin_base"].astype(bool), df_e["inside_cos_base"].astype(bool),
                    df_e["inside_range_base"].astype(bool), g("L_sin_base"), g("U_sin_base"),
                    g("L_cos_base"), g("U_cos_base"), g("L_range_base"), g("U_range_base"))


def _spatial_metrics_evid_conf(df_e, tag):
    g = lambda c: df_e[c].values
    return _spatial(df_e[f"inside_sin_{tag}"].astype(bool), df_e[f"inside_cos_{tag}"].astype(bool),
                    df_e[f"inside_range_{tag}"].astype(bool), g(f"L_sin_{tag}"), g(f"U_sin_{tag}"),
                    g(f"L_cos_{tag}"), g(f"U_cos_{tag}"), g(f"L_range_{tag}"), g(f"U_range_{tag}"))


def _spatial_metrics_plain_conf(df_b, tag):
    g = lambda c: df_b[c].values
    return _spatial(df_b[f"cov_sin_{tag}"].astype(bool), df_b[f"cov_cos_{tag}"].astype(bool),
                    df_b[f"cov_range_{tag}"].astype(bool), g(f"Lp_sin_{tag}"), g(f"Up_sin_{tag}"),
                    g(f"Lp_cos_{tag}"), g(f"Up_cos_{tag}"), g(f"Lp_range_{tag}"), g(f"Up_range_{tag}"))


def compute_spatial_rows(hp_suffixes, df_evids, df_b):
    rows = []
    r = {"method": "Plain+Conf", "hp": "—", "star": ""}
    for tag, _ in LEVELS:
        for k, v in _spatial_metrics_plain_conf(df_b, tag).items():
            r[f"{k}_{tag}"] = v
    rows.append(r)
    for name, fn in [("NIG Base", _spatial_metrics_evid_base), ("Evid+Conf", _spatial_metrics_evid_conf)]:
        for i, (hp, df_e) in enumerate(zip(hp_suffixes, df_evids)):
            r = {"method": name, "hp": hp, "star": " ★" if i == 0 else ""}
            for tag, _ in LEVELS:
                for k, v in fn(df_e, tag).items():
                    r[f"{k}_{tag}"] = v
            rows.append(r)
    return rows


def save_spatial_table(rows, tbl_dir, exp):
    cols = ["Method", "HP"]
    for _, t in LEVELS:
        cols += [f"Keyhole ({t:.2f}%)", f"S+R¬C ({t:.2f}%)", f"C+R¬S ({t:.2f}%)", f"Area m2 ({t:.2f}%)",
                 f"Area% ({t:.2f}%)", f"Empty ({t:.2f}%)"]
    data = []
    for r in rows:
        row = [r["method"] + r["star"], r["hp"]]
        for tag, _ in LEVELS:
            row += [f"{r[f'full_{tag}']:.4f}", f"{r[f's_not_c_{tag}']:.4f}", f"{r[f'c_not_s_{tag}']:.4f}",
                    f"{r[f'area_m2_{tag}']:.0f}", f"{r[f'area_pct_{tag}']:.2f}", f"{r[f'empty_{tag}']:.4f}"]
        data.append(row)
    pd.DataFrame(data, columns=cols).to_csv(tbl_dir / "spatial_coverage_table.csv", index=False)
    tex = [[x for j, x in enumerate(r) if j < 2 or (j - 2) % 6 != 5] for r in data]   # empty share: csv only
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Spatial coverage at all three nominal levels (external prior, {exp}; true clipped bounds). "
             r"Keyhole~=~GT inside all three intervals simultaneously. S+R$\neg$C: sin$\cap$range without cos. "
             r"C+R$\neg$S: cos$\cap$range without sin. $\star$~=~selected HP.}",
             rf"\label{{tab:spatial_{exp}}}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll" + "rrrrr" * 3 + "}", r"\toprule",
             r"& & \multicolumn{5}{c}{68.27\%} & \multicolumn{5}{c}{95.45\%} & \multicolumn{5}{c}{99.73\%} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}",
             r"Method & HP & Keyhole & S+R$\neg$C & C+R$\neg$S & Area (m$^2$) & Area (\%) "
             r"& Keyhole & S+R$\neg$C & C+R$\neg$S & Area (m$^2$) & Area (\%) "
             r"& Keyhole & S+R$\neg$C & C+R$\neg$S & Area (m$^2$) & Area (\%) \\", r"\midrule"]
    _tex_rows(tex, lines)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (tbl_dir / "spatial_coverage_table.tex").write_text("\n".join(lines))
    print("  Saved spatial_coverage_table.{csv,tex}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Coverage vs Area
# ══════════════════════════════════════════════════════════════════════════════

def _cva_metrics(hp_suffixes, df_evids, df_b):
    rows = []
    df_e0 = df_evids[0]
    s = _spatial_metrics_evid_base(df_e0)
    comp = np.mean([df_e0[f"inside_{h}_base"].mean() for h in HEADS]) * 100
    rows.append({"method": "NIG Base", "level": "—", "comp_cov": comp, "spatial_cov": s["full"] * 100, "area_m2": s["area_m2"]})
    for tag, target in LEVELS:
        s = _spatial_metrics_plain_conf(df_b, tag)
        comp = np.mean([df_b[f"cov_{h}_{tag}"].mean() for h in HEADS]) * 100
        rows.append({"method": "Plain+Conf", "level": f"{target:.2f}%", "comp_cov": comp,
                     "spatial_cov": s["full"] * 100, "area_m2": s["area_m2"]})
    for tag, target in LEVELS:
        s = _spatial_metrics_evid_conf(df_e0, tag)
        comp = np.mean([df_e0[f"inside_{h}_{tag}"].mean() for h in HEADS]) * 100
        rows.append({"method": "Evid+Conf", "level": f"{target:.2f}%", "comp_cov": comp,
                     "spatial_cov": s["full"] * 100, "area_m2": s["area_m2"]})
    return rows


def _draw_cva(ax, rows):
    STYLE = {"NIG Base": dict(color="#555555", marker="*", ms=13, zorder=5),
             "Plain+Conf": dict(color="#2878b5", marker="o", ms=8, zorder=4),
             "Evid+Conf": dict(color="#c82423", marker="D", ms=7, zorder=4)}
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    for _, target in LEVELS:
        ax.axhline(target, color="#cccccc", lw=0.9, ls="--", zorder=1)
        offset = -1.2 if target == 99.73 else 0.5
        ax.text(1.8e4, target + offset, f"{target:.2f}%", color="#999999", fontsize=7,
                va="top" if offset < 0 else "bottom", ha="right")
    for method in ["NIG Base", "Plain+Conf", "Evid+Conf"]:
        mrs = by_method.get(method, [])
        if not mrs:
            continue
        st = STYLE[method]
        areas = [r["area_m2"] for r in mrs]
        ax.plot(areas, [r["comp_cov"] for r in mrs], color=st["color"], marker=st["marker"],
                markersize=st["ms"], lw=1.8, zorder=st["zorder"])
        ax.plot(areas, [r["spatial_cov"] for r in mrs], color=st["color"], marker=st["marker"],
                markersize=st["ms"], lw=1.8, ls=":", markerfacecolor="white", markeredgewidth=1.5, zorder=st["zorder"])
        if method != "NIG Base":
            for r in mrs:
                ax.annotate(r["level"], xy=(r["area_m2"], r["comp_cov"]), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=7.5, color=st["color"], fontweight="bold")
    if "Plain+Conf" in by_method and "Evid+Conf" in by_method:
        for pr, er in zip(by_method["Plain+Conf"], by_method["Evid+Conf"]):
            if er["area_m2"] > 0:
                ratio = pr["area_m2"] / er["area_m2"]
                ax.text((pr["area_m2"] * er["area_m2"]) ** 0.5, (pr["comp_cov"] + er["comp_cov"]) / 2 - 5,
                        f"{ratio:.1f}× smaller", ha="center", va="center", fontsize=7, color="#666666",
                        fontstyle="italic", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("Mean Interval Area  (m²)", fontsize=10)
    ax.set_ylabel("Coverage  (%)", fontsize=10)
    ax.set_xlim(80, 3e4)
    ax.set_ylim(30, 105)
    ax.set_yticks([40, 50, 60, 68.27, 70, 80, 90, 95.45, 99.73])
    ax.set_yticklabels(["40", "50", "60", "68.27", "70", "80", "90", "95.45", "99.73"], fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", which="both", color="#eeeeee", lw=0.5, zorder=0)
    l1 = ax.legend(handles=[Line2D([0], [0], color="k", lw=1.8, label="Component cov."),
                            Line2D([0], [0], color="k", lw=1.8, ls=":", marker="o", markersize=6, markerfacecolor="white",
                                   markeredgewidth=1.3, label="Spatial cov.")],
                   loc="lower right", fontsize=8, framealpha=0.92, title="Coverage type", title_fontsize=8)
    ax.add_artist(l1)
    ax.legend(handles=[Line2D([0], [0], color=STYLE[m]["color"], lw=2, marker=STYLE[m]["marker"],
                              markersize=STYLE[m]["ms"] * 0.8, label=m) for m in ["NIG Base", "Plain+Conf", "Evid+Conf"]
                       if m in by_method], loc="upper left", fontsize=8, framealpha=0.92, title="Method", title_fontsize=8)


def make_coverage_vs_area(hp_suffixes, df_evids, df_b, fig_dir, exp):
    rows = _cva_metrics(hp_suffixes, df_evids, df_b)
    pd.DataFrame(rows).to_csv(fig_dir / "coverage_vs_area.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    _draw_cva(ax, rows)
    ax.set_title(f"Coverage vs Area Efficiency  (external prior, {exp}, clean)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage_vs_area.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved coverage_vs_area.{png,csv}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Range-bearing polar heatmaps (unchanged logic: boolean coverage)
# ══════════════════════════════════════════════════════════════════════════════

def _cov_grid(y_sin, y_cos, y_range, cov_vals, min_count=10):
    gt_b = np.rad2deg(np.arctan2(y_sin, y_cos)) % 360.0
    gt_r = y_range * R_MAX
    r_bin = np.clip(np.digitize(gt_r, RANGE_EDGES) - 1, 0, N_RANGE - 1)
    b_bin = np.clip(np.digitize(gt_b, BEARING_EDGES) - 1, 0, N_BEARING - 1)
    grid = np.full((N_RANGE, N_BEARING), np.nan)
    for i in range(N_RANGE):
        for j in range(N_BEARING):
            m = (r_bin == i) & (b_bin == j)
            if m.sum() >= min_count:
                grid[i, j] = cov_vals[m].mean() * 100.0
    return grid


def _spatial_cov_series(df_e, df_b, mtype, tag):
    if mtype == "nig":
        cols, d = [f"inside_{h}_base" for h in HEADS], df_e
    elif mtype == "plain":
        cols, d = [f"cov_{h}_{tag}" for h in HEADS], df_b
    else:
        cols, d = [f"inside_{h}_{tag}" for h in HEADS], df_e
    return np.logical_and.reduce([d[c].astype(bool).values for c in cols]).astype(float)


def _polar_panel(ax, grid, norm):
    ax.pcolormesh(_T_GRID, _R_GRID, np.ma.masked_invalid(grid), cmap=_COV_CMAP, norm=norm, shading="flat")
    for i in range(N_RANGE):
        for j in range(N_BEARING):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(_T_CENTERS[j], _R_CENTERS[i], f"{v:.1f}%", ha="center", va="center", fontsize=4.5,
                        color="black", fontweight="bold")
    ax.set_rlim(0, R_MAX)
    ax.set_yticks([50, 100, 150, 200])
    ax.set_yticklabels(["50m", "100m", "150m", "200m"], fontsize=5)
    ax.set_rlabel_position(135)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"], fontsize=5)
    ax.grid(True, lw=0.4, alpha=0.5)


def make_range_bearing_figures(hp_suffixes, df_evids, df_b, fig_dir, exp):
    rb_dir = fig_dir / "range_bearing"
    rb_dir.mkdir(exist_ok=True)
    norm = mcolors.Normalize(vmin=0, vmax=100)
    df_e0 = df_evids[0]
    y_sin, y_cos, y_rng = df_e0["y_sin"].values, df_e0["y_cos"].values, df_e0["y_range"].values
    for mtype, label, slug in [("nig", "NIG Base", "nig_base"), ("plain", "Plain+Conf", "plain_conf"),
                               ("evid", "Evid+Conf ★", "evid_conf")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
        for col_j, (tag, target) in enumerate(LEVELS):
            grid = _cov_grid(y_sin, y_cos, y_rng, _spatial_cov_series(df_e0, df_b, mtype, tag))
            overall = float(np.nanmean(grid[~np.isnan(grid)]))
            _polar_panel(axes[col_j], grid, norm)
            axes[col_j].set_title(f"Target {target:.2f}%  (overall {overall:.1f}%)", fontsize=8, pad=10)
            fig_i, ax_i = plt.subplots(1, 1, figsize=(5.5, 5.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
            _polar_panel(ax_i, grid, norm)
            ax_i.set_title(f"{label} — {target:.2f}%  (overall {overall:.1f}%)", fontsize=8, pad=10)
            sm = plt.cm.ScalarMappable(cmap=_COV_CMAP, norm=norm); sm.set_array([])
            fig_i.colorbar(sm, ax=ax_i, fraction=0.04, pad=0.08, shrink=0.7, label="Coverage (%)")
            fig_i.savefig(rb_dir / f"{slug}_{tag}.png", dpi=200, bbox_inches="tight")
            plt.close(fig_i)
        sm = plt.cm.ScalarMappable(cmap=_COV_CMAP, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=list(axes), fraction=0.015, pad=0.04, shrink=0.8, label="Spatial coverage (%)")
        fig.suptitle(f"Spatial coverage by range & bearing — {label}  (external prior, {exp}, clean)",
                     fontsize=9, fontweight="bold")
        fig.savefig(rb_dir / f"range_bearing_{slug}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print("  Saved range_bearing/ (3 combined + 9 individual panels)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Compact comparison per drone class (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

def _rect_grid(y_sin, y_cos, y_range, cov_vals, min_count=5):
    return _cov_grid(y_sin, y_cos, y_range, cov_vals, min_count)


def _rect_panel(ax, grid, norm, title):
    ax.pcolormesh(np.arange(N_BEARING + 1), np.arange(N_RANGE + 1), np.ma.masked_invalid(grid),
                  cmap=_COV_CMAP, norm=norm, shading="flat")
    ax.set_xticks(np.arange(N_BEARING) + 0.5)
    ax.set_xticklabels([f"{int(b)}°" for b in BEARING_EDGES[:-1]], fontsize=7)
    ax.set_yticks(np.arange(N_RANGE) + 0.5)
    ax.set_yticklabels([f"{int(RANGE_EDGES[i])}-{int(RANGE_EDGES[i+1])} m" for i in range(N_RANGE)], fontsize=7)
    ax.set_xlabel("Bearing", fontsize=8)
    ax.set_ylabel("Range", fontsize=8)
    ax.set_title(title, fontsize=8, fontweight="bold")


def make_compact_comparison(hp_suffixes, df_evids, df_b, meta, fig_dir, exp):
    comp_dir = fig_dir / "compact_comparison"
    comp_dir.mkdir(exist_ok=True)
    norm = mcolors.Normalize(vmin=0, vmax=100)
    df_e0 = df_evids[0]
    id2cls = dict(zip(meta["sample_id"].values, meta["drone_class"].values))
    cols = {"sample_id": df_e0["sample_id"].values, "y_sin": df_e0["y_sin"].values,
            "y_cos": df_e0["y_cos"].values, "y_range": df_e0["y_range"].values}
    for tag, _ in LEVELS:
        for m in ["nig", "plain", "evid"]:
            cols[f"cov_{m}_{tag}"] = _spatial_cov_series(df_e0, df_b, m, tag)
    df_all = pd.DataFrame(cols)
    df_all["drone_class"] = df_all["sample_id"].map(id2cls)
    for drone_class in ["MINI", "PRO4"]:
        sub = df_all[df_all["drone_class"] == drone_class]
        if len(sub) < 20:
            print(f"  [SKIP compact] {drone_class}: {len(sub)} samples")
            continue
        fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
        fig.suptitle(f"Spatial coverage by bearing × range  —  {drone_class}  (external prior, {exp}, clean)",
                     fontsize=11, fontweight="bold")
        for row_i, (mkey, _) in enumerate([("nig", "NIG Base"), ("plain", "Plain+Conf"), ("evid", "Evid+Conf ★")]):
            for col_j, (tag, _) in enumerate(LEVELS):
                grid = _rect_grid(sub["y_sin"].values, sub["y_cos"].values, sub["y_range"].values,
                                  sub[f"cov_{mkey}_{tag}"].values)
                overall = float(np.nanmean(grid[~np.isnan(grid)]))
                _rect_panel(axes[row_i, col_j], grid, norm, f"Overall: {overall:.1f}%")
                fig_i, ax_i = plt.subplots(figsize=(7, 5))
                _rect_panel(ax_i, grid, norm, f"Overall: {overall:.1f}%")
                sm = plt.cm.ScalarMappable(cmap=_COV_CMAP, norm=norm); sm.set_array([])
                fig_i.colorbar(sm, ax=ax_i, label="Spatial coverage (%)")
                fig_i.tight_layout()
                fig_i.savefig(comp_dir / f"{drone_class}_{mkey}_{tag}.png", dpi=150, bbox_inches="tight")
                plt.close(fig_i)
        sm = plt.cm.ScalarMappable(cmap=_COV_CMAP, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.007, pad=0.01, shrink=0.6, label="Spatial coverage (%)")
        fig.savefig(comp_dir / f"compact_{drone_class}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved compact_{drone_class}.png + 9 individual panels ({len(sub):,} samples)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Polar qualitative (true bounds)
# ══════════════════════════════════════════════════════════════════════════════

QUAL_SCENARIOS = [("inside_q6827", "Inside 68.27%"), ("out68_in95", "Out 68%, In 95%"),
                  ("out95_in99", "Out 95%, In 99%"), ("outside_all", "Outside 99.73%")]


def _joint(df_e, tag):
    return (df_e[f"inside_sin_{tag}"].astype(bool) & df_e[f"inside_cos_{tag}"].astype(bool)
            & df_e[f"inside_range_{tag}"].astype(bool))


def _scenario_mask(df_e, scenario):
    in68, in95, in99 = _joint(df_e, "q6827"), _joint(df_e, "q9545"), _joint(df_e, "q9973")
    return {"inside_q6827": in68, "out68_in95": (~in68) & in95, "out95_in99": (~in95) & in99}.get(scenario, ~in99)


def _pick_representative(df_e, mask):
    idxs = np.where(mask.values)[0]
    if len(idxs) == 0:
        return None
    errs = _euclidean_err(df_e)[idxs]
    return int(idxs[np.argsort(errs)[len(errs) // 2]])


def _arc_angles_from_LU(L_sin, U_sin, L_cos, U_cos, n=720):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    m = ((np.sin(th) >= float(np.clip(L_sin, -1, 1))) & (np.sin(th) <= float(np.clip(U_sin, -1, 1)))
         & (np.cos(th) >= float(np.clip(L_cos, -1, 1))) & (np.cos(th) <= float(np.clip(U_cos, -1, 1))))
    return th[m]


def _fill_annular(ax, angles, r_inner, r_outer, color, alpha, label=None):
    if len(angles) < 2 or r_outer <= max(r_inner, 0):
        return
    first = True
    for seg in np.split(angles, np.where(np.diff(angles) > 0.06)[0] + 1):
        if len(seg) < 2:
            continue
        ax.fill(np.concatenate([seg, seg[::-1]]),
                np.concatenate([np.full(len(seg), r_outer), np.full(len(seg), max(r_inner, 0.0))]),
                color=color, alpha=alpha, label=label if first else None)
        first = False


def _draw_polar_sample(ax, se, sb, tag):
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_rlim(0, R_MAX * 1.1)
    ax.set_yticks([50, 100, 150, 200, 250])
    ax.set_yticklabels(["50", "100", "150", "200", "250"], fontsize=5)
    ax.grid(True, lw=0.4, alpha=0.4)
    for (Ls, Us, Lc, Uc, Lr, Ur), color, alpha, label in [
        ((se["L_sin_base"], se["U_sin_base"], se["L_cos_base"], se["U_cos_base"], se["L_range_base"], se["U_range_base"]),
         "#555555", 0.35, "NIG Base"),
        ((sb[f"Lp_sin_{tag}"], sb[f"Up_sin_{tag}"], sb[f"Lp_cos_{tag}"], sb[f"Up_cos_{tag}"],
          sb[f"Lp_range_{tag}"], sb[f"Up_range_{tag}"]), "#2878b5", 0.4, "Plain+Conf"),
        ((se[f"L_sin_{tag}"], se[f"U_sin_{tag}"], se[f"L_cos_{tag}"], se[f"U_cos_{tag}"],
          se[f"L_range_{tag}"], se[f"U_range_{tag}"]), "#c82423", 0.5, "Evid+Conf ★"),
    ]:
        _fill_annular(ax, _arc_angles_from_LU(Ls, Us, Lc, Uc), float(np.clip(Lr, 0, 1)) * R_MAX,
                      float(np.clip(Ur, 0, 1)) * R_MAX, color, alpha, label=label)
    ax.scatter([float(np.arctan2(se["y_sin"], se["y_cos"]))], [float(se["y_range"]) * R_MAX], marker="*", s=130,
               color="gold", edgecolors="black", linewidths=0.5, zorder=10)
    ax.scatter([float(se["theta_hat_rad"])], [float(se["mu_range"]) * R_MAX], marker="D", s=30, color="#c82423",
               edgecolors="white", linewidths=0.5, zorder=9)
    ax.scatter([float(np.arctan2(sb["pred_sin"], sb["pred_cos"]))], [float(sb["pred_range_norm"]) * R_MAX], marker="o",
               s=30, color="#2878b5", edgecolors="white", linewidths=0.5, zorder=9)


_QUAL_LEGEND = [
    Line2D([0], [0], color="#555555", lw=4, alpha=0.5, label="NIG Base"),
    Line2D([0], [0], color="#2878b5", lw=4, alpha=0.5, label="Plain+Conf"),
    Line2D([0], [0], color="#c82423", lw=4, alpha=0.5, label="Evid+Conf ★"),
    Line2D([0], [0], marker="*", color="gold", markeredgecolor="black", markersize=10, lw=0, label="GT location"),
    Line2D([0], [0], marker="D", color="#c82423", markersize=6, lw=0, label="Pred. Evid"),
    Line2D([0], [0], marker="o", color="#2878b5", markersize=6, lw=0, label="Pred. Plain"),
]


def make_polar_qualitative(hp_suffixes, df_evids, df_b, fig_dir, exp):
    qual_dir = fig_dir / "polar_qualitative"
    qual_dir.mkdir(exist_ok=True)
    df_e0 = df_evids[0]
    idx_of = {}
    for scen_key, _ in QUAL_SCENARIOS:
        mask = _scenario_mask(df_e0, scen_key)
        idx_of[scen_key] = _pick_representative(df_e0, mask)
        print(f"  {scen_key}: {int(mask.sum())} samples  ->  idx={idx_of[scen_key]}")
    fig, axes = plt.subplots(len(QUAL_SCENARIOS), 3, figsize=(15, 20), subplot_kw={"projection": "polar"},
                             constrained_layout=True)
    for row_i, (scen_key, scen_label) in enumerate(QUAL_SCENARIOS):
        idx = idx_of[scen_key]
        for col_j, (tag, target) in enumerate(LEVELS):
            ax = axes[row_i, col_j]
            if idx is None:
                ax.set_visible(False)
                continue
            se, sb = df_e0.iloc[idx], df_b.iloc[idx]
            assert se["sample_id"] == sb["sample_id"]
            _draw_polar_sample(ax, se, sb, tag)
            ax.set_title(f"{scen_label}\n{target:.2f}%" if col_j == 0 else f"{target:.2f}%", fontsize=8, pad=8)
            fig_i, ax_i = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": "polar"})
            _draw_polar_sample(ax_i, se, sb, tag)
            ax_i.set_title(f"{scen_label}  |  {target:.2f}%", fontsize=9)
            ax_i.legend(handles=_QUAL_LEGEND, loc="lower right", fontsize=6, framealpha=0.9)
            fig_i.tight_layout()
            fig_i.savefig(qual_dir / f"{scen_key}_{tag}.png", dpi=200, bbox_inches="tight")
            plt.close(fig_i)
    fig.legend(handles=_QUAL_LEGEND, loc="lower center", ncol=6, fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"Qualitative polar analysis — representative samples per scenario (external prior, {exp}, clean)\n"
                 "Rows: coverage scenario  |  Cols: nominal level", fontsize=10, fontweight="bold")
    fig.savefig(qual_dir / "polar_qualitative.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved polar_qualitative.png + {len(QUAL_SCENARIOS) * 3} individual panels")

# ══════════════════════════════════════════════════════════════════════════════
# NEW — Noise robustness (clean-trained, clean-calibrated, tested on noise)
# ══════════════════════════════════════════════════════════════════════════════

def noise_metrics(exp, hp_suffixes):
    rows = []
    for i, hp in enumerate(hp_suffixes):
        for cond in CONDITIONS:
            df = load_evid(exp, hp, cond)
            err = _euclidean_err(df)
            for method in ["NIG Base", "Evid+Conf"]:
                for tag, target in LEVELS:
                    if method == "NIG Base" and tag != "q9545":
                        continue   # the raw NIG interval has a single width (k = 1)
                    sfx = "base" if method == "NIG Base" else tag
                    s = _spatial_metrics_evid_base(df) if method == "NIG Base" else _spatial_metrics_evid_conf(df, tag)
                    mwis = _mwis_evid_base(df, ALPHA_MAP[tag]) if method == "NIG Base" else _mwis_evid_conf(df, tag, ALPHA_MAP[tag])
                    rows.append(dict(exp=exp, hp=hp, star=i == 0, rank=i + 1, method=method,
                                     level="base" if method == "NIG Base" else f"{target:.2f}%", tag=sfx,
                                     condition=cond, **{f"cov_{h}": float(df[f"inside_{h}_{sfx}"].mean()) for h in HEADS},
                                     keyhole=s["full"], empty=s["empty"], area_m2=s["area_m2"], area_pct=s["area_pct"],
                                     mwis=mwis, euclidean_mae_m=float(err.mean())))
    return pd.DataFrame(rows)


def save_noise_tables(nm, tbl_dir, exp):
    nm.to_csv(tbl_dir / "noise_full_metrics.csv", index=False)
    c = nm[(nm.method == "Evid+Conf") & (nm.tag == "q9545")]
    cols = ["HP", "Output"] + [COND_LABEL[x].replace("$-$", "-") for x in CONDITIONS]
    data = []
    for hp in c.hp.unique():
        x = c[c.hp == hp].set_index("condition").loc[CONDITIONS]
        star = " ★" if bool(x.star.iloc[0]) else ""
        for key, lab in [("cov_sin", "sin"), ("cov_cos", "cos"), ("cov_range", "range"), ("keyhole", "keyhole"),
                         ("empty", "empty region"), ("mwis", "MWIS"), ("area_pct", "area (%)"),
                         ("euclidean_mae_m", "Euclid. MAE (m)")]:
            fmt = (lambda v: f"{v:.2f}") if key in ("mwis", "area_pct", "euclidean_mae_m") else (lambda v: f"{v*100:.1f}")
            data.append([hp + star, lab] + [fmt(v) for v in x[key]])
    pd.DataFrame(data, columns=cols).to_csv(tbl_dir / "noise_coverage_table.csv", index=False)
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{Robustness of the clean-trained, clean-calibrated external-prior model ({exp}) at the "
             r"95.45\% level under AWGN and clutter (SNR in dB). Coverage, keyhole (all three outputs inside) "
             r"and empty-region share (sin and cos intervals admit no bearing) in \%. $\star$~=~selected HP.}",
             rf"\label{{tab:noise_{exp}}}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll" + "r" * 7 + "}",
             r"\toprule", "HP & Quantity & " + " & ".join(COND_LABEL[x] for x in CONDITIONS) + r" \\", r"\midrule"]
    prev = None
    for r in data:
        if prev is not None and r[0] != prev:
            lines.append(r"\midrule")
        prev = r[0]
        lines.append(" & ".join(v.replace("_", " ").replace("★", r"$\star$") for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (tbl_dir / "noise_coverage_table.tex").write_text("\n".join(lines))
    print("  Saved noise_coverage_table.{csv,tex} + noise_full_metrics.csv")


def make_noise_figure(nm, fig_dir, exp):
    c = nm[(nm.method == "Evid+Conf") & (nm.tag == "q9545")]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)
    xlab = [COND_LABEL[x].replace("$-$", "−") for x in CONDITIONS]
    colors = ["#c82423", "#2878b5", "#1baf7a"]
    for ax, (key, title) in zip(axes, [("cov_sin", "sin coverage"), ("cov_cos", "cos coverage"),
                                       ("cov_range", "range coverage"), ("keyhole", "keyhole coverage"),
                                       ("empty", "empty regions")]):
        for (hp, g), col in zip(c.groupby("rank"), colors):
            g = g.set_index("condition").loc[CONDITIONS]
            ax.plot(range(len(CONDITIONS)), g[key] * 100, marker="o", color=col, lw=1.8,
                    label=f"{g.hp.iloc[0]}{' ★' if g.star.iloc[0] else ''}")
        if key != "empty":
            ax.axhline(95.45, color="#999999", ls="--", lw=1)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(xlab, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#eeeeee")
    axes[0].set_ylabel("% of test recordings")
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Clean-trained, clean-calibrated external prior ({exp}) under noise — 95.45% level "
                 "(dashed = target)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "noise_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved noise_robustness.png")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(exp):
    print("=" * 70)
    print(f"  External prior — {exp}")
    print("=" * 70)
    out = OUT_ROOT / exp
    tbl_dir, fig_dir = out / "tables", out / "figures"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n[HP selection — validation]")
    hp_suffixes = select_top3_hps(exp, tbl_dir)

    print("\n[Loading data]")
    df_evids = [load_evid(exp, hp) for hp in hp_suffixes]
    df_b = load_base()
    meta = load_meta()
    df_evids, df_b = align_all(df_evids, df_b)
    print(f"  Aligned samples: {len(df_evids[0]):,}")

    print("\n[Tables]")
    acc = compute_accuracy_rows(hp_suffixes, df_evids, df_b)
    cov = compute_coverage_rows(hp_suffixes, df_evids, df_b)
    spa = compute_spatial_rows(hp_suffixes, df_evids, df_b)
    save_accuracy_table(acc, tbl_dir, exp)
    save_coverage_table(cov, tbl_dir, exp)
    save_spatial_table(spa, tbl_dir, exp)

    print("\n[Figure 1 — Coverage vs Area]")
    make_coverage_vs_area(hp_suffixes, df_evids, df_b, fig_dir, exp)
    print("\n[Figure 2 — Range-bearing polar heatmaps]")
    make_range_bearing_figures(hp_suffixes, df_evids, df_b, fig_dir, exp)
    print("\n[Figure 3 — Compact comparison]")
    make_compact_comparison(hp_suffixes, df_evids, df_b, meta, fig_dir, exp)
    print("\n[Figure 4 — Polar qualitative]")
    make_polar_qualitative(hp_suffixes, df_evids, df_b, fig_dir, exp)

    print("\n[Noise robustness]")
    nm = noise_metrics(exp, hp_suffixes)
    save_noise_tables(nm, tbl_dir, exp)
    make_noise_figure(nm, fig_dir, exp)

    best = {"exp": exp, "hp": hp_suffixes[0]}
    a = next(r for r in acc if r["method"] == "Evid+Conf")
    c = next(r for r in cov if r["method"] == "Evid+Conf")
    s = next(r for r in spa if r["method"] == "Evid+Conf")
    best.update(clean_euclid_mae=a["mae_euclidean_m"], **{f"clean_{h}_q9545": c[f"{h}_q9545"] for h in HEADS},
                clean_mwis_q9545=c["mwis_q9545"], clean_keyhole_q9545=s["full_q9545"], clean_area_pct_q9545=s["area_pct_q9545"])
    n = nm[(nm.hp == hp_suffixes[0]) & (nm.method == "Evid+Conf") & (nm.tag == "q9545") & (nm.condition != "plain")]
    best.update(noisy_sin_q9545=n.cov_sin.mean(), noisy_cos_q9545=n.cov_cos.mean(), noisy_range_q9545=n.cov_range.mean(),
                noisy_keyhole_q9545=n["keyhole"].mean(), noisy_empty_q9545=n["empty"].mean(),
                noisy_euclid_mae=n["euclidean_mae_m"].mean())
    return best


# ══════════════════════════════════════════════════════════════════════════════
# NEW — every (coeff, kl) setting of an experiment, clean and noisy
# ══════════════════════════════════════════════════════════════════════════════

def all_settings_tables(exp):
    """One row per setting x condition x method/level: coverage per output, keyhole, empty share, MWIS,
    area and point errors; plus the validation metrics. Written to <exp>/all_settings/."""
    out = OUT_ROOT / exp / "all_settings"
    out.mkdir(parents=True, exist_ok=True)
    val = val_metrics(exp)
    val["hp"] = [hp_tag(c, k) for c, k in zip(val.coeff, val.kl_weight)]
    rows = []
    for hp in exp_hps(exp):
        c, k = [float(x.replace("p", ".")) for x in hp[1:].split("_kl")]
        for cond in CONDITIONS:
            df = load_evid(exp, hp, cond)
            err = _euclidean_err(df)
            gt_b, pr_b = np.arctan2(df["y_sin"].values, df["y_cos"].values), df["theta_hat_rad"].values
            common = dict(exp=exp, hp=hp, coeff=c, kl_weight=k, condition=cond,
                          euclidean_mae_m=float(err.mean()),
                          bearing_mae_deg=float(np.abs(np.rad2deg(np.arctan2(np.sin(pr_b - gt_b), np.cos(pr_b - gt_b)))).mean()),
                          range_mae_m=float(np.abs(df["mu_range"] - df["y_range"]).mean() * R_MAX))
            s = _spatial_metrics_evid_base(df)
            rows.append(dict(common, method="NIG Base", level="base",
                             **{f"cov_{h}": float(df[f"inside_{h}_base"].mean()) for h in HEADS},
                             keyhole=s["full"], empty=s["empty"], area_m2=s["area_m2"], area_pct=s["area_pct"],
                             mwis=_mwis_evid_base(df, ALPHA_MAP["q9545"])))
            for tag, target in LEVELS:
                s = _spatial_metrics_evid_conf(df, tag)
                rows.append(dict(common, method="Evid+Conf", level=f"{target:.2f}%",
                                 **{f"cov_{h}": float(df[f"inside_{h}_{tag}"].mean()) for h in HEADS},
                                 keyhole=s["full"], empty=s["empty"], area_m2=s["area_m2"], area_pct=s["area_pct"],
                                 mwis=_mwis_evid_conf(df, tag, ALPHA_MAP[tag])))
    long = pd.DataFrame(rows)
    long.to_csv(out / "all_settings_long.csv", index=False)
    val.to_csv(out / "all_settings_validation.csv", index=False)

    c95 = long[(long.method == "Evid+Conf") & (long.level == "95.45%")]
    clean = c95[c95.condition == "plain"].set_index("hp")
    noisy = c95[c95.condition != "plain"].groupby("hp").mean(numeric_only=True)
    v = val.set_index("hp")
    wide = pd.DataFrame({
        "coeff": clean.coeff, "kl_weight": clean.kl_weight,
        **{f"val_cov_{h}": v[f"cov_{h}_9545"] for h in HEADS}, "val_mae_m": v["euclidean_mae_m"],
        **{f"clean_cov_{h}": clean[f"cov_{h}"] for h in HEADS}, "clean_keyhole": clean["keyhole"],
        "clean_mwis": clean["mwis"], "clean_area_pct": clean["area_pct"], "clean_mae_m": clean["euclidean_mae_m"],
        **{f"noisy_cov_{h}": noisy[f"cov_{h}"] for h in HEADS}, "noisy_keyhole": noisy["keyhole"],
        "noisy_empty": noisy["empty"], "noisy_mwis": noisy["mwis"], "noisy_mae_m": noisy["euclidean_mae_m"],
    }).sort_values(["coeff", "kl_weight"])
    wide.to_csv(out / "all_settings_summary_9545.csv")

    p = lambda x: f"{x*100:.1f}"
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{All (coeff, $\eta$) settings of the external-prior model ({exp}) at the 95.45\% level, trained "
             r"and calibrated on clean audio. Validation and clean-test component coverage, keyhole coverage (all three "
             r"outputs inside), MWIS and Euclidean MAE; noisy = mean over AWGN and clutter at 15, 5 and $-$5 dB with the "
             r"clean $\hat q$. Coverage in \%.}",
             rf"\label{{tab:all_settings_{exp}}}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{rr rrrr rrrrrr rrrrrr}", r"\toprule",
             r"& & \multicolumn{4}{c}{Validation} & \multicolumn{6}{c}{Clean test} & \multicolumn{6}{c}{Noisy test (mean of 6)} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-12}\cmidrule(lr){13-18}",
             r"$c$ & $\eta$ & sin & cos & range & MAE & sin & cos & range & keyhole & MWIS & MAE "
             r"& sin & cos & range & keyhole & empty & MAE \\", r"\midrule"]
    for r in wide.itertuples():
        lines.append(f"{r.coeff:g} & {r.kl_weight:g} & {p(r.val_cov_sin)} & {p(r.val_cov_cos)} & {p(r.val_cov_range)} & "
                     f"{r.val_mae_m:.2f} & {p(r.clean_cov_sin)} & {p(r.clean_cov_cos)} & {p(r.clean_cov_range)} & "
                     f"{p(r.clean_keyhole)} & {r.clean_mwis:.3f} & {r.clean_mae_m:.2f} & {p(r.noisy_cov_sin)} & "
                     f"{p(r.noisy_cov_cos)} & {p(r.noisy_cov_range)} & {p(r.noisy_keyhole)} & {p(r.noisy_empty)} & "
                     f"{r.noisy_mae_m:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (out / "all_settings_summary_9545.tex").write_text("\n".join(lines))

    for key, lab in [("cov_sin", "sin"), ("cov_cos", "cos"), ("cov_range", "range"), ("keyhole", "keyhole"),
                     ("empty", "empty"), ("mwis", "mwis"), ("euclidean_mae_m", "mae")]:
        piv = c95.pivot_table(index=["coeff", "kl_weight"], columns="condition", values=key)[CONDITIONS]
        piv.to_csv(out / f"per_condition_{lab}_9545.csv")
    print(f"  Saved {out}/ (all_settings_long.csv, all_settings_summary_9545.{{csv,tex}}, per_condition_*.csv)")
    return wide


def set_roots(results_root, runs_root=None):
    global RESULTS_ROOT, RUNS_ROOT, OUT_ROOT, EXPERIMENTS
    RESULTS_ROOT = Path(results_root)
    RUNS_ROOT = Path(runs_root) if runs_root else RUNS_ROOT
    OUT_ROOT = RESULTS_ROOT.parent
    EXPERIMENTS = [RESULTS_ROOT.name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default=str(config.RESULTS_ROOT))
    ap.add_argument("--runs_root", default=str(config.RUNS_ROOT), help="only needed if qhat_table.csv is missing")
    ap.add_argument("--all_settings", action="store_true", help="also write tables for every (coeff, kl) setting")
    args = ap.parse_args()
    set_roots(args.results_root, args.runs_root)
    exp = EXPERIMENTS[0]
    run_experiment(exp)
    if args.all_settings:
        print(f"\n[All settings — {exp}]")
        all_settings_tables(exp)


if __name__ == "__main__":
    main()
