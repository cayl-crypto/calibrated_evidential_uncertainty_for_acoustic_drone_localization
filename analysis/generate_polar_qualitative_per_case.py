"""
generate_polar_qualitative_per_case.py

Per-case polar qualitative figures for the selected model: one figure per case with
three polar panels side by side -- Evidential (raw NIG), Base Model + Conformal, Evidential +
Conformal -- nested 68.27 / 95.45 / 99.73% regions as dotted / dashed / solid outlines, the same
markers (black star ground truth, x evidential prediction, + base prediction, square microphone
array) and legend.

 * Regions use the TRUE clipped interval bounds (clip(mu +/- qhat*sigma); clip(pred +/- q) for the
   plain baseline; stored L/U for raw NIG).
 * Cases:
     1-4  clean test, the four coverage scenarios
     5-7  AWGN:    15 dB raw NIG misses but Evid+Conf (95.45%) covers | 15 dB typical | -5 dB typical
     8-10 clutter: same three
   "typical" = the recording closest to the median evidential localization error under that
   condition, whatever its coverage outcome.
 * The plain baseline has no noisy-condition results, so its panel shows a placeholder in cases 5-10.
 * Within each case the recording closest to the median EVIDENTIAL localization error is drawn
   (the original used the plain baseline's error column).
Models are trained and calibrated on clean audio only; noisy cases reuse the clean q-hat.

Output: <results_root>/figures/polar_qualitative/per_case/{NN}_{case}.png, combined_grid.png and
        per_case_samples.csv (chosen recording, its coverage, how many recordings qualified).
Usage:  python analysis/generate_polar_qualitative_per_case.py [--results_root R] [--hp c0p7_kl0p01]
        (default HP: rank 1 in <results_root>/tables/hp_selection.csv; run generate_all_results.py first)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_all_results as g  # noqa: E402

R_MAX = g.R_MAX
N_THETA = 2000
LEVEL_CFG = [("q6827", "68.27%", ":", 1.0), ("q9545", "95.45%", "--", 1.7), ("q9973", "99.73%", "-", 2.2)]

CASES = [
    dict(idx=1, name="plain_inside_q6827", cond="plain", criterion="inside_q6827",
         label="Clean · GT inside 68.27%"),
    dict(idx=2, name="plain_out68_in95", cond="plain", criterion="out68_in95",
         label="Clean · GT outside 68.27%, inside 95.45%"),
    dict(idx=3, name="plain_out95_in99", cond="plain", criterion="out95_in99",
         label="Clean · GT outside 95.45%, inside 99.73%"),
    dict(idx=4, name="plain_outside_all", cond="plain", criterion="outside_all",
         label="Clean · GT outside all levels"),
    dict(idx=5, name="awgn_15dB_nig_fail_evid_ok", cond="awgn_15dB", criterion="nig_fail_evid_ok",
         label="AWGN 15 dB · NIG misses, Evid+Conf covers"),
    dict(idx=6, name="awgn_15dB_typical", cond="awgn_15dB", criterion="typical",
         label="AWGN 15 dB · typical (median error)"),
    dict(idx=7, name="awgn_-5dB_typical", cond="awgn_-5dB", criterion="typical",
         label="AWGN −5 dB · typical (median error)"),
    dict(idx=8, name="clutter_15dB_nig_fail_evid_ok", cond="clutter_15dB", criterion="nig_fail_evid_ok",
         label="Clutter 15 dB · NIG misses, Evid+Conf covers"),
    dict(idx=9, name="clutter_15dB_typical", cond="clutter_15dB", criterion="typical",
         label="Clutter 15 dB · typical (median error)"),
    dict(idx=10, name="clutter_-5dB_typical", cond="clutter_-5dB", criterion="typical",
         label="Clutter −5 dB · typical (median error)"),
]

# ── Selection ─────────────────────────────────────────────────────────────────

def _joint(df, tag):
    return (df[f"inside_sin_{tag}"].astype(bool) & df[f"inside_cos_{tag}"].astype(bool)
            & df[f"inside_range_{tag}"].astype(bool))


def _nig_joint(df):
    return (df["inside_sin_base"].astype(bool) & df["inside_cos_base"].astype(bool)
            & df["inside_range_base"].astype(bool))


def select_sample(df, criterion):
    """Return (row, pool_size); row = recording closest to the median evidential error in the pool."""
    i68, i95, i99 = _joint(df, "q6827"), _joint(df, "q9545"), _joint(df, "q9973")
    pools = {"inside_q6827": i68, "out68_in95": ~i68 & i95, "out95_in99": ~i95 & i99, "outside_all": ~i99,
             "nig_fail_evid_ok": ~_nig_joint(df) & i95, "typical": pd.Series(True, index=df.index)}
    pool = df[pools[criterion].values]
    if pool.empty:
        return None, 0
    err = g._euclidean_err(pool)
    return pool.iloc[int(np.argmin(np.abs(err - np.median(err))))], len(pool)

# ── Drawing (style copied from TIM_manuscript_20260616/generate_polar_qualitative.py) ──

def _style_ax(ax):
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "−135°", "−90°", "−45°"], fontsize=7)
    ax.set_yticks([50, 100, 150, 200, 250])
    ax.set_yticklabels(["50", "100", "150", "200", "250"], fontsize=6)
    ax.set_ylim(0, R_MAX)
    ax.set_rlabel_position(135)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)


def _draw_keyhole(ax, L_sin, U_sin, L_cos, U_cos, L_r, U_r, color, ls, lw, label, zorder=3):
    theta = np.linspace(0, 2 * np.pi, N_THETA)
    sg, cg = np.sin(theta), np.cos(theta)
    mask = (sg >= L_sin) & (sg <= U_sin) & (cg >= L_cos) & (cg <= U_cos)
    ax.plot(theta, np.where(mask, L_r, np.nan), color=color, ls=ls, lw=lw, label=label, zorder=zorder)
    ax.plot(theta, np.where(mask, U_r, np.nan), color=color, ls=ls, lw=lw, zorder=zorder)
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return
    eps = 1e-6
    for seg in np.split(valid, np.where(np.diff(valid) > 1)[0] + 1):
        if len(seg) < 2:
            continue
        ts, te = theta[seg[0]], theta[seg[-1]]
        if ts > eps:
            ax.plot([ts, ts], [L_r, U_r], color=color, ls=ls, lw=lw, zorder=zorder)
        if te < 2 * np.pi - eps:
            ax.plot([te, te], [L_r, U_r], color=color, ls=ls, lw=lw, zorder=zorder)


def _add_markers(ax, row, show_base_pred, show_evid_pred=True):
    ax.scatter([float(np.arctan2(row["y_sin"], row["y_cos"]))], [float(row["y_range"]) * R_MAX],
               marker="*", s=130, c="black", zorder=8, label="GT")
    if show_evid_pred:
        ax.scatter([float(row["theta_hat_rad"])], [float(row["mu_range"]) * R_MAX],
                   marker="x", s=80, c="#c0392b", linewidths=2, zorder=7, label="Evid. pred.")
    if show_base_pred:
        ax.scatter([float(np.arctan2(row["pred_sin"], row["pred_cos"]))], [float(row["pred_range_norm"]) * R_MAX],
                   marker="+", s=80, c="#1a5276", linewidths=2, zorder=7, label="Base pred.")
    ax.scatter([0.0], [0.0], marker="s", s=45, c="black", zorder=6)


def plot_nig_panel(ax, row, title=True):
    _draw_keyhole(ax, float(row["L_sin_base"]), float(row["U_sin_base"]), float(row["L_cos_base"]),
                  float(row["U_cos_base"]), float(row["L_range_base"]) * R_MAX, float(row["U_range_base"]) * R_MAX,
                  color="#7f8c8d", ls="-", lw=2.0, label="NIG Base")
    _add_markers(ax, row, show_base_pred=False, show_evid_pred=True)
    _style_ax(ax)
    if title:
        ax.set_title("Evidential", fontsize=9, pad=6)


def plot_plain_panel(ax, row, title=True):
    if row is None or "pred_sin" not in row.index or pd.isna(row.get("pred_sin")):
        _style_ax(ax)
        ax.set_yticklabels([])
        ax.text(0.5, 0.5, "Not available under noise\n(no noisy test results\nfor the base model)",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#555555",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbbbbb"))
    else:
        for tag, lbl, ls, lw in LEVEL_CFG:
            _draw_keyhole(ax, float(row[f"Lp_sin_{tag}"]), float(row[f"Up_sin_{tag}"]), float(row[f"Lp_cos_{tag}"]),
                          float(row[f"Up_cos_{tag}"]), float(row[f"Lp_range_{tag}"]) * R_MAX,
                          float(row[f"Up_range_{tag}"]) * R_MAX, color="#4878d0", ls=ls, lw=lw, label=f"Plain+Conf {lbl}")
        _add_markers(ax, row, show_base_pred=True, show_evid_pred=False)
        _style_ax(ax)
    if title:
        ax.set_title("Base Model + Conformal", fontsize=9, pad=6)


def plot_evid_panel(ax, row, title=True):
    for tag, lbl, ls, lw in LEVEL_CFG:
        _draw_keyhole(ax, float(row[f"L_sin_{tag}"]), float(row[f"U_sin_{tag}"]), float(row[f"L_cos_{tag}"]),
                      float(row[f"U_cos_{tag}"]), float(row[f"L_range_{tag}"]) * R_MAX,
                      float(row[f"U_range_{tag}"]) * R_MAX, color="#ee854a", ls=ls, lw=lw, label=f"Evid+Conf {lbl}")
    _add_markers(ax, row, show_base_pred=False, show_evid_pred=True)
    _style_ax(ax)
    if title:
        ax.set_title("Evidential + Conformal", fontsize=9, pad=6)


LEGEND_FULL = [
    mlines.Line2D([], [], color="#7f8c8d", ls="-", lw=2.0, label="Evidential"),
    mlines.Line2D([], [], color="#4878d0", ls=":", lw=1.0, label="Base Model + Conformal 68.27%"),
    mlines.Line2D([], [], color="#4878d0", ls="--", lw=1.7, label="Base Model + Conformal 95.45%"),
    mlines.Line2D([], [], color="#4878d0", ls="-", lw=2.2, label="Base Model + Conformal 99.73%"),
    mlines.Line2D([], [], color="#ee854a", ls=":", lw=1.0, label="Evidential + Conformal 68.27%"),
    mlines.Line2D([], [], color="#ee854a", ls="--", lw=1.7, label="Evidential + Conformal 95.45%"),
    mlines.Line2D([], [], color="#ee854a", ls="-", lw=2.2, label="Evidential + Conformal 99.73%"),
    mlines.Line2D([], [], marker="*", color="black", lw=0, ms=9, label="Ground truth"),
    mlines.Line2D([], [], marker="x", color="#c0392b", lw=0, ms=7, markeredgewidth=2, label="Evidential Prediction"),
    mlines.Line2D([], [], marker="+", color="#1a5276", lw=0, ms=7, markeredgewidth=2, label="Base Prediction"),
    mlines.Line2D([], [], marker="s", color="black", lw=0, ms=7, label="Microphone Array"),
]


def make_case_figure(row, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), subplot_kw={"projection": "polar"}, constrained_layout=True)
    plot_nig_panel(axes[0], row)
    plot_plain_panel(axes[1], row)
    plot_evid_panel(axes[2], row)
    fig.legend(handles=LEGEND_FULL, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.07), framealpha=0.95)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def make_combined_grid(rows, cases, out_path):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n), subplot_kw={"projection": "polar"}, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, (row, case) in enumerate(zip(rows, cases)):
        plot_nig_panel(axes[i, 0], row, title=False)
        plot_plain_panel(axes[i, 1], row, title=False)
        plot_evid_panel(axes[i, 2], row, title=False)
        axes[i, 0].set_ylabel(f"Case {case['idx']}\n{case['label']}", labelpad=40, fontsize=7)
    for ax, title in zip(axes[0], ["Evidential", "Base Model + Conformal", "Evidential + Conformal"]):
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ld = [LEGEND_FULL[0], mlines.Line2D([], [], color="#4878d0", ls="--", lw=1.5, label="Base Model + Conformal 95%"),
          mlines.Line2D([], [], color="#ee854a", ls="--", lw=1.5, label="Evidential + Conformal 95%")] + LEGEND_FULL[7:]
    fig.legend(handles=ld, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.015), framealpha=0.95)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default=str(g.config.RESULTS_ROOT))
    ap.add_argument("--hp", default=None, help="default: rank 1 in <results_root>/tables/hp_selection.csv")
    args = ap.parse_args()
    g.set_roots(args.results_root)
    exp = g.EXPERIMENTS[0]
    hp = args.hp or pd.read_csv(g.OUT_ROOT / exp / "tables" / "hp_selection.csv").hp.iloc[0]
    print(f"External prior {exp} — {hp}")

    out_dir = g.OUT_ROOT / exp / "figures" / "polar_qualitative" / "per_case"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = g.load_base()
    data = {}
    rows, cases, log = [], [], []
    for case in CASES:
        cond = case["cond"]
        if cond not in data:
            df = g.load_evid(exp, hp, cond)
            if cond == "plain":
                df = df.merge(base, on="sample_id", how="inner")
            data[cond] = df
        row, n_pool = select_sample(data[cond], case["criterion"])
        print(f"── Case {case['idx']}: {case['name']}  (pool {n_pool} of {len(data[cond])})")
        if row is None:
            print("  [SKIP] no recording matches")
            log.append(dict(case=case["idx"], name=case["name"], condition=cond, pool=0))
            continue
        fname = f"{case['idx']:02d}_{case['name']}.png"
        make_case_figure(row, out_dir / fname)
        err = float(g._euclidean_err(pd.DataFrame([row]))[0])
        log.append(dict(case=case["idx"], name=case["name"], condition=cond, criterion=case["criterion"],
                        pool=n_pool, pool_share=n_pool / len(data[cond]), sample_id=int(row["sample_id"]),
                        evid_error_m=err, **{f"inside_{t}": bool(_joint(pd.DataFrame([row]), t).iloc[0]) for t, _, _, _ in LEVEL_CFG},
                        nig_inside=bool(_nig_joint(pd.DataFrame([row])).iloc[0]), file=fname))
        rows.append(row)
        cases.append(case)
    pd.DataFrame(log).to_csv(out_dir / "per_case_samples.csv", index=False)
    make_combined_grid(rows, cases, out_dir.parent / "combined_grid.png")
    print(f"\nAll outputs -> {out_dir.parent}")


if __name__ == "__main__":
    main()
