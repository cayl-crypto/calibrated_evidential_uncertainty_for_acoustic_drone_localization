"""
conformal.py

Per-head split conformal calibration on the clean validation calibration split (2,912 recordings),
at 68.27 / 95.45 / 99.73% (tags 6827 / 9545 / 9973).

  Evidential + Conformal: score = |y - mu| / sigma, sigma^2 = beta (1 + v) / (v (alpha - 1))
                          interval = clip(mu +/- q * sigma)
  Base Model + Conformal: score = |y - pred| (range in metres), interval = clip(pred +/- q)
q-hat is the ceil((n + 1)(1 - alpha))-th smallest score. Intervals are clipped to [-1, 1] for
sin/cos and [0, 1] for normalised range.

From calibrate_evidential_external_prior_tim_coverage_levels.py and
calibrate_base_localization_v2_tim_coverage_levels.py (same math).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from .config import MAX_RANGE

ALPHA_TAGS = ["6827", "9545", "9973"]
ALPHAS: Dict[str, float] = {"6827": 0.3173, "9545": 0.0455, "9973": 0.0027}
HEADS = ["sin", "cos", "range"]
BASE_SCORE_TYPES = ["sin", "cos", "range", "bearing", "euclidean"]
BASE_UNITS = {"sin": "unit", "cos": "unit", "range": "m", "bearing": "deg", "euclidean": "m"}


def compute_qhat(scores: torch.Tensor, alpha: float) -> float:
    scores_sorted, _ = torch.sort(scores)
    n = scores_sorted.numel()
    k = min(max(math.ceil((n + 1) * (1.0 - alpha)), 1), n)
    return float(scores_sorted[k - 1].item())


def nig_predictive_sigma_from_pack(pack: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = torch.clamp(pack[:, 1], min=eps)
    a = torch.clamp(pack[:, 2], min=1.0 + eps)
    b = torch.clamp(pack[:, 3], min=eps)
    return torch.sqrt(torch.clamp(b * (1.0 + v) / (v * (a - 1.0)), min=eps))


@torch.no_grad()
def collect_evidential_scores(model, calib_loader, device, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    model = model.to(device).eval()
    chunks: Dict[str, List[torch.Tensor]] = {h: [] for h in HEADS}
    for audio, bearing_vec, range_to_target, _ in tqdm(calib_loader, desc="calib scores", leave=False):
        audio = audio.to(device).float()
        bearing_vec = bearing_vec.to(device).float()
        y = {"sin": bearing_vec[:, 1], "cos": bearing_vec[:, 0], "range": range_to_target.to(device).float().view(-1)}
        means, packs = model.forward_all(audio)
        for h, mu, pack in zip(HEADS, means, packs):
            sigma = nig_predictive_sigma_from_pack(pack, eps)
            chunks[h].append((torch.abs(y[h] - mu) / (sigma + eps)).cpu())
    return {h: torch.cat(v, dim=0) for h, v in chunks.items()}


def _wrap_rad(d: torch.Tensor) -> torch.Tensor:
    return (d + math.pi) % (2 * math.pi) - math.pi


@torch.inference_mode()
def collect_base_scores(model, calib_loader, device) -> Dict[str, torch.Tensor]:
    model = model.to(device).eval()
    chunks: Dict[str, List[torch.Tensor]] = {st: [] for st in BASE_SCORE_TYPES}
    for x, bvec, r_t, _ in tqdm(calib_loader, desc="calib scores (base)", leave=False):
        x = x.to(device).float()
        r_t = r_t.to(device).float().view(-1)
        cos_t, sin_t = bvec[:, 0].to(device).float(), bvec[:, 1].to(device).float()
        sin_p, cos_p, rng_p = model(x)
        rng_p = rng_p.view(-1)
        r_p_m, r_t_m = rng_p * MAX_RANGE, r_t * MAX_RANGE
        chunks["sin"].append(torch.abs(sin_p - sin_t).cpu())
        chunks["cos"].append(torch.abs(cos_p - cos_t).cpu())
        chunks["range"].append(torch.abs(r_p_m - r_t_m).cpu())
        chunks["bearing"].append(torch.abs(torch.rad2deg(_wrap_rad(torch.atan2(sin_p, cos_p) - torch.atan2(sin_t, cos_t)))).cpu())
        chunks["euclidean"].append(torch.sqrt((r_p_m * cos_p - r_t_m * cos_t) ** 2 + (r_p_m * sin_p - r_t_m * sin_t) ** 2).cpu())
    return {st: torch.cat(v, dim=0) for st, v in chunks.items()}


def save_qhats(scores: Dict[str, torch.Tensor], out_dir, units: Dict[str, str], **meta) -> Dict[str, Dict[str, float]]:
    """Writes conformal_q_{head}_{tag}.pt for every head and level; returns {tag: {head: qhat}}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = {}
    for tag in ALPHA_TAGS:
        q[tag] = {}
        for head, s in scores.items():
            q[tag][head] = compute_qhat(s, ALPHAS[tag])
            torch.save({"score_type": head, "units": units[head], "alpha_tag": tag, "alpha": float(ALPHAS[tag]),
                        "qhat": q[tag][head], "n_calib": int(s.numel()), **meta},
                       out_dir / f"conformal_q_{head}_{tag}.pt")
    return q


def load_qhats(q_dir, heads=HEADS) -> Dict[str, Dict[str, float]]:
    """{tag: {head: qhat}} from conformal_q_{head}_{tag}.pt files."""
    return {tag: {h: float(torch.load(Path(q_dir) / f"conformal_q_{h}_{tag}.pt", weights_only=False)["qhat"])
                  for h in heads} for tag in ALPHA_TAGS}
