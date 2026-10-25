"""
evaluation.py

  evaluate_evidential_localization_fn   validation metrics during training (NLL used for early stopping)
                                        -- from evaluate_evidential_localization.py
  evaluate_one_cached                   per-sample CSV for the evidential model: mu, NIG parameters, sigma,
                                        raw +/-1 sigma ("base") interval and conformal intervals at the three
                                        levels -- from document_evidential_real_kl_ood_style_fast.py
  evaluate_plain_localization_fn        validation metrics for the Base Model during training
                                        -- from evaluate_base_localization.py
  evaluate_base_per_sample              per-sample CSV for Base Model + Conformal
                                        -- from document_base_localization_v5_tim_coverage_levels.py
The numerical code is copied unchanged.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.distributions import StudentT
from tqdm import tqdm

from .config import MAX_RANGE

# ── evidential: validation metrics during training ─────────────────────────

def nig_nll_from_pack(pack: torch.Tensor, y: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    pack: (B,4) = [mu, lam, alpha, beta]
    y:    (B,)
    Returns NLL per-sample: (B,)
    Predictive distribution is Student-t with:
        ν   = 2α
        loc = μ
        scale^2 = β * (1 + 1/λ) / α
    """
    mu    = pack[:, 0]
    lam   = pack[:, 1]
    alpha = pack[:, 2]
    beta  = pack[:, 3]

    nu = 2.0 * alpha
    scale2 = beta * (1.0 + 1.0 / (lam + eps)) / (alpha + eps)
    scale = torch.sqrt(torch.clamp(scale2, min=eps))

    dist = StudentT(df=nu, loc=mu, scale=scale)
    return -dist.log_prob(y)


def nig_std_from_pack(pack: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    pack: (B,4) = [mu, lam, alpha, beta]
    Predictive std of Student-t:
        Var[y] = β * (1 + 1/λ) / (α - 1), α > 1
    """
    lam   = pack[:, 1]
    alpha = pack[:, 2]
    beta  = pack[:, 3]

    var = beta * (1.0 + 1.0 / (lam + eps)) / (torch.clamp(alpha - 1.0, min=eps))
    return torch.sqrt(torch.clamp(var, min=eps))


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap angle to (-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


@torch.no_grad()
def evaluate_evidential_localization_fn(
    model,
    dataloader,
    device: str = "cuda",
    max_range: float = 250.0,
    eps: float = 1e-8,
):
    """
    Evaluate EvidentialLocalization model on a localization dataloader.

    Assumes each batch from `dataloader` is:
        audio_data, bearing_vec, range_to_target, sample_id
    where
        bearing_vec.shape == (B, 2)  -> [sin, cos]
        range_to_target.shape == (B,) or (B,1)  -> normalized in [0, 1]

    Returns a dictionary with:
      - NLLs (per-sample mean): total, sin, cos, range
      - MAEs: bearing (deg), range (m), euclidean (m)
      - Coverage (fractions) for 1, 2, 3 std for:
            sin, cos, range, bearing, overall
    """
    model = model.to(device)
    model.eval()



    # ----- accumulators -----

    total_samples = 0.0

    nll_sin_sum = 0.0
    nll_cos_sum = 0.0
    nll_range_sum = 0.0
    nll_total_sum = 0.0

    bearing_mae_sum = 0.0
    range_mae_sum = 0.0
    euc_mae_sum = 0.0

    sigmas = [1, 2, 3]
    coverage_counts = {
        "sin":     {k: 0.0 for k in sigmas},
        "cos":     {k: 0.0 for k in sigmas},
        "range":   {k: 0.0 for k in sigmas},
        "bearing": {k: 0.0 for k in sigmas},
        "overall": {k: 0.0 for k in sigmas},
    }

    # ----- main loop -----

    for batch in dataloader:
        # Unpack batch from TestCustomAudioDataset
        audio_data, bearing_vec, range_to_target, sample_id = batch

        audio_data = audio_data.to(device)
        bearing_vec = bearing_vec.to(device)
        range_to_target = range_to_target.to(device).view(-1)

        B = audio_data.size(0)
        total_samples += B

        y_sin = bearing_vec[:, 1]
        y_cos = bearing_vec[:, 0]
        y_range = range_to_target

        # Get evidential outputs: means and (mu, lam, alpha, beta) packs
        means, packs = model.forward_all(audio_data)
        sin_mu, cos_mu, rng_mu = means
        sin_pack, cos_pack, rng_pack = packs  # each (B,4)

        # ----- NLLs -----
        nll_sin = nig_nll_from_pack(sin_pack, y_sin)       # (B,)
        nll_cos = nig_nll_from_pack(cos_pack, y_cos)
        nll_rng = nig_nll_from_pack(rng_pack, y_range)

        nll_sin_sum   += nll_sin.sum().item()
        nll_cos_sum   += nll_cos.sum().item()
        nll_range_sum += nll_rng.sum().item()
        nll_total_sum += (nll_sin + nll_cos + nll_rng).sum().item()

        # ----- point-estimate metrics using your helper -----
        metrics = model.compute_metrics_from_means(
            sin_mu, cos_mu, rng_mu,
            y_sin,  y_cos,  y_range,
            max_range=max_range,
        )
        bearing_mae_sum += metrics["bearing_mae_deg"] * B
        range_mae_sum   += metrics["range_mae_m"] * B
        euc_mae_sum     += metrics["euclidean_mae_m"] * B

        # ----- predictive stds -----
        sin_std = nig_std_from_pack(sin_pack)    # (B,)
        cos_std = nig_std_from_pack(cos_pack)
        rng_std = nig_std_from_pack(rng_pack)

        # ----- bearing std from sin/cos (error propagation) -----
        # θ = atan2(sin, cos)
        theta_mu = torch.atan2(sin_mu, cos_mu)
        theta_true = torch.atan2(y_sin, y_cos)

        denom = sin_mu**2 + cos_mu**2 + eps
        dtheta_dsin =  cos_mu / denom      # ∂θ/∂sin
        dtheta_dcos = -sin_mu / denom      # ∂θ/∂cos

        var_theta = (dtheta_dsin**2) * (sin_std**2) + (dtheta_dcos**2) * (cos_std**2)
        theta_std = torch.sqrt(torch.clamp(var_theta, min=eps))

        angle_diff = wrap_angle(theta_true - theta_mu)  # (B,)

        # ----- coverages for 1σ, 2σ, 3σ -----
        for k in sigmas:
            kf = float(k)

            sin_mask = (torch.abs(y_sin   - sin_mu)  <= kf * sin_std)
            cos_mask = (torch.abs(y_cos   - cos_mu)  <= kf * cos_std)
            rng_mask = (torch.abs(y_range - rng_mu)  <= kf * rng_std)
            bearing_mask = (torch.abs(angle_diff)    <= kf * theta_std)

            overall_mask = rng_mask & bearing_mask

            coverage_counts["sin"][k]     += sin_mask.sum().item()
            coverage_counts["cos"][k]     += cos_mask.sum().item()
            coverage_counts["range"][k]   += rng_mask.sum().item()
            coverage_counts["bearing"][k] += bearing_mask.sum().item()
            coverage_counts["overall"][k] += overall_mask.sum().item()

    # ----- normalize to per-sample means -----

    results = {
        "nll": {
            "total":  nll_total_sum / total_samples,
            "sin":    nll_sin_sum   / total_samples,
            "cos":    nll_cos_sum   / total_samples,
            "range":  nll_range_sum / total_samples,
        },
        "mae": {
            "bearing_deg":  bearing_mae_sum / total_samples,
            "range_m":      range_mae_sum   / total_samples,
            "euclidean_m":  euc_mae_sum     / total_samples,
        },
        "coverage": {
            "sin":     {f"{k}std": coverage_counts["sin"][k]     / total_samples for k in sigmas},
            "cos":     {f"{k}std": coverage_counts["cos"][k]     / total_samples for k in sigmas},
            "range":   {f"{k}std": coverage_counts["range"][k]   / total_samples for k in sigmas},
            "bearing": {f"{k}std": coverage_counts["bearing"][k] / total_samples for k in sigmas},
            "overall": {f"{k}std": coverage_counts["overall"][k] / total_samples for k in sigmas},
        },
        "num_samples": int(total_samples),
    }

    return results


def export_evidential_results_to_csv(results: dict, output_path: str):
    """
    Flatten the nested results dictionary produced by evaluate_evidential_localization
    and save it into a CSV file with two columns: metric, value.

    Args:
        results: dict returned by evaluate_evidential_localization
        output_path: full path to CSV file
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # flatten the dict into key → value
    flat_rows = {}

    # NLL
    for k, v in results["nll"].items():
        flat_rows[f"nll_{k}"] = v

    # MAE
    for k, v in results["mae"].items():
        flat_rows[f"mae_{k}"] = v

    # coverage
    for category, subdict in results["coverage"].items():
        for k, v in subdict.items():
            flat_rows[f"coverage_{category}_{k}"] = v

    # num samples
    flat_rows["num_samples"] = results["num_samples"]

    # write CSV
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in flat_rows.items():
            writer.writerow([metric, value])

    print(f"Results exported to: {out_file}")


# ── evidential: per-sample results ─────────────────────────────────────────

def nig_unpack_params(pack: torch.Tensor, eps: float = 1e-8):
    v_raw = pack[:, 1]
    alpha_raw = pack[:, 2]
    beta_raw = pack[:, 3]

    v = torch.clamp(v_raw, min=eps)
    alpha = torch.clamp(alpha_raw, min=1.0 + eps)
    beta = torch.clamp(beta_raw, min=eps)

    v_clamped = v_raw <= eps
    alpha_clamped = alpha_raw <= (1.0 + eps)
    beta_clamped = beta_raw <= eps

    return v, alpha, beta, v_clamped, alpha_clamped, beta_clamped


def nig_variance_decomposition(v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-8):
    denom = torch.clamp(alpha - 1.0, min=eps)
    v_safe = torch.clamp(v, min=eps)

    aleatoric_var = beta / denom
    epistemic_var = beta / (v_safe * denom)
    total_var = aleatoric_var + epistemic_var
    return aleatoric_var, epistemic_var, total_var


def safe_atan2(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):
    y2 = torch.where(y.abs() < eps, y * 0.0, y)
    x2 = torch.where(x.abs() < eps, x * 0.0, x)
    return torch.atan2(y2, x2)


@torch.no_grad()
def evaluate_one_cached(model, test_loader, q_values, device, run_name: str, desc: str) -> pd.DataFrame:
    """Same math as evaluate_one() in document_ood_ablation_results_v5_tim_coverage_levels.py,
    but takes an already-loaded model and an already-built test_loader instead of
    constructing them internally."""
    q_labels = sorted(q_values.keys())
    rows = []

    for batch in tqdm(test_loader, desc=desc, leave=False):
        audio_data, bearing_vec, range_to_target, sample_id = batch
        audio_data = audio_data.to(device)
        bearing_vec = bearing_vec.to(device)
        range_to_target = range_to_target.to(device).view(-1)
        B = audio_data.size(0)

        y_sin = bearing_vec[:, 1]
        y_cos = bearing_vec[:, 0]
        y_range = range_to_target

        means, packs = model.forward_all(audio_data)
        mu_sin, mu_cos, mu_range = means
        sin_pack, cos_pack, range_pack = packs

        v_sin, alpha_sin, beta_sin, v_sin_cl, a_sin_cl, b_sin_cl = nig_unpack_params(sin_pack)
        v_cos, alpha_cos, beta_cos, v_cos_cl, a_cos_cl, b_cos_cl = nig_unpack_params(cos_pack)
        v_rng, alpha_rng, beta_rng, v_rng_cl, a_rng_cl, b_rng_cl = nig_unpack_params(range_pack)

        nu_sin = 2.0 * alpha_sin
        nu_cos = 2.0 * alpha_cos
        nu_rng = 2.0 * alpha_rng

        alea_var_sin, epi_var_sin, tot_var_sin = nig_variance_decomposition(v_sin, alpha_sin, beta_sin)
        alea_var_cos, epi_var_cos, tot_var_cos = nig_variance_decomposition(v_cos, alpha_cos, beta_cos)
        alea_var_rng, epi_var_rng, tot_var_rng = nig_variance_decomposition(v_rng, alpha_rng, beta_rng)

        sigma_sin = torch.sqrt(torch.clamp(tot_var_sin, min=1e-8))
        sigma_cos = torch.sqrt(torch.clamp(tot_var_cos, min=1e-8))
        sigma_range = torch.sqrt(torch.clamp(tot_var_rng, min=1e-8))

        strength_sin = v_sin + alpha_sin
        strength_cos = v_cos + alpha_cos
        strength_rng = v_rng + alpha_rng

        vacuity_sin = 1.0 / (1.0 + v_sin)
        vacuity_cos = 1.0 / (1.0 + v_cos)
        vacuity_rng = 1.0 / (1.0 + v_rng)

        alpha_near_1_sin = alpha_sin < 1.05
        alpha_near_1_cos = alpha_cos < 1.05
        alpha_near_1_rng = alpha_rng < 1.05

        v_tiny_sin = v_sin < 1e-3
        v_tiny_cos = v_cos < 1e-3
        v_tiny_rng = v_rng < 1e-3

        theta_hat = safe_atan2(mu_sin, mu_cos)
        mu_norm = torch.sqrt(torch.clamp(mu_sin ** 2 + mu_cos ** 2, min=1e-12))

        bearing_alea_var = alea_var_sin + alea_var_cos
        bearing_epi_var = epi_var_sin + epi_var_cos
        bearing_tot_var = tot_var_sin + tot_var_cos

        base_L_sin = torch.clamp(mu_sin - sigma_sin, -1.0, 1.0)
        base_U_sin = torch.clamp(mu_sin + sigma_sin, -1.0, 1.0)
        base_L_cos = torch.clamp(mu_cos - sigma_cos, -1.0, 1.0)
        base_U_cos = torch.clamp(mu_cos + sigma_cos, -1.0, 1.0)
        base_L_range = torch.clamp(mu_range - sigma_range, 0.0, 1.0)
        base_U_range = torch.clamp(mu_range + sigma_range, 0.0, 1.0)

        base_inside_sin = (y_sin >= base_L_sin) & (y_sin <= base_U_sin)
        base_inside_cos = (y_cos >= base_L_cos) & (y_cos <= base_U_cos)
        base_inside_range = (y_range >= base_L_range) & (y_range <= base_U_range)

        conformal_data = {}
        for label in q_labels:
            qs = q_values[label]
            R_sin = qs["sin"] * sigma_sin
            R_cos = qs["cos"] * sigma_cos
            R_rng = qs["range"] * sigma_range

            L_sin = torch.clamp(mu_sin - R_sin, -1.0, 1.0)
            U_sin = torch.clamp(mu_sin + R_sin, -1.0, 1.0)
            L_cos = torch.clamp(mu_cos - R_cos, -1.0, 1.0)
            U_cos = torch.clamp(mu_cos + R_cos, -1.0, 1.0)
            L_rng = torch.clamp(mu_range - R_rng, 0.0, 1.0)
            U_rng = torch.clamp(mu_range + R_rng, 0.0, 1.0)

            inside_sin = (y_sin >= L_sin) & (y_sin <= U_sin)
            inside_cos = (y_cos >= L_cos) & (y_cos <= U_cos)
            inside_rng = (y_range >= L_rng) & (y_range <= U_rng)

            conformal_data[label] = (L_sin, U_sin, inside_sin, L_cos, U_cos, inside_cos, L_rng, U_rng, inside_rng)

        if isinstance(sample_id, torch.Tensor):
            sample_id_list = sample_id.detach().cpu().numpy().tolist()
        else:
            sample_id_list = list(sample_id)

        def c(t):
            return t.detach().cpu()

        y_sin_c, y_cos_c, y_rng_c = c(y_sin), c(y_cos), c(y_range)
        mu_sin_c, mu_cos_c, mu_rng_c = c(mu_sin), c(mu_cos), c(mu_range)
        sigma_sin_c, sigma_cos_c, sigma_rng_c = c(sigma_sin), c(sigma_cos), c(sigma_range)

        v_sin_c, a_sin_c, b_sin_c, nu_sin_c = c(v_sin), c(alpha_sin), c(beta_sin), c(nu_sin)
        v_cos_c, a_cos_c, b_cos_c, nu_cos_c = c(v_cos), c(alpha_cos), c(beta_cos), c(nu_cos)
        v_rng_c, a_rng_c, b_rng_c, nu_rng_c = c(v_rng), c(alpha_rng), c(beta_rng), c(nu_rng)

        alea_sin_c, epi_sin_c, tot_sin_c = c(alea_var_sin), c(epi_var_sin), c(tot_var_sin)
        alea_cos_c, epi_cos_c, tot_cos_c = c(alea_var_cos), c(epi_var_cos), c(tot_var_cos)
        alea_rng_c, epi_rng_c, tot_rng_c = c(alea_var_rng), c(epi_var_rng), c(tot_var_rng)

        strength_sin_c, strength_cos_c, strength_rng_c = c(strength_sin), c(strength_cos), c(strength_rng)
        vac_sin_c, vac_cos_c, vac_rng_c = c(vacuity_sin), c(vacuity_cos), c(vacuity_rng)

        theta_c, mu_norm_c = c(theta_hat), c(mu_norm)
        b_alea_c, b_epi_c, b_tot_c = c(bearing_alea_var), c(bearing_epi_var), c(bearing_tot_var)

        base_L_sin_c, base_U_sin_c = c(base_L_sin), c(base_U_sin)
        base_L_cos_c, base_U_cos_c = c(base_L_cos), c(base_U_cos)
        base_L_rng_c, base_U_rng_c = c(base_L_range), c(base_U_range)

        base_in_sin_c, base_in_cos_c, base_in_rng_c = c(base_inside_sin), c(base_inside_cos), c(base_inside_range)

        v_sin_cl_c, a_sin_cl_c, b_sin_cl_c = c(v_sin_cl), c(a_sin_cl), c(b_sin_cl)
        v_cos_cl_c, a_cos_cl_c, b_cos_cl_c = c(v_cos_cl), c(a_cos_cl), c(b_cos_cl)
        v_rng_cl_c, a_rng_cl_c, b_rng_cl_c = c(v_rng_cl), c(a_rng_cl), c(b_rng_cl)

        an1_sin_c, an1_cos_c, an1_rng_c = c(alpha_near_1_sin), c(alpha_near_1_cos), c(alpha_near_1_rng)
        vt_sin_c, vt_cos_c, vt_rng_c = c(v_tiny_sin), c(v_tiny_cos), c(v_tiny_rng)

        conformal_cpu = {}
        for label in q_labels:
            Ls, Us, ins, Lc, Uc, inc, Lr, Ur, inr = conformal_data[label]
            conformal_cpu[label] = (c(Ls), c(Us), c(ins), c(Lc), c(Uc), c(inc), c(Lr), c(Ur), c(inr))

        for i in range(B):
            row = {
                "model_source": "plain", "calib_source": "plain", "test_setting": "plain",
                "run_name": run_name, "sample_id": sample_id_list[i],

                "y_sin": float(y_sin_c[i]), "y_cos": float(y_cos_c[i]), "y_range": float(y_rng_c[i]),
                "mu_sin": float(mu_sin_c[i]), "mu_cos": float(mu_cos_c[i]), "mu_range": float(mu_rng_c[i]),
                "sigma_sin": float(sigma_sin_c[i]), "sigma_cos": float(sigma_cos_c[i]), "sigma_range": float(sigma_rng_c[i]),

                "v_sin": float(v_sin_c[i]), "alpha_sin": float(a_sin_c[i]), "beta_sin": float(b_sin_c[i]), "nu_sin": float(nu_sin_c[i]),
                "v_cos": float(v_cos_c[i]), "alpha_cos": float(a_cos_c[i]), "beta_cos": float(b_cos_c[i]), "nu_cos": float(nu_cos_c[i]),
                "v_range": float(v_rng_c[i]), "alpha_range": float(a_rng_c[i]), "beta_range": float(b_rng_c[i]), "nu_range": float(nu_rng_c[i]),

                "aleatoric_var_sin": float(alea_sin_c[i]), "epistemic_var_sin": float(epi_sin_c[i]), "total_var_sin": float(tot_sin_c[i]),
                "aleatoric_var_cos": float(alea_cos_c[i]), "epistemic_var_cos": float(epi_cos_c[i]), "total_var_cos": float(tot_cos_c[i]),
                "aleatoric_var_range": float(alea_rng_c[i]), "epistemic_var_range": float(epi_rng_c[i]), "total_var_range": float(tot_rng_c[i]),

                "strength_sin": float(strength_sin_c[i]), "strength_cos": float(strength_cos_c[i]), "strength_range": float(strength_rng_c[i]),
                "vacuity_sin": float(vac_sin_c[i]), "vacuity_cos": float(vac_cos_c[i]), "vacuity_range": float(vac_rng_c[i]),

                "theta_hat_rad": float(theta_c[i]), "mu_norm": float(mu_norm_c[i]),
                "bearing_aleatoric_var": float(b_alea_c[i]), "bearing_epistemic_var": float(b_epi_c[i]), "bearing_total_var": float(b_tot_c[i]),

                "v_sin_clamped": int(v_sin_cl_c[i]), "alpha_sin_clamped": int(a_sin_cl_c[i]), "beta_sin_clamped": int(b_sin_cl_c[i]),
                "v_cos_clamped": int(v_cos_cl_c[i]), "alpha_cos_clamped": int(a_cos_cl_c[i]), "beta_cos_clamped": int(b_cos_cl_c[i]),
                "v_range_clamped": int(v_rng_cl_c[i]), "alpha_range_clamped": int(a_rng_cl_c[i]), "beta_range_clamped": int(b_rng_cl_c[i]),

                "alpha_near_1_sin": int(an1_sin_c[i]), "alpha_near_1_cos": int(an1_cos_c[i]), "alpha_near_1_range": int(an1_rng_c[i]),
                "v_tiny_sin": int(vt_sin_c[i]), "v_tiny_cos": int(vt_cos_c[i]), "v_tiny_range": int(vt_rng_c[i]),

                "L_sin_base": float(base_L_sin_c[i]), "U_sin_base": float(base_U_sin_c[i]),
                "inside_sin_base": int(base_in_sin_c[i]), "W_sin_base": float(base_U_sin_c[i] - base_L_sin_c[i]),

                "L_cos_base": float(base_L_cos_c[i]), "U_cos_base": float(base_U_cos_c[i]),
                "inside_cos_base": int(base_in_cos_c[i]), "W_cos_base": float(base_U_cos_c[i] - base_L_cos_c[i]),

                "L_range_base": float(base_L_rng_c[i]), "U_range_base": float(base_U_rng_c[i]),
                "inside_range_base": int(base_in_rng_c[i]), "W_range_base": float(base_U_rng_c[i] - base_L_rng_c[i]),
            }

            for label in q_labels:
                Ls, Us, ins, Lc, Uc, inc, Lr, Ur, inr = conformal_cpu[label]
                row[f"inside_sin_q{label}"] = int(ins[i])
                row[f"inside_cos_q{label}"] = int(inc[i])
                row[f"inside_range_q{label}"] = int(inr[i])
                row[f"W_sin_q{label}"] = float(Us[i] - Ls[i])
                row[f"W_cos_q{label}"] = float(Uc[i] - Lc[i])
                row[f"W_range_q{label}"] = float(Ur[i] - Lr[i])

            rows.append(row)

    return pd.DataFrame(rows)


# ── Base Model: validation metrics during training ─────────────────────────

def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angle to (-pi, pi]."""
    return (x + math.pi) % (2 * math.pi) - math.pi


@torch.no_grad()
def evaluate_plain_localization_fn(
    model,
    loader,
    device: Optional[torch.device] = None,
    max_range: float = 250.0,
    desc: str = "Eval (plain loc)",
) -> Dict[str, Any]:
    """
    Plain localization evaluator with the SAME "results dict" vibe as evidential.

    Returns a dict with:
      results = {
        "loss": {"total": float, "sin": float, "cos": float, "range": float},
        "mae":  {"bearing_deg": float, "range_m": float, "euclidean_m": float},
        "n_samples": int,
        "rows": [ {per-sample fields...}, ... ],
      }

    Notes:
    - No evidential params -> no NLL or sigma coverage. We export what exists.
    - If your dataloader yields ids, we keep them; otherwise we auto-index.
    """

    model.eval()

    if device is None:
        device = next(model.parameters()).device

    total_n = 0

    # loss parts accumulators (MSE)
    sum_mse_sin = 0.0
    sum_mse_cos = 0.0
    sum_mse_rng = 0.0
    sum_loss_total = 0.0

    # MAE accumulators
    sum_bearing_mae_deg = 0.0
    sum_range_mae_m = 0.0
    sum_euclidean_mae_m = 0.0

    rows: List[Dict[str, Any]] = []
    running_id = 0  # fallback id if not provided by loader

    for batch in tqdm(loader, desc=desc, leave=False):
        # Support (x, bearing_vec, r_t) OR (x, bearing_vec, r_t, ids)
        if len(batch) == 3:
            x, bearing_vec, r_t = batch
            ids = None
        elif len(batch) == 4:
            x, bearing_vec, r_t, ids = batch
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}. Expected 3 or 4 items.")

        x = x.to(device, non_blocking=True).float()
        r_t = r_t.to(device, non_blocking=True).float()

        # bearing_vec is (cos, sin) by your convention
        cos_t = bearing_vec[:, 0].to(device, non_blocking=True).float()
        sin_t = bearing_vec[:, 1].to(device, non_blocking=True).float()

        # forward
        sin_p, cos_p, r_p = model(x)

        bsz = x.size(0)
        total_n += bsz

        # ----- compute MSE components
        mse_sin = torch.mean((sin_p - sin_t) ** 2)
        mse_cos = torch.mean((cos_p - cos_t) ** 2)
        mse_rng = torch.mean((r_p - r_t) ** 2)

        # if your model has a compute_loss() that includes extra penalties (unit norm), use it
        if hasattr(model, "compute_loss"):
            loss_total = model.compute_loss((sin_p, cos_p, r_p), (sin_t, cos_t, r_t))
        else:
            loss_total = mse_sin + mse_cos + mse_rng

        sum_mse_sin += mse_sin.item() * bsz
        sum_mse_cos += mse_cos.item() * bsz
        sum_mse_rng += mse_rng.item() * bsz
        sum_loss_total += loss_total.item() * bsz

        # ----- compute MAEs
        pred_bearing = torch.atan2(sin_p, cos_p)
        true_bearing = torch.atan2(sin_t, cos_t)
        d = _wrap_pi(pred_bearing - true_bearing)
        bearing_mae_deg = torch.rad2deg(torch.abs(d))  # (B,)

        r_hat_m = r_p * max_range
        r_true_m = r_t * max_range
        range_mae_m = torch.abs(r_hat_m - r_true_m)  # (B,)

        px = r_hat_m * cos_p
        py = r_hat_m * sin_p
        tx = r_true_m * cos_t
        ty = r_true_m * sin_t
        euclidean_mae_m = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2)  # (B,)

        sum_bearing_mae_deg += bearing_mae_deg.mean().item() * bsz
        sum_range_mae_m += range_mae_m.mean().item() * bsz
        sum_euclidean_mae_m += euclidean_mae_m.mean().item() * bsz

        # ----- per-sample rows (schema similar to your evidential per-sample CSV)
        if ids is None:
            # auto ids
            batch_ids = list(range(running_id, running_id + bsz))
            running_id += bsz
        else:
            # ids can be tensor/list/tuple
            if torch.is_tensor(ids):
                batch_ids = ids.detach().cpu().tolist()
            else:
                batch_ids = list(ids)

        # Convert bearings to degrees in [0, 360) or (-180,180]? keep consistent with your evaluator
        bearing_pred_deg = torch.rad2deg(pred_bearing)
        bearing_gt_deg = torch.rad2deg(true_bearing)

        for i in range(bsz):
            rows.append(
                {
                    "id": batch_ids[i],
                    "range_pred": float(r_hat_m[i].item()),
                    "range_true": float(r_true_m[i].item()),
                    "bearing_pred_deg": float(bearing_pred_deg[i].item()),
                    "bearing_gt_deg": float(bearing_gt_deg[i].item()),
                    "range_mae": float(range_mae_m[i].item()),
                    "bearing_mae_deg": float(bearing_mae_deg[i].item()),
                    "euclidean_mae": float(euclidean_mae_m[i].item()),
                    # placeholders to keep compatibility with "evidential" CSV schema if you reuse it
                    "covered_1sigma": "",
                    "covered_2sigma": "",
                    "covered_3sigma": "",
                    "outside_3sigma": "",
                    "coverage_category": "",
                }
            )

    denom = max(total_n, 1)
    results: Dict[str, Any] = {
        "loss": {
            "total": sum_loss_total / denom,
            "sin": sum_mse_sin / denom,
            "cos": sum_mse_cos / denom,
            "range": sum_mse_rng / denom,
        },
        "mae": {
            "bearing_deg": sum_bearing_mae_deg / denom,
            "range_m": sum_range_mae_m / denom,
            "euclidean_m": sum_euclidean_mae_m / denom,
        },
        "n_samples": int(total_n),
        "rows": rows,
    }
    return results


def export_plain_results_to_csv(results: Dict[str, Any], out_csv_path: str) -> None:
    """
    Exports per-sample rows just like your evidential exporter does.
    """
    rows = results.get("rows", [])
    if not rows:
        return

    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── Base Model + Conformal: per-sample results ─────────────────────────────

ALPHA_TAGS = ["6827", "9545", "9973"]
SCORE_TYPES = ["sin", "cos", "range", "bearing", "euclidean"]


def _wrap_rad_base(d: torch.Tensor) -> torch.Tensor:
    return (d + math.pi) % (2 * math.pi) - math.pi


def _bearing_deg(sin_v: torch.Tensor, cos_v: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.atan2(sin_v, cos_v))


def _euclidean_m(
    ps: torch.Tensor, pc: torch.Tensor, pr: torch.Tensor,
    ts: torch.Tensor, tc: torch.Tensor, tr: torch.Tensor,
) -> torch.Tensor:
    r_p = pr * MAX_RANGE
    r_t = tr * MAX_RANGE
    return torch.sqrt((r_p * pc - r_t * tc) ** 2 + (r_p * ps - r_t * ts) ** 2)


def _eff_width(pred: torch.Tensor, q: float, lo: float, hi: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Effective interval width after clipping to [lo, hi], and fraction of domain."""
    clip_lo = torch.clamp(pred - q, lo, hi)
    clip_hi = torch.clamp(pred + q, lo, hi)
    eff_w = clip_hi - clip_lo
    return eff_w, eff_w / float(hi - lo)


@torch.inference_mode()
def evaluate_base_per_sample(model, test_loader, qhats, device, run_name: str, test_setting: str = "plain",
                             model_source: str = "plain", calib_source: str = "plain"):
    """qhats: {score_type: {alpha_tag: qhat}} (range and euclidean in metres, bearing in degrees).
    Returns (per-sample DataFrame, one-row summary DataFrame)."""
    model = model.to(device).eval()
    rows = []

    for batch in tqdm(
        test_loader,
        desc=f"base @ {test_setting}",
        leave=False,
    ):
        audio, bvec, r_t, sample_id = batch

        audio = audio.to(device).float()
        bvec = bvec.to(device).float()
        r_t = r_t.to(device).float().view(-1)

        cos_t = bvec[:, 0]
        sin_t = bvec[:, 1]

        sin_p, cos_p, r_p = model(audio)
        r_p = r_p.view(-1)

        B = audio.size(0)

        r_p_m = r_p * MAX_RANGE
        r_t_m = r_t * MAX_RANGE

        b_pred = _bearing_deg(sin_p, cos_p)
        b_true = _bearing_deg(sin_t, cos_t)
        b_err = torch.abs(torch.rad2deg(
            _wrap_rad_base(torch.deg2rad(b_pred) - torch.deg2rad(b_true))
        ))

        err_sin = torch.abs(sin_p - sin_t)
        err_cos = torch.abs(cos_p - cos_t)
        err_r_norm = torch.abs(r_p - r_t)
        err_r_m = torch.abs(r_p_m - r_t_m)
        err_euc = _euclidean_m(sin_p, cos_p, r_p, sin_t, cos_t, r_t)

        if isinstance(sample_id, torch.Tensor):
            id_list = sample_id.detach().cpu().numpy().tolist()
        else:
            id_list = list(sample_id)

        def c(t): return t.detach().cpu()

        sin_p_c, cos_p_c, r_p_c = c(sin_p), c(cos_p), c(r_p)
        r_p_m_c, r_t_m_c = c(r_p_m), c(r_t_m)
        sin_t_c, cos_t_c, r_t_c = c(sin_t), c(cos_t), c(r_t)
        b_pred_c, b_true_c, b_err_c = c(b_pred), c(b_true), c(b_err)
        err_sin_c, err_cos_c = c(err_sin), c(err_cos)
        err_r_c, err_r_m_c = c(err_r_norm), c(err_r_m)
        err_euc_c = c(err_euc)

        for i in range(B):
            row: dict = {
                "model_source":    model_source,
                "calib_source":    calib_source,
                "test_setting":    test_setting,
                "run_name":        run_name,
                "sample_id":       id_list[i],
                "pred_sin":        float(sin_p_c[i]),
                "pred_cos":        float(cos_p_c[i]),
                "pred_range_norm": float(r_p_c[i]),
                "pred_range_m":    float(r_p_m_c[i]),
                "pred_bearing_deg":float(b_pred_c[i]),
                "gt_sin":          float(sin_t_c[i]),
                "gt_cos":          float(cos_t_c[i]),
                "gt_range_norm":   float(r_t_c[i]),
                "gt_range_m":      float(r_t_m_c[i]),
                "gt_bearing_deg":  float(b_true_c[i]),
                "err_sin":         float(err_sin_c[i]),
                "err_cos":         float(err_cos_c[i]),
                "err_range_norm":  float(err_r_c[i]),
                "err_range_m":     float(err_r_m_c[i]),
                "err_bearing_deg": float(b_err_c[i]),
                "err_euclidean_m": float(err_euc_c[i]),
            }

            for tag in ALPHA_TAGS:
                q_sin = qhats["sin"][tag]
                q_cos = qhats["cos"][tag]
                q_rng = qhats["range"][tag]      # metres
                q_brg = qhats["bearing"][tag]    # degrees
                q_euc = qhats["euclidean"][tag]  # metres

                # Coverage
                row[f"cov_sin_q{tag}"]       = int(float(err_sin_c[i])   <= q_sin)
                row[f"cov_cos_q{tag}"]       = int(float(err_cos_c[i])   <= q_cos)
                row[f"cov_range_q{tag}"]     = int(float(err_r_m_c[i])   <= q_rng)
                row[f"cov_bearing_q{tag}"]   = int(float(b_err_c[i])     <= q_brg)
                row[f"cov_euclidean_q{tag}"] = int(float(err_euc_c[i])   <= q_euc)

                # Raw interval half-widths (= q_hat)
                row[f"q_sin_{tag}"]       = float(q_sin)
                row[f"q_cos_{tag}"]       = float(q_cos)
                row[f"q_range_{tag}"]     = float(q_rng)
                row[f"q_bearing_{tag}"]   = float(q_brg)
                row[f"q_euclidean_{tag}"] = float(q_euc)

                # Effective widths (clipped to domain)
                ew_sin,  ef_sin  = _eff_width(sin_p_c[i:i+1], q_sin,        -1.0, 1.0)
                ew_cos,  ef_cos  = _eff_width(cos_p_c[i:i+1], q_cos,        -1.0, 1.0)
                q_rng_norm = q_rng / MAX_RANGE
                ew_rn,   ef_rn   = _eff_width(r_p_c[i:i+1],  q_rng_norm,   0.0,  1.0)

                row[f"eff_width_sin_q{tag}"]        = float(ew_sin.item())
                row[f"eff_width_cos_q{tag}"]        = float(ew_cos.item())
                row[f"eff_width_range_norm_q{tag}"] = float(ew_rn.item())
                row[f"eff_frac_sin_q{tag}"]         = float(ef_sin.item())
                row[f"eff_frac_cos_q{tag}"]         = float(ef_cos.item())
                row[f"eff_frac_range_norm_q{tag}"]  = float(ef_rn.item())

            rows.append(row)

    df = pd.DataFrame(rows)

    # Summary row
    summary: dict = {
        "model_source": model_source,
        "calib_source": calib_source,
        "test_setting": test_setting,
        "run_name":     run_name,
        "n":            len(df),
        "mae_bearing_deg":  float(df["err_bearing_deg"].mean()),
        "mae_range_m":      float(df["err_range_m"].mean()),
        "mae_euclidean_m":  float(df["err_euclidean_m"].mean()),
        "mae_sin":          float(df["err_sin"].mean()),
        "mae_cos":          float(df["err_cos"].mean()),
    }
    for tag in ALPHA_TAGS:
        for st in SCORE_TYPES:
            summary[f"cov_{st}_q{tag}"]     = float(df[f"cov_{st}_q{tag}"].mean())
        summary[f"mean_q_sin_{tag}"]        = float(df[f"q_sin_{tag}"].iloc[0])
        summary[f"mean_q_cos_{tag}"]        = float(df[f"q_cos_{tag}"].iloc[0])
        summary[f"mean_q_range_{tag}"]      = float(df[f"q_range_{tag}"].iloc[0])
        summary[f"mean_q_bearing_{tag}"]    = float(df[f"q_bearing_{tag}"].iloc[0])
        summary[f"mean_q_euclidean_{tag}"]  = float(df[f"q_euclidean_{tag}"].iloc[0])
        summary[f"eff_width_sin_q{tag}"]    = float(df[f"eff_width_sin_q{tag}"].mean())
        summary[f"eff_width_cos_q{tag}"]    = float(df[f"eff_width_cos_q{tag}"].mean())

    return df, pd.DataFrame([summary])
