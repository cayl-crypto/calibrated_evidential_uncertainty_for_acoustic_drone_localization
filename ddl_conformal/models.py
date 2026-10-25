"""
models.py

Conformer8Mic encoder (conformer_model.py), the evidential model trained with the external-prior
loss (evidential_model_external_prior.py) and the non-evidential Base Model (localization_model.py).
Layer definitions are unchanged, so the released state_dicts load into these classes directly.

Checkpoints are saved as {"model_class", "config", "state_dict"} (see save_checkpoint /
load_checkpoint) instead of pickling the whole module.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .losses import EvidentialNIGLossExternalPrior


class Conformer8Mic(nn.Module):
    """(B, 8, 128, T) log-mel -> (B, 192) embedding: 2-layer CNN, linear projection, 6-layer Conformer, mean pool."""

    def __init__(self, in_channels=8, input_freq_bins=128, input_time_frames=28, cnn_channels=128,
                 conformer_dim=192, num_heads=6, ff_mult=4, num_layers=6, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
        )
        self.proj = nn.Linear(cnn_channels * input_freq_bins, conformer_dim)
        self.encoder = torchaudio.models.Conformer(
            input_dim=conformer_dim, num_heads=num_heads, ffn_dim=conformer_dim * ff_mult,
            num_layers=num_layers, dropout=dropout, depthwise_conv_kernel_size=31,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed_dim = conformer_dim

    def forward(self, x):
        x = self.cnn(x)                              # (B, C, F, T)
        B, C, F_, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F_)
        x = self.proj(x)                             # (B, T, D)
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        x, _ = self.encoder(x, lengths)
        return self.pool(x.transpose(1, 2)).squeeze(-1)   # (B, D)


def _metrics_from_means(sin_mu, cos_mu, range_mu, y_sin, y_cos, y_range, max_range: float = 250.0) -> dict:
    pred_bearing = torch.atan2(sin_mu, cos_mu)
    true_bearing = torch.atan2(y_sin, y_cos)
    d = (pred_bearing - true_bearing + math.pi) % (2 * math.pi) - math.pi
    r_hat_m, r_true_m = range_mu * max_range, y_range * max_range
    px, py = r_hat_m * cos_mu, r_hat_m * sin_mu
    tx, ty = r_true_m * y_cos, r_true_m * y_sin
    return {
        "bearing_mae_deg": float(torch.rad2deg(d.abs()).mean().item()),
        "range_mae_m": float(torch.mean(torch.abs(r_hat_m - r_true_m)).item()),
        "euclidean_mae_m": float(torch.norm(torch.stack([px - tx, py - ty], dim=-1), dim=-1).mean().item()),
    }


class EvidentialLocalizationExternalPrior(nn.Module):
    """Three NIG heads (sin, cos, range), each (mu, v, alpha, beta).

    Only mu is bounded in the model (tanh for sin/cos, sigmoid for range); v/alpha/beta stay raw and
    EvidentialNIGLossExternalPrior.activate() applies softplus, both in the loss (training) and in
    forward_all() (inference), so both use the same activation.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, coeff: float = 0.1, kl_weight: float = 0.1,
                 prior_mu: float = 0.0, prior_v: float = 1.0, prior_alpha: float = 1.5, prior_beta: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.encoder = encoder
        self.eps = eps
        self.sin_head = nn.Linear(embed_dim, 4)
        self.cos_head = nn.Linear(embed_dim, 4)
        self.range_head = nn.Linear(embed_dim, 4)
        self.sin_loss = EvidentialNIGLossExternalPrior(coeff, kl_weight, prior_mu, prior_v, prior_alpha, prior_beta, eps)
        self.cos_loss = EvidentialNIGLossExternalPrior(coeff, kl_weight, prior_mu, prior_v, prior_alpha, prior_beta, eps)
        self.range_loss = EvidentialNIGLossExternalPrior(coeff, kl_weight, prior_mu, prior_v, prior_alpha, prior_beta, eps)

    def _raw_pack(self, four_vec: torch.Tensor, is_range: bool) -> torch.Tensor:
        mu_raw, v_raw, alpha_raw, beta_raw = four_vec.chunk(4, dim=-1)
        mu = torch.sigmoid(mu_raw) if is_range else torch.tanh(mu_raw)
        return torch.cat([mu, v_raw, alpha_raw, beta_raw], dim=-1)

    def forward_params_raw(self, x: torch.Tensor):
        z = self.encoder(x)
        return (self._raw_pack(self.sin_head(z), False), self._raw_pack(self.cos_head(z), False),
                self._raw_pack(self.range_head(z), True))

    def compute_loss(self, raw_packs, targets):
        sin_pack, cos_pack, rng_pack = raw_packs
        y_sin, y_cos, y_rng = targets
        return self.sin_loss(sin_pack, y_sin) + self.cos_loss(cos_pack, y_cos) + self.range_loss(rng_pack, y_rng)

    def forward_all(self, x: torch.Tensor):
        """means (sin, cos, range) and activated (B, 4) packs [mu, v, alpha, beta] per head."""
        sin_raw, cos_raw, rng_raw = self.forward_params_raw(x)
        sin_mu, sin_v, sin_a, sin_b = self.sin_loss.activate(sin_raw)
        cos_mu, cos_v, cos_a, cos_b = self.cos_loss.activate(cos_raw)
        rng_mu, rng_v, rng_a, rng_b = self.range_loss.activate(rng_raw)
        means = (sin_mu, cos_mu, rng_mu)
        packs = (torch.stack([sin_mu, sin_v, sin_a, sin_b], dim=-1),
                 torch.stack([cos_mu, cos_v, cos_a, cos_b], dim=-1),
                 torch.stack([rng_mu, rng_v, rng_a, rng_b], dim=-1))
        return means, packs

    def forward(self, x: torch.Tensor):
        return self.forward_all(x)[0]

    compute_metrics_from_means = staticmethod(_metrics_from_means)


class PlainLocalization(nn.Module):
    """Base Model: one linear head -> (tanh sin, tanh cos, sigmoid range), trained with MSE."""

    def __init__(self, encoder: nn.Module, embed_dim: int,
                 loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0), eps: float = 1e-8):
        super().__init__()
        self.encoder = encoder
        self.eps = eps
        self.head = nn.Linear(embed_dim, 3)
        self.w_sin, self.w_cos, self.w_range = loss_weights

    def forward(self, x: torch.Tensor):
        out = self.head(self.encoder(x))
        return torch.tanh(out[:, 0]), torch.tanh(out[:, 1]), torch.sigmoid(out[:, 2])

    def compute_loss(self, preds, targets) -> torch.Tensor:
        sin_p, cos_p, range_p = preds
        y_sin, y_cos, y_range = (t.to(p.dtype).to(p.device) for t, p in zip(targets, preds))
        return (self.w_sin * F.mse_loss(sin_p, y_sin) + self.w_cos * F.mse_loss(cos_p, y_cos)
                + self.w_range * F.mse_loss(range_p, y_range))

    compute_metrics_from_means = staticmethod(_metrics_from_means)


# ── checkpoints ──────────────────────────────────────────────────────────────

def build_evidential(coeff: float, kl_weight: float) -> EvidentialLocalizationExternalPrior:
    encoder = Conformer8Mic()
    return EvidentialLocalizationExternalPrior(encoder, encoder.embed_dim, coeff=coeff, kl_weight=kl_weight)


def build_base() -> PlainLocalization:
    encoder = Conformer8Mic()
    return PlainLocalization(encoder, encoder.embed_dim)


def save_checkpoint(model: nn.Module, path, **config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_class": type(model).__name__, "config": config, "state_dict": model.state_dict()}, path)


def load_checkpoint(path, map_location="cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt["model_class"] == "EvidentialLocalizationExternalPrior":
        model = build_evidential(ckpt["config"]["coeff"], ckpt["config"]["kl_weight"])
    elif ckpt["model_class"] == "PlainLocalization":
        model = build_base()
    else:
        raise ValueError(f"Unknown model_class {ckpt['model_class']!r}")
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
