"""
losses.py  (from evidential_loss_review/evidential_nig_loss_aligned.py, unchanged code)

1. Reference evidential regression (ERN) port, line-for-line from
       https://github.com/aamini/evidential-deep-learning
       https://github.com/hxu296/torch-evidental-deep-learning
   loss = NIG NLL + coeff * |y - mu| * (2v + alpha)

2. EvidentialNIGLossExternalPrior -- the loss used in the paper:
       loss = NLL + c * |y - mu| * (2v + alpha) + eta * |y - mu| * KL(NIG(mu, v, alpha, beta) || NIG_0)
   - NLL and the evidence regularizer are exactly the reference ERN terms (error not detached).
   - KL against a FIXED external prior NIG_0 = NIG(mu0=0, v0=1, alpha0=1.5, beta0=1), exact form:
     v-terms log(v/v0) + v0/v, mu-term scaled by E[1/sigma^2] = alpha/beta.
   - The KL is multiplied by |y - mu|, like the reference's reg = error * kl.
   - Activation: v = softplus + eps, alpha = softplus + 1 + eps, beta = softplus + eps (eps = 1e-6).
   With eta = 0 this is the reference ERN (plus eps in the activation).
   c is `coeff`, eta is `kl_weight`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════
# 1. REFERENCE PORT -- faithful to both aamini/evidential-deep-learning and
#    hxu296/torch-evidental-deep-learning (identical in both repos)
# ════════════════════════════════════════════════════════════════════════

def reference_activate(raw: torch.Tensor):
    """
    DenseNormalGamma.forward() in both repos, verbatim:
        v     = softplus(logv)
        alpha = softplus(logalpha) + 1
        beta  = softplus(logbeta)
    No epsilon anywhere -- neither repo adds one here.
    """
    mu, logv, logalpha, logbeta = raw.unbind(dim=-1)
    v = F.softplus(logv)
    alpha = F.softplus(logalpha) + 1.0
    beta = F.softplus(logbeta)
    return mu, v, alpha, beta


def reference_nig_nll(y: torch.Tensor, gamma: torch.Tensor, v: torch.Tensor,
                       alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """NIG_NLL() in both repos, verbatim (identical to our existing nig_nll)."""
    two_blambda = 2 * beta * (1 + v)
    nll = (
        0.5 * torch.log(torch.pi / v)
        - alpha * torch.log(two_blambda)
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll


def reference_kl_nig(mu1, v1, a1, b1, mu2, v2, a2, b2) -> torch.Tensor:
    """
    KL_NIG() in both repos, verbatim. General KL(NIG(mu1,v1,a1,b1) ||
    NIG(mu2,v2,a2,b2)) for ANY two NIG parameter sets (not specific to the
    self-referential call below -- that's just how NIG_Reg happens to call
    it).

    NOTE this uses (a1-1)/b1 for the mu-term's precision-scaling factor --
    i.e. 1/mean(InvGamma(a1,b1)), NOT the exact E[1/sigma^2] = a1/b1 (which
    is what 1/sigma^2 ~ Gamma(a1,b1) actually averages to, verified via
    direct Monte Carlo simulation of the true KL divergence in conversation
    -- see EvidentialNIGLossExternalPrior.kl_nig below, which uses a1/b1).
    This repo's choice is an approximation, like the version in our
    evidential_loss.py (prior to the term1 fix) was -- just using the
    CURRENT (a1,b1) instead of the PRIOR's (a0,b0) for that approximation.
    The v-ratio and log terms here (log(v1/v2) + v2/v1) ARE exact -- they
    match our Monte-Carlo-verified fix precisely.
    """
    KL = (
        0.5 * (a1 - 1) / b1 * (v2 * torch.square(mu2 - mu1))
        + 0.5 * v2 / v1
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1))
        - 0.5
        + a2 * torch.log(b1 / b2)
        - (torch.lgamma(a1) - torch.lgamma(a2))
        + (a1 - a2) * torch.digamma(a1)
        - (b1 - b2) * a1 / b1
    )
    return KL


def reference_nig_reg(y, gamma, v, alpha, beta, omega: float = 0.01,
                       kl: bool = False) -> torch.Tensor:
    """
    NIG_Reg() in both repos, verbatim. Two MUTUALLY EXCLUSIVE regularizer
    modes selected by the `kl` flag -- never both at once:
      kl=False (the default used by EvidentialRegression): the simple
        evidence regularizer  error * (2v + alpha),  error = |y-gamma|
        NOT detached (neither repo stops the gradient here -- our existing
        evidential_loss.py DOES call .detach() on this, which the
        reference code does not).
      kl=True: a SELF-REFERENTIAL KL -- the model's own (gamma,v,alpha,beta)
        against a "minimal evidence" version of ITSELF: same gamma, same
        beta, but v replaced by a small fixed omega and alpha replaced by
        1+omega. This is NOT a KL against an external, informative prior
        with its own mu0/beta0 -- both repos never construct one. The
        resulting kl is then MULTIPLIED by error (not added on its own):
        reg = error * kl.
    """
    error = torch.abs(y - gamma)  # NOT detached, matching both repos

    if kl:
        kl_val = reference_kl_nig(gamma, v, alpha, beta, gamma, torch.as_tensor(omega, dtype=v.dtype), torch.as_tensor(1.0 + omega, dtype=alpha.dtype), beta)
        reg = error * kl_val
    else:
        evidence = 2 * v + alpha
        reg = error * evidence

    return reg


def evidential_regression_reference(y_true: torch.Tensor, raw_output: torch.Tensor,
                                      coeff: float = 1.0, use_kl_reg: bool = False,
                                      omega: float = 0.01) -> torch.Tensor:
    """
    EvidentialRegression() in both repos, verbatim structure:
        loss = NLL + coeff * Reg
    where Reg is EITHER the simple evidence regularizer OR the
    self-referential-KL regularizer (never both, never a third additive
    KL-to-external-prior term -- that structure does not exist in either
    reference repo).
    """
    gamma, v, alpha, beta = reference_activate(raw_output)
    y_true = y_true.view_as(gamma)

    loss_nll = reference_nig_nll(y_true, gamma, v, alpha, beta).mean()
    loss_reg = reference_nig_reg(y_true, gamma, v, alpha, beta, omega=omega, kl=use_kl_reg).mean()
    return loss_nll + coeff * loss_reg


# ════════════════════════════════════════════════════════════════════════
# 2. External-prior variant (used in the paper) -- external fixed prior,
#    additive third term. Not part of either reference repo.
# ════════════════════════════════════════════════════════════════════════

class EvidentialNIGLossExternalPrior(nn.Module):
    """
    NLL + coeff*evidence_reg(error) + kl_weight*(error*KL),
    where KL is against an EXTERNAL fixed prior NIG(mu0,v0,alpha0,beta0),
    all three ADDED together. Uses the exact a/b mu-term scaling
    (Monte-Carlo-verified), not the reference's (a-1)/b approximation.

    `error` (= |y-mu|) is NOT detached -- matches both reference repos,
    which never stop the gradient here either. Gradients from both the
    evidence regularizer and the KL term flow into `mu` through `error`,
    in addition to whatever the NLL term already contributes.

    The KL term is multiplied by `error` before being weighted by
    kl_weight -- aligned with the reference repos' own `reg = error * kl`
    pattern in NIG_Reg(kl=True), even though we still keep this as a THIRD
    additive term (coeff*reg + kl_weight*(error*kl)) rather than the
    reference's design of using error*kl as a straight REPLACEMENT for the
    simple evidence regularizer.
    """
    def __init__(self, coeff: float = 0.1, kl_weight: float = 0.1,
                 prior_mu: float = 0.0, prior_v: float = 1.0,
                 prior_alpha: float = 1.5, prior_beta: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.coeff = coeff
        self.kl_weight = kl_weight
        self.mu0, self.v0, self.alpha0, self.beta0 = prior_mu, prior_v, prior_alpha, prior_beta
        self.eps = eps

    def activate(self, raw: torch.Tensor):
        mu, v_raw, alpha_raw, beta_raw = raw.unbind(dim=-1)
        v = F.softplus(v_raw) + self.eps
        alpha = F.softplus(alpha_raw) + 1.0 + self.eps
        beta = F.softplus(beta_raw) + self.eps
        return mu, v, alpha, beta

    def kl_nig(self, mu, v, alpha, beta):
        mu0 = torch.as_tensor(self.mu0, device=mu.device, dtype=mu.dtype)
        v0 = torch.as_tensor(self.v0, device=v.device, dtype=v.dtype)
        alpha0 = torch.as_tensor(self.alpha0, device=alpha.device, dtype=alpha.dtype)
        beta0 = torch.as_tensor(self.beta0, device=beta.device, dtype=beta.dtype)

        term1 = 0.5 * (
            torch.log(v / v0) + v0 / v
            + v0 * (mu - mu0) ** 2 * (alpha / beta)  # exact E[1/sigma^2], NOT (alpha-1)/beta
            - 1.0
        )
        term2 = (
            alpha0 * torch.log(beta / beta0)
            - torch.lgamma(alpha) + torch.lgamma(alpha0)
            + (alpha - alpha0) * torch.digamma(alpha)
            - (beta - beta0) * (alpha / beta)
        )
        return term1 + term2

    def forward(self, y_pred_raw: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mu, v, alpha, beta = self.activate(y_pred_raw)
        y_true = y_true.view_as(mu)

        nll = reference_nig_nll(y_true, mu, v, alpha, beta)
        error = torch.abs(y_true - mu)  # NOT detached -- now matches reference exactly
        reg = error * (2.0 * v + alpha)
        kl = self.kl_nig(mu, v, alpha, beta)
        kl_reg = error * kl  # multiply by error, matching reference's reg = error * kl pattern

        return nll.mean() + self.coeff * reg.mean() + self.kl_weight * kl_reg.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    raw = torch.randn(8, 4)
    y = torch.rand(8)

    ref_loss = evidential_regression_reference(y, raw, coeff=0.1, use_kl_reg=False)
    ref_loss_kl = evidential_regression_reference(y, raw, coeff=0.1, use_kl_reg=True)
    print("reference loss (simple evidence reg):", ref_loss.item())
    print("reference loss (self-referential KL reg):", ref_loss_kl.item())

    ours = EvidentialNIGLossExternalPrior(coeff=0.1, kl_weight=0.1)
    print("our external-prior variant:", ours(raw, y).item())
