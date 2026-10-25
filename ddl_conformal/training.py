"""
training.py

Training loops (from train_evidential_localization_external_prior_ablation.py and
train_localization.py, same loop): AdamW (lr 5e-5), cosine schedule (T_max 20), batch 64,
mixed precision on GPU, gradient clipping at 1.0, early stopping with patience 3 on the
validation evaluation split.
  evidential  stops on total validation NLL (sin + cos + range Student-t NLL)
  Base Model  stops on validation loss (sum of the three MSEs)
The best epoch is saved as {run_dir}/model.pt (state_dict + config).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from .data import get_localization_loaders
from .evaluation import (evaluate_evidential_localization_fn, evaluate_plain_localization_fn,
                         export_evidential_results_to_csv, export_plain_results_to_csv)
from .models import build_base, build_evidential, save_checkpoint


def _train_epoch(model, train_loader, optimizer, scaler, device, loss_fn):
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    model.train()
    total_loss, total_n = 0.0, 0
    for x, bearing_vec, r_t in tqdm(train_loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True).float()
        r_t = r_t.to(device, non_blocking=True).float()
        sin_t = bearing_vec[:, 1].to(device, non_blocking=True).float()
        cos_t = bearing_vec[:, 0].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(amp_device, enabled=(device.type == "cuda")):
            loss = loss_fn(x, (sin_t, cos_t, r_t))

        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    return total_loss / max(total_n, 1)


def train_evidential(data_root, run_dir, device, coeff: float, kl_weight: float, batch_size: int = 64,
                     learning_rate: float = 5e-5, patience: int = 3, num_workers: int = 4,
                     pin_memory: bool = True) -> Dict:
    run_dir = Path(run_dir)
    val_epoch_dir = run_dir / "val_epoch_csv"
    val_epoch_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\nEvidential localization (external prior)  coeff={coeff}  kl_weight={kl_weight}\n"
          f"  run_dir: {run_dir}\n{'=' * 70}")

    train_loader, val_evaluation_loader, _, _ = get_localization_loaders(
        data_root, "plain", batch_size, num_workers=num_workers, pin_memory=pin_memory)
    model = build_evidential(coeff, kl_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))

    def loss_fn(x, targets):
        return model.compute_loss(model.forward_params_raw(x), targets)

    best_total_nll, nll_tol, epochs_no_improve, epoch = float("inf"), 1e-4, 0, 0
    while epochs_no_improve < patience:
        epoch += 1
        train_loss = _train_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        scheduler.step()
        results = evaluate_evidential_localization_fn(model, val_evaluation_loader, device=device)
        total_nll = results["nll"]["total"]
        export_evidential_results_to_csv(results, str(val_epoch_dir / f"val_results_epoch_{epoch}.csv"))
        print(f"Epoch {epoch}: train loss {train_loss:.6f}  val NLL {total_nll:.6f}")
        if total_nll < best_total_nll - nll_tol:
            best_total_nll, epochs_no_improve = total_nll, 0
            save_checkpoint(model, run_dir / "model.pt", coeff=coeff, kl_weight=kl_weight)
        else:
            epochs_no_improve += 1

    return dict(feature_source="plain", feature_combo="logmel", batch_size=batch_size, learning_rate=learning_rate,
                patience=patience, evidential_coeff=coeff, evidential_kl_weight=kl_weight,
                best_total_nll=best_total_nll, epochs=epoch, run_dir=str(run_dir))


def train_base(data_root, run_dir, device, batch_size: int = 64, learning_rate: float = 5e-5, patience: int = 3,
               num_workers: int = 4, pin_memory: bool = True) -> Dict:
    run_dir = Path(run_dir)
    val_epoch_dir = run_dir / "val_epoch_csv"
    val_epoch_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\nBase Model (MSE)\n  run_dir: {run_dir}\n{'=' * 70}")

    train_loader, val_evaluation_loader, _, _ = get_localization_loaders(
        data_root, "plain", batch_size, num_workers=num_workers, pin_memory=pin_memory)
    model = build_base().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))

    def loss_fn(x, targets):
        return model.compute_loss(model(x), targets)

    best_score, tol, epochs_no_improve, epoch = float("inf"), 1e-6, 0, 0
    while epochs_no_improve < patience:
        epoch += 1
        train_loss = _train_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        scheduler.step()
        val = evaluate_plain_localization_fn(model, val_evaluation_loader, device=device)
        export_plain_results_to_csv(val, str(val_epoch_dir / f"val_results_epoch_{epoch}.csv"))
        score = val["loss"]["total"]
        print(f"Epoch {epoch}: train loss {train_loss:.6f}  val loss {score:.6f}  "
              f"val Euclidean {val['mae']['euclidean_m']:.3f} m")
        if score < best_score - tol:
            best_score, epochs_no_improve = score, 0
            save_checkpoint(model, run_dir / "model.pt")
        else:
            epochs_no_improve += 1

    return dict(feature_source="plain", feature_combo="logmel", batch_size=batch_size, learning_rate=learning_rate,
                patience=patience, selection_metric="loss", best_score=best_score, epochs=epoch, run_dir=str(run_dir))
