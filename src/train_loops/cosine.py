# src/train_loops/cosine.py
from __future__ import annotations
import math, torch
from typing import Dict
from . import register
from .standard import _run_epoch  # reuse

def _build_scheduler(optimizer, total_epochs: int, warmup: int = 5):
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, total_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

@register("cosine")
def fit(model, train_dl, val_dl, device, cfg, norm, pos_weight, optimizer) -> Dict[str, float]:
    scheduler = _build_scheduler(optimizer, total_epochs=cfg.epochs, warmup=5)
    best_val, bad = math.inf, 0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_mets = _run_epoch(model, train_dl, device, optimizer=optimizer, pos_weight=pos_weight, norm=norm)
        va_loss, va_mets = _run_epoch(model, val_dl,   device, optimizer=None,      pos_weight=pos_weight, norm=norm)
        scheduler.step()
        print(
            f"[cosine] Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_mets['acc']:.3f} | "
            f"val  loss {va_loss:.4f} acc {va_mets['acc']:.3f}"
        )
        if va_loss < best_val - 1e-4:
            best_val, bad = va_loss, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping."); break
    return {"best_val_loss": best_val}
