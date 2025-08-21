from __future__ import annotations
import math, torch
from typing import Dict
from . import register
from src.utils.metrics import binary_metrics

def _run_epoch(model, loader, device, optimizer=None, pos_weight=None, norm=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None \
              else torch.nn.BCEWithLogitsLoss()
    total_loss, total_n = 0.0, 0
    all_probs, all_targets = [], []
    for batch in loader:
        x_seq    = batch["x_seq"].to(device)
        x_static = batch["x_static"].to(device)
        mask     = batch["mask"].to(device)
        y        = batch["y"].to(device)
        if norm is not None:
            mean_seq, std_seq, mean_sta, std_sta = norm
            x_seq    = (x_seq - mean_seq) / std_seq
            x_static = (x_static - mean_sta) / std_sta
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_seq, x_static, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x_seq, x_static, mask)
                loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        total_n    += y.size(0)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach().cpu()); all_targets.append(y.detach().cpu())
    mean_loss = total_loss / max(1, total_n)
    probs   = torch.cat(all_probs) if all_probs else torch.zeros(0)
    targets = torch.cat(all_targets) if all_targets else torch.zeros(0)
    mets = binary_metrics(probs, targets, thresh=0.5)
    return mean_loss, mets

@register("standard")
def fit(
    model, train_dl, val_dl, device, cfg, norm, pos_weight, optimizer,
    ckpt_best: str | None = None,
) -> Dict[str, float]:
    import os
    from pathlib import Path

    def _atomic_save(state, path: str):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, str(p))

    best_val, bad = math.inf, 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_mets = _run_epoch(model, train_dl, device, optimizer=optimizer,
                                      pos_weight=pos_weight, norm=norm)
        va_loss, va_mets = _run_epoch(model, val_dl,   device, optimizer=None,
                                      pos_weight=pos_weight, norm=norm)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_mets['acc']:.3f} | "
            f"val  loss {va_loss:.4f} acc {va_mets['acc']:.3f} "
            f"(prec {va_mets['precision']:.3f}, rec {va_mets['recall']:.3f})"
        )

        if va_loss < best_val - 1e-4:
            best_val, bad = va_loss, 0
            if ckpt_best:
                _atomic_save(model.state_dict(), ckpt_best)
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping."); break

    return {"best_val_loss": best_val}

