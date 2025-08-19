# src/utils/thresholds.py
import torch
from src.utils.metrics import binary_metrics, mcc_at_threshold

@torch.no_grad()
def sweep_thresholds(probs: torch.Tensor, y: torch.Tensor, metric: str = "f1"):
    """Return best threshold in [0.05..0.95] and its score."""
    probs = probs.view(-1).cpu()
    y = y.view(-1).cpu()
    candidates = torch.linspace(0.05, 0.95, steps=19)
    best_thr, best_score = 0.5, -1.0
    for thr in candidates.tolist():
        if metric.lower() == "mcc":
            score = mcc_at_threshold(probs, y, thr)
        else:
            mets = binary_metrics((probs>=thr).float(), y.float(), thresh=0.5)
            score = mets["f1"]
        if score > best_score:
            best_score, best_thr = score, thr
    return float(best_thr), float(best_score)
