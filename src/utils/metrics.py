import torch

@torch.no_grad()
def binary_metrics(probs, targets, thresh=0.5):
    preds = (probs >= thresh).to(dtype=torch.long)
    t = targets.long()
    acc = (preds == t).float().mean().item()
    tp = ((preds == 1) & (t == 1)).sum().item()
    fp = ((preds == 1) & (t == 0)).sum().item()
    fn = ((preds == 0) & (t == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"acc": acc, "precision": precision, "recall": recall}

@torch.no_grad()
def roc_auc_score_fast(probs: torch.Tensor, y: torch.Tensor) -> float:
    """Binary ROC-AUC with torch only."""
    # sort by score descending
    s, idx = torch.sort(probs.view(-1), descending=True)
    t = y.view(-1).float()[idx]
    # cum positives/negatives
    P = t.sum().item()
    N = (len(t) - P)
    if P == 0 or N == 0:
        return float("nan")
    # true positive rate and false positive rate at each threshold
    tp = torch.cumsum(t, dim=0)
    fp = torch.arange(1, len(t)+1, device=t.device) - tp
    tpr = tp / P
    fpr = fp / N
    # trapz integrate over FPR
    # need to prepend (0,0)
    fpr = torch.cat([torch.tensor([0.0], device=fpr.device), fpr])
    tpr = torch.cat([torch.tensor([0.0], device=tpr.device), tpr])
    return torch.trapz(tpr, fpr).item()

@torch.no_grad()
def pr_auc_score_fast(probs: torch.Tensor, y: torch.Tensor) -> float:
    """Binary PR-AUC (area under precision-recall)."""
    s, idx = torch.sort(probs.view(-1), descending=True)
    t = y.view(-1).float()[idx]
    P = t.sum().item()
    if P == 0:
        return float("nan")
    tp = torch.cumsum(t, dim=0)
    fp = torch.arange(1, len(t)+1, device=t.device) - tp
    precision = tp / torch.clamp(tp + fp, min=1)
    recall    = tp / P
    # prepend (recall=0, precision=1) for nicer curve
    recall = torch.cat([torch.tensor([0.0], device=recall.device), recall])
    precision = torch.cat([torch.tensor([1.0], device=precision.device), precision])
    return torch.trapz(precision, recall).item()

@torch.no_grad()
def mcc_at_threshold(probs: torch.Tensor, y: torch.Tensor, thresh: float) -> float:
    p = (probs >= thresh).long()
    t = y.long()
    tp = ((p==1)&(t==1)).sum().item()
    tn = ((p==0)&(t==0)).sum().item()
    fp = ((p==1)&(t==0)).sum().item()
    fn = ((p==0)&(t==1)).sum().item()
    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if denom == 0:
        return 0.0
    return (tp*tn - fp*fn) / (denom ** 0.5)