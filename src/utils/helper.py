import json
import torch

@torch.no_grad()
def collect_hits_simple(
    dl, model, device,
    threshold=0.65, norm=None, out_path=None, round_to=4,
    head_idx=None, label_names=None,    # NEW: optional selector for multi-label
):
    """
    Collect items with predicted prob >= threshold for a single selected head.
    Works with single-label ([B]) and multi-label ([B,K]) outputs.
    Returns list[{"prob": float, "label": 0|1}] sorted by prob desc.
    """
    model.eval()
    hits = []
    # choose default head if not provided
    default_idx = 0
    if head_idx is None and label_names:
        if "2_2" in label_names:
            default_idx = label_names.index("2_2")
    if head_idx is None:
        head_idx = default_idx

    for b in dl:
        x_seq    = b["x_seq"].to(device)
        x_static = b["x_static"].to(device)
        mask     = b["mask"].to(device)
        y_full   = b["y"]  # [B] or [B,K]; keep on CPU until we pick head

        if norm is not None:
            mean_seq, std_seq, mean_sta, std_sta = norm
            x_seq    = (x_seq - mean_seq) / std_seq
            x_static = (x_static - mean_sta) / std_sta

        # logits -> probs
        probs = torch.sigmoid(model(x_seq, x_static, mask)).cpu()  # [B] or [B,K]

        # If multi-label, select the requested head
        if probs.ndim == 2:
            probs = probs[:, head_idx]
            y     = y_full[:, head_idx].cpu()
        else:
            y     = y_full.cpu()

        for p, yy in zip(probs.tolist(), y.tolist()):
            if p >= threshold:
                hits.append({"prob": round(p, round_to), "label": int(yy)})

    hits.sort(key=lambda r: r["prob"], reverse=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(hits, f, indent=2)

    print(f"Collected {len(hits)} items with prob ≥ {int(threshold*100)}%")
    print("Top 100:", hits[:100])  # fixed label to match slice
    return hits
