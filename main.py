from pathlib import Path
import math
import torch
from torch.utils.data import random_split
from src.data.datasets import MinSnipDataset, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH
from src.models.snip_lstm01 import SnipLSTM
from src.utils.helper import collect_hits_simple




# -----------------------
# Device (M1-friendly)
# -----------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():  # no args
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def binary_metrics(probs, targets, thresh=0.5):
    preds = (probs >= thresh).to(dtype=torch.long)
    t = targets.long()
    correct = (preds == t).sum().item()
    acc = correct / len(t)
    # simple precision/recall (safe against /0)
    tp = ((preds == 1) & (t == 1)).sum().item()
    fp = ((preds == 1) & (t == 0)).sum().item()
    fn = ((preds == 0) & (t == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"acc": acc, "precision": precision, "recall": recall}

@torch.no_grad()
def frac_positive(loader):
    pos, tot = 0, 0
    for b in loader:
        y = b["y"]
        pos += int((y == 1).sum())
        tot += int(y.numel())
    return pos / max(1, tot)

@torch.no_grad()
def compute_stats_seq_static(loader):
    """
    Compute z-score stats over the TRAIN loader.
    Returns mean/std for seq channels [C_seq] and static features [C_static].
    """
    n_seq = 0
    sum_seq = sumsq_seq = None
    n_sta = 0
    sum_sta = sumsq_sta = None

    for b in loader:
        xs = b["x_seq"]       # [B,T,C_seq]
        xm = b["mask"]        # [B,T]
        st = b["x_static"]    # [B,C_static]

        # keep only valid timesteps for seq stats
        valid = xm.bool().unsqueeze(-1)          # [B,T,1]
        vals = xs[valid.expand_as(xs)]           # [N_valid*C_seq]
        vals = vals.view(-1, xs.shape[-1])       # [N_valid, C_seq]

        if sum_seq is None:
            sum_seq   = vals.sum(dim=0)
            sumsq_seq = (vals**2).sum(dim=0)
        else:
            sum_seq   += vals.sum(dim=0)
            sumsq_seq += (vals**2).sum(dim=0)
        n_seq += vals.shape[0]

        if sum_sta is None:
            sum_sta   = st.sum(dim=0)
            sumsq_sta = (st**2).sum(dim=0)
        else:
            sum_sta   += st.sum(dim=0)
            sumsq_sta += (st**2).sum(dim=0)
        n_sta += st.shape[0]

    mean_seq = sum_seq / max(1, n_seq)
    var_seq  = sumsq_seq / max(1, n_seq) - mean_seq**2
    std_seq  = torch.sqrt(torch.clamp(var_seq, min=1e-8))

    mean_sta = sum_sta / max(1, n_sta)
    var_sta  = sumsq_sta / max(1, n_sta) - mean_sta**2
    std_sta  = torch.sqrt(torch.clamp(var_sta, min=1e-8))

    return mean_seq, std_seq, mean_sta, std_sta



# -----------------------
# Train / Eval loops
# -----------------------
def run_epoch(model, loader, device, optimizer=None, pos_weight=None, norm=None):
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
        all_probs.append(probs.detach().cpu())
        all_targets.append(y.detach().cpu())

    mean_loss = total_loss / max(1, total_n)
    probs   = torch.cat(all_probs) if all_probs else torch.zeros(0)
    targets = torch.cat(all_targets) if all_targets else torch.zeros(0)
    mets = binary_metrics(probs, targets, thresh=0.5)
    return mean_loss, mets
    
def estimate_pos_weight(loader, device):
    """pos_weight = (#neg / #pos) for BCEWithLogitsLoss if classes are imbalanced."""
    pos = 0
    neg = 0
    for b in loader:
        y = b["y"]
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    if pos == 0:
        return None
    ratio = neg / pos
    return torch.tensor([ratio], device=device, dtype=torch.float32)


# -----------------------
# Quick train script
# -----------------------
def main():
    torch.manual_seed(42)

    train_path = DEFAULT_TRAIN_PATH
    val_path   = DEFAULT_TEST_PATH
    assert Path(train_path).exists(), f"Missing {train_path}"
    assert Path(val_path).exists(),   f"Missing {val_path}"

    # 1) Device FIRST
    device = get_device()
    print(f"Using device: {device}")

    # 2) Build loaders
    train_dl = MinSnipDataset.make_dataloader(
        train_path, batch_size=128, shuffle=True, num_workers=0, seed=42,
        label_keys=["2_2"]   # <-- or whichever label actually has positives
    )
    val_dl = MinSnipDataset.make_dataloader(
        val_path, batch_size=128, shuffle=False, num_workers=0,
        label_keys=["2_2"]
    )

    # 3) Build model
    model = SnipLSTM(
        seq_input_size=2,
        static_input_size=5,
        lstm_hidden=64,
        lstm_layers=1,
        bidirectional=False,
        mlp_hidden=64,
        dropout=0.1,
    ).to(device)

    # 4) (Optional) compute normalization stats on TRAIN and move them to device
    mean_seq, std_seq, mean_sta, std_sta = compute_stats_seq_static(train_dl)  # returns CPU tensors
    mean_seq, std_seq = mean_seq.to(device), std_seq.to(device)
    mean_sta, std_sta = mean_sta.to(device), std_sta.to(device)
    norm = (mean_seq, std_seq, mean_sta, std_sta)

    torch.save(
        {
            "mean_seq": mean_seq.detach().cpu(),
            "std_seq":  std_seq.detach().cpu(),
            "mean_sta": mean_sta.detach().cpu(),
            "std_sta":  std_sta.detach().cpu(),
        },
        "norm_stats_standard25.pt"
    )

    # 5) (Optional) init output bias to base rate to stabilize early training
    pi = frac_positive(train_dl)
    pi = min(max(pi, 1e-4), 1 - 1e-4)
    with torch.no_grad():
        model.head[-1].bias.fill_(math.log(pi/(1-pi)))

    # 6) Rest as before...
    pos_weight = estimate_pos_weight(train_dl, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val, patience, bad, epochs = math.inf, 30, 0, 400
    for epoch in range(1, epochs + 1):
        tr_loss, tr_mets = run_epoch(model, train_dl, device, optimizer=optimizer, pos_weight=pos_weight, norm=norm)
        va_loss, va_mets = run_epoch(model, val_dl,   device, optimizer=None,      pos_weight=pos_weight, norm=norm)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_mets['acc']:.3f} | "
            f"val loss {va_loss:.4f} acc {va_mets['acc']:.3f} "
            f"(prec {va_mets['precision']:.3f}, rec {va_mets['recall']:.3f})"
        )
        if va_loss < best_val - 1e-4:
            best_val, bad = va_loss, 0
            torch.save(model.state_dict(), "checkpoint_snip_standard25k.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Done. Best val loss:", best_val)
    print()

    # Quick prob preview (no thresholding here)
    model.eval()
    batch = next(iter(val_dl))
    with torch.no_grad():
        probs = model.predict_proba(
            batch["x_seq"].to(device),
            batch["x_static"].to(device),
            batch["mask"].to(device),
        ).cpu()
    print("Sample probs (val[0:8]):", probs[:8].tolist())

    hits = collect_hits_simple(
        val_dl, model, device,
        threshold=0.65,
        norm=norm if 'norm' in locals() else None,   # pass your (mean/std) tuple if you computed it
        out_path="val_hits_065_simple.json"          # or None to skip writing
    )
    print("Top 100 hits:", hits[:100])


if __name__ == "__main__":
    main()






# from pathlib import Path
# import math
# import torch
# from torch.utils.data import random_split
# from src.data.datasets import MinSnipDataset, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH
# from src.models.snip_lstm01 import SnipLSTM
# from src.utils.helper import collect_hits_simple

# LABELS = ["5_5"]   # K = 6

# # -----------------------
# # Device (M1-friendly)
# # -----------------------
# def get_device():
#     if torch.backends.mps.is_available():
#         return torch.device("mps")
#     elif torch.cuda.is_available():  # no args
#         return torch.device("cuda")
#     else:
#         return torch.device("cpu")


# # -----------------------
# # Metrics
# # -----------------------
# @torch.no_grad()
# def binary_metrics_multi(probs, targets, thresh=0.5, label_names=None):
#     """
#     probs, targets: [N, K]
#     returns {"macro": {...}, "per_label": {name: {...}}}
#     """
#     K = probs.shape[1]
#     preds = (probs >= thresh).long()
#     t = targets.long()
#     per = {}
#     accs = []; precs = []; recs = []

#     for k in range(K):
#         pk, tk = preds[:, k], t[:, k]
#         correct = (pk == tk).sum().item()
#         acc = correct / len(tk)
#         tp = ((pk == 1) & (tk == 1)).sum().item()
#         fp = ((pk == 1) & (tk == 0)).sum().item()
#         fn = ((pk == 0) & (tk == 1)).sum().item()
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         name = label_names[k] if label_names else str(k)
#         per[name] = {"acc": acc, "precision": precision, "recall": recall}
#         accs.append(acc); precs.append(precision); recs.append(recall)

#     macro = {"acc": sum(accs)/K, "precision": sum(precs)/K, "recall": sum(recs)/K}
#     return {"macro": macro, "per_label": per}

# @torch.no_grad()
# def frac_positive_multi(loader):
#     pos = None
#     tot = 0
#     for b in loader:
#         y = b["y"]                     # [B, K]
#         pos = y.sum(dim=0) if pos is None else pos + y.sum(dim=0)
#         tot += y.shape[0]
#     pi = pos / max(1, tot)             # [K]
#     return torch.clamp(pi, 1e-4, 1-1e-4)

# @torch.no_grad()
# def estimate_pos_weight_multi(loader, device):
#     pos = None; neg = None
#     for b in loader:
#         y = b["y"]                     # [B, K]
#         y0 = (y == 0).sum(dim=0)       # [K]
#         y1 = (y == 1).sum(dim=0)       # [K]
#         neg = y0 if neg is None else neg + y0
#         pos = y1 if pos is None else pos + y1
#     pos = pos.to(torch.float32).clamp_min(1.0)
#     neg = neg.to(torch.float32)
#     return (neg / pos).to(device)      # [K]


# @torch.no_grad()
# def compute_stats_seq_static(loader):
#     """
#     Compute z-score stats over the TRAIN loader.
#     Returns mean/std for seq channels [C_seq] and static features [C_static].
#     """
#     n_seq = 0
#     sum_seq = sumsq_seq = None
#     n_sta = 0
#     sum_sta = sumsq_sta = None

#     for b in loader:
#         xs = b["x_seq"]       # [B,T,C_seq]
#         xm = b["mask"]        # [B,T]
#         st = b["x_static"]    # [B,C_static]

#         # keep only valid timesteps for seq stats
#         valid = xm.bool().unsqueeze(-1)          # [B,T,1]
#         vals = xs[valid.expand_as(xs)]           # [N_valid*C_seq]
#         vals = vals.view(-1, xs.shape[-1])       # [N_valid, C_seq]

#         if sum_seq is None:
#             sum_seq   = vals.sum(dim=0)
#             sumsq_seq = (vals**2).sum(dim=0)
#         else:
#             sum_seq   += vals.sum(dim=0)
#             sumsq_seq += (vals**2).sum(dim=0)
#         n_seq += vals.shape[0]

#         if sum_sta is None:
#             sum_sta   = st.sum(dim=0)
#             sumsq_sta = (st**2).sum(dim=0)
#         else:
#             sum_sta   += st.sum(dim=0)
#             sumsq_sta += (st**2).sum(dim=0)
#         n_sta += st.shape[0]

#     mean_seq = sum_seq / max(1, n_seq)
#     var_seq  = sumsq_seq / max(1, n_seq) - mean_seq**2
#     std_seq  = torch.sqrt(torch.clamp(var_seq, min=1e-8))

#     mean_sta = sum_sta / max(1, n_sta)
#     var_sta  = sumsq_sta / max(1, n_sta) - mean_sta**2
#     std_sta  = torch.sqrt(torch.clamp(var_sta, min=1e-8))

#     return mean_seq, std_seq, mean_sta, std_sta



# # -----------------------
# # Train / Eval loops
# # -----------------------
# def run_epoch(model, loader, device, optimizer=None, pos_weight=None, norm=None, labels=None):
#     is_train = optimizer is not None
#     model.train(is_train)

#     loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None \
#               else torch.nn.BCEWithLogitsLoss()

#     total_loss, total_n = 0.0, 0
#     all_probs, all_targets = [], []

#     for batch in loader:
#         x_seq    = batch["x_seq"].to(device)
#         x_static = batch["x_static"].to(device)
#         mask     = batch["mask"].to(device)
#         y        = batch["y"].to(device)            # [B, K]

#         if norm is not None:
#             mean_seq, std_seq, mean_sta, std_sta = norm
#             x_seq    = (x_seq - mean_seq) / std_seq
#             x_static = (x_static - mean_sta) / std_sta

#         if is_train:
#             optimizer.zero_grad(set_to_none=True)
#             logits = model(x_seq, x_static, mask)   # [B, K]
#             loss = loss_fn(logits, y)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#         else:
#             with torch.no_grad():
#                 logits = model(x_seq, x_static, mask)
#                 loss = loss_fn(logits, y)

#         total_loss += loss.item() * y.size(0)
#         total_n    += y.size(0)
#         probs = torch.sigmoid(logits)
#         all_probs.append(probs.detach().cpu())
#         all_targets.append(y.detach().cpu())

#     mean_loss = total_loss / max(1, total_n)
#     probs   = torch.cat(all_probs) if all_probs else torch.zeros(0, len(labels or []))
#     targets = torch.cat(all_targets) if all_targets else torch.zeros(0, len(labels or []))
#     mets = binary_metrics_multi(probs, targets, thresh=0.5, label_names=labels)
#     return mean_loss, mets


# # -----------------------
# # Quick train script
# # -----------------------
# def main():
#     torch.manual_seed(42)

#     train_path = DEFAULT_TRAIN_PATH
#     val_path   = DEFAULT_TEST_PATH
#     assert Path(train_path).exists(), f"Missing {train_path}"
#     assert Path(val_path).exists(),   f"Missing {val_path}"

#     # 1) Device FIRST
#     device = get_device()
#     print(f"Using device: {device}")

#     # 2) Build loaders
#     train_dl = MinSnipDataset.make_dataloader(
#         DEFAULT_TRAIN_PATH, batch_size=128, shuffle=True, num_workers=0, seed=42,
#         label_keys=LABELS
#     )
#     val_dl = MinSnipDataset.make_dataloader(
#         DEFAULT_TEST_PATH, batch_size=128, shuffle=False, num_workers=0,
#         label_keys=LABELS
#     )

#     # 3) Build model
#     model = SnipLSTM(
#         seq_input_size=2,
#         static_input_size=5,
#         lstm_hidden=64,
#         lstm_layers=1,
#         bidirectional=False,
#         mlp_hidden=64,
#         dropout=0.1,
#         num_tasks=len(LABELS),
#     ).to(device)

#     # 4) (Optional) compute normalization stats on TRAIN and move them to device
#     mean_seq, std_seq, mean_sta, std_sta = compute_stats_seq_static(train_dl)  # returns CPU tensors
#     mean_seq, std_seq = mean_seq.to(device), std_seq.to(device)
#     mean_sta, std_sta = mean_sta.to(device), std_sta.to(device)
#     norm = (mean_seq, std_seq, mean_sta, std_sta)

#     torch.save(
#         {
#             "mean_seq": mean_seq.detach().cpu(),
#             "std_seq":  std_seq.detach().cpu(),
#             "mean_sta": mean_sta.detach().cpu(),
#             "std_sta":  std_sta.detach().cpu(),
#         },
#         "norm_stats_standard25k.pt"
#     )

#     # 5) (Optional) init output bias to base rate to stabilize early training
#     pi = frac_positive_multi(train_dl).to(device)          # [K]
#     with torch.no_grad():
#         bias = (pi.clamp(1e-4, 1-1e-4) / (1 - pi.clamp(1e-4, 1-1e-4))).log()
#         model.head[-1].bias.copy_(bias.to(model.head[-1].bias.device))   # vector [K]

#     # 6) Rest as before...
#     pos_weight = estimate_pos_weight_multi(train_dl, device)  # [K]
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

#     best_val, patience, bad, epochs = math.inf, 30, 0, 400
#     for epoch in range(1, epochs + 1):
#         tr_loss, tr_mets = run_epoch(model, train_dl, device,
#                                      optimizer=optimizer, pos_weight=pos_weight, norm=norm, labels=LABELS)
#         va_loss, va_mets = run_epoch(model, val_dl, device,
#                                      optimizer=None,      pos_weight=pos_weight, norm=norm, labels=LABELS)
    
#         m = va_mets["macro"]
#         h22 = va_mets["per_label"]["5_5"]
    
#         print(
#             f"Epoch {epoch:02d} | "
#             f"train loss {tr_loss:.4f} | val loss {va_loss:.4f} | "
#             f"macro acc {m['acc']:.3f} prec {m['precision']:.3f} rec {m['recall']:.3f} | "
#             f"2_2 acc {h22['acc']:.3f} prec {h22['precision']:.3f} rec {h22['recall']:.3f}"
#         )
    
#         if va_loss < best_val - 1e-4:
#             best_val, bad = va_loss, 0
#             torch.save(model.state_dict(), "checkpoint_snip_standard25k.pt")
#         else:
#             bad += 1
#             if bad >= patience:
#                 print("Early stopping.")
#                 break

#     print("Done. Best val loss:", best_val)

#     # Quick prob preview (no thresholding here)
#     model.eval()
#     batch = next(iter(val_dl))
#     with torch.no_grad():
#         probs = model.predict_proba(
#             batch["x_seq"].to(device),
#             batch["x_static"].to(device),
#             batch["mask"].to(device),
#         ).cpu()
#     print("Sample probs (val[0:8]):", probs[:8].tolist())

#     hits = collect_hits_simple(
#         val_dl, model, device,
#         threshold=0.65,
#         norm=norm if 'norm' in locals() else None,   # pass your (mean/std) tuple if you computed it
#         out_path="val_hits_065_simple.json"          # or None to skip writing
#     )
#     print("Top 100 hits:", hits[:100])


# if __name__ == "__main__":
#     main()


