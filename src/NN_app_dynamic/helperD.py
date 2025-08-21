import json
import re
import torch
from pathlib import Path


@torch.no_grad()
def scoreboard_from_loader(
    loader,
    model,
    device,
    threshold,
    norm=None,
    verbose: bool = True,
    list_high_conf: bool = True,  # print only high-confidence positives
):
    """
    Run the model over the loader and PRINT ONLY one-liners for items with prob >= threshold:
      '#   <idx>   p=<prob>   y=<true> -> OK/WRONG'

    Also prints a small summary header with counts/precision for the selected set.
    """
    model.eval()

    all_probs, all_ytrue, all_idx = [], [], []
    seen = 0

    for batch in loader:
        x_seq    = batch["x_seq"].to(device)
        x_static = batch["x_static"].to(device)
        mask     = batch["mask"].to(device)
        y        = batch["y"].to(device).float()

        if y.ndim == 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)

        if norm is not None:
            mean_seq, std_seq, mean_sta, std_sta = norm
            x_seq    = (x_seq - mean_seq) / std_seq
            x_static = (x_static - mean_sta) / std_sta

        logits = model(x_seq, x_static, mask)  # [B] or [B,1]
        if logits.ndim == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        probs = torch.sigmoid(logits).detach().cpu()
        ycpu  = y.detach().cpu().to(torch.int64)

        B = probs.shape[0]
        idx = torch.arange(seen, seen + B, dtype=torch.int64)
        seen += B

        all_probs.append(probs)
        all_ytrue.append(ycpu)
        all_idx.append(idx)

    if not all_probs:
        if verbose:
            print("Loader is empty. No samples to evaluate.")
        return {
            "n_total": 0, "n_selected": 0, "selected_precision": None,
            "selected_indices": torch.empty(0, dtype=torch.int64),
            "selected_probs": torch.empty(0),
            "selected_ytrue": torch.empty(0, dtype=torch.int64),
            "threshold": float(threshold) if not isinstance(threshold, (tuple, list)) else tuple(threshold),
        }

    probs = torch.cat(all_probs)   # [N]
    ytrue = torch.cat(all_ytrue)   # [N]
    idxs  = torch.cat(all_idx)     # [N]

    # filter to predicted positives at/above threshold

      # --- NEW: allow (low, high) band ---
    if isinstance(threshold, (tuple, list)):
        lo, hi = threshold
        lo = 0.0 if lo is None else float(lo)
        hi = 1.0 if hi is None else float(hi)
        if hi < lo:
            lo, hi = hi, lo
        mask = (probs >= lo) & (probs <= hi)
        thresh_desc = f"[{lo:.3f}, {hi:.3f}]"
    else:
        lo = float(threshold)
        hi = None
        mask = probs >= lo
        thresh_desc = f"{lo:.3f}"
    # -----------------------------------

    sel_probs = probs[mask]
    sel_ytrue = ytrue[mask]
    sel_idxs  = idxs[mask]

    n_total    = int(probs.numel())
    n_selected = int(mask.sum())
    selected_correct   = int((sel_ytrue == 1).sum())
    selected_incorrect = int((sel_ytrue == 0).sum())
    selected_precision = (selected_correct / n_selected) if n_selected > 0 else None

    if verbose:
        print("\n=== High-Confidence Scoreboard ===")
        print(f"Total samples        : {n_total}")
        if hi is None:
            print(f"Threshold            : {thresh_desc}")
            print(f"Selected (prob>=τ)   : {n_selected}")
        else:
            print(f"Threshold band       : {thresh_desc}")
            print(f"Selected (lo<=p<=hi) : {n_selected}")
        if n_selected > 0:
            print(f"  Correct (label=1)  : {selected_correct}")
            print(f"  Incorrect (=0)     : {selected_incorrect}")
            print(f"  Precision@sel      : {selected_precision:.4f}")
        else:
            print("  (No samples met the criterion)")
        print("=================================\n")

        # one-liners only, sorted by probability descending
        if list_high_conf and n_selected > 0:
            order = torch.argsort(sel_probs, descending=True)
            for i, p, y in zip(sel_idxs[order].tolist(),
                               sel_probs[order].tolist(),
                               sel_ytrue[order].tolist()):
                tag = "OK" if y == 1 else "WRONG"
                print(f"  #{i:>7}   p={p:.4f}   y={y} -> {tag}")

    return {
        "n_total": n_total,
        "n_selected": n_selected,
        "selected_correct": selected_correct,
        "selected_incorrect": selected_incorrect,
        "selected_precision": float(selected_precision) if selected_precision is not None else None,
        "threshold": (float(lo), float(hi)) if hi is not None else float(lo),
        "threshold_low": float(lo),
        "threshold_high": float(hi) if hi is not None else None,
        "selected_indices": sel_idxs,
        "selected_probs": sel_probs,
        "selected_ytrue": sel_ytrue,
        "all_probs": probs,
        "all_ytrue": ytrue,
        "all_indices": idxs,
    }

def _extract_state_dict(obj):
    """
    Accept either a raw state_dict or a dict wrapper like {'model': sd} / {'state_dict': sd}.
    """
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj  # assume already a state_dict

def detect_model_config_from_state(state_dict):
    """
    Infer SnipLSTM constructor args from a PyTorch state_dict.
    Robust to prefixes (e.g., 'module.') and to bidirectional keys ('_reverse').
    """
    ih_entries = []
    for k, v in state_dict.items():
        m = re.search(r"lstm\.weight_ih_l(\d+)(?:_reverse)?$", k)
        if m:
            layer = int(m.group(1))
            is_rev = k.endswith("_reverse")
            ih_entries.append((layer, is_rev, k, v))
    if not ih_entries:
        for k, v in state_dict.items():
            m = re.search(r"weight_ih_l(\d+)(?:_reverse)?$", k)
            if m:
                layer = int(m.group(1))
                is_rev = k.endswith("_reverse")
                ih_entries.append((layer, is_rev, k, v))
    if not ih_entries:
        raise RuntimeError("Cannot find any 'lstm.weight_ih_l*' parameters in checkpoint.")

    fwd = [(layer, k, w) for (layer, is_rev, k, w) in ih_entries if not is_rev]
    if not fwd:
        fwd = [(layer, k, w) for (layer, is_rev, k, w) in ih_entries]
    fwd.sort(key=lambda x: x[0])
    _, _, w0 = fwd[0]
    hidden = w0.shape[0] // 4
    seq_input_size = w0.shape[1]
    lstm_layers = len({layer for (layer, _, _) in fwd})
    bidirectional = any(is_rev for (_, is_rev, _, _) in ih_entries)

    static_key = None
    if "norm_static.weight" in state_dict:
        static_key = "norm_static.weight"
    else:
        for k in state_dict.keys():
            if k.endswith("norm_static.weight"):
                static_key = k
                break
    if static_key is None:
        raise RuntimeError("Cannot infer static_input_size (missing 'norm_static.weight').")
    static_input_size = state_dict[static_key].shape[0]

    mlp_hidden = state_dict.get("head.0.weight", None)
    if mlp_hidden is not None:
        mlp_hidden = mlp_hidden.shape[0]
    else:
        head_ws = [state_dict[k] for k in state_dict if k.startswith("head.") and k.endswith(".weight")]
        mlp_hidden = head_ws[0].shape[0] if head_ws else 64

    return dict(
        seq_input_size=seq_input_size,
        static_input_size=static_input_size,
        lstm_hidden=hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        mlp_hidden=mlp_hidden,
        dropout=0.0,
    )

def read_first_obj(p: Path):
    with p.open("r", encoding="utf-8") as f:
        head = f.read(4096).lstrip()
        f.seek(0)
        if head.startswith("["):
            arr = json.load(f)
            if not arr:
                raise ValueError(f"{p} is an empty array.")
            return arr[0]
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"Could not read one JSON object from {p}.")

def infer_keys_from_data(data_path: Path):
    row = read_first_obj(data_path)
    seq_keys_in_file    = list((row.get("seq") or {}).keys())
    static_keys_in_file = list((row.get("static") or {}).keys())
    return seq_keys_in_file, static_keys_in_file

def maybe_add_tod(static_keys, want):
    """Add tod_sin/tod_cos if file lacks them and the checkpoint expects more static features."""
    sk = list(static_keys)
    need = want - len(sk)
    if need <= 0:
        return sk
    extras = []
    if "tod_sin" not in sk: extras.append("tod_sin")
    if "tod_cos" not in sk: extras.append("tod_cos")
    return sk + extras[:need]