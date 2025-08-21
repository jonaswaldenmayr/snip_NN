from pathlib import Path
import sys
import json
import torch
import re
from src.NN_app_dynamic.helperD import _extract_state_dict, detect_model_config_from_state, infer_keys_from_data, maybe_add_tod

from src.data.datasets import MinSnipDataset
from src.models.snip_lstm01 import SnipLSTM
from src.NN_app.helper import scoreboard_from_loader

"""
source venv/bin/activate
python -m src.NN_app_dynamic.inferD
"""

# ---------- Config (paths & thresholds only) ----------
THRESH_LOW  = 0.8
THRESH_HIGH = 0.999
CKPT = Path("runs/20250821_120210_midfull_55/model.pt")
NORM = Path("runs/20250821_120210_midfull_55/norm_stats.pt")
DATA = Path("data/processed/validationSNIPS_midfull_55.json")  # set the file you actually want to evaluate


# ---------- Utilities ----------
def get_device():
    if torch.backends.mps.is_available():   return torch.device("mps")
    if torch.cuda.is_available():           return torch.device("cuda")
    return torch.device("cpu")



@torch.no_grad()
def load_norm(device, expected_seq: int, expected_sta: int):
    if not NORM.exists():
        print("norm_stats.pt not found -> proceeding without normalization")
        return None
    d = torch.load(NORM, map_location="cpu")
    mean_seq, std_seq = d["mean_seq"], d["std_seq"]
    mean_sta, std_sta = d["mean_sta"], d["std_sta"]
    if mean_seq.numel() != expected_seq or mean_sta.numel() != expected_sta:
        print(
            f"[WARN] Normalization shape mismatch. "
            f"norm(seq={mean_seq.numel()}, static={mean_sta.numel()}) "
            f"!= expected(seq={expected_seq}, static={expected_sta}). Skipping normalization."
        )
        return None
    return (mean_seq.to(device), std_seq.to(device), mean_sta.to(device), std_sta.to(device))


# ---------- Main ----------
def main():
    device = get_device()
    print("Device:", device)

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")
    if not DATA.exists():
        raise FileNotFoundError(f"DATA file not found: {DATA}")

    # 1) Load checkpoint & detect expected sizes
    raw = torch.load(CKPT, map_location=device)
    state = _extract_state_dict(raw)
    ck_cfg = detect_model_config_from_state(state)
    print(
        "Checkpoint expects -> "
        f"seq: {ck_cfg['seq_input_size']}, static: {ck_cfg['static_input_size']}, "
        f"hidden: {ck_cfg['lstm_hidden']}, layers: {ck_cfg['lstm_layers']}, "
        f"bidir: {ck_cfg['bidirectional']}, mlp_hidden: {ck_cfg['mlp_hidden']}"
    )

    # 2) Infer keys from the DATA file (no reliance on run config)
    seq_keys_file, static_keys_file = infer_keys_from_data(DATA)

    # 3) Reconcile with checkpoint sizes (only allow adding TOD if needed)
    seq_keys    = list(seq_keys_file)
    static_keys = maybe_add_tod(static_keys_file, ck_cfg["static_input_size"])

    print(f"[infer] Using DATA={DATA}")
    print(f"[infer] file seq_keys   ({len(seq_keys_file)}): {seq_keys_file}")
    print(f"[infer] file static_keys({len(static_keys_file)}): {static_keys_file}")
    if static_keys != static_keys_file:
        print(f"[infer] static_keys after TOD add ({len(static_keys)}): {static_keys}")

    # 4) Hard guard: sizes must match exactly now
    if len(seq_keys) != ck_cfg["seq_input_size"] or len(static_keys) != ck_cfg["static_input_size"]:
        raise RuntimeError(
            "Feature schema mismatch (file vs. checkpoint):\n"
            f"  DATA file: {DATA}\n"
            f"  file seq_keys      ({len(seq_keys_file)}): {seq_keys_file}\n"
            f"  file static_keys   ({len(static_keys_file)}): {static_keys_file}\n"
            f"  (after TOD add)    ({len(static_keys)}): {static_keys}\n"
            f"  checkpoint expects -> seq={ck_cfg['seq_input_size']}, static={ck_cfg['static_input_size']}\n"
            "Regenerate the DATA file with matching keys, or choose a file built with the checkpoint's schema."
        )

    # 5) Build dataloader with EXPLICIT keys (prevents 2/5 fallback)
    test_dl = MinSnipDataset.make_dataloader(
        DATA.as_posix(),
        batch_size=512,
        shuffle=False,
        num_workers=0,
        seq_keys=seq_keys,
        static_keys=static_keys,
        label_keys=["2_2"],
    )

    # 6) Sanity check from a real batch
    try:
        first_batch = next(iter(test_dl))
    except StopIteration:
        print("Loader is empty. Nothing to evaluate.")
        sys.exit(0)
    ds_seq_in  = first_batch["x_seq"].shape[-1]
    ds_sta_in  = first_batch["x_static"].shape[-1]
    print(f"Dataset features -> seq: {ds_seq_in}, static: {ds_sta_in}")

    # 7) Instantiate model & load weights
    model = SnipLSTM(**ck_cfg).to(device)
    model.load_state_dict(state, strict=True)

    # 8) Load normalization (only if shapes match)
    norm = load_norm(device, ck_cfg["seq_input_size"], ck_cfg["static_input_size"])

    # 9) Run scoreboard
    _ = scoreboard_from_loader(
        test_dl,
        model,
        device,
        threshold=(THRESH_LOW, THRESH_HIGH),
        norm=norm,
        verbose=True,
        list_high_conf=True,
    )

if __name__ == "__main__":
    main()
