from pathlib import Path
from src.data.datasets import MinSnipDataset
import torch
from src.models.snip_lstm01 import SnipLSTM
from src.NN_app.helper import scoreboard_from_loader

"""
source venv/bin/activate
python -m src.NN_app.infer
"""

THRESH_LOW  = 0.8
THRESH_HIGH = 0.999
CKPT = "runs/20250820_141653_APPLY/model.pt"
NORM = "runs/20250820_141653_APPLY/norm_stats.pt"
TEST_PATH = "data/processed/validationSET.json"  # direct path to your test_min.json

#TEST_PATH = "data/processed/mid_2025_short_test01.json"  # direct path to your test_min.json


def get_device():
    if torch.backends.mps.is_available():   return torch.device("mps")
    if torch.cuda.is_available():           return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def load_norm(device):
    if not Path(NORM).exists():
        print("norm_stats.pt not found -> proceeding without normalization")
        return None
    d = torch.load(NORM, map_location="cpu")
    return (d["mean_seq"].to(device), d["std_seq"].to(device),
            d["mean_sta"].to(device), d["std_sta"].to(device))

def main():
    device = get_device()
    print("Device:", device)

    # Only use the test data
    test_dl = MinSnipDataset.make_dataloader(TEST_PATH, batch_size=512, shuffle=False, num_workers=0)

    # Model + weights
    model = SnipLSTM(seq_input_size=2, static_input_size=5, lstm_hidden=64, lstm_layers=1,
                     bidirectional=False, mlp_hidden=64, dropout=0.0).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)

    norm = load_norm(device)

    # High-confidence scoreboard (pred >= THRESH)
    _ = scoreboard_from_loader(
    test_dl, model, device,
    threshold=(THRESH_LOW, THRESH_HIGH),  # <-- pass tuple now
    norm=norm,
    verbose=True,
    list_high_conf=True,
)

if __name__ == "__main__":
    main()
