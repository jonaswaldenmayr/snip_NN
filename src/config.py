
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json, time

@dataclass
class Config:

    # ----- data -----
    train_path: str = "data/processed/04_train_22-24_midfull_55.json"
    val_path:   str = "data/processed/04_test_22-24_midfull_55.json"
    label_keys: tuple[str, ...] = ("5_5",)
    seq_keys:   tuple[str, ...] = ("cumLogVsBuy","rollVol","logRet","drawdown","runup","volNormRet")
    static_keys:tuple[str, ...] = ("dailyDrops","buySpread","buyPrice","dropSize","volatility","tod_sin","tod_cos","hist.daily.mean","hist.daily.vol","hist.daily.mdd","hist.weekly.mom","hist.monthly.mdd")


    # # ----- data -----
    # train_path: str = "data/processed/04_train_22-24_midfull_22.json"
    # val_path:   str = "data/processed/04_test_22-24_midfull_22.json"
    # label_keys: tuple[str, ...] = ("2_2",)
    # seq_keys:   tuple[str, ...] = ("cumLogVsBuy","rollVol","logRet","drawdown","runup","volNormRet")
    # static_keys:tuple[str, ...] = ("dailyDrops","buySpread","buyPrice","dropSize","volatility","tod_sin","tod_cos","hist.daily.mean","hist.daily.vol","hist.daily.mdd","hist.weekly.mom","hist.monthly.mdd")

    # # ----- data - BASE SETUP -----
    # train_path: str = "data/processed/03_train_new25k.json"
    # val_path:   str = "data/processed/03_test_new25k.json"
    # label_keys: tuple[str, ...] = ("2_2",)                          #choose 2_2
    # seq_keys:   tuple[str, ...] = ("cumLogVsBuy","rollVol")
    # static_keys:tuple[str, ...] = ("dailyDrops","buySpread","buyPrice","dropSize","volatility")

    # ----- model -----
    model_name: str = "lstm01"   # lstm01, "hybrid01"
    seq_input_size: int = 2
    static_input_size: int = 5
    lstm_hidden: int = 64
    lstm_layers: int = 1
    bidirectional: bool = False
    mlp_hidden: int = 64
    dropout: float = 0.10
    num_tasks: int = 1           # keep 1 label for now

    # ----- training -----
    loop_name: str = "standard"
    epochs: int = 20
    patience: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    use_pos_weight: bool = True
    pos_weight_cap: float | None = None  # e.g. 4.0 or None

    # ----- normalization -----
    compute_norms: bool = True

    # ----- eval -----
    threshold: float = 0.5

    # ----- artifacts (auto-filled run folder) -----
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    run_dir: str = "runs"  # base folder

    def paths(self) -> dict[str, str]:
        rd = str(Path(self.run_dir) / self.run_id)
        return {
            "run_dir": rd,
            "ckpt":    str(Path(rd) / "model.pt"),
            "norm":    str(Path(rd) / "norm_stats.pt"),
            "config":  str(Path(rd) / "config.used.json"),
            "metrics": str(Path(rd) / "metrics_val.json"),
        }

CFG = Config()

CFG.seq_input_size    = len(CFG.seq_keys)
CFG.static_input_size = len(CFG.static_keys)


def save_used_config(cfg: Config, path: str | None = None):
    p = Path(path) if path else Path(cfg.paths()["config"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(asdict(cfg), f, indent=2)
