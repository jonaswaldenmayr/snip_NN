from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class DataConfig:
        seq_keys:   Tuple[str, ...] = ("cumLogVsBuy","rollVol")
        static_keys:Tuple[str, ...] = ("dailyDrops","buySpread","buyPrice","dropSize","volatility")
        label_keys: Tuple[str, ...] = ("2_2","5_5","8_8","2_5","5_2","2_1")