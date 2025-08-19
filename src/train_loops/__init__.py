# src/train_loops/__init__.py
from __future__ import annotations
from typing import Callable, Dict, Any
import torch

# Signature the loops will implement
# norm: (mean_seq, std_seq, mean_sta, std_sta) or None
TrainLoopFn = Callable[
    [torch.nn.Module, torch.utils.data.DataLoader, torch.utils.data.DataLoader,
     torch.device, Any, tuple[torch.Tensor, ...] | None, torch.Tensor | None, torch.optim.Optimizer],
    Dict[str, float]
]

_REG: Dict[str, TrainLoopFn] = {}

def register(name: str):
    def deco(fn: TrainLoopFn):
        _REG[name] = fn
        return fn
    return deco

def get(name: str) -> TrainLoopFn:
    if name not in _REG:
        raise ValueError(f"Unknown training loop: {name} (available: {list(_REG)})")
    return _REG[name]

# Import built-ins to auto-register
from .standard import fit as _fit_standard  # noqa
from .cosine import fit as _fit_cosine      # noqa (optional example)
