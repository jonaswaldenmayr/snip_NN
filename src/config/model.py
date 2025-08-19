from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    lstm_hidden: int = 64
    lstm_layers: int = 1
    bidirectional: bool = False
    mlp_hidden: int = 64
    dropout: float = 0.1
    pooling: str = "last"   # "last" | "mean" | "attn"
