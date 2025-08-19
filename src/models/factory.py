from .snip_lstm01 import SnipLSTM
from .snip_hybrid01 import SnipHybrid01


def get_model(cfg):
    """
    Return a model instance based on cfg.model_name
    Reuses cfg.lstm_hidden / cfg.lstm_layers as GRU sizes for the hybrid.
    """
    num_tasks = len(cfg.label_keys)

    if cfg.model_name == "lstm01":
        return SnipLSTM(
            seq_input_size=cfg.seq_input_size,
            static_input_size=cfg.static_input_size,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            bidirectional=cfg.bidirectional,
            mlp_hidden=cfg.mlp_hidden,
            dropout=cfg.dropout,
            num_tasks=num_tasks,
        )

    if cfg.model_name == "hybrid01":
        # Use GRU sizes from LSTM fields to avoid adding new config keys now.
        return SnipHybrid01(
            seq_input_size=cfg.seq_input_size,
            static_input_size=cfg.static_input_size,
            gru_hidden=cfg.lstm_hidden,
            gru_layers=cfg.lstm_layers,
            mlp_hidden=cfg.mlp_hidden,
            dropout=cfg.dropout,
            num_tasks=num_tasks,
            # cnn_channels / kernel_sizes keep sensible defaults; add to cfg later if you want.
        )

    raise ValueError(f"Unknown model_name: {cfg.model_name}")
