import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SnipLSTM(nn.Module):
    """
    seq: [B, T, C_seq] (C_seq = 2 from DEFAULT_SEQ_KEYS)
    static: [B, C_static] (C_static = 5 from DEFAULT_STATIC_KEYS)
    mask: [B, T] with 1.0 for valid, 0.0 for pad
    """
    def __init__(
        self,
        seq_input_size: int = 2,    #cumLogVsBuy, rollVol
        static_input_size: int = 5,  # dailyDrops, buySpread, buyPrice, dropSize, volatility
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
        num_tasks: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_size = lstm_hidden * (2 if bidirectional else 1)

        self.norm_static = nn.LayerNorm(static_input_size) #-> clean up
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size + static_input_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_tasks), #K logit
            )
    
    def forward(self, x_seq, x_static, mask):
        # x_seq: [B,T,C], mask: [B,T]
        lengths = mask.sum(dim=1).to(dtype=torch.int64) #B
        lengths = torch.clamp(lengths, min=1) #avoid zero-length...

        # pack -> LSTM
        packed = pack_padded_sequence(x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed)
        # hn: [num_layers * num_dirs, B, H]
        lstm_summary = hn[-1]

        x_static = self.norm_static(x_static) #<- clean up
        h = torch.cat([lstm_summary, x_static], dim=1)   # [B, H + C_static]
        logits = self.head(h)                             # [B, num_tasks]  <-- no squeeze
        return logits

    @torch.no_grad()
    def predict_proba(self, x_seq, x_static, mask):
        logits = self.forward(x_seq, x_static, mask)  # [B, num_tasks]
        return torch.sigmoid(logits)
        

