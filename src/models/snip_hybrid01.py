import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionPooling(nn.Module):
    """
    Mask-aware additive attention:
      scores_t = v^T tanh(W h_t)
      α = softmax(scores) over valid timesteps only
      ctx = Σ_t α_t · h_t
    """
    def __init__(self, d_in: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_in)
        self.v    = nn.Linear(d_in, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h:    [B, T, D]
        mask: [B, T]  (1.0 valid, 0.0 pad)
        returns: context [B, D]
        """
        # scores: [B, T]
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1)
        # mask padded positions to -inf before softmax
        scores = scores.masked_fill(mask <= 0, float('-inf'))
        attn = torch.softmax(scores, dim=1)  # [B, T]
        ctx = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # [B, D]
        return ctx


class SnipHybrid01(nn.Module):
    """
    ModelV2: CNN -> BiGRU -> Attention (seq)  ⊕  Static tower  -> head

    Inputs:
      x_seq:    [B, T, C_seq]    (e.g., cumLogVsBuy, rollVol)
      x_static: [B, C_static]    (dailyDrops, buySpread, buyPrice, dropSize, volatility)
      mask:     [B, T]           (1.0 valid, 0.0 padded)

    Output:
      logits: [B, num_tasks]  (use BCEWithLogitsLoss)
    """
    def __init__(
        self,
        seq_input_size: int = 2,
        static_input_size: int = 5,
        # CNN
        cnn_channels: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        # RNN
        gru_hidden: int = 64,
        gru_layers: int = 1,
        # Heads
        mlp_hidden: int = 64,
        dropout: float = 0.10,
        num_tasks: int = 1,
    ):
        super().__init__()
        self.seq_input_size = seq_input_size
        self.static_input_size = static_input_size
        self.dropout = nn.Dropout(dropout)

        # ----- Temporal CNN (1D over time) -----
        # Conv1d expects [B, C_in, T]; we’ll transpose in forward.
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=seq_input_size,
                      out_channels=cnn_channels,
                      kernel_size=k,
                      padding=k // 2)
            for k in kernel_sizes
        ])
        cnn_out_dim = cnn_channels * len(kernel_sizes)
        self.seq_ln = nn.LayerNorm(cnn_out_dim)  # per time step LN

        # ----- BiGRU over CNN features -----
        self.gru = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True
        )
        rnn_out_dim = gru_hidden * 2  # bidirectional

        # ----- Mask-aware attention pooling -----
        self.attn = AttentionPooling(rnn_out_dim)

        # ----- Static tower -----
        self.norm_static  = nn.LayerNorm(static_input_size)
        static_hidden = max(32, mlp_hidden // 2)  # small tower; adjustable
        self.static_tower = nn.Sequential(
            nn.Linear(static_input_size, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----- Fusion + Head (keep name 'head' so bias init code works) -----
        fusion_in = rnn_out_dim + static_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_tasks),  # <- index -1 exists for bias init
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, T, C_seq], x_static: [B, C_static], mask: [B, T]
        """
        B, T, _ = x_seq.shape

        # Ensure valid lengths (avoid zero-length pack)
        lengths = mask.sum(dim=1).to(dtype=torch.int64)
        lengths = torch.clamp(lengths, min=1)

        # Zero-out padded timesteps before convolutions (limits pad leakage)
        x_seq = x_seq * mask.unsqueeze(-1)

        # ----- CNN over time -----
        # to [B, C_in, T]
        z = x_seq.transpose(1, 2)
        conv_feats = []
        for conv in self.convs:
            c = conv(z)                 # [B, cnn_ch, T]
            c = torch.relu(c)
            c = self.dropout(c)
            conv_feats.append(c)
        zc = torch.cat(conv_feats, dim=1)      # [B, cnn_out_dim, T]
        zc = zc.transpose(1, 2)                # back to [B, T, cnn_out_dim]
        zc = self.seq_ln(zc)

        # ----- BiGRU (packed) -----
        packed = pack_padded_sequence(zc, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        h, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # [B, T, 2H]

        # ----- Attention pooling (mask-aware) -----
        ctx = self.attn(h, mask)  # [B, 2H]

        # ----- Static tower -----
        xs = self.norm_static(x_static)
        xs = self.static_tower(xs)

        # ----- Fusion + head -----
        fused = torch.cat([ctx, xs], dim=1)  # [B, 2H + static_hidden]
        logits = self.head(fused)            # [B, num_tasks]
        return logits

    @torch.no_grad()
    def predict_proba(self, x_seq: torch.Tensor, x_static: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x_seq, x_static, mask)
        return torch.sigmoid(logits)
