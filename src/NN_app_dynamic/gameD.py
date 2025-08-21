import torch

@torch.no_grad()
def compute_stats_seq_static(loader):
    """Z-score stats for seq channels and static features from TRAIN loader."""
    n_seq = 0
    sum_seq = sumsq_seq = None
    n_sta = 0
    sum_sta = sumsq_sta = None

    for b in loader:
        xs, xm = b["x_seq"], b["mask"]     # [B,T,C], [B,T]
        st     = b["x_static"]             # [B,C_static]

        # valid timesteps across batch
        valid = xm.bool().unsqueeze(-1)    # [B,T,1]
        vals  = xs[valid.expand_as(xs)].view(-1, xs.shape[-1])  # [N_valid, C]

        if sum_seq is None:
            sum_seq   = vals.sum(0)
            sumsq_seq = (vals**2).sum(0)
        else:
            sum_seq   += vals.sum(0)
            sumsq_seq += (vals**2).sum(0)
        n_seq += vals.shape[0]

        if sum_sta is None:
            sum_sta   = st.sum(0)
            sumsq_sta = (st**2).sum(0)
        else:
            sum_sta   += st.sum(0)
            sumsq_sta += (st**2).sum(0)
        n_sta += st.shape[0]

    mean_seq = sum_seq / max(1, n_seq)
    var_seq  = sumsq_seq / max(1, n_seq) - mean_seq**2
    std_seq  = torch.sqrt(torch.clamp(var_seq, min=1e-8))

    mean_sta = sum_sta / max(1, n_sta)
    var_sta  = sumsq_sta / max(1, n_sta) - mean_sta**2
    std_sta  = torch.sqrt(torch.clamp(var_sta, min=1e-8))

    return mean_seq, std_seq, mean_sta, std_sta
