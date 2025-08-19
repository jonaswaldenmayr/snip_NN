import torch

@torch.no_grad()
def frac_positive(loader):
    pos = tot = 0
    for b in loader:
        y = b["y"]
        pos += int((y == 1).sum())
        tot += int(y.numel())
    return pos / max(1, tot)

@torch.no_grad()
def compute_stats_seq_static(loader):
    n_seq = n_sta = 0
    sum_seq = sumsq_seq = None
    sum_sta = sumsq_sta = None
    for b in loader:
        xs, xm, st = b["x_seq"], b["mask"], b["x_static"]
        valid = xm.bool().unsqueeze(-1)
        vals = xs[valid.expand_as(xs)].view(-1, xs.shape[-1])
        if sum_seq is None:
            sum_seq, sumsq_seq = vals.sum(dim=0), (vals**2).sum(dim=0)
        else:
            sum_seq += vals.sum(dim=0); sumsq_seq += (vals**2).sum(dim=0)
        n_seq += vals.shape[0]
        if sum_sta is None:
            sum_sta, sumsq_sta = st.sum(dim=0), (st**2).sum(dim=0)
        else:
            sum_sta += st.sum(dim=0); sumsq_sta += (st**2).sum(dim=0)
        n_sta += st.shape[0]
    mean_seq = sum_seq / max(1, n_seq)
    var_seq  = sumsq_seq / max(1, n_seq) - mean_seq**2
    std_seq  = torch.sqrt(torch.clamp(var_seq, min=1e-8))
    mean_sta = sum_sta / max(1, n_sta)
    var_sta  = sumsq_sta / max(1, n_sta) - mean_sta**2
    std_sta  = torch.sqrt(torch.clamp(var_sta, min=1e-8))
    return mean_seq, std_seq, mean_sta, std_sta

@torch.no_grad()
def estimate_pos_weight(loader, device, cap=None):
    pos = neg = 0
    for b in loader:
        y = b["y"]
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    if pos == 0: 
        return None
    ratio = neg / pos
    if cap is not None:
        ratio = min(ratio, cap)
    return torch.tensor([ratio], device=device, dtype=torch.float32)
