from __future__ import annotations
from pathlib import Path
import math, json, torch
from src.data.datasets import MinSnipDataset
from src.models.factory import get_model
from src.config import CFG, save_used_config
from src.train_loops import get as get_loop
from src.utils.device import get_device                 # <--- moved
from src.utils.stats import (                           # <--- moved
    compute_stats_seq_static, frac_positive, estimate_pos_weight
)





def main():
    torch.manual_seed(CFG.seed)

    # Resolve paths
    train_path = CFG.train_path
    val_path   = CFG.val_path
    assert Path(train_path).exists(), f"Missing {train_path}"
    assert Path(val_path).exists(),   f"Missing {val_path}"

    # Run dir + save config used
    paths = CFG.paths()
    Path(paths["run_dir"]).mkdir(parents=True, exist_ok=True)
    save_used_config(CFG, paths["config"])

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Loaders
    train_dl = MinSnipDataset.make_dataloader(
        train_path,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
        seed=CFG.seed,
        label_keys=CFG.label_keys,
        seq_keys=list(CFG.seq_keys),
        static_keys=list(CFG.static_keys),
    )

    val_dl = MinSnipDataset.make_dataloader(
        val_path,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,
        label_keys=CFG.label_keys,
        seq_keys=list(CFG.seq_keys),
        static_keys=list(CFG.static_keys),
    )


    # Model (num_tasks from labels)
    num_tasks = len(CFG.label_keys)
    model = get_model(CFG).to(device)   

    # Norms
    norm = None
    if CFG.compute_norms:
        mean_seq, std_seq, mean_sta, std_sta = compute_stats_seq_static(train_dl)
        mean_seq, std_seq = mean_seq.to(device), std_seq.to(device)
        mean_sta, std_sta = mean_sta.to(device), std_sta.to(device)
        norm = (mean_seq, std_seq, mean_sta, std_sta)
        torch.save(
            {"mean_seq": mean_seq.cpu(), "std_seq": std_seq.cpu(),
             "mean_sta": mean_sta.cpu(), "std_sta": std_sta.cpu()},
            paths["norm"]
        )

    # Bias init to base rate
    pi = frac_positive(train_dl)
    pi = min(max(pi, 1e-4), 1 - 1e-4)
    with torch.no_grad():
        model.head[-1].bias.fill_(math.log(pi/(1-pi)))

    # Optimizer and pos_weight
    pos_weight = estimate_pos_weight(train_dl, device, cap=CFG.pos_weight_cap) if CFG.use_pos_weight else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    # ---- choose and run the loop via registry ----
    loop_fn = get_loop(CFG.loop_name)   # "standard", "cosine"
    metrics = loop_fn(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        device=device,
        cfg=CFG,
        norm=norm,
        pos_weight=pos_weight,
        optimizer=optimizer,
        ckpt_best=paths["ckpt"],   # <--- add this line

    )
    # ---------------------------------------------------

    # Save model 
    # torch.save(model.state_dict(), paths["ckpt"])

     # === Save basic loop metrics first ===
    with open(paths["metrics"], "w") as f:
        json.dump(metrics or {}, f, indent=2)



if __name__ == "__main__":
    main()
