from __future__ import annotations
from pathlib import Path
import math, json, torch
from src.data.datasets import MinSnipDataset
from src.models.factory import get_model
from src.utils.helper import collect_hits_simple
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
        train_path, batch_size=CFG.batch_size, shuffle=True, num_workers=0, seed=CFG.seed,
        label_keys=CFG.label_keys
    )
    val_dl = MinSnipDataset.make_dataloader(
        val_path, batch_size=CFG.batch_size, shuffle=False, num_workers=0,
        label_keys=CFG.label_keys
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
    )
    # ---------------------------------------------------

    # Save model 
    torch.save(model.state_dict(), paths["ckpt"])

     # === Save basic loop metrics first ===
    with open(paths["metrics"], "w") as f:
        json.dump(metrics or {}, f, indent=2)








    # === Rich validation metrics + threshold sweep === -> if unnecessary, remove it & utils functions
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for b in val_dl:
            x_seq    = b["x_seq"].to(device)
            x_static = b["x_static"].to(device)
            mask     = b["mask"].to(device)
            y        = b["y"].to(device)
            logits = model(x_seq, x_static, mask)
            probs  = torch.sigmoid(logits)
            all_probs.append(probs.cpu()); all_targets.append(y.cpu())
    probs_val   = torch.cat(all_probs)
    targets_val = torch.cat(all_targets)

    from src.utils.metrics import pr_auc_score_fast, roc_auc_score_fast, binary_metrics
    from src.utils.thresholds import sweep_thresholds

    pr_auc  = pr_auc_score_fast(probs_val, targets_val)
    roc_auc = roc_auc_score_fast(probs_val, targets_val)
    best_thr, best_score = sweep_thresholds(probs_val, targets_val, metric="f1")
    mets_at_best = binary_metrics((probs_val>=best_thr).float(), targets_val.float(), thresh=0.5)

    # Save thresholds
    thresholds_path = str(Path(paths["metrics"]).with_name("thresholds.json"))
    with open(thresholds_path, "w") as f:
        json.dump({"best_threshold": best_thr, "metric": "f1", "score": best_score}, f, indent=2)

    # Append richer metrics
    with open(paths["metrics"], "w") as f:
        json.dump({
            **(metrics or {}),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_threshold": best_thr,
            "f1_at_best_thr": mets_at_best["f1"],
            "precision_at_best_thr": mets_at_best["precision"],
            "recall_at_best_thr": mets_at_best["recall"],
        }, f, indent=2)

    print(f"[VAL] PR-AUC {pr_auc:.4f} | ROC-AUC {roc_auc:.4f} | "
          f"best thr {best_thr:.2f} → "
          f"F1 {mets_at_best['f1']:.3f} "
          f"(P {mets_at_best['precision']:.3f}, R {mets_at_best['recall']:.3f})")

    print("Done.")
if __name__ == "__main__":
    main()
