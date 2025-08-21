# src/data_analysis.py
from pathlib import Path
import argparse, time, ijson
import argparse

from typing import List, Dict, Any


# def label_dist_from_json(json_path: str, label_key: str = "2_2"):
#     path = Path(json_path)
#     pos = neg = tot = 0
#     with path.open("r", encoding="utf-8") as f:
#         for obj in ijson.items(f, "item"):
#             v = obj.get("y", {}).get(label_key, None)
#             if v is None:
#                 continue
#             tot += 1
#             if float(v) >= 0.5: pos += 1
#             else:               neg += 1
#     pos_frac = (pos / tot) if tot else 0.0
#     print(f"{path.name}: total={tot}  pos(1)={pos} ({pos_frac:.1%})  neg(0)={neg} ({1-pos_frac:.1%})")
#     return {"total": tot, "pos": pos, "neg": neg, "pos_frac": pos_frac, "neg_frac": 1 - pos_frac}

# def run_once(train="data/processed/train_min.json",
#              test="data/processed/test_min.json",
#              label_key="2_2"):
#     label_dist_from_json(train, label_key)
#     label_dist_from_json(test,  label_key)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train", default="data/processed/train_min.json")
#     ap.add_argument("--test",  default="data/processed/test_min.json")
#     ap.add_argument("--label-key", default="2_2")
#     ap.add_argument("--watch", action="store_true", help="repeat every N seconds")
#     ap.add_argument("--every", type=float, default=15.0)
#     args = ap.parse_args()

#     if not args.watch:
#         run_once(args.train, args.test, args.label_key)
#         return

#     try:
#         while True:
#             run_once(args.train, args.test, args.label_key)
#             time.sleep(args.every)
#     except KeyboardInterrupt:
#         pass

# if __name__ == "__main__":
#     main()

RAW_DIR01 = Path("data/raw")
from typing import Iterator, Dict, Any, Optional, Set, Union, List

def iter_snips01(name_or_path: str) -> Iterator[Dict[str, Any]]:
    p = Path(name_or_path)
    path = p if p.exists() else RAW_DIR01 / name_or_path
    if not path.exists():
        raise FileNotFoundError(f"Could not find {name_or_path} or {path}")
    with path.open("r", encoding="utf-8") as f:
        for obj in ijson.items(f, "item", use_float=True):
            yield obj


def label_dists_from_raw(
    filename: str,
    label_keys: List[str] = None,
    y_field: str = "y",
) -> Dict[str, Dict[str, int]]:
    """
    Streams the raw file and counts TRUE/FALSE per label.
    Supports both:
      - top-level labels: obj["2_2"], obj["5_5"], ...
      - nested labels:    obj["y"]["2_2"], ...
    Treats >= 0.5 as positive (1).
    """
    if label_keys is None:
        label_keys = ["2_2", "5_5", "8_8", "2_5", "5_2", "2_1"]

    stats = {k: {"pos": 0, "neg": 0, "total": 0} for k in label_keys}
    total_rows = 0
    rows_with_any_label = 0

    for obj in iter_snips01(filename):  # keep your iterator
        total_rows += 1
        found_this_row = False

        # optional nested dict
        ydict = obj.get(y_field, None)
        ydict = ydict if isinstance(ydict, dict) else None

        for k in label_keys:
            v = obj.get(k, None)
            if v is None and ydict is not None:
                v = ydict.get(k, None)
            if v is None:
                continue

            found_this_row = True
            stats[k]["total"] += 1
            if float(v) >= 0.5:
                stats[k]["pos"] += 1
            else:
                stats[k]["neg"] += 1

        if found_this_row:
            rows_with_any_label += 1

    print(f"Scanned rows: {total_rows} | rows with any label: {rows_with_any_label}")
    for k in label_keys:
        s = stats[k]
        if s["total"] == 0:
            print(f"{k:>4}: total=0  pos(1)=0 (0.0%)  neg(0)=0 (0.0%)")
        else:
            pos_frac = s["pos"] / s["total"]
            neg_frac = s["neg"] / s["total"]
            print(
                f"{k:>4}: total={s['total']:6d}  pos(1)={s['pos']:6d} ({pos_frac:.1%})  "
                f"neg(0)={s['neg']:6d} ({neg_frac:.1%})"
            )
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="validationSNIPS.json", help="raw filename under data/raw/")
    ap.add_argument("--keys", nargs="*", default=["2_2", "5_5", "8_8", "2_5", "5_2"])
    args = ap.parse_args()
    label_dists_from_raw(args.raw, args.keys)

if __name__ == "__main__":
    main()