


from typing import Any


def _get_by_path(d: Any, path: str) -> Any:
    """
    Supports dotted dict paths and numeric indices for lists:
      "hist.daily.mean", "hist.daily.hist.0"
    Root is the full item (not only item['static']).
    """
    cur = d
    for p in path.split('.'):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, (list, tuple)) and p.isdigit():
            idx = int(p)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                raise KeyError(f"Index out of range in path: {path}")
        else:
            raise KeyError(f"Missing key path: {path}")
    return cur