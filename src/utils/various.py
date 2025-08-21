from typing import Iterator, Dict, Any, List, Optional, Set, Union
import math


def _tod_features(snip: Dict[str, Any]) -> Dict[str, float]:
    """Compute tod_sin/tod_cos from HH:MM (snip['buyTime'] or snip['meta']['buyTime'])."""
    bt = snip.get("buyTime")
    if isinstance(snip.get("meta"), dict):
        bt = snip["meta"].get("buyTime", bt)

    mins = None
    if isinstance(bt, str) and ":" in bt:
        try:
            h, m = bt.strip().split(":")
            mins = int(h) * 60 + int(m)
        except Exception:
            mins = None

    if mins is None:
        return {"tod_sin": 0.0, "tod_cos": 1.0}
    ang = 2 * math.pi * (mins / 1440.0)
    return {"tod_sin": math.sin(ang), "tod_cos": math.cos(ang)}