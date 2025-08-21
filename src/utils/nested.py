# src/utils/nested.py
from typing import Any, Mapping

def get_nested(d: Mapping[str, Any], dotted: str, default=None):
    """
    Fetch nested values like 'hist.daily.mean' from nested dicts.
    Whitespace around parts is ignored.
    """
    cur: Any = d
    for part in dotted.split("."):
        key = part.strip()
        if not key:
            return default
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur
