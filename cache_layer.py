# cache_layer.py
# Ultra-light JSON map cache with TTL (hours).
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _path(name: str) -> Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-", "."))
    return CACHE_DIR / f"{safe}.json"

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_map(name: str, ttl_hours: Optional[int] = None) -> Dict[str, Any]:
    p = _path(name)
    if not p.exists():
        return {}
    try:
        if ttl_hours is not None:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if datetime.now(timezone.utc) - mtime > timedelta(hours=ttl_hours):
                return {}
        return json.loads(p.read_text())
    except Exception:
        return {}

def save_map(name: str, obj: Dict[str, Any]) -> None:
    p = _path(name)
    try:
        p.write_text(json.dumps(obj, ensure_ascii=False))
    except Exception:
        pass

def get_cached(name: str, key: str) -> Any:
    m = load_map(name, ttl_hours=None)
    return m.get(key)

def set_cached(name: str, key: str, value: Any, autosave: bool = True) -> None:
    m = load_map(name, ttl_hours=None)
    m[key] = value
    if autosave:
        save_map(name, m)
