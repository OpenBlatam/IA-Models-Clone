import json
import os
import time
from typing import Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("truthgpt_collected/cache")
CACHE_FILE = CACHE_DIR / "semantic_cache.json"

def _load_semantic_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}

SEMANTIC_CACHE = _load_semantic_cache()

def _save_semantic_cache(cache_data: dict):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache_data, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass

def get_cached_response(prompt: str) -> Optional[str]:
    import hashlib
    h = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    if h in SEMANTIC_CACHE:
        entry = SEMANTIC_CACHE[h]
        if time.time() - entry.get("ts", 0) < 86400 * 3: # 3 days validity
            return entry.get("response")
    return None

def set_cached_response(prompt: str, response: str):
    import hashlib
    h = hashlib.md5(prompt.encode('utf-8')).hexdigest()
    SEMANTIC_CACHE[h] = {"response": response, "ts": time.time()}
    _save_semantic_cache(SEMANTIC_CACHE)