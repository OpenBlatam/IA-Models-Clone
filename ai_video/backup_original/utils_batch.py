from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Callable, Any, Optional
import asyncio

from typing import Any, List, Dict, Optional
import logging
# --- Helpers batch para status ---
def batch_get_status(ids: List[str], use_onyx: bool, cache) -> (Dict[str, Any], List[str], Callable):
    """Obtiene los jobs en caché y la lista de IDs no cacheados."""
    cache_key = lambda rid: f"onyx:{rid}" if use_onyx else f"local:{rid}"
    cached = {rid: cache.get(cache_key(rid)) for rid in ids}
    uncached = [rid for rid, val in cached.items() if val is None]
    return cached, uncached, cache_key

async async def batch_fetch_status(uncached: List[str], use_onyx: bool, get_system, VIDEO_STATUS, cache, cache_key, max_concurrency: int = 10) -> Dict[str, Any]:
    """
    Obtiene el status de los jobs no cacheados, actualiza caché y retorna dict.
    Usa asyncio.Semaphore para limitar la concurrencia máxima.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    async def get_status(rid) -> Optional[Dict[str, Any]]:
        async with semaphore:
            try:
                if use_onyx:
                    system = await get_system()
                    status = await system.get_job_status(rid)
                else:
                    status = VIDEO_STATUS.get(rid)
                    if status:
                        status = {"request_id": rid, **status}
                if status and status.get("status") in ["completed", "failed"]:
                    cache[cache_key(rid)] = status
                return rid, {k: v for k, v in (status or {"error": "not found"}).items() if k in ("request_id", "status", "error")}
            except Exception as e:
                return rid, {"error": str(e)}
    if uncached:
        results = await asyncio.gather(*(get_status(rid) for rid in uncached), return_exceptions=True)
        return {rid: status for res in results if isinstance(res, tuple) for rid, status in [res]}
    return {}

def batch_serialize_status(cached: Dict[str, Any], fetched: Dict[str, Any]) -> Dict[str, Any]:
    """Serializa el resultado final de status batch."""
    statuses = {**{rid: {k: v for k, v in val.items() if k in ("request_id", "status", "error")} for rid, val in cached.items() if val is not None}, **fetched}
    return statuses

# --- Helpers batch para logs ---
def batch_get_logs(ids: List[str], use_onyx: bool, cache) -> (Dict[str, Any], List[str], Callable):
    """Obtiene los logs en caché y la lista de IDs no cacheados."""
    cache_key = lambda rid: f"onyx:{rid}" if use_onyx else f"local:{rid}"
    cached = {rid: cache.get(cache_key(rid)) for rid in ids}
    uncached = [rid for rid, val in cached.items() if val is None]
    return cached, uncached, cache_key

async async def batch_fetch_logs(uncached: List[str], use_onyx: bool, get_system, VIDEO_LOGS, cache, cache_key, max_concurrency: int = 10) -> Dict[str, Any]:
    """
    Obtiene los logs de los jobs no cacheados, actualiza caché y retorna dict.
    Usa asyncio.Semaphore para limitar la concurrencia máxima.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    async def get_logs(rid) -> Optional[Dict[str, Any]]:
        async with semaphore:
            try:
                if use_onyx:
                    system = await get_system()
                    logs = await system.get_job_logs(rid)
                else:
                    logs = VIDEO_LOGS.get(rid, [])
                if logs and any(l.get("event") in ["completed", "failed"] for l in logs):
                    cache[cache_key(rid)] = logs
                return rid, [{k: v for k, v in (log or {}).items() if k in ("timestamp", "event", "error")} for log in (logs or [])]
            except Exception as e:
                return rid, [{"error": str(e)}]
    if uncached:
        results = await asyncio.gather(*(get_logs(rid) for rid in uncached), return_exceptions=True)
        return {rid: logs for res in results if isinstance(res, tuple) for rid, logs in [res]}
    return {}

def batch_serialize_logs(cached: Dict[str, Any], fetched: Dict[str, Any]) -> Dict[str, Any]:
    """Serializa el resultado final de logs batch."""
    logs_dict = {**{rid: [{k: v for k, v in log.items() if k in ("timestamp", "event", "error")} for log in val] for rid, val in cached.items() if val is not None}, **fetched}
    return logs_dict 