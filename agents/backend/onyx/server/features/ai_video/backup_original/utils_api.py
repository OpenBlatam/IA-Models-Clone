from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import HTTPException, Depends
from functools import wraps
from typing import Callable, List, Optional
import datetime

from typing import Any, List, Dict, Optional
import logging
import asyncio
# --- Constantes de producción ---
MAX_BATCH_IDS = 50
ERROR_INVALID_IDS = "IDs debe ser lista de strings (máx 50)"
ERROR_UNAUTHORIZED = "Unauthorized"

# --- Validación batch ---
def validate_batch_ids(ids: list) -> None:
    """Valida que ids sea una lista de strings y no supere el límite."""
    if not isinstance(ids, list) or len(ids) > MAX_BATCH_IDS or not all(isinstance(rid, str) for rid in ids):
        raise HTTPException(400, ERROR_INVALID_IDS)

# --- Decorador de autorización ---
def require_user(func: Callable) -> Callable:
    """Requiere usuario autenticado (vía Depends)."""
    @wraps(func)
    async def wrapper(*args, user=Depends(lambda: None), **kwargs):
        if not user:
            raise HTTPException(401, ERROR_UNAUTHORIZED)
        return await func(*args, user=user, **kwargs)
    return wrapper

# --- Decorador de logging estructurado y manejo de errores ---
def endpoint_protected(endpoint_name: str, logger, envelope):
    """
    Decorador desacoplado para logging estructurado y manejo de errores.
    Loggea inicio, éxito y error, siempre con trace_id y usuario si está disponible.
    Devuelve EnvelopeResponse consistente en caso de error, incluyendo trace_id y timestamp.
    Uso:
        @endpoint_protected("/video/status/batch", logger, envelope)
        async def endpoint(...) -> Any: ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, x_trace_id: Optional[str] = None, **kwargs):
            
    """wrapper function."""
trace_id = x_trace_id or "no-trace-id"
            user = kwargs.get("user") or (args[1] if len(args) > 1 else None)
            logger.info({"endpoint": endpoint_name, "event": "start", "trace_id": trace_id, "user": getattr(user, 'sub', None) if user else None})
            try:
                result = await func(*args, x_trace_id=trace_id, **kwargs)
                logger.info({"endpoint": endpoint_name, "event": "success", "trace_id": trace_id, "user": getattr(user, 'sub', None) if user else None})
                return result
            except Exception as e:
                logger.error({"endpoint": endpoint_name, "event": "error", "trace_id": trace_id, "user": getattr(user, 'sub', None) if user else None, "error": str(e)})
                return envelope(
                    success=False,
                    error={"message": str(e)},
                    data=None,
                    trace_id=trace_id,
                    timestamp=datetime.datetime.utcnow().isoformat() + "Z"
                )
        return wrapper
    return decorator 