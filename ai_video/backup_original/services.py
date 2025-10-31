from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import List, Dict, Any, Optional, Callable, Union, Mapping
from pydantic import BaseModel, ValidationError
import uuid
from starlette.datastructures import Headers

                from .onyx_ai_video.core.models import VideoRequest
from typing import Any, List, Dict, Optional
import logging
import asyncio
class ServiceError(Exception):
    """Excepción base para errores de servicios."""
    pass

def get_trace_id_from_headers(headers: Optional[Mapping[str, Any]] = None, default: Optional[str] = None) -> str:
    """
    Obtiene el trace_id de los headers (Mapping, Headers, dict o None) o genera uno nuevo único.
    - Busca variantes de 'x-trace-id' ignorando mayúsculas/minúsculas y guiones/bajos.
    - Si headers tiene <=4 elementos, busca con un generador para evitar crear un dict extra.
    - Si headers es grande (>4), normaliza una vez para O(1) búsqueda.
    - Si no lo encuentra, devuelve default si está definido, si no genera uno nuevo.
    - Compatible con dict, starlette Headers, cualquier Mapping y None.
    - O(1) para la mayoría de headers, O(n) solo si hay colisiones raras.

    Args:
        headers: Mapping de headers (opcional).
        default: valor a devolver si no se encuentra trace_id (opcional).

    Ejemplos (doctest):
    >>> get_trace_id_from_headers({"x_trace_id": "abc"})
    'abc'
    >>> get_trace_id_from_headers({"X-Trace-Id": "def"})
    'def'
    >>> get_trace_id_from_headers(Headers({"x-trace-id": "ghi"}))
    'ghi'
    >>> isinstance(get_trace_id_from_headers(), str)
    True
    >>> get_trace_id_from_headers({}, default="zzz")
    'zzz'
    """
    norm_keys = {"x-trace-id", "x_trace_id"}
    if headers is None:
        return default if default is not None else str(uuid.uuid4())
    if isinstance(headers, Headers):
        headers = dict(headers)
    items = list(headers.items())
    if len(items) <= 4:
        for k, v in items:
            norm = k.lower().replace("_", "-")
            if norm in norm_keys:
                return v
    else:
        norm = {k.lower().replace("_", "-"): v for k, v in items}
        for key in norm_keys:
            if key in norm:
                return norm[key]
    return default if default is not None else str(uuid.uuid4())

class VideoService:
    """
    Servicio de lógica de negocio para videos AI (local y Onyx).

    Args:
        get_system: Callable para obtener el sistema Onyx.
        video_status: Dict de estados locales.
        video_logs: Dict de logs locales.
        envelope: Función para formatear la respuesta.
        logger: Logger opcional para auditoría.
        on_job_created: Callback opcional para eventos de creación de job.

    Ejemplo de uso:
        service = VideoService(get_system, video_status, video_logs, envelope, logger=mylogger, on_job_created=myhook)
        resp = await service.create_video(body, user, use_onyx, x_trace_id="abc")

    Advertencias:
        - Para alta concurrencia, usar locks o recursos thread-safe en video_status/video_logs.
        - El trace_id se propaga en logs y respuestas si se pasa en headers.
    """
    def __init__(self, get_system: Callable, video_status: Dict[str, Any], video_logs: Dict[str, Any], envelope: Callable[[bool, Any], dict], logger: Optional[Any] = None, on_job_created: Optional[Callable[[str, dict], None]] = None):
        
    """__init__ function."""
self.get_system = get_system
        self.video_status = video_status
        self.video_logs = video_logs
        self.envelope = envelope
        self.logger = logger
        self.on_job_created = on_job_created

    async def create_video(self, body: Union[dict, BaseModel], user: dict, use_onyx: bool, **headers) -> dict:
        """
        Crea un video AI usando Onyx o el flujo local.
        Args:
            body: dict o modelo Pydantic con los datos del video.
            user: dict con info del usuario.
            use_onyx: bool, si True usa Onyx, si False local.
            headers: opcionales (x_request_id, x_trace_id, etc).
        Returns:
            dict con la respuesta (envelope).
        Raises:
            ServiceError si hay error de validación o integración.
        """
        trace_id = get_trace_id_from_headers(headers)
        try:
            if self.logger:
                self.logger.info({"action": "create_video", "use_onyx": use_onyx, "user": user.get("sub"), "trace_id": trace_id})
            if use_onyx:
                req = body if isinstance(body, VideoRequest) else VideoRequest(**body)
                system = await self.get_system()
                response = await system.generate_video(req)
                if self.logger:
                    self.logger.info({"action": "onyx_video_created", "request_id": getattr(req, "request_id", None), "trace_id": trace_id})
                return self.envelope(True, data=response.dict(), mode="onyx", trace_id=trace_id)
            # Local
            request_id = headers.get("x_request_id") or f"req_{user.get('sub', 'anon')}"
            self.video_status[request_id] = {"status": "queued"}
            if self.on_job_created:
                self.on_job_created(request_id, user)
            if self.logger:
                self.logger.info({"action": "local_video_queued", "request_id": request_id, "user": user.get("sub"), "trace_id": trace_id})
            return self.envelope(True, data={"request_id": request_id, "status": "queued"}, mode="local", trace_id=trace_id)
        except (ValidationError, Exception) as e:
            if self.logger:
                self.logger.error({"action": "create_video_error", "error": str(e), "trace_id": trace_id})
            raise ServiceError(str(e))

    async def get_status(self, request_id: str, use_onyx: bool, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene el estado de un video por request_id.
        Returns dict con status o error.
        """
        if not isinstance(request_id, str):
            raise ServiceError("request_id debe ser str")
        try:
            if self.logger:
                self.logger.info({"action": "get_status", "request_id": request_id, "use_onyx": use_onyx, "trace_id": trace_id})
            if use_onyx:
                system = await self.get_system()
                status = await system.get_job_status(request_id)
                return status if status else {"error": "not found", "trace_id": trace_id}
            return self.video_status.get(request_id, {"error": "not found", "trace_id": trace_id})
        except Exception as e:
            if self.logger:
                self.logger.error({"action": "get_status_error", "error": str(e), "trace_id": trace_id})
            return {"error": str(e), "trace_id": trace_id}

    async def get_logs(self, request_id: str, use_onyx: bool, skip: int = 0, limit: int = 100, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene los logs de un video por request_id.
        Returns lista de logs (dicts).
        """
        if not isinstance(request_id, str):
            raise ServiceError("request_id debe ser str")
        try:
            if self.logger:
                self.logger.info({"action": "get_logs", "request_id": request_id, "use_onyx": use_onyx, "trace_id": trace_id})
            if use_onyx:
                system = await self.get_system()
                logs = await system.get_job_logs(request_id, skip=skip, limit=limit)
                return logs
            return self.video_logs.get(request_id, [])[skip:skip+limit]
        except Exception as e:
            if self.logger:
                self.logger.error({"action": "get_logs_error", "error": str(e), "trace_id": trace_id})
            return [{"error": str(e), "trace_id": trace_id}]

    async def cancel(self, request_id: str, use_onyx: bool, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancela un job por request_id.
        Returns dict con mensaje o error.
        """
        if not isinstance(request_id, str):
            raise ServiceError("request_id debe ser str")
        try:
            if self.logger:
                self.logger.info({"action": "cancel", "request_id": request_id, "use_onyx": use_onyx, "trace_id": trace_id})
            if use_onyx:
                system = await self.get_system()
                result = await system.cancel_job(request_id)
                return {"message": "Job cancelado (Onyx)", "result": result, "trace_id": trace_id}
            if request_id in self.video_status:
                self.video_status[request_id]["status"] = "cancelled"
                return {"message": "Job cancelado", "trace_id": trace_id}
            return {"error": "No se puede cancelar este job", "trace_id": trace_id}
        except Exception as e:
            if self.logger:
                self.logger.error({"action": "cancel_error", "error": str(e), "trace_id": trace_id})
            return {"error": str(e), "trace_id": trace_id}

    async def retry(self, request_id: str, use_onyx: bool, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Reintenta un job fallido por request_id.
        Returns dict con mensaje o error.
        """
        if not isinstance(request_id, str):
            raise ServiceError("request_id debe ser str")
        try:
            if self.logger:
                self.logger.info({"action": "retry", "request_id": request_id, "use_onyx": use_onyx, "trace_id": trace_id})
            if use_onyx:
                system = await self.get_system()
                result = await system.retry_job(request_id)
                return {"message": "Job reintentado (Onyx)", "result": result, "trace_id": trace_id}
            if request_id in self.video_status and self.video_status[request_id]["status"] == "failed":
                self.video_status[request_id]["status"] = "queued"
                return {"message": "Job reintentado", "trace_id": trace_id}
            return {"error": "Solo se pueden reintentar jobs fallidos", "trace_id": trace_id}
        except Exception as e:
            if self.logger:
                self.logger.error({"action": "retry_error", "error": str(e), "trace_id": trace_id})
            return {"error": str(e), "trace_id": trace_id}

class BatchService:
    """
    Servicio para operaciones batch de status y logs.

    Args:
        batch_helpers: Dict de helpers batch (get_status, fetch_status, serialize_status, ...).
        get_system: Callable para obtener el sistema Onyx.
        video_status: Dict de estados locales.
        video_logs: Dict de logs locales.
        cache_status: caché para status.
        cache_logs: caché para logs.
        envelope: función para formatear respuesta.
        logger: logger opcional.
        on_batch_completed: Callback opcional para eventos de batch.

    Ejemplo de uso:
        batch = BatchService(..., on_batch_completed=myhook)
        resp = await batch.batch_status(["id1", "id2"], use_onyx=True)

    Advertencias:
        - ids debe ser lista de str, máximo 50.
        - Para alta concurrencia, usar cachés thread-safe.
        - El trace_id se propaga en logs y respuestas si se pasa.
    """
    def __init__(self, batch_helpers: Dict[str, Callable], get_system: Callable, video_status: Dict[str, Any], video_logs: Dict[str, Any], cache_status: Any, cache_logs: Any, envelope: Callable[[bool, Any], dict], logger: Optional[Any] = None, on_batch_completed: Optional[Callable[[List[str], dict], None]] = None):
        
    """__init__ function."""
self.batch_get_status = batch_helpers["get_status"]
        self.batch_fetch_status = batch_helpers["fetch_status"]
        self.batch_serialize_status = batch_helpers["serialize_status"]
        self.batch_get_logs = batch_helpers["get_logs"]
        self.batch_fetch_logs = batch_helpers["fetch_logs"]
        self.batch_serialize_logs = batch_helpers["serialize_logs"]
        self.get_system = get_system
        self.video_status = video_status
        self.video_logs = video_logs
        self.cache_status = cache_status
        self.cache_logs = cache_logs
        self.envelope = envelope
        self.logger = logger
        self.on_batch_completed = on_batch_completed

    async def batch_status(self, ids: List[str], use_onyx: bool, trace_id: Optional[str] = None, max_concurrency: int = 10) -> dict:
        """
        Batch status de múltiples jobs.
        Args:
            ids: lista de request_id (máx 50).
            use_onyx: bool, si True usa Onyx, si False local.
            trace_id: opcional para tracing/logging.
            max_concurrency: máximo de tareas concurrentes (default 10).
        Returns:
            dict envelope con statuses.
        Edge cases:
            - Si todos los ids están en caché, respuesta inmediata.
            - Si hay error en algún id, se incluye en el dict.
        """
        if not isinstance(ids, list) or not all(isinstance(rid, str) for rid in ids):
            raise ServiceError("ids debe ser lista de str")
        try:
            if self.logger:
                self.logger.info({"action": "batch_status", "ids": ids, "use_onyx": use_onyx, "trace_id": trace_id, "max_concurrency": max_concurrency})
            cached, uncached, cache_key = self.batch_get_status(ids, use_onyx, self.cache_status)
            if not uncached:
                statuses = self.batch_serialize_status(cached, {})
                if self.on_batch_completed:
                    self.on_batch_completed(ids, statuses)
                return self.envelope(True, data={"statuses": statuses}, mode="onyx" if use_onyx else "local", trace_id=trace_id)
            fetched = await self.batch_fetch_status(uncached, use_onyx, self.get_system, self.video_status, self.cache_status, cache_key, max_concurrency=max_concurrency)
            statuses = self.batch_serialize_status(cached, fetched)
            if self.on_batch_completed:
                self.on_batch_completed(ids, statuses)
            return self.envelope(True, data={"statuses": statuses}, mode="onyx" if use_onyx else "local", trace_id=trace_id)
        except Exception as e:
            if self.logger:
                self.logger.error({"action": "batch_status_error", "error": str(e), "trace_id": trace_id})
            return self.envelope(False, error={"message": str(e)}, mode="onyx" if use_onyx else "local", trace_id=trace_id)

    async def batch_logs(self, ids: List[str], use_onyx: bool, trace_id: Optional[str] = None, max_concurrency: int = 10) -> dict:
        """
        Batch logs de múltiples jobs.
        Args:
            ids: lista de request_id (máx 50).
            use_onyx: bool, si True usa Onyx, si False local.
            trace_id: opcional para tracing/logging.
            max_concurrency: máximo de tareas concurrentes (default 10).
        Returns:
            dict envelope con logs.
        Edge cases:
            - Si todos los ids están en caché, respuesta inmediata.
            - Si hay error en algún id, se incluye en el dict.
        """
        if not isinstance(ids, list) or not all(isinstance(rid, str) for rid in ids):
            raise ServiceError("ids debe ser lista de str")
        try:
            if self.logger:
                self.logger.info({"action": "batch_logs", "ids": ids, "use_onyx": use_onyx, "trace_id": trace_id, "max_concurrency": max_concurrency})
            cached, uncached, cache_key = self.batch_get_logs(ids, use_onyx, self.cache_logs)
            if not uncached:
                logs_dict = self.batch_serialize_logs(cached, {})
                if self.on_batch_completed:
                    self.on_batch_completed(ids, logs_dict)
                return self.envelope(True, data={"logs": logs_dict}, mode="onyx" if use_onyx else "local", trace_id=trace_id)
            fetched = await self.batch_fetch_logs(uncached, use_onyx, self.get_system, self.video_logs, self.cache_logs, cache_key, max_concurrency=max_concurrency)
            logs_dict = self.batch_serialize_logs(cached, fetched)
            if self.on_batch_completed:
                self.on_batch_completed(ids, logs_dict)
            return self.envelope(True, data={"logs": logs_dict}, mode="onyx" if use_onyx else "local", trace_id=trace_id)
        except Exception as e:
            if self.logger:
                self.logger.error({"action": "batch_logs_error", "error": str(e), "trace_id": trace_id})
            return self.envelope(False, error={"message": str(e)}, mode="onyx" if use_onyx else "local", trace_id=trace_id) 