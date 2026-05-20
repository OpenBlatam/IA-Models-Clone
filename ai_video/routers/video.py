from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, Body, Query, Header, Request, status
from typing import List, Optional
from ..models import EnvelopeResponse, VideoRequestInput
from ..services import video_service, batch_service
from ..auth import get_current_user
from ..utils_api import endpoint_protected

from typing import Any, List, Dict, Optional
import logging
import asyncio
logger = logging.getLogger("video_router")

video_router = APIRouter(prefix="/video", tags=["Video"])

@video_router.post(
    "/",
    response_model=EnvelopeResponse,
    response_model_exclude_unset=True,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Solicita la generación de un video AI (local o Onyx)",
    responses={
        202: {"content": {"application/json": {"example": {"success": True, "data": {"request_id": "req_123", "status": "queued"}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        400: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Validation error", "details": {"duration": ["Duración fuera de rango"]}}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        422: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Validation error", "details": {"quality": ["quality debe ser 'low', 'medium' o 'high'"], "input_text": ["ensure this value has at least 1 characters"]}}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        500: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Internal server error"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}}
    },
)
@endpoint_protected("/video", logger, EnvelopeResponse)
async def create_video(
    body: VideoRequestInput = Body(...),
    user=Depends(get_current_user),
    x_request_id: Optional[str] = Header(None, description="ID único opcional para la request"),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    x_span_id: Optional[str] = Header(None, description="ID de span para tracing"),
    use_onyx: bool = Query(False, description="Si es True, delega a Onyx; si es False, usa el microservicio local."),
    request: Request = None
) -> EnvelopeResponse:
    """
    Crea un video AI usando Onyx o el flujo local.
    - Valida duración, calidad, input_text.
    - Devuelve EnvelopeResponse con trace_id y timestamp.
    - Errores posibles: validación, integración Onyx, error interno.
    """
    return await video_service.create_video(body, user, use_onyx, x_request_id=x_request_id, x_trace_id=x_trace_id, x_span_id=x_span_id)

@video_router.get("/{request_id}/status",
    response_model=EnvelopeResponse,
    response_model_exclude_unset=True,
    summary="Consulta el estado de un video por request_id",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"request_id": "req_123", "status": "processing", "progress": 0.5}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        404: {"content": {"application/json": {"example": {"success": False, "error": {"message": "No existe el request_id"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        500: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Internal server error"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}}
    })
@endpoint_protected("/video/{request_id}/status", logger, EnvelopeResponse)
async def get_video_status(
    request_id: str,
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    user=Depends(get_current_user)
) -> EnvelopeResponse:
    """
    Consulta el estado de un video AI por su request_id.
    - Devuelve progreso, estado y trace_id.
    - Errores: no existe el request_id, error interno.
    """
    return await video_service.get_status(request_id, user, x_trace_id)

@video_router.get("/{request_id}/logs",
    response_model=EnvelopeResponse,
    response_model_exclude_unset=True,
    summary="Obtiene los logs de procesamiento de un video",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"logs": ["Started", "Processing", "Done"]}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        404: {"content": {"application/json": {"example": {"success": False, "error": {"message": "No existe el request_id"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        500: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Internal server error"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}}
    })
@endpoint_protected("/video/{request_id}/logs", logger, EnvelopeResponse)
async def get_video_logs(
    request_id: str,
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    user=Depends(get_current_user)
) -> EnvelopeResponse:
    """
    Obtiene los logs de procesamiento de un video AI por su request_id.
    - Devuelve lista de logs y trace_id.
    - Errores: no existe el request_id, error interno.
    """
    return await video_service.get_logs(request_id, user, x_trace_id)

@video_router.post("/{request_id}/cancel",
    response_model=EnvelopeResponse,
    response_model_exclude_unset=True,
    summary="Cancela un trabajo de video AI en curso",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"request_id": "req_123", "status": "cancelled"}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        404: {"content": {"application/json": {"example": {"success": False, "error": {"message": "No existe el request_id"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        409: {"content": {"application/json": {"example": {"success": False, "error": {"message": "No se puede cancelar un trabajo finalizado"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        500: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Internal server error"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}}
    })
@endpoint_protected("/video/{request_id}/cancel", logger, EnvelopeResponse)
async def cancel_video(
    request_id: str,
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    user=Depends(get_current_user)
) -> EnvelopeResponse:
    """
    Cancela un trabajo de video AI en curso por su request_id.
    - Devuelve estado actualizado y trace_id.
    - Errores: no existe el request_id, ya finalizado, error interno.
    """
    return await video_service.cancel(request_id, user, x_trace_id)

@video_router.post("/{request_id}/retry",
    response_model=EnvelopeResponse,
    response_model_exclude_unset=True,
    summary="Reintenta un trabajo de video AI fallido",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"request_id": "req_123", "status": "queued"}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        404: {"content": {"application/json": {"example": {"success": False, "error": {"message": "No existe el request_id"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        409: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Solo se puede reintentar trabajos fallidos"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        500: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Internal server error"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}}
    })
@endpoint_protected("/video/{request_id}/retry", logger, EnvelopeResponse)
async def retry_video(
    request_id: str,
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    user=Depends(get_current_user)
) -> EnvelopeResponse:
    """
    Reintenta un trabajo de video AI fallido por su request_id.
    - Devuelve estado actualizado y trace_id.
    - Errores: no existe el request_id, no es fallido, error interno.
    """
    return await video_service.retry(request_id, user, x_trace_id)

@video_router.post("/status/batch",
    response_model=EnvelopeResponse,
    response_model_exclude_unset=True,
    summary="Consulta el estado de múltiples videos por batch",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"statuses": [{"request_id": "req_1", "status": "done"}, {"request_id": "req_2", "status": "processing"}]}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        400: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Lista de request_ids vacía"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        500: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Internal server error"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}}
    })
@endpoint_protected("/video/status/batch", logger, EnvelopeResponse)
async def batch_status(
    request_ids: List[str] = Body(..., embed=True, description="Lista de request_ids a consultar"),
    max_concurrency: int = Query(10, ge=1, le=50, description="Máximo de tareas concurrentes para el batch (default 10, máximo 50)"),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    user=Depends(get_current_user)
) -> EnvelopeResponse:
    """
    Consulta el estado de múltiples videos AI por batch, procesando en paralelo hasta max_concurrency tareas.
    - Devuelve lista de estados y trace_id.
    - Errores: lista vacía, error interno.
    """
    return await batch_service.status(request_ids, user, x_trace_id, max_concurrency=max_concurrency)

@video_router.post("/logs/batch",
    response_model=EnvelopeResponse,
    response_model_exclude_unset=True,
    summary="Obtiene los logs de múltiples videos por batch",
    responses={
        200: {"content": {"application/json": {"example": {"success": True, "data": {"logs": [{"request_id": "req_1", "logs": ["Started", "Done"]}, {"request_id": "req_2", "logs": ["Started", "Processing"]}]}, "error": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        400: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Lista de request_ids vacía"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}},
        500: {"content": {"application/json": {"example": {"success": False, "error": {"message": "Internal server error"}, "data": None, "timestamp": "2024-05-01T12:00:00Z", "trace_id": "abc-123"}}}}
    })
@endpoint_protected("/video/logs/batch", logger, EnvelopeResponse)
async def batch_logs(
    request_ids: List[str] = Body(..., embed=True, description="Lista de request_ids a consultar"),
    max_concurrency: int = Query(10, ge=1, le=50, description="Máximo de tareas concurrentes para el batch (default 10, máximo 50)"),
    x_trace_id: Optional[str] = Header(None, description="ID de trazabilidad distribuida"),
    user=Depends(get_current_user)
) -> EnvelopeResponse:
    """
    Obtiene los logs de múltiples videos AI por batch, procesando en paralelo hasta max_concurrency tareas.
    - Devuelve lista de logs y trace_id.
    - Errores: lista vacía, error interno.
    """
    return await batch_service.logs(request_ids, user, x_trace_id, max_concurrency=max_concurrency) 