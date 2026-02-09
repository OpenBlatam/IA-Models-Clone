"""
Endpoints para generación de canciones con IA

Este módulo proporciona endpoints para generar música usando modelos de IA.
Incluye generación desde prompts directos y desde conversaciones en lenguaje natural.

Características:
- Generación asíncrona en background
- Soporte para chat conversacional
- Validación robusta de inputs
- Caching inteligente de estados
- Manejo de errores completo
- Métricas y logging detallado
"""

import logging
import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, BackgroundTasks, status, Depends, Response, HTTPException, Query, Request

from ..schemas import ChatMessage, SongGenerationRequest, SongResponse, StatusResponse
from ..dependencies import SongServiceDep, ChatProcessorDep, MetricsServiceDep, NotificationServiceDep
from ..helpers import generate_song_id, create_song_info_from_request, notify_song_started
from ..business_logic import extract_song_info_from_chat
from ..validators import validate_song_id
from ..utils.response_cache import cache_response
from ..utils.request_helpers import get_request_metadata, add_cache_headers
from ..utils.performance_monitor import measure_time
from ..utils.decorators import log_request, measure_performance, rate_limit_decorator
from ...config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/generate",
    tags=["generation"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        429: {"description": "Too many requests - Rate limit excedido"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)


@router.post(
    "/chat/create-song",
    response_model=SongResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generar canción desde chat",
    description="Crea una canción desde un mensaje de chat en lenguaje natural usando procesamiento de lenguaje natural"
)
@log_request
@measure_performance
@rate_limit_decorator(max_requests=20, window_seconds=60)
async def create_song_from_chat(
    chat_message: ChatMessage,
    background_tasks: BackgroundTasks,
    request: Request,
    chat_processor: ChatProcessorDep = Depends(),
    notification_service: Optional[NotificationServiceDep] = Depends(),
    metrics_service: Optional[MetricsServiceDep] = Depends()
) -> SongResponse:
    """
    Crea una canción desde un mensaje de chat en lenguaje natural.
    
    El sistema procesa el mensaje del usuario usando NLP para extraer:
    - Descripción de la canción (prompt)
    - Género musical
    - Duración deseada
    - Estado de ánimo
    - Otros parámetros relevantes
    
    Args:
        chat_message: Mensaje del usuario con historial opcional
        background_tasks: Tareas en background de FastAPI
        chat_processor: Procesador de chat (inyectado)
        notification_service: Servicio de notificaciones (inyectado, opcional)
        metrics_service: Servicio de métricas (inyectado, opcional)
    
    Returns:
        SongResponse con song_id y estado "processing"
    
    Raises:
        HTTPException 400: Si el mensaje es inválido
        HTTPException 429: Si se excede el rate limit
        HTTPException 500: Si hay error en el procesamiento
    
    Example:
        ```json
        POST /suno/generate/chat/create-song
        {
            "message": "quiero una canción de rock energética de 30 segundos",
            "user_id": "user123",
            "chat_history": []
        }
        ```
    """
    try:
        start_time = time.time()
        
        # Log request metadata para debugging
        request_metadata = get_request_metadata(request)
        logger.debug(f"Request metadata: {request_metadata}")
        
        # Extraer información de la canción desde el chat
        with measure_time("extract_song_info"):
            song_info = extract_song_info_from_chat(
                chat_message.message,
                chat_message.chat_history or [],
                chat_processor
            )
        
        song_id = generate_song_id()
        
        # Importar aquí para evitar circular imports
        from ..background_tasks import generate_song_background
        
        # Agregar tarea en background
        background_tasks.add_task(
            generate_song_background,
            song_id,
            song_info,
            chat_message.user_id
        )
        
        # Notificar inicio (opcional)
        if notification_service:
            try:
                await notify_song_started(notification_service, chat_message.user_id, song_id)
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
        
        # Registrar métrica (opcional)
        if metrics_service:
            try:
                metrics_service.record_generation_start(
                    song_id=song_id,
                    user_id=chat_message.user_id,
                    source="chat",
                    prompt_length=len(chat_message.message)
                )
            except Exception as e:
                logger.warning(f"Failed to record metric: {e}")
        
        processing_time = time.time() - start_time
        logger.info(
            f"Song generation started from chat: {song_id} for user: {chat_message.user_id} "
            f"(processing time: {processing_time:.3f}s)"
        )
        
        return SongResponse(
            song_id=song_id,
            status="processing",
            message="Canción en proceso de generación",
            metadata={
                **song_info,
                "source": "chat",
                "processing_time": round(processing_time, 3)
            }
        )
    except ValueError as e:
        logger.warning(f"Invalid chat message: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid chat message: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error creating song from chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}"
        )


@router.post(
    "",
    response_model=SongResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generar canción desde prompt",
    description="Genera una canción directamente desde un prompt estructurado"
)
@log_request
@measure_performance
@rate_limit_decorator(max_requests=20, window_seconds=60)
async def generate_song(
    request: SongGenerationRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    notification_service: Optional[NotificationServiceDep] = Depends(),
    metrics_service: Optional[MetricsServiceDep] = Depends()
) -> SongResponse:
    """
    Genera una canción directamente desde un prompt estructurado.
    
    Este endpoint acepta parámetros específicos para la generación:
    - prompt: Descripción de la canción
    - duration: Duración en segundos (opcional)
    - genre: Género musical (opcional)
    - mood: Estado de ánimo (opcional)
    
    Args:
        request: Request con parámetros de generación
        background_tasks: Tareas en background de FastAPI
        notification_service: Servicio de notificaciones (inyectado, opcional)
        metrics_service: Servicio de métricas (inyectado, opcional)
    
    Returns:
        SongResponse con song_id y estado "processing"
    
    Raises:
        HTTPException 400: Si los parámetros son inválidos
        HTTPException 429: Si se excede el rate limit
        HTTPException 500: Si hay error en la generación
    
    Example:
        ```json
        POST /suno/generate
        {
            "prompt": "una canción de rock energética",
            "duration": 30,
            "genre": "rock",
            "mood": "energetic",
            "user_id": "user123"
        }
        ```
    """
    try:
        start_time = time.time()
        
        # Log request metadata
        request_metadata = get_request_metadata(http_request)
        logger.debug(f"Request metadata: {request_metadata}")
        
        song_id = generate_song_id()
        with measure_time("create_song_info"):
            song_info = create_song_info_from_request(request)
        
        # Importar aquí para evitar circular imports
        from ..background_tasks import generate_song_background
        
        # Agregar tarea en background
        background_tasks.add_task(
            generate_song_background,
            song_id,
            song_info,
            request.user_id
        )
        
        # Notificar inicio (opcional)
        if notification_service:
            try:
                await notify_song_started(notification_service, request.user_id, song_id)
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
        
        # Registrar métrica (opcional)
        if metrics_service:
            try:
                metrics_service.record_generation_start(
                    song_id=song_id,
                    user_id=request.user_id,
                    source="direct",
                    prompt_length=len(request.prompt),
                    duration=request.duration,
                    genre=request.genre
                )
            except Exception as e:
                logger.warning(f"Failed to record metric: {e}")
        
        processing_time = time.time() - start_time
        logger.info(
            f"Song generation started: {song_id} for user: {request.user_id} "
            f"(processing time: {processing_time:.3f}s)"
        )
        
        return SongResponse(
            song_id=song_id,
            status="processing",
            message="Canción en proceso de generación",
            metadata={
                **song_info,
                "source": "direct",
                "processing_time": round(processing_time, 3)
            }
        )
    except ValueError as e:
        logger.warning(f"Invalid generation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error generating song: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting generation: {str(e)}"
        )


@router.get(
    "/status/{task_id}",
    response_model=StatusResponse,
    summary="Obtener estado de generación",
    description="Obtiene el estado actual de una generación en proceso"
)
@cache_response(ttl=5)  # Cache corto para estados que cambian frecuentemente
@log_request
@rate_limit_decorator(max_requests=100, window_seconds=60)  # Más permisivo para status checks
async def get_generation_status(
    task_id: str,
    response: Optional[Response] = None,
    song_service: SongServiceDep = Depends(),
    include_progress: bool = Query(False, description="Incluir información de progreso")
) -> StatusResponse:
    """
    Obtiene el estado de una generación en proceso (optimizado).
    
    Usa caching corto para reducir carga en la base de datos.
    Soporta información de progreso opcional.
    
    Args:
        task_id: ID de la tarea de generación (UUID)
        response: Response object para headers (inyectado)
        song_service: Servicio de canciones (inyectado)
        include_progress: Incluir información detallada de progreso
    
    Returns:
        StatusResponse con estado actual de la generación
    
    Raises:
        HTTPException 400: Si el task_id es inválido
        HTTPException 404: Si la tarea no existe
    
    Example:
        ```
        GET /suno/generate/status/123e4567-e89b-12d3-a456-426614174000?include_progress=true
        ```
    """
    try:
        validate_song_id(task_id)
        
        # Intentar usar servicio async si está disponible
        try:
            from ..helpers.service_helpers import get_song_async_or_sync
            song = await get_song_async_or_sync(song_service, 'get_song', task_id)
        except (ImportError, AttributeError):
            # Fallback a servicio síncrono
            song = song_service.get_song(task_id)
        
        # Early return si no existe
        if not song:
            return StatusResponse(
                status="not_found",
                song_id=task_id,
                message="Task not found"
            )
        
        status_value = song.get("status", "unknown")
        message = song.get("message", "")
        
        # Agregar información de progreso si se solicita
        if include_progress:
            metadata = song.get("metadata", {})
            progress_info = {
                "progress_percentage": metadata.get("progress_percentage", 0),
                "estimated_time_remaining": metadata.get("estimated_time_remaining"),
                "current_step": metadata.get("current_step"),
                "total_steps": metadata.get("total_steps")
            }
            message = f"{message} | Progress: {progress_info}"
        
        # Headers de cache corto usando helper
        if response:
            add_cache_headers(response, max_age=5, public=True)
            response.headers["X-Generation-Status"] = status_value
        
        return StatusResponse(
            status=status_value,
            song_id=task_id,
            message=message
        )
    except ValueError as e:
        logger.warning(f"Invalid task_id format: {task_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid task ID format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting generation status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving status: {str(e)}"
        )


@router.get(
    "/batch-status",
    summary="Obtener estados de múltiples generaciones",
    description="Obtiene el estado de múltiples generaciones en una sola petición (optimizado)",
    response_model=Dict[str, Any]
)
@cache_response(ttl=5)  # Cache corto para batch status
async def get_batch_generation_status(
    task_ids: str = Query(..., description="IDs de tareas separados por coma (máx 50)"),
    response: Optional[Response] = None,
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene el estado de múltiples generaciones en una sola petición.
    
    Útil para monitorear múltiples generaciones simultáneas.
    
    Args:
        task_ids: IDs de tareas separados por coma (máximo 50)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Diccionario con estados de todas las tareas
    
    Raises:
        HTTPException 400: Si hay demasiados IDs o formato inválido
    
    Example:
        ```
        GET /suno/generate/batch-status?task_ids=id1,id2,id3
        ```
    """
    try:
        # Parsear y validar IDs usando helper optimizado
        from ..utils.validation_helpers import parse_comma_separated_ids
        from ..exceptions import InvalidInputError
        
        try:
            id_list = parse_comma_separated_ids(
                task_ids,
                max_items=50,
                field_name="task IDs"
            )
        except ValueError as e:
            raise InvalidInputError(str(e))
        
        # Obtener estados usando batch processing optimizado
        from ..utils.batch_processor import batch_get_songs
        
        songs = await batch_get_songs(song_service, id_list, max_concurrent=10)
        songs_dict = {song.get("song_id"): song for song in songs if song}
        
        statuses = {}
        for task_id in id_list:
            song = songs_dict.get(task_id)
            if song:
                statuses[task_id] = {
                    "status": song.get("status", "unknown"),
                    "message": song.get("message", ""),
                    "song_id": task_id
                }
            else:
                statuses[task_id] = {
                    "status": "not_found",
                    "message": "Task not found",
                    "song_id": task_id
                }
        
        # Headers de cache usando helper
        if response:
            add_cache_headers(response, max_age=5, public=True)
        
        found_count = len([s for s in statuses.values() if s["status"] != "not_found"])
        
        return {
            "total_requested": len(id_list),
            "found": found_count,
            "not_found": len(id_list) - found_count,
            "statuses": statuses
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving batch status: {str(e)}"
        )

