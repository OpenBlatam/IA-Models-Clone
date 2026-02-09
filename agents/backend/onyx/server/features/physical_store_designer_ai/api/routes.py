"""
API Routes for Physical Store Designer AI
"""

from fastapi import APIRouter, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from ..core.models import (
    StoreDesignRequest,
    StoreDesign,
    ChatMessage,
    ChatSession
)
from ..core.exceptions import (
    NotFoundError,
    ValidationError,
    StorageError
)
from ..core.logging_config import get_logger
from ..core.factories import ServiceFactory
from ..core.route_helpers import handle_route_errors, track_route_metrics

logger = get_logger(__name__)

router = APIRouter()

# Inicializar servicios usando factory
store_designer_service = ServiceFactory.get_designer_service()
chat_service = ServiceFactory.get_chat_service()
storage_service = ServiceFactory.get_storage_service()

# Almacenamiento temporal en memoria (backup)
designs_store: dict[str, StoreDesign] = {}


@router.post("/chat/session", response_model=dict)
@handle_route_errors
@track_route_metrics("chat.create_session")
async def create_chat_session():
    """Crear nueva sesión de chat"""
    session_id = chat_service.create_session()
    session = chat_service.get_session(session_id)
    return {
        "session_id": session_id,
        "messages": [msg.dict() for msg in session.messages] if session else []
    }


@router.post("/chat/{session_id}/message", response_model=dict)
@handle_route_errors
@track_route_metrics("chat.send_message")
async def send_chat_message(session_id: str, message: ChatMessage):
    """
    Enviar mensaje en el chat.
    
    Args:
        session_id: ID de la sesión de chat (debe ser UUID válido)
        message: Mensaje del usuario con contenido y rol
        
    Returns:
        Dict con respuesta de la IA, mensajes de la sesión e información del store
        
    Raises:
        NotFoundError: Si la sesión no existe
        ValidationError: Si el session_id o message no son válidos
    """
    from ..core.validators import Validator
    
    # Validar session_id
    if not Validator.validate_session_id(session_id):
        raise ValidationError(
            "Invalid session ID format",
            details={"session_id": session_id, "expected": "UUID format"}
        )
    
    # Validar contenido del mensaje
    if not message.content or not Validator.validate_string_length(message.content, min_len=1, max_len=5000):
        raise ValidationError(
            "Message content must be between 1 and 5000 characters",
            details={"content_length": len(message.content) if message.content else 0}
        )
    
    response = await chat_service.generate_response(session_id, message.content)
    session = chat_service.get_session(session_id)
    if not session:
        raise NotFoundError("Chat session", session_id)
    return {
        "session_id": session_id,
        "response": response,
        "messages": [msg.dict() for msg in session.messages],
        "store_info": session.store_info
    }


@router.get("/chat/{session_id}", response_model=dict)
@handle_route_errors
@track_route_metrics("chat.get_session")
async def get_chat_session(session_id: str):
    """
    Obtener sesión de chat.
    
    Args:
        session_id: ID de la sesión de chat (debe ser UUID válido)
        
    Returns:
        Dict con información de la sesión, mensajes e información del store
        
    Raises:
        NotFoundError: Si la sesión no existe
        ValidationError: Si el session_id no es válido
    """
    from ..core.validators import Validator
    
    # Validar session_id
    if not Validator.validate_session_id(session_id):
        raise ValidationError(
            "Invalid session ID format",
            details={"session_id": session_id, "expected": "UUID format"}
        )
    
    session = chat_service.get_session(session_id)
    if not session:
        raise NotFoundError("Chat session", session_id)
    return {
        "session_id": session_id,
        "messages": [msg.dict() for msg in session.messages],
        "store_info": session.store_info
    }


@router.post("/design/generate", response_model=StoreDesign)
@handle_route_errors
@track_route_metrics("store_designer.generate")
async def generate_store_design(request: StoreDesignRequest):
    """
    Generar diseño completo del local.
    
    Args:
        request: Request con información del store (nombre, tipo, estilo, presupuesto, etc.)
        
    Returns:
        StoreDesign completo con layout, marketing plan, decoration plan, etc.
        
    Raises:
        ValidationError: Si los datos del request no son válidos
        StorageError: Si hay error guardando el diseño (no crítico, solo loguea)
    """
    from ..core.validators import Validator
    
    # Validar dimensiones si están presentes
    if request.dimensions:
        if not Validator.validate_dimensions(request.dimensions):
            raise ValidationError(
                "Invalid dimensions. Must include width, length, and height (all positive numbers)",
                details={"dimensions": request.dimensions}
            )
    
    # Validar presupuesto si está presente
    if request.budget_range and not Validator.validate_budget_range(request.budget_range):
        raise ValidationError(
            "Invalid budget range. Must be one of: bajo, medio, alto, premium",
            details={"budget_range": request.budget_range}
        )
    
    # Validar nombre del store
    if not request.store_name or not Validator.validate_string_length(request.store_name, min_len=1, max_len=200):
        raise ValidationError(
            "Store name must be between 1 and 200 characters",
            details={"store_name": request.store_name}
        )
    
    design = await store_designer_service.generate_store_design(request)
    designs_store[design.store_id] = design
    # Guardar en almacenamiento persistente (no crítico si falla)
    try:
        storage_service.save_design(design)
    except StorageError as e:
        logger.warning(f"Error guardando diseño en storage: {e.message}", extra={"store_id": design.store_id})
    except Exception as e:
        logger.warning(f"Error inesperado guardando diseño: {e}", extra={"store_id": design.store_id})
    return design


@router.get("/design/{store_id}", response_model=StoreDesign)
@handle_route_errors
@track_route_metrics("store_designer.get")
async def get_store_design(store_id: str):
    """
    Obtener diseño por ID.
    
    Args:
        store_id: ID del diseño a obtener
        
    Returns:
        StoreDesign completo con toda la información
        
    Raises:
        NotFoundError: Si el diseño no existe
        ValidationError: Si el store_id no es válido
    """
    from ..core.validators import Validator
    
    # Validar store_id
    if not Validator.validate_store_id(store_id):
        raise ValidationError(
            "Invalid store ID format",
            details={"store_id": store_id}
        )
    
    # Intentar desde memoria primero
    design = designs_store.get(store_id)
    if not design:
        # Intentar desde almacenamiento persistente
        try:
            design = storage_service.load_design(store_id)
            if design:
                designs_store[store_id] = design
        except StorageError as e:
            logger.warning(f"Error cargando diseño desde storage: {e.message}", extra={"store_id": store_id})
        except Exception as e:
            logger.warning(f"Error inesperado cargando diseño: {e}", extra={"store_id": store_id})
    
    if not design:
        raise NotFoundError("Store design", store_id)
    return design


@router.get("/designs", response_model=List[dict])
@handle_route_errors
@track_route_metrics("store_designer.list")
async def list_store_designs():
    """Listar todos los diseños"""
    # Combinar diseños de memoria y almacenamiento
    designs_list = []
    
    # De memoria
    for design in designs_store.values():
        designs_list.append({
            "store_id": design.store_id,
            "store_name": design.store_name,
            "store_type": design.store_type.value,
            "style": design.style.value,
            "created_at": design.created_at.isoformat() if isinstance(design.created_at, datetime) else str(design.created_at)
        })
    
    # De almacenamiento (evitar duplicados)
    stored_ids = {d["store_id"] for d in designs_list}
    for stored_design in storage_service.list_designs():
        if stored_design["store_id"] not in stored_ids:
            designs_list.append(stored_design)
    
    return designs_list


@router.post("/design/from-chat/{session_id}", response_model=StoreDesign)
@handle_route_errors
@track_route_metrics("store_designer.from_chat")
async def generate_design_from_chat(session_id: str):
    """
    Generar diseño basado en información del chat.
    
    Args:
        session_id: ID de la sesión de chat (debe ser UUID válido)
        
    Returns:
        StoreDesign completo generado desde la información del chat
        
    Raises:
        NotFoundError: Si la sesión no existe
        ValidationError: Si el session_id no es válido o falta información
    """
    from ..core.validators import Validator
    
    # Validar session_id
    if not Validator.validate_session_id(session_id):
        raise ValidationError(
            "Invalid session ID format",
            details={"session_id": session_id, "expected": "UUID format"}
        )
    
    session = chat_service.get_session(session_id)
    if not session:
        raise NotFoundError("Chat session", session_id)
    
    # Asegurar que la información esté actualizada
    store_info = await chat_service.extract_store_info(session_id)
    
    if not store_info.get("store_type"):
        raise ValidationError(
            "No hay suficiente información. Necesito saber el tipo de tienda.",
            details={"session_id": session_id, "store_info": store_info}
        )
    
    # Crear request desde la información del chat
    from ..core.models import StoreType, DesignStyle
    
    try:
        store_type = StoreType(store_info["store_type"])
    except ValueError:
        raise ValidationError(
            f"Tipo de tienda inválido: {store_info.get('store_type')}",
            details={"store_type": store_info.get("store_type")}
        )
    
    style_preference = None
    if store_info.get("style_preference"):
        try:
            style_preference = DesignStyle(store_info["style_preference"])
        except ValueError:
            logger.warning(f"Estilo inválido ignorado: {store_info.get('style_preference')}")
    
    request = StoreDesignRequest(
        store_name=store_info.get("store_name", "Mi Tienda"),
        store_type=store_type,
        style_preference=style_preference,
        budget_range=store_info.get("budget_range"),
        location=store_info.get("location"),
        target_audience=store_info.get("target_audience"),
        additional_info=store_info.get("additional_info")
    )
    
    design = await store_designer_service.generate_store_design(request)
    designs_store[design.store_id] = design
    try:
        storage_service.save_design(design)
    except StorageError as e:
        logger.warning(f"Error guardando diseño: {e.message}", extra={"store_id": design.store_id})
    except Exception as e:
        logger.warning(f"Error inesperado guardando diseño: {e}", extra={"store_id": design.store_id})
    return design


@router.get("/design/{store_id}/export")
@handle_route_errors
@track_route_metrics("store_designer.export")
async def export_design(store_id: str, format: str = "json"):
    """Exportar diseño en formato específico"""
    if format not in ["json", "markdown", "html"]:
        raise ValidationError(
            "Formato no soportado. Use: json, markdown, html",
            details={"format": format, "supported_formats": ["json", "markdown", "html"]}
        )
    
    exported = None
    try:
        exported = storage_service.export_design(store_id, format)
    except StorageError as e:
        logger.warning(f"Error exportando desde storage: {e.message}", extra={"store_id": store_id, "format": format})
    except Exception as e:
        logger.warning(f"Error inesperado exportando diseño: {e}", extra={"store_id": store_id, "format": format})
    
    if not exported:
        # Intentar desde memoria
        design = designs_store.get(store_id)
        if design:
            try:
                storage_service.save_design(design)
                exported = storage_service.export_design(store_id, format)
            except StorageError as e:
                logger.error(f"Error guardando y exportando diseño: {e.message}", extra={"store_id": store_id})
            except Exception as e:
                logger.error(f"Error inesperado guardando y exportando diseño: {e}", extra={"store_id": store_id})
    
    if not exported:
        raise NotFoundError("Store design", store_id)
    
    content_type = {
        "json": "application/json",
        "markdown": "text/markdown",
        "html": "text/html"
    }.get(format, "text/plain")
    
    from fastapi.responses import Response
    return Response(content=exported, media_type=content_type)


@router.delete("/design/{store_id}")
@handle_route_errors
@track_route_metrics("store_designer.delete")
async def delete_store_design(store_id: str):
    """Eliminar diseño"""
    # Verificar si existe
    exists_in_memory = store_id in designs_store
    
    # Eliminar de memoria
    designs_store.pop(store_id, None)
    
    # Eliminar de almacenamiento
    deleted = False
    try:
        deleted = storage_service.delete_design(store_id)
    except StorageError as e:
        logger.warning(f"Error eliminando diseño desde storage: {e.message}", extra={"store_id": store_id})
    except Exception as e:
        logger.warning(f"Error inesperado eliminando diseño: {e}", extra={"store_id": store_id})
    
    if not deleted and not exists_in_memory:
        raise NotFoundError("Store design", store_id)
    
    logger.info(f"Diseño eliminado: {store_id}", extra={"store_id": store_id})
    return {"message": "Diseño eliminado exitosamente", "store_id": store_id}

