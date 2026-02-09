"""
Helper functions for webhook operations.
Simplifies webhook sending patterns.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


async def send_webhook(
    event: str,
    data: Dict[str, Any],
    webhook_service: Optional[Any] = None
) -> None:
    """
    Envía un webhook de forma segura (no lanza excepciones).
    
    Args:
        event: Tipo de evento (ej: "identity_created", "content_generated")
        data: Datos del evento
        webhook_service: Instancia del servicio de webhooks (opcional, se obtiene automáticamente)
        
    Usage:
        await send_webhook("identity_created", {
            "identity_id": identity.profile_id,
            "username": identity.username
        })
    """
    try:
        if webhook_service is None:
            from ..services.webhook_service import get_webhook_service
            webhook_service = get_webhook_service()
        
        await webhook_service.send_webhook(event, data)
        logger.debug(f"Webhook sent: {event}")
    except Exception as e:
        logger.warning(f"Failed to send webhook {event}: {e}", exc_info=True)
        # No re-raise: webhooks no deben romper el flujo principal


def webhook_event(event_name: str):
    """
    Decorador para enviar webhook automáticamente después de una operación.
    
    Args:
        event_name: Nombre del evento a enviar
        
    Usage:
        @webhook_event("identity_created")
        async def create_identity(data):
            identity = # ... crear identidad ...
            return identity
    """
    def decorator(func):
        from functools import wraps
        import asyncio
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Intentar extraer datos del resultado
            if isinstance(result, dict):
                webhook_data = result
            elif hasattr(result, 'model_dump'):
                webhook_data = result.model_dump()
            elif hasattr(result, '__dict__'):
                webhook_data = result.__dict__
            else:
                webhook_data = {"result": str(result)}
            
            await send_webhook(event_name, webhook_data)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Para funciones sync, ejecutar webhook en background
            if asyncio.iscoroutinefunction(send_webhook):
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(send_webhook(event_name, {"result": str(result)}))
                else:
                    loop.run_until_complete(send_webhook(event_name, {"result": str(result)}))
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator








