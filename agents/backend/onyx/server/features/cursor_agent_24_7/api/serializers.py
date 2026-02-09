"""
API Serializers - Utilidades para serialización de respuestas
=============================================================

Utilidades para convertir objetos del dominio a formatos de respuesta API.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


def serialize_notification(notification: Any) -> Dict[str, Any]:
    """
    Serializar notificación a diccionario.
    
    Args:
        notification: Objeto de notificación.
    
    Returns:
        Diccionario con datos de la notificación.
    """
    return {
        "id": notification.id,
        "title": notification.title,
        "message": notification.message,
        "level": notification.level.value if hasattr(notification.level, 'value') else str(notification.level),
        "timestamp": notification.timestamp.isoformat() if isinstance(notification.timestamp, datetime) else str(notification.timestamp),
        "read": notification.read,
        "metadata": notification.metadata if hasattr(notification, 'metadata') else {}
    }


def serialize_notifications(notifications: List[Any]) -> List[Dict[str, Any]]:
    """
    Serializar lista de notificaciones.
    
    Args:
        notifications: Lista de objetos de notificación.
    
    Returns:
        Lista de diccionarios con datos de notificaciones.
    """
    return [serialize_notification(n) for n in notifications]


def serialize_task(task: Any) -> Dict[str, Any]:
    """
    Serializar tarea a diccionario.
    
    Args:
        task: Objeto de tarea o diccionario.
    
    Returns:
        Diccionario con datos de la tarea.
    """
    if isinstance(task, dict):
        return task
    
    return {
        "id": task.id,
        "command": task.command,
        "status": task.status,
        "timestamp": task.timestamp.isoformat() if isinstance(task.timestamp, datetime) else str(task.timestamp),
        "result": task.result,
        "error": task.error,
    }


def serialize_tasks(tasks: List[Any]) -> List[Dict[str, Any]]:
    """
    Serializar lista de tareas.
    
    Args:
        tasks: Lista de objetos de tarea o diccionarios.
    
    Returns:
        Lista de diccionarios con datos de tareas.
    """
    return [serialize_task(t) for t in tasks]


def serialize_search_result(search_result: Any) -> Dict[str, Any]:
    """
    Serializar resultado de búsqueda a diccionario.
    
    Args:
        search_result: Objeto SearchResultInput o diccionario.
    
    Returns:
        Diccionario con datos del resultado de búsqueda.
    """
    if isinstance(search_result, dict):
        return search_result
    
    return {
        'title': search_result.title,
        'url': search_result.url,
        'snippet': search_result.snippet,
        'content': getattr(search_result, 'content', None),
        'source': getattr(search_result, 'source', None),
        'timestamp': getattr(search_result, 'timestamp', None)
    }


def serialize_search_results(search_results: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Serializar lista de resultados de búsqueda.
    
    Args:
        search_results: Lista de objetos SearchResultInput o None.
    
    Returns:
        Lista de diccionarios o None.
    """
    if search_results is None:
        return None
    
    return [serialize_search_result(sr) for sr in search_results]


def serialize_webhook(webhook_id: str, config: Any) -> Dict[str, Any]:
    """
    Serializar webhook a diccionario.
    
    Args:
        webhook_id: ID del webhook.
        config: Configuración del webhook.
    
    Returns:
        Diccionario con datos del webhook.
    """
    return {
        "webhook_id": webhook_id,
        "url": config.url if hasattr(config, 'url') else str(config.url),
        "events": config.events if hasattr(config, 'events') else [],
        "enabled": config.enabled if hasattr(config, 'enabled') else True
    }


def serialize_webhooks(webhooks: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Serializar diccionario de webhooks a lista.
    
    Args:
        webhooks: Diccionario de webhook_id -> config.
    
    Returns:
        Lista de diccionarios con datos de webhooks.
    """
    return [
        serialize_webhook(webhook_id, config)
        for webhook_id, config in webhooks.items()
    ]




