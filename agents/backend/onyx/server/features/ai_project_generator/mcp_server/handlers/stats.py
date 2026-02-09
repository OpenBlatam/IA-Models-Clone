"""
Stats handlers for MCP Server
==============================

Handlers para endpoints de estadísticas detalladas.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..observability import MCPObservability
from ..services import ResourceService

logger = logging.getLogger(__name__)


async def get_stats(
    resource_service: Optional[ResourceService] = None,
    observability: Optional[MCPObservability] = None
) -> Dict[str, Any]:
    """
    Obtener estadísticas detalladas del servidor.
    
    Args:
        resource_service: Resource service (opcional)
        observability: Observability manager (opcional)
        
    Returns:
        Diccionario con estadísticas detalladas
    """
    stats: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {}
    }
    
    # Estadísticas de observabilidad
    if observability:
        try:
            if hasattr(observability, 'get_metrics_summary'):
                obs_stats = observability.get_metrics_summary()
                stats["observability"] = obs_stats
            elif hasattr(observability, 'metrics') and observability.metrics:
                # Fallback: obtener métricas directamente
                stats["observability"] = {
                    "enabled": observability.enable_metrics,
                    "tracing_enabled": observability.enable_tracing,
                    "summary": "available"
                }
            else:
                stats["observability"] = {
                    "enabled": False,
                    "tracing_enabled": False
                }
        except Exception as e:
            logger.warning(f"Error getting observability stats: {e}")
            stats["observability"] = {"error": str(e)}
    
    # Estadísticas de recursos (si el servicio está disponible)
    if resource_service:
        try:
            # Obtener estadísticas del servicio si está disponible
            if hasattr(resource_service, 'get_stats'):
                service_stats = resource_service.get_stats()
                stats["resources"] = service_stats
        except Exception as e:
            logger.debug(f"Resource service stats not available: {e}")
    
    return stats

