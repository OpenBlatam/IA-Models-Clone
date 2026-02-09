"""
Metrics handlers for MCP Server
================================

Handlers para endpoints de métricas y estadísticas.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..observability import MCPObservability
from ..connectors import ConnectorRegistry
from ..manifests import ManifestRegistry
from ..services import ResourceService

logger = logging.getLogger(__name__)


async def get_metrics(
    observability: Optional[MCPObservability] = None,
    connector_registry: Optional[ConnectorRegistry] = None,
    manifest_registry: Optional[ManifestRegistry] = None
) -> Dict[str, Any]:
    """
    Obtener métricas del servidor MCP.
    
    Args:
        observability: Observability manager (opcional)
        connector_registry: Connector registry (opcional)
        manifest_registry: Manifest registry (opcional)
        
    Returns:
        Diccionario con métricas del servidor
    """
    metrics: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "server": {
            "version": "1.0.0",
            "status": "running"
        }
    }
    
    # Métricas de observabilidad
    if observability:
        try:
            if hasattr(observability, 'get_metrics_summary'):
                obs_metrics = observability.get_metrics_summary()
                metrics["observability"] = obs_metrics
            elif hasattr(observability, 'metrics') and observability.metrics:
                # Fallback: obtener métricas directamente
                metrics["observability"] = {
                    "enabled": observability.enable_metrics,
                    "tracing_enabled": observability.enable_tracing
                }
            else:
                metrics["observability"] = {
                    "enabled": False,
                    "tracing_enabled": False
                }
        except Exception as e:
            logger.warning(f"Error getting observability metrics: {e}")
            metrics["observability"] = {"error": "unavailable"}
    
    # Métricas de conectores
    if connector_registry:
        try:
            connectors = connector_registry.list_connectors()
            metrics["connectors"] = {
                "count": len(connectors),
                "types": connectors
            }
        except Exception as e:
            logger.warning(f"Error getting connector metrics: {e}")
            metrics["connectors"] = {"error": "unavailable"}
    
    # Métricas de recursos
    if manifest_registry:
        try:
            resources = manifest_registry.get_all()
            metrics["resources"] = {
                "count": len(resources),
                "by_connector": {}
            }
            
            # Agrupar por tipo de conector
            for resource in resources:
                connector_type = resource.connector_type
                if connector_type not in metrics["resources"]["by_connector"]:
                    metrics["resources"]["by_connector"][connector_type] = 0
                metrics["resources"]["by_connector"][connector_type] += 1
        except Exception as e:
            logger.warning(f"Error getting resource metrics: {e}")
            metrics["resources"] = {"error": "unavailable"}
    
    return metrics


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
    if observability and hasattr(observability, 'get_stats'):
        try:
            obs_stats = observability.get_stats()
            stats["observability"] = obs_stats
        except Exception as e:
            logger.warning(f"Error getting observability stats: {e}")
    
    return stats

