"""
Health check handler for MCP Server
===================================

Handler para health checks del servidor, incluyendo verificación
de conectores, manifests y estado general del sistema.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from ..connectors import ConnectorRegistry, BaseConnector
from ..manifests import ManifestRegistry

logger = logging.getLogger(__name__)


async def health_check(
    connector_registry: ConnectorRegistry,
    manifest_registry: ManifestRegistry
) -> Dict[str, Any]:
    """
    Health check endpoint con verificación completa del sistema.
    
    Args:
        connector_registry: Connector registry (injected)
        manifest_registry: Manifest registry (injected)
        
    Returns:
        Health status information con detalles del sistema
        
    Raises:
        ValueError: Si algún registry es None
    """
    if connector_registry is None:
        logger.error("Connector registry not configured for health check")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "Connector registry not configured",
        }
    
    if manifest_registry is None:
        logger.error("Manifest registry not configured for health check")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "Manifest registry not configured",
        }
    
    try:
        # Contar recursos y conectores con manejo de errores
        try:
            resources_count = manifest_registry.count()
        except Exception as e:
            logger.warning(f"Error counting resources: {e}")
            resources_count = 0
        
        try:
            connectors_count = connector_registry.count()
        except Exception as e:
            logger.warning(f"Error counting connectors: {e}")
            connectors_count = 0
        
        # Verificar salud de conectores
        connector_health: Dict[str, bool] = {}
        connector_types: List[str] = []
        
        try:
            connector_types = connector_registry.list_connectors()
        except Exception as e:
            logger.warning(f"Error listing connectors: {e}")
            connector_types = []
        
        for connector_type in connector_types:
            try:
                connector = connector_registry.get(connector_type)
                if connector and hasattr(connector, 'health_check'):
                    try:
                        is_healthy = await connector.health_check()
                        connector_health[connector_type] = is_healthy
                    except Exception as e:
                        logger.warning(
                            f"Error executing health_check for connector "
                            f"{connector_type}: {e}"
                        )
                        connector_health[connector_type] = False
                else:
                    # Si no tiene health_check, asumir saludable
                    connector_health[connector_type] = True
            except Exception as e:
                logger.warning(f"Error checking health of connector {connector_type}: {e}")
                connector_health[connector_type] = False
        
        # Determinar estado general
        all_healthy = all(connector_health.values()) if connector_health else True
        health_status = "healthy" if all_healthy and resources_count > 0 else "degraded"
        
        if not all_healthy:
            health_status = "unhealthy"
        
        return {
            "status": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resources_count": resources_count,
            "connectors_count": connectors_count,
            "connector_health": connector_health,
            "details": {
                "all_connectors_healthy": all_healthy,
                "connector_types": connector_types,
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error in health check: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "Health check failed",
            "error_details": str(e),
        }

