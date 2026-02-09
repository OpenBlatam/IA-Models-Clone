"""
Info handlers for MCP Server
============================

Handlers para endpoints de información del servidor.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..connectors import ConnectorRegistry
from ..manifests import ManifestRegistry
from ..security import MCPSecurityManager

logger = logging.getLogger(__name__)


async def get_server_info(
    connector_registry: Optional[ConnectorRegistry] = None,
    manifest_registry: Optional[ManifestRegistry] = None,
    security_manager: Optional[MCPSecurityManager] = None
) -> Dict[str, Any]:
    """
    Obtener información del servidor MCP.
    
    Args:
        connector_registry: Connector registry (opcional)
        manifest_registry: Manifest registry (opcional)
        security_manager: Security manager (opcional)
        
    Returns:
        Diccionario con información del servidor
    """
    info: Dict[str, Any] = {
        "server": {
            "name": "MCP Server",
            "version": "1.0.0",
            "protocol": "MCP v1",
            "timestamp": datetime.utcnow().isoformat()
        },
        "capabilities": {
            "connectors": [],
            "operations": []
        }
    }
    
    # Información de conectores
    if connector_registry:
        try:
            connectors = connector_registry.list_connectors()
            info["capabilities"]["connectors"] = connectors
            
            # Obtener operaciones soportadas por cada conector
            for connector_type in connectors:
                connector = connector_registry.get(connector_type)
                if connector:
                    try:
                        operations = connector.get_supported_operations()
                        if operations:
                            info["capabilities"]["operations"].extend(operations)
                    except Exception as e:
                        logger.debug(f"Error getting operations for {connector_type}: {e}")
            
            # Eliminar duplicados
            info["capabilities"]["operations"] = list(set(info["capabilities"]["operations"]))
        except Exception as e:
            logger.warning(f"Error getting connector info: {e}")
    
    # Información de recursos
    if manifest_registry:
        try:
            resources = manifest_registry.get_all()
            info["resources"] = {
                "total": len(resources),
                "by_type": {}
            }
            
            for resource in resources:
                connector_type = resource.connector_type
                if connector_type not in info["resources"]["by_type"]:
                    info["resources"]["by_type"][connector_type] = 0
                info["resources"]["by_type"][connector_type] += 1
        except Exception as e:
            logger.warning(f"Error getting resource info: {e}")
    
    # Información de seguridad
    if security_manager:
        try:
            info["security"] = {
                "enabled": True,
                "authentication": "Bearer Token",
                "authorization": "Scope-based"
            }
        except Exception as e:
            logger.warning(f"Error getting security info: {e}")
    
    return info

