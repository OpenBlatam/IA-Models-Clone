"""
Resource Helpers - Utilidades para gestión de recursos
======================================================

Funciones helper para facilitar la gestión y operación de recursos MCP.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from contextlib import asynccontextmanager

from ..connectors import BaseConnector, ConnectorRegistry
from ..manifests import ManifestRegistry, ResourceManifest
from ..exceptions import MCPResourceNotFoundError, MCPConnectorError

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Gestor de recursos MCP.
    
    Facilita la gestión y operación de recursos.
    """
    
    def __init__(
        self,
        connector_registry: ConnectorRegistry,
        manifest_registry: ManifestRegistry
    ):
        """
        Inicializar gestor de recursos.
        
        Args:
            connector_registry: Registry de conectores
            manifest_registry: Registry de manifests
        """
        self.connector_registry = connector_registry
        self.manifest_registry = manifest_registry
    
    def get_resource_manifest(self, resource_id: str) -> ResourceManifest:
        """
        Obtener manifest de un recurso.
        
        Args:
            resource_id: ID del recurso
        
        Returns:
            ResourceManifest del recurso
        
        Raises:
            MCPResourceNotFoundError: Si el recurso no existe
        """
        manifest = self.manifest_registry.get(resource_id)
        if not manifest:
            raise MCPResourceNotFoundError(
                resource_id=resource_id,
                message=f"Resource {resource_id} not found"
            )
        return manifest
    
    def get_connector_for_resource(self, resource_id: str) -> BaseConnector:
        """
        Obtener conector para un recurso.
        
        Args:
            resource_id: ID del recurso
        
        Returns:
            BaseConnector para el recurso
        
        Raises:
            MCPResourceNotFoundError: Si el recurso no existe
            MCPConnectorError: Si el conector no está disponible
        """
        manifest = self.get_resource_manifest(resource_id)
        connector = self.connector_registry.get_connector(manifest.connector_type)
        
        if not connector:
            raise MCPConnectorError(
                connector_name=manifest.connector_type,
                message=f"Connector {manifest.connector_type} not found for resource {resource_id}"
            )
        
        return connector
    
    async def execute_operation(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        context: Optional[Any] = None
    ) -> Any:
        """
        Ejecutar operación en un recurso.
        
        Args:
            resource_id: ID del recurso
            operation: Operación a ejecutar
            parameters: Parámetros de la operación
            context: Contexto adicional (opcional)
        
        Returns:
            Resultado de la operación
        
        Raises:
            MCPResourceNotFoundError: Si el recurso no existe
            MCPConnectorError: Si hay error del conector
        """
        manifest = self.get_resource_manifest(resource_id)
        connector = self.get_connector_for_resource(resource_id)
        
        # Validar que la operación esté soportada
        if not connector.validate_operation(operation):
            raise MCPConnectorError(
                connector_name=manifest.connector_type,
                message=f"Operation {operation} not supported by connector {manifest.connector_type}"
            )
        
        # Ejecutar operación
        return await connector.execute(
            resource_id=resource_id,
            operation=operation,
            parameters=parameters,
            context=context
        )
    
    def list_resources_by_type(self, connector_type: str) -> List[ResourceManifest]:
        """
        Listar recursos por tipo de conector.
        
        Args:
            connector_type: Tipo de conector
        
        Returns:
            Lista de manifests del tipo especificado
        """
        all_resources = self.manifest_registry.get_all()
        return [
            resource for resource in all_resources
            if resource.connector_type == connector_type
        ]
    
    def get_resource_summary(self, resource_id: str) -> Dict[str, Any]:
        """
        Obtener resumen de un recurso.
        
        Args:
            resource_id: ID del recurso
        
        Returns:
            Diccionario con resumen del recurso
        
        Raises:
            MCPResourceNotFoundError: Si el recurso no existe
        """
        manifest = self.get_resource_manifest(resource_id)
        connector = self.get_connector_for_resource(resource_id)
        
        return {
            "resource_id": resource_id,
            "name": manifest.name,
            "connector_type": manifest.connector_type,
            "supported_operations": list(connector.get_supported_operations()),
            "description": getattr(manifest, "description", None),
            "metadata": getattr(manifest, "metadata", {})
        }


@asynccontextmanager
async def resource_operation(
    resource_manager: ResourceManager,
    resource_id: str,
    operation: str,
    parameters: Dict[str, Any],
    context: Optional[Any] = None
):
    """
    Context manager para operaciones sobre recursos.
    
    Args:
        resource_manager: ResourceManager instance
        resource_id: ID del recurso
        operation: Operación a ejecutar
        parameters: Parámetros
        context: Contexto adicional
    
    Yields:
        Resultado de la operación
    """
    try:
        result = await resource_manager.execute_operation(
            resource_id=resource_id,
            operation=operation,
            parameters=parameters,
            context=context
        )
        yield result
    except Exception as e:
        logger.error(
            f"Error in resource operation {operation} on {resource_id}: {e}",
            exc_info=True
        )
        raise


def validate_resource_access(
    manifest: ResourceManifest,
    user_scopes: List[str],
    operation: str
) -> bool:
    """
    Validar acceso de usuario a recurso.
    
    Args:
        manifest: ResourceManifest del recurso
        user_scopes: Scopes del usuario
        operation: Operación a validar
    
    Returns:
        True si el usuario tiene acceso
    """
    # Verificar que la operación esté soportada
    if operation not in manifest.supported_operations:
        return False
    
    # Verificar permisos (simplificado)
    required_permission = "read" if operation in ["read", "query", "list"] else "write"
    
    # Verificar scopes del usuario
    if required_permission not in user_scopes:
        return False
    
    return True


def build_resource_filter(
    connector_type: Optional[str] = None,
    operation: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Callable[[ResourceManifest], bool]:
    """
    Construir filtro para recursos.
    
    Args:
        connector_type: Tipo de conector (opcional)
        operation: Operación soportada (opcional)
        tags: Tags del recurso (opcional)
    
    Returns:
        Función de filtro
    """
    def filter_func(manifest: ResourceManifest) -> bool:
        if connector_type and manifest.connector_type != connector_type:
            return False
        
        if operation and operation not in manifest.supported_operations:
            return False
        
        if tags:
            manifest_tags = getattr(manifest, "tags", [])
            if not any(tag in manifest_tags for tag in tags):
                return False
        
        return True
    
    return filter_func


def group_resources_by_connector(
    manifests: List[ResourceManifest]
) -> Dict[str, List[ResourceManifest]]:
    """
    Agrupar recursos por tipo de conector.
    
    Args:
        manifests: Lista de manifests
    
    Returns:
        Diccionario agrupado por connector_type
    """
    grouped: Dict[str, List[ResourceManifest]] = {}
    
    for manifest in manifests:
        connector_type = manifest.connector_type
        if connector_type not in grouped:
            grouped[connector_type] = []
        grouped[connector_type].append(manifest)
    
    return grouped


def get_resource_statistics(
    manifest_registry: ManifestRegistry,
    connector_registry: ConnectorRegistry
) -> Dict[str, Any]:
    """
    Obtener estadísticas de recursos.
    
    Args:
        manifest_registry: Registry de manifests
        connector_registry: Registry de conectores
    
    Returns:
        Diccionario con estadísticas
    """
    all_resources = manifest_registry.get_all()
    connectors = connector_registry.list_connectors()
    
    # Agrupar por tipo
    by_connector = group_resources_by_connector(all_resources)
    
    # Contar operaciones
    operations_count: Dict[str, int] = {}
    for resource in all_resources:
        for op in resource.supported_operations:
            operations_count[op] = operations_count.get(op, 0) + 1
    
    return {
        "total_resources": len(all_resources),
        "total_connectors": len(connectors),
        "by_connector_type": {
            k: len(v) for k, v in by_connector.items()
        },
        "operations_count": operations_count,
        "connector_types": list(set(r.connector_type for r in all_resources))
    }

