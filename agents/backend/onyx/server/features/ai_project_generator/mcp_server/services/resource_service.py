"""
Resource Service - Business logic for resource operations
=========================================================

Servicio que encapsula la lógica de negocio para operaciones sobre recursos,
incluyendo validación, autorización y ejecución de operaciones.
"""

import logging
from typing import Any, Dict, List, Optional

from ..connectors import ConnectorRegistry, BaseConnector
from ..manifests import ManifestRegistry, ResourceManifest
from ..security import MCPSecurityManager, Scope
from ..contracts import ContextFrame
from ..exceptions import (
    MCPResourceNotFoundError,
    MCPConnectorError,
    MCPOperationError,
    MCPAuthorizationError,
)
from ..utils.validators import (
    validate_resource_id,
    validate_operation,
    validate_user,
    validate_parameters,
)

logger = logging.getLogger(__name__)


class ResourceService:
    """Service for managing resource operations"""
    
    def __init__(
        self,
        connector_registry: ConnectorRegistry,
        manifest_registry: ManifestRegistry,
        security_manager: MCPSecurityManager
    ):
        """Initialize resource service
        
        Args:
            connector_registry: Registry of connectors
            manifest_registry: Registry of resource manifests
            security_manager: Security manager for access control
        """
        self.connector_registry = connector_registry
        self.manifest_registry = manifest_registry
        self.security_manager = security_manager
    
    def list_available_resources(self, user: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        List all resources available to the user.
        
        Args:
            user: User information dictionary (debe contener 'sub' con user_id)
            
        Returns:
            List of resource manifests as dictionaries
            
        Raises:
            ValueError: Si user es inválido
        """
        validated_user = validate_user(user)
        
        available_resources = []
        for manifest in self.manifest_registry.get_all():
            try:
                if self.security_manager.has_access(validated_user, manifest.resource_id, Scope.READ):
                    available_resources.append(manifest.to_dict())
            except Exception as e:
                logger.warning(
                    f"Error checking access for resource {manifest.resource_id}: {e}",
                    exc_info=True
                )
                # Continuar con otros recursos en caso de error
                continue
        
        logger.debug(f"Listed {len(available_resources)} resources for user {validated_user.get('sub', 'unknown')}")
        return available_resources
    
    def get_resource(self, resource_id: str, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get resource information.
        
        Args:
            resource_id: ID of the resource
            user: User information dictionary (debe contener 'sub' con user_id)
            
        Returns:
            Resource manifest as dictionary
            
        Raises:
            MCPAuthorizationError: If user doesn't have access
            MCPResourceNotFoundError: If resource not found
            ValueError: Si resource_id o user son inválidos
        """
        validated_resource_id = validate_resource_id(resource_id)
        validated_user = validate_user(user)
        
        # Verificar acceso
        if not self.security_manager.has_access(validated_user, validated_resource_id, Scope.READ):
            raise MCPAuthorizationError(f"Access denied to resource {validated_resource_id}")
        
        # Obtener manifest
        manifest = self.manifest_registry.get(validated_resource_id)
        if not manifest:
            raise MCPResourceNotFoundError(f"Resource '{validated_resource_id}' not found")
        
        logger.debug(f"Retrieved resource {validated_resource_id} for user {validated_user.get('sub', 'unknown')}")
        return manifest.to_dict()
    
    async def execute_operation(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        user: Dict[str, Any],
        context: Optional[ContextFrame] = None,
        required_scope: Optional[Scope] = None
    ) -> Any:
        """
        Execute an operation on a resource.
        
        Args:
            resource_id: ID of the resource
            operation: Operation to execute
            parameters: Operation parameters
            user: User information dictionary (debe contener 'sub' con user_id)
            context: Optional context frame
            required_scope: Required scope (if None, will be determined)
            
        Returns:
            Operation result
            
        Raises:
            MCPAuthorizationError: If user doesn't have access
            MCPResourceNotFoundError: If resource not found
            MCPConnectorError: If connector not available
            MCPOperationError: If operation execution fails
            ValueError: Si los parámetros son inválidos
        """
        # Validar inputs usando utilidades compartidas
        validated_resource_id = validate_resource_id(resource_id)
        validated_operation = validate_operation(operation)
        validated_parameters = validate_parameters(parameters)
        validated_user = validate_user(user)
        
        # Determine required scope if not provided
        if required_scope is None:
            from .scope_service import ScopeService
            required_scope = ScopeService.get_scope_for_operation(validated_operation)
        
        # Check access
        if not self.security_manager.has_access(validated_user, validated_resource_id, required_scope):
            raise MCPAuthorizationError(
                f"Access denied to {validated_operation} on resource {validated_resource_id}"
            )
        
        # Get manifest
        manifest = self.manifest_registry.get(validated_resource_id)
        if not manifest:
            raise MCPResourceNotFoundError(f"Resource '{validated_resource_id}' not found")
        
        # Get connector
        connector = self.connector_registry.get(manifest.connector_type)
        if not connector:
            raise MCPConnectorError(
                connector_name=manifest.connector_type,
                message=f"Connector '{manifest.connector_type}' not available for resource '{validated_resource_id}'"
            )
        
        # Validate operation is supported
        if not connector.validate_operation(validated_operation):
            raise MCPOperationError(
                operation=validated_operation,
                message=f"Operation '{validated_operation}' not supported by connector '{manifest.connector_type}'"
            )
        
        # Execute operation
        try:
            logger.debug(
                f"Executing {validated_operation} on {validated_resource_id} "
                f"for user {validated_user.get('sub', 'unknown')}"
            )
            result = await connector.execute(
                resource_id=validated_resource_id,
                operation=validated_operation,
                parameters=validated_parameters,
                context=context,
            )
            logger.debug(f"Successfully executed {validated_operation} on {validated_resource_id}")
            return result
        except (MCPAuthorizationError, MCPResourceNotFoundError, MCPConnectorError, MCPOperationError):
            # Re-raise MCP exceptions as-is
            raise
        except Exception as e:
            logger.error(
                f"Error executing {validated_operation} on {validated_resource_id}: {e}",
                exc_info=True
            )
            raise MCPOperationError(
                operation=validated_operation,
                message=f"Operation '{validated_operation}' failed: {e}"
            ) from e

