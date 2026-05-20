"""
Operation Service - Wraps resource operations with observability
================================================================

Servicio que envuelve operaciones de recursos con observabilidad,
incluyendo tracing, métricas y manejo de errores estructurado.
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional

from .resource_service import ResourceService
from .scope_service import ScopeService
from ..contracts import ContextFrame
from ..observability import MCPObservability
from ..exceptions import (
    MCPAuthorizationError,
    MCPResourceNotFoundError,
    MCPConnectorError,
    MCPOperationError,
)
from ..utils.validators import (
    validate_resource_id,
    validate_operation,
    validate_user,
    validate_parameters,
)

logger = logging.getLogger(__name__)


class OperationService:
    """Service for executing operations with observability"""
    
    def __init__(
        self,
        resource_service: ResourceService,
        observability: Optional[MCPObservability] = None
    ):
        """Initialize operation service
        
        Args:
            resource_service: Resource service instance
            observability: Observability manager (optional)
        """
        self.resource_service = resource_service
        self.observability = observability
    
    async def execute_with_observability(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        user: Dict[str, Any],
        context: Optional[ContextFrame] = None
    ) -> Dict[str, Any]:
        """
        Execute operation with observability tracking.
        
        Args:
            resource_id: ID of the resource
            operation: Operation to execute
            parameters: Operation parameters
            user: User information dictionary (debe contener 'sub' con user_id)
            context: Optional context frame
            
        Returns:
            Dictionary with success status and result/error
            
        Raises:
            MCPAuthorizationError: Si el usuario no tiene acceso
            MCPResourceNotFoundError: Si el recurso no existe
            MCPConnectorError: Si el conector no está disponible
            MCPOperationError: Si la operación falla
        """
        # Validar inputs usando utilidades compartidas
        validated_resource_id = validate_resource_id(resource_id)
        validated_operation = validate_operation(operation)
        validated_parameters = validate_parameters(parameters)
        validated_user = validate_user(user)
        
        trace_name = f"{validated_operation}_{validated_resource_id}"
        
        ctx = self.observability.trace(trace_name, resource_id=validated_resource_id) if self.observability else nullcontext()
        with ctx:
            try:
                required_scope = ScopeService.get_scope_for_operation(validated_operation)
                
                result = await self.resource_service.execute_operation(
                    resource_id=validated_resource_id,
                    operation=validated_operation,
                    parameters=validated_parameters,
                    user=validated_user,
                    context=context,
                    required_scope=required_scope
                )
                
                # Record success metric
                if self.observability:
                    self.observability.record_metric(
                        "mcp_operation_success",
                        1,
                        resource_id=validated_resource_id,
                        operation=validated_operation,
                    )
                
                logger.debug(
                    f"Operation {validated_operation} on {validated_resource_id} "
                    f"succeeded for user {validated_user.get('sub', 'unknown')}"
                )
                
                return {
                    "success": True,
                    "data": result,
                    "metadata": {
                        "resource_id": validated_resource_id,
                        "operation": validated_operation,
                    }
                }
                
            except (
                MCPAuthorizationError,
                MCPResourceNotFoundError,
                MCPConnectorError,
                MCPOperationError
            ) as e:
                # Re-raise MCP exceptions as-is (ya tienen observabilidad en ResourceService)
                if self.observability:
                    error_type = type(e).__name__
                    self.observability.record_error(
                        f"mcp_{error_type.lower()}",
                        error=str(e),
                        resource_id=validated_resource_id,
                        operation=validated_operation,
                    )
                raise
                
            except Exception as e:
                logger.error(
                    f"Unexpected error in operation {validated_operation} on {validated_resource_id}: {e}",
                    exc_info=True
                )
                if self.observability:
                    self.observability.record_error(
                        "mcp_unexpected_error",
                        error=str(e),
                        error_type=type(e).__name__,
                        resource_id=validated_resource_id,
                        operation=validated_operation,
                    )
                raise MCPOperationError(
                    operation=validated_operation,
                    message=f"Unexpected error: {e}"
                ) from e

