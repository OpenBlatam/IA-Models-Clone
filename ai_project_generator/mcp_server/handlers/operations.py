"""
Operation handlers for MCP Server
==================================

Handlers para operaciones sobre recursos, incluyendo validación,
manejo de errores y construcción de respuestas.
"""

import logging
from typing import Dict, Any
from fastapi import HTTPException, status

from ..models import MCPRequest, MCPResponse
from ..services import OperationService
from ..manifests import ManifestRegistry
from ..exceptions import (
    MCPAuthorizationError,
    MCPResourceNotFoundError,
    MCPConnectorError,
    MCPOperationError,
    MCPValidationError,
)
from ..utils.response_builder import build_response_from_exception
from ..utils.error_handlers import get_error_type
from ..utils.validators import validate_resource_id
from ..utils.error_handlers import handle_mcp_exception

logger = logging.getLogger(__name__)


async def query_resource(
    resource_id: str,
    request: MCPRequest,
    user: Dict[str, Any],
    operation_service: OperationService,
    manifest_registry: ManifestRegistry
) -> MCPResponse:
    """
    Query/execute operation on a resource.
    
    Args:
        resource_id: ID of the resource
        request: MCP request with operation details
        user: Current authenticated user (debe contener 'sub' con user_id)
        operation_service: Operation service (injected)
        manifest_registry: Manifest registry (injected)
        
    Returns:
        MCP response with result or error
        
    Raises:
        HTTPException: Si los servicios no están configurados o hay error de validación
    """
    # Validar servicios inyectados
    if operation_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Operation service not configured"
        )
    
    if manifest_registry is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Manifest registry not configured"
        )
    
    # Validar resource_id
    validated_resource_id = validate_resource_id(resource_id)
    
    # Validar resource_id coincide con request
    if request.resource_id != validated_resource_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Resource ID mismatch: path={validated_resource_id}, request={request.resource_id}"
        )
    
    # Validar request
    if not request.operation or not isinstance(request.operation, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Operation must be a non-empty string"
        )
    
    try:
        result = await operation_service.execute_with_observability(
            resource_id=validated_resource_id,
            operation=request.operation,
            parameters=request.parameters or {},
            user=user,
            context=request.context
        )
        
        # Get manifest for metadata
        manifest = manifest_registry.get(validated_resource_id)
        connector_type = manifest.connector_type if manifest else "unknown"
        
        return MCPResponse(
            success=result["success"],
            data=result.get("data"),
            metadata={
                **result.get("metadata", {}),
                "connector_type": connector_type,
            }
        )
        
    except (
        MCPAuthorizationError,
        MCPResourceNotFoundError,
        MCPConnectorError,
        MCPOperationError,
        MCPValidationError
    ) as e:
        # Usar utilidad compartida para manejar excepciones MCP
        return handle_mcp_exception(
            exception=e,
            resource_id=validated_resource_id,
            operation=request.operation,
            user_id=user.get("sub"),
            log_level="warning" if isinstance(e, (MCPAuthorizationError, MCPResourceNotFoundError, MCPValidationError)) else "error"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error querying resource {validated_resource_id}: {e}",
            exc_info=True
        )
        return handle_mcp_exception(
            exception=e,
            resource_id=validated_resource_id,
            operation=request.operation,
            user_id=user.get("sub"),
            log_level="error"
        )

