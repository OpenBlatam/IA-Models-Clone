"""
Resource handlers for MCP Server
=================================

Handlers para operaciones sobre recursos, incluyendo listado
y obtención de información de recursos.
"""

import logging
from contextlib import nullcontext
from typing import Dict, List, Any, Optional
from fastapi import HTTPException, status

from ..models import MCPResponse
from ..services import ResourceService
from ..observability import MCPObservability
from ..exceptions import (
    MCPAuthorizationError,
    MCPResourceNotFoundError,
    MCPValidationError,
)
from ..utils.validators import validate_user, validate_resource_id
from ..utils.error_handlers import handle_mcp_exception, raise_http_exception_from_mcp

logger = logging.getLogger(__name__)


async def list_resources(
    user: Dict[str, Any],
    resource_service: ResourceService,
    observability: Optional[MCPObservability] = None
) -> List[Dict[str, Any]]:
    """
    List all available resources.
    
    Args:
        user: Current authenticated user (debe contener 'sub' con user_id)
        resource_service: Resource service (injected)
        observability: Observability manager (injected)
        
    Returns:
        List of available resources
        
    Raises:
        HTTPException: Si el servicio no está configurado o hay error
    """
    if resource_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resource service not configured"
        )
    
    # Validar user usando utilidad compartida
    try:
        validated_user = validate_user(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    ctx = observability.trace("list_resources") if observability else nullcontext()
    with ctx:
        try:
            resources = resource_service.list_available_resources(validated_user)
            
            if observability:
                observability.record_metric("mcp_resources_listed", len(resources))
            
            logger.debug(
                f"Listed {len(resources)} resources for user {validated_user.get('sub', 'unknown')}"
            )
            return resources
            
        except (MCPAuthorizationError, MCPValidationError) as e:
            raise_http_exception_from_mcp(e)
        except ValueError as e:
            logger.warning(f"Validation error listing resources: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error listing resources: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error listing resources"
            )


async def get_resource(
    resource_id: str,
    user: Dict[str, Any],
    resource_service: ResourceService
) -> Dict[str, Any]:
    """
    Get information about a specific resource.
    
    Args:
        resource_id: ID of the resource
        user: Current authenticated user (debe contener 'sub' con user_id)
        resource_service: Resource service (injected)
        
    Returns:
        Resource information
        
    Raises:
        HTTPException: Si el servicio no está configurado, acceso denegado o recurso no encontrado
    """
    if resource_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resource service not configured"
        )
    
    # Validar inputs usando utilidades compartidas
    try:
        validated_resource_id = validate_resource_id(resource_id)
        validated_user = validate_user(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    try:
        return resource_service.get_resource(validated_resource_id, validated_user)
    except (MCPAuthorizationError, MCPResourceNotFoundError, MCPValidationError) as e:
        raise_http_exception_from_mcp(e)
    except ValueError as e:
        logger.warning(f"Validation error getting resource {validated_resource_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error getting resource {validated_resource_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving resource"
        )

