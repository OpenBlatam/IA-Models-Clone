"""
Testing Helpers - Funciones helper para testing
===============================================

Funciones auxiliares para crear mocks y datos de prueba.
"""

import logging
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

from ..connectors import BaseConnector
from ..manifests import ResourceManifest
from ..security import MCPSecurityManager

logger = logging.getLogger(__name__)


def create_mock_connector(
    connector_type: str = "filesystem",
    supported_operations: Optional[List[str]] = None,
    execute_result: Any = None
) -> Mock:
    """
    Crea un mock connector para testing.
    
    Args:
        connector_type: Tipo de connector
        supported_operations: Lista de operaciones soportadas
        execute_result: Resultado por defecto de execute()
        
    Returns:
        Mock connector configurado
    """
    mock = MagicMock(spec=BaseConnector)
    mock.connector_type = connector_type
    mock.get_connector_type.return_value = connector_type
    
    if supported_operations is None:
        supported_operations = ["read", "write", "list"]
    
    mock.get_supported_operations.return_value = supported_operations
    mock.validate_operation = lambda op: op in supported_operations
    
    if execute_result is None:
        execute_result = {"result": "mock_data", "status": "success"}
    
    async def mock_execute(*args, **kwargs):
        return execute_result
    
    mock.execute = mock_execute
    mock.health_check = MagicMock(return_value=True)
    mock.close = MagicMock(return_value=None)
    
    return mock


def create_mock_manifest(
    resource_id: str = "test-resource",
    connector_type: str = "filesystem",
    name: Optional[str] = None,
    description: Optional[str] = None,
    permissions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Crea un mock manifest para testing.
    
    Args:
        resource_id: ID del recurso
        connector_type: Tipo de connector
        name: Nombre del recurso (opcional)
        description: Descripción (opcional)
        permissions: Permisos (opcional)
        
    Returns:
        Diccionario con manifest mock
    """
    if name is None:
        name = f"Test {resource_id}"
    
    if description is None:
        description = f"Test resource: {resource_id}"
    
    if permissions is None:
        permissions = ["read", "write"]
    
    return {
        "resource_id": resource_id,
        "name": name,
        "description": description,
        "connector_type": connector_type,
        "type": connector_type,
        "endpoint": f"/test/{resource_id}",
        "schema": {},
        "permissions": permissions,
        "metadata": {
            "created_at": datetime.utcnow().isoformat(),
            "test": True
        }
    }


def create_test_user(
    user_id: str = "test_user",
    email: Optional[str] = None,
    scopes: Optional[List[str]] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crea un diccionario de usuario para testing.
    
    Args:
        user_id: ID del usuario
        email: Email del usuario (opcional)
        scopes: Scopes del usuario (opcional)
        additional_claims: Claims adicionales (opcional)
        
    Returns:
        Diccionario con información del usuario
    """
    if email is None:
        email = f"{user_id}@test.com"
    
    if scopes is None:
        scopes = ["read", "write"]
    
    user = {
        "sub": user_id,
        "email": email,
        "scopes": scopes,
        "iat": int(datetime.utcnow().timestamp()),
        "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
    }
    
    if additional_claims:
        user.update(additional_claims)
    
    return user


def create_test_token(
    security_manager: MCPSecurityManager,
    user_id: str = "test_user",
    scopes: Optional[List[str]] = None,
    expires_in_minutes: int = 60
) -> str:
    """
    Crea un token JWT de prueba.
    
    Args:
        security_manager: Security manager para generar token
        user_id: ID del usuario
        scopes: Scopes del token (opcional)
        expires_in_minutes: Minutos hasta expiración
        
    Returns:
        Token JWT
    """
    user = create_test_user(user_id=user_id, scopes=scopes)
    
    # Intentar generar token usando el security manager
    try:
        if hasattr(security_manager, 'create_token'):
            return security_manager.create_token(user, expires_in_minutes=expires_in_minutes)
        elif hasattr(security_manager, 'generate_token'):
            return security_manager.generate_token(user, expires_in_minutes=expires_in_minutes)
        else:
            # Fallback: crear token simple
            logger.warning("Security manager doesn't have create_token or generate_token method")
            return f"mock_token_{user_id}"
    except Exception as e:
        logger.warning(f"Error creating token: {e}, using mock token")
        return f"mock_token_{user_id}"


def create_test_request(
    resource_id: str = "test-resource",
    operation: str = "read",
    parameters: Optional[Dict[str, Any]] = None,
    context: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Crea un request MCP para testing.
    
    Args:
        resource_id: ID del recurso
        operation: Operación a ejecutar
        parameters: Parámetros (opcional)
        context: Contexto (opcional)
        
    Returns:
        Diccionario con request
    """
    request = {
        "resource_id": resource_id,
        "operation": operation,
        "parameters": parameters or {},
    }
    
    if context is not None:
        request["context"] = context
    
    return request

