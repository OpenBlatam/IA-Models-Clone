"""
MCP Testing Utilities - Utilidades de testing (Legacy)
=======================================================

Este módulo mantiene compatibilidad hacia atrás.
Para nuevas funcionalidades, usar el módulo testing/.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from unittest.mock import Mock, MagicMock
import pytest

logger = logging.getLogger(__name__)


class MCPTestClient:
    """
    Cliente de testing para MCP (Legacy).
    
    Para funcionalidades mejoradas, usar testing.client.MCPTestClient
    """
    
    def __init__(self, app):
        """
        Args:
            app: Aplicación FastAPI
        """
        from fastapi.testclient import TestClient
        self.client = TestClient(app)
    
    def list_resources(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Lista recursos"""
        response = self.client.get("/mcp/v1/resources", headers=headers or {})
        return response.json()
    
    def get_resource(self, resource_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Obtiene recurso"""
        response = self.client.get(
            f"/mcp/v1/resources/{resource_id}",
            headers=headers or {},
        )
        return response.json()
    
    def query_resource(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Consulta recurso"""
        response = self.client.post(
            f"/mcp/v1/resources/{resource_id}/query",
            json={
                "operation": operation,
                "parameters": parameters,
            },
            headers=headers or {},
        )
        return response.json()


def create_mock_connector(connector_type: str = "filesystem") -> Mock:
    """
    Crea mock connector para testing (legacy).
    
    Para funcionalidades mejoradas, usar testing.helpers.create_mock_connector
    
    Args:
        connector_type: Tipo de connector
        
    Returns:
        Mock connector
    """
    mock = MagicMock()
    mock.connector_type = connector_type
    mock.execute = MagicMock(return_value={"result": "mock_data"})
    return mock


def create_mock_manifest(resource_id: str = "test-resource") -> Dict[str, Any]:
    """
    Crea mock manifest para testing (legacy).
    
    Para funcionalidades mejoradas, usar testing.helpers.create_mock_manifest
    
    Args:
        resource_id: ID del recurso
        
    Returns:
        Mock manifest
    """
    return {
        "resource_id": resource_id,
        "name": "Test Resource",
        "type": "filesystem",
        "endpoint": "/test",
        "schema": {},
        "permissions": ["read"],
    }


@pytest.fixture
def mcp_test_client(app):
    """
    Fixture de test client (legacy).
    
    Para funcionalidades mejoradas, usar testing.fixtures.mcp_test_client
    """
    return MCPTestClient(app)


@pytest.fixture
def mock_connector():
    """Fixture de mock connector"""
    return create_mock_connector()


@pytest.fixture
def mock_manifest():
    """Fixture de mock manifest"""
    return create_mock_manifest()


def assert_mcp_response(response: Dict[str, Any], success: bool = True):
    """
    Assert para validar respuesta MCP (legacy).
    
    Para funcionalidades mejoradas, usar testing.assertions.assert_mcp_response
    
    Args:
        response: Respuesta MCP
        success: Si debe ser exitosa
    """
    assert "success" in response
    assert response["success"] == success
    
    if success:
        assert "data" in response
    else:
        assert "error" in response
