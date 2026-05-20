"""
Tests para MCP Server
=====================
"""

import pytest
from fastapi.testclient import TestClient

from ..mcp_server.mocks import MockMCPServer, mcp_server_fixture


def test_mcp_server_health_check(mcp_server_fixture):
    """Test health check del servidor MCP"""
    app = mcp_server_fixture.get_app()
    client = TestClient(app)
    
    response = client.get("/mcp/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "resources_count" in data


def test_list_resources(mcp_server_fixture):
    """Test listar recursos"""
    # Agregar recurso mock
    mcp_server_fixture.add_mock_resource("test_resource", "filesystem")
    
    app = mcp_server_fixture.get_app()
    client = TestClient(app)
    
    # Crear token
    token = mcp_server_fixture.create_test_token()
    
    response = client.get(
        "/mcp/v1/resources",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    resources = response.json()
    assert isinstance(resources, list)
    assert len(resources) >= 1


def test_get_resource(mcp_server_fixture):
    """Test obtener recurso específico"""
    mcp_server_fixture.add_mock_resource("test_resource", "filesystem")
    
    app = mcp_server_fixture.get_app()
    client = TestClient(app)
    
    token = mcp_server_fixture.create_test_token()
    
    response = client.get(
        "/mcp/v1/resources/test_resource",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    resource = response.json()
    assert resource["resource_id"] == "test_resource"


def test_query_resource_filesystem(mcp_server_fixture):
    """Test consultar recurso filesystem"""
    mcp_server_fixture.add_mock_resource("test_fs", "filesystem")
    
    # Agregar archivo mock
    fs_connector = mcp_server_fixture.connector_registry.get("filesystem")
    fs_connector.add_file("test.txt", "Hello, World!")
    
    app = mcp_server_fixture.get_app()
    client = TestClient(app)
    
    token = mcp_server_fixture.create_test_token()
    
    response = client.post(
        "/mcp/v1/resources/test_fs/query",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "resource_id": "test_fs",
            "operation": "read",
            "parameters": {"path": "test.txt"},
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["content"] == "Hello, World!"


def test_query_resource_without_auth(mcp_server_fixture):
    """Test que requiere autenticación"""
    app = mcp_server_fixture.get_app()
    client = TestClient(app)
    
    response = client.get("/mcp/v1/resources")
    assert response.status_code == 403  # Unauthorized


def test_query_resource_invalid_token(mcp_server_fixture):
    """Test con token inválido"""
    app = mcp_server_fixture.get_app()
    client = TestClient(app)
    
    response = client.get(
        "/mcp/v1/resources",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 403


def test_query_resource_missing_scope(mcp_server_fixture):
    """Test con scope insuficiente"""
    mcp_server_fixture.add_mock_resource("test_resource", "filesystem")
    
    app = mcp_server_fixture.get_app()
    client = TestClient(app)
    
    # Token sin scopes
    token = mcp_server_fixture.create_test_token(scopes=[])
    
    response = client.get(
        "/mcp/v1/resources/test_resource",
        headers={"Authorization": f"Bearer {token}"}
    )
    # Debería fallar por falta de scope
    assert response.status_code in [403, 404]

