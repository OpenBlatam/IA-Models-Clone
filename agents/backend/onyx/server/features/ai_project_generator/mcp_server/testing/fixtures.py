"""
Pytest Fixtures - Fixtures mejoradas para testing MCP
======================================================

Fixtures pytest para facilitar testing del servidor MCP.
"""

import pytest
import logging
from typing import Generator, Optional
from fastapi import FastAPI

from ..server import MCPServer
from ..connectors import ConnectorRegistry
from ..manifests import ManifestRegistry
from ..security import MCPSecurityManager
from ..mocks import MockMCPServer
from .client import MCPTestClient, AsyncMCPTestClient
from .helpers import create_test_token

logger = logging.getLogger(__name__)


@pytest.fixture
def mcp_test_client(app: FastAPI) -> MCPTestClient:
    """
    Fixture que proporciona un cliente de testing MCP.
    
    Args:
        app: Aplicación FastAPI
        
    Returns:
        MCPTestClient configurado
    """
    return MCPTestClient(app)


@pytest.fixture
async def async_mcp_test_client_fixture(app: FastAPI) -> AsyncMCPTestClient:
    """
    Fixture que proporciona un cliente de testing asíncrono MCP.
    
    Args:
        app: Aplicación FastAPI
        
    Yields:
        AsyncMCPTestClient configurado
    """
    async with AsyncMCPTestClient(app) as client:
        yield client


@pytest.fixture
def mock_mcp_server() -> Generator[MockMCPServer, None, None]:
    """
    Fixture que proporciona un servidor MCP mock.
    
    Yields:
        MockMCPServer configurado
    """
    server = MockMCPServer(enable_observability=False)
    yield server
    # Cleanup si es necesario
    if hasattr(server, 'cleanup'):
        server.cleanup()


@pytest.fixture
def mock_connector_registry() -> Generator[ConnectorRegistry, None, None]:
    """
    Fixture que proporciona un registry de conectores mock.
    
    Yields:
        ConnectorRegistry vacío
    """
    registry = ConnectorRegistry()
    yield registry
    registry.clear()


@pytest.fixture
def mock_manifest_registry() -> Generator[ManifestRegistry, None, None]:
    """
    Fixture que proporciona un registry de manifests mock.
    
    Yields:
        ManifestRegistry vacío
    """
    registry = ManifestRegistry()
    yield registry
    registry.clear()


@pytest.fixture
def mock_security_manager() -> Generator[MCPSecurityManager, None, None]:
    """
    Fixture que proporciona un security manager mock.
    
    Yields:
        MCPSecurityManager configurado
    """
    # Crear security manager con secret key de prueba
    manager = MCPSecurityManager(
        secret_key="test-secret-key-for-testing-only-" + "x" * 32,
        token_expire_minutes=60
    )
    yield manager


@pytest.fixture
def test_token(mock_security_manager: MCPSecurityManager) -> str:
    """
    Fixture que proporciona un token JWT de prueba.
    
    Args:
        mock_security_manager: Security manager mock
        
    Returns:
        Token JWT válido
    """
    return create_test_token(
        security_manager=mock_security_manager,
        user_id="test_user",
        scopes=["read", "write"]
    )


@pytest.fixture
def authenticated_client(mcp_test_client: MCPTestClient, test_token: str) -> MCPTestClient:
    """
    Fixture que proporciona un cliente autenticado.
    
    Args:
        mcp_test_client: Cliente de testing
        test_token: Token de prueba
        
    Returns:
        Cliente con token configurado
    """
    mcp_test_client.set_auth_token(test_token)
    return mcp_test_client

