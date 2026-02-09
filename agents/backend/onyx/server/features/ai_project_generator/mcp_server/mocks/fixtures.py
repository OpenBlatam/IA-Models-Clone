"""
Pytest Fixtures - Fixtures para testing MCP
============================================
"""

import pytest
from typing import Generator

from .mock_server import MockMCPServer
from ..connectors import ConnectorRegistry
from ..manifests import ManifestRegistry
from ..security import MCPSecurityManager


@pytest.fixture
def mcp_server_fixture() -> Generator[MockMCPServer, None, None]:
    """
    Fixture que proporciona un servidor MCP mock
    
    Yields:
        MockMCPServer configurado
    """
    server = MockMCPServer(enable_observability=False)
    yield server


@pytest.fixture
def mock_connector_registry() -> Generator[ConnectorRegistry, None, None]:
    """
    Fixture que proporciona un registry de conectores mock
    
    Yields:
        ConnectorRegistry con conectores mock
    """
    from .mock_connectors import (
        MockFileSystemConnector,
        MockDatabaseConnector,
        MockAPIConnector,
    )
    
    registry = ConnectorRegistry()
    registry.register("filesystem", MockFileSystemConnector())
    registry.register("database", MockDatabaseConnector())
    registry.register("api", MockAPIConnector())
    
    yield registry


@pytest.fixture
def mock_manifest_registry() -> Generator[ManifestRegistry, None, None]:
    """
    Fixture que proporciona un registry de manifests mock
    
    Yields:
        ManifestRegistry vacío (puede poblarse en tests)
    """
    registry = ManifestRegistry()
    yield registry


@pytest.fixture
def mock_security_manager() -> Generator[MCPSecurityManager, None, None]:
    """
    Fixture que proporciona un security manager mock
    
    Yields:
        MCPSecurityManager configurado para testing
    """
    manager = MCPSecurityManager(
        secret_key="test-secret-key",
        algorithm="HS256",
        access_token_expire_minutes=60,
    )
    yield manager


@pytest.fixture
def test_token(mock_security_manager: MCPSecurityManager) -> str:
    """
    Fixture que proporciona un token de prueba
    
    Args:
        mock_security_manager: Security manager mock
        
    Returns:
        Token JWT válido
    """
    return mock_security_manager.create_access_token({
        "sub": "test_user",
        "scopes": ["read", "write"],
    })

