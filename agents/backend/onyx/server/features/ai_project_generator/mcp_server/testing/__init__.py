"""
MCP Testing Utilities - Utilidades mejoradas de testing
========================================================

Utilidades completas para testing del servidor MCP, incluyendo:
- Cliente de testing mejorado
- Fixtures pytest
- Helpers para mocks
- Assertions personalizadas
"""

from .client import MCPTestClient, AsyncMCPTestClient
from .fixtures import (
    mcp_test_client_fixture as mcp_test_client,
    async_mcp_test_client_fixture as async_mcp_test_client,
    mock_mcp_server,
    mock_connector_registry,
    mock_manifest_registry,
    mock_security_manager,
    test_token,
    authenticated_client,
)
from .helpers import (
    create_mock_connector,
    create_mock_manifest,
    create_test_user,
    create_test_token,
)
from .assertions import (
    assert_mcp_response,
    assert_mcp_success,
    assert_mcp_error,
    assert_mcp_authorized,
    assert_mcp_forbidden,
)

__all__ = [
    # Clients
    "MCPTestClient",
    "AsyncMCPTestClient",
    # Fixtures
    "mcp_test_client",
    "async_mcp_test_client",
    "mock_mcp_server",
    "mock_connector_registry",
    "mock_manifest_registry",
    "mock_security_manager",
    "test_token",
    # Helpers
    "create_mock_connector",
    "create_mock_manifest",
    "create_test_user",
    "create_test_token",
    # Assertions
    "assert_mcp_response",
    "assert_mcp_success",
    "assert_mcp_error",
    "assert_mcp_authorized",
    "assert_mcp_forbidden",
]

