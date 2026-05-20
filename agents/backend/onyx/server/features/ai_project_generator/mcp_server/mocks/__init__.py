"""
MCP Mocks - Mocks y emulación local para desarrollo y tests
============================================================

Fixtures pytest y mcp_mock para simular responses de recursos
(DB, búsquedas, archivos) sin acceso a sistemas reales.
"""

from .mock_server import MockMCPServer
from .mock_connectors import (
    MockFileSystemConnector,
    MockDatabaseConnector,
    MockAPIConnector,
)
from .fixtures import (
    mcp_server_fixture,
    mock_connector_registry,
    mock_manifest_registry,
    mock_security_manager,
)

__all__ = [
    "MockMCPServer",
    "MockFileSystemConnector",
    "MockDatabaseConnector",
    "MockAPIConnector",
    "mcp_server_fixture",
    "mock_connector_registry",
    "mock_manifest_registry",
    "mock_security_manager",
]

