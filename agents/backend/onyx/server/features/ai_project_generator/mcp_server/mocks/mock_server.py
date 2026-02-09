"""
Mock MCP Server - Servidor MCP mock para testing
=================================================
"""

from ..server import MCPServer
from ..connectors import ConnectorRegistry
from ..manifests import ManifestRegistry
from ..security import MCPSecurityManager
from ..observability import MCPObservability
from .mock_connectors import (
    MockFileSystemConnector,
    MockDatabaseConnector,
    MockAPIConnector,
)


class MockMCPServer:
    """
    Servidor MCP mock para testing
    
    Configura un servidor MCP completo con conectores mock.
    """
    
    def __init__(self, enable_observability: bool = False):
        """
        Inicializa servidor mock
        
        Args:
            enable_observability: Habilitar observabilidad (opcional)
        """
        # Crear registries
        self.connector_registry = ConnectorRegistry()
        self.manifest_registry = ManifestRegistry()
        
        # Registrar conectores mock
        self.connector_registry.register("filesystem", MockFileSystemConnector())
        self.connector_registry.register("database", MockDatabaseConnector())
        self.connector_registry.register("api", MockAPIConnector())
        
        # Crear security manager mock
        self.security_manager = MCPSecurityManager(
            secret_key="mock-secret-key-for-testing",
            algorithm="HS256",
        )
        
        # Crear observability (opcional)
        self.observability = MCPObservability(
            enable_tracing=enable_observability,
            enable_metrics=enable_observability,
        ) if enable_observability else None
        
        # Crear servidor
        self.server = MCPServer(
            connector_registry=self.connector_registry,
            manifest_registry=self.manifest_registry,
            security_manager=self.security_manager,
            observability=self.observability,
        )
    
    def get_app(self):
        """Retorna aplicación FastAPI"""
        return self.server.get_app()
    
    def create_test_token(self, user_id: str = "test_user", scopes: list[str] = None) -> str:
        """
        Crea token de prueba
        
        Args:
            user_id: ID del usuario
            scopes: Lista de scopes
            
        Returns:
            Token JWT
        """
        if scopes is None:
            scopes = ["read", "write"]
        
        return self.security_manager.create_access_token({
            "sub": user_id,
            "scopes": scopes,
        })
    
    def add_mock_resource(self, resource_id: str, connector_type: str = "filesystem"):
        """
        Agrega recurso mock al registry
        
        Args:
            resource_id: ID del recurso
            connector_type: Tipo de conector
        """
        from ..manifests import ResourceManifest, ResourceType
        
        manifest = ResourceManifest(
            resource_id=resource_id,
            name=f"Mock {resource_id}",
            type=ResourceType(connector_type),
            connector_type=connector_type,
            description=f"Mock resource for testing: {resource_id}",
        )
        
        self.manifest_registry.register(manifest)

