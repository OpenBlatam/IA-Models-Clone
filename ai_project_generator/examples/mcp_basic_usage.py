"""
Ejemplo básico de uso del servidor MCP
=======================================

Muestra cómo:
1. Exponer un recurso al MCP
2. El LLM pide contexto
3. Auditar la llamada
"""

import asyncio
from mcp_server import (
    MCPServer,
    ConnectorRegistry,
    ManifestRegistry,
    MCPSecurityManager,
    ResourceManifest,
    ResourceType,
    FileSystemConnector,
)
from mcp_server.observability import MCPObservability


async def main():
    """Ejemplo principal"""
    
    # 1. Configurar registries
    connector_registry = ConnectorRegistry()
    manifest_registry = ManifestRegistry()
    
    # 2. Registrar conector de filesystem
    fs_connector = FileSystemConnector(base_path="./examples/data")
    connector_registry.register("filesystem", fs_connector)
    
    # 3. Crear manifest de recurso
    manifest = ResourceManifest(
        resource_id="example_files",
        name="Example Files",
        type=ResourceType.FILESYSTEM,
        connector_type="filesystem",
        description="Example filesystem resource for demonstration",
        supported_operations=["read", "list", "search"],
        permissions=ResourcePermissions(read=True, write=False),
    )
    manifest_registry.register(manifest)
    
    # 4. Configurar seguridad
    security_manager = MCPSecurityManager(
        secret_key="example-secret-key-change-in-production",
        algorithm="HS256",
    )
    
    # 5. Configurar observabilidad
    observability = MCPObservability(
        enable_tracing=True,
        enable_metrics=True,
    )
    
    # 6. Crear servidor MCP
    server = MCPServer(
        connector_registry=connector_registry,
        manifest_registry=manifest_registry,
        security_manager=security_manager,
        observability=observability,
    )
    
    # 7. Obtener aplicación FastAPI
    app = server.get_app()
    
    print("✅ Servidor MCP configurado exitosamente")
    print("📡 Servidor disponible en http://0.0.0.0:8020")
    print("📊 Health check: http://0.0.0.0:8020/mcp/v1/health")
    print("\nPara usar el servidor:")
    print("1. Crear token: security_manager.create_access_token({'sub': 'user_id', 'scopes': ['read']})")
    print("2. Hacer request: GET /mcp/v1/resources con Authorization: Bearer <token>")
    print("3. Consultar recurso: POST /mcp/v1/resources/example_files/query")
    
    return app


if __name__ == "__main__":
    app = asyncio.run(main())

