# Quick Start - MCP Server

## Inicio Rápido

### 1. Instalación

```bash
cd agents/backend/onyx/server/features/ai_project_generator
pip install -r requirements.txt
```

### 2. Configurar Servidor MCP

```python
from mcp_server import (
    MCPServer,
    ConnectorRegistry,
    ManifestRegistry,
    MCPSecurityManager,
    FileSystemConnector,
    ResourceManifest,
    ResourceType,
)
from mcp_server.observability import MCPObservability

# 1. Crear registries
connector_registry = ConnectorRegistry()
manifest_registry = ManifestRegistry()

# 2. Registrar conector
fs_connector = FileSystemConnector(base_path="./data")
connector_registry.register("filesystem", fs_connector)

# 3. Crear manifest
manifest = ResourceManifest(
    resource_id="my_files",
    name="My Files",
    type=ResourceType.FILESYSTEM,
    connector_type="filesystem",
    supported_operations=["read", "list"],
)
manifest_registry.register(manifest)

# 4. Configurar seguridad
security_manager = MCPSecurityManager(
    secret_key="your-secret-key-here",
)

# 5. Configurar observabilidad
observability = MCPObservability(
    enable_tracing=True,
    enable_metrics=True,
)

# 6. Crear servidor
server = MCPServer(
    connector_registry=connector_registry,
    manifest_registry=manifest_registry,
    security_manager=security_manager,
    observability=observability,
)

app = server.get_app()
```

### 3. Ejecutar Servidor

```bash
# Opción 1: Con uvicorn
uvicorn main:app --host 0.0.0.0 --port 8020

# Opción 2: Con Docker
docker-compose up -d

# Opción 3: Integrado en aplicación principal
python main.py
```

### 4. Usar el Servidor

```bash
# 1. Crear token
python -c "
from mcp_server.security import MCPSecurityManager
m = MCPSecurityManager('your-secret-key')
print(m.create_access_token({'sub': 'user1', 'scopes': ['read', 'write']}))
"

# 2. Listar recursos
curl -H "Authorization: Bearer <token>" \
  http://localhost:8020/mcp/v1/resources

# 3. Consultar recurso
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "read",
    "parameters": {"path": "file.txt"}
  }' \
  http://localhost:8020/mcp/v1/resources/my_files/query
```

## Cargar Manifests desde Archivos

```python
from mcp_server.manifests import ManifestLoader, ManifestRegistry

registry = ManifestRegistry()

# Cargar desde archivo
ManifestLoader.register_from_file(
    registry,
    "mcp_server/manifests/examples/filesystem_resource.yaml"
)

# Cargar desde directorio
ManifestLoader.register_from_directory(
    registry,
    "mcp_server/manifests/examples/"
)
```

## Testing

```bash
# Tests unitarios
pytest tests/test_mcp_server.py -v

# Tests E2E
pytest tests/test_mcp_e2e.py -v

# Con mocks
pytest tests/ -v --fixtures
```

## Ejemplos

Ver `examples/` para:
- `mcp_basic_usage.py`: Uso básico del servidor
- `mcp_contracts_example.py`: Uso de ContextFrame y PromptFrame

## Documentación Completa

Ver `mcp_server/README.md` para documentación completa.

