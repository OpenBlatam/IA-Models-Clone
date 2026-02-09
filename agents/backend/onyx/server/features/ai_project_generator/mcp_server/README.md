# MCP Server - Model Context Protocol Integration Layer

## Introducción

El servidor MCP (Model Context Protocol) proporciona una capa de integración estandarizada para que los modelos de IA puedan consultar fuentes autorizadas de contexto de forma segura y observable.

## Características

- ✅ **Conectores estandarizados**: Filesystem, Database, API
- ✅ **Manifiestos de recursos**: Descriptores JSON/YAML con validación Pydantic
- ✅ **Seguridad**: OAuth2/JWT, scopes, auditoría de acceso
- ✅ **Observabilidad**: Tracing OpenTelemetry, métricas Prometheus, logs estructurados
- ✅ **Contratos estandarizados**: ContextFrame y PromptFrame para entrada/salida
- ✅ **Mocks para testing**: Fixtures pytest y servidor mock
- ✅ **Tests E2E**: Pipelines completos de integración
- ✅ **Docker**: Empaquetado reproducible
- ✅ **CI/CD**: Checks automáticos y validación

## Arquitectura

```
┌─────────────────┐
│   LLM/Model     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MCP Server    │
│  (FastAPI)      │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬─────────┐
    ▼        ▼          ▼         ▼
┌────────┐ ┌──────┐ ┌──────┐ ┌──────────┐
│ Files  │ │  DB  │ │ APIs │ │  Cache   │
└────────┘ └──────┘ └──────┘ └──────────┘
```

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# O con Docker
docker-compose up -d
```

## Uso Rápido

### 1. Crear un recurso

```python
from mcp_server import (
    ResourceManifest,
    ResourceType,
    ResourcePermissions,
    ManifestRegistry,
)

manifest = ResourceManifest(
    resource_id="my_files",
    name="My Files",
    type=ResourceType.FILESYSTEM,
    connector_type="filesystem",
    description="My filesystem resource",
    supported_operations=["read", "list"],
    permissions=ResourcePermissions(read=True),
)

registry = ManifestRegistry()
registry.register(manifest)
```

### 2. Consultar un recurso

```python
from mcp_server import MCPServer, ConnectorRegistry, MCPSecurityManager
from mcp_server.observability import MCPObservability

# Configurar servidor
connector_registry = ConnectorRegistry()
manifest_registry = ManifestRegistry()
security_manager = MCPSecurityManager(secret_key="your-secret-key")
observability = MCPObservability()

server = MCPServer(
    connector_registry=connector_registry,
    manifest_registry=manifest_registry,
    security_manager=security_manager,
    observability=observability,
)

app = server.get_app()
```

### 3. Hacer request

```bash
# Obtener token
TOKEN=$(python -c "from mcp_server.security import MCPSecurityManager; m = MCPSecurityManager('secret'); print(m.create_access_token({'sub': 'user', 'scopes': ['read']}))")

# Listar recursos
curl -H "Authorization: Bearer $TOKEN" http://localhost:8020/mcp/v1/resources

# Consultar recurso
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"operation": "read", "parameters": {"path": "file.txt"}}' \
  http://localhost:8020/mcp/v1/resources/my_files/query
```

## Estructura del Proyecto

```
mcp_server/
├── __init__.py              # Exports principales
├── server.py                 # Servidor MCP principal
├── connectors/              # Conectores (filesystem, database, api)
├── manifests/                # Manifiestos de recursos
├── security/                 # Seguridad y autenticación
├── contracts/                # Contratos de entrada/salida
├── observability/            # Observabilidad (tracing, métricas)
└── mocks/                    # Mocks para testing
```

## Documentación

- [Contratos MCP](docs/mcp_contracts.md): Documentación de ContextFrame y PromptFrame
- [Ejemplos](examples/): Ejemplos de uso
- [Tests](tests/): Tests unitarios y E2E

## Testing

```bash
# Tests unitarios
pytest tests/test_mcp_server.py -v

# Tests E2E
pytest tests/test_mcp_e2e.py -v

# Con coverage
pytest tests/test_mcp*.py --cov=mcp_server --cov-report=html
```

## Desarrollo

```bash
# Con Docker (incluye mock)
docker-compose --profile dev up

# Con mocks locales
pytest tests/ -v --fixtures
```

## Seguridad

- **Tokens JWT**: Todos los endpoints requieren autenticación
- **Scopes**: Control de acceso por recurso y operación
- **Auditoría**: Logs de todas las operaciones
- **Políticas**: Políticas de acceso configurables por recurso

## Observabilidad

- **Métricas**: Prometheus en `/metrics`
- **Tracing**: OpenTelemetry (configurable)
- **Logs**: Logging estructurado

## CI/CD

El pipeline incluye:
- Lint (Black, Ruff, Pylint)
- Type checking (MyPy)
- Tests (unit + E2E)
- Security scan (Bandit)
- Validación de manifiestos
- Docker build

Ver `.github/workflows/mcp-ci.yml` para detalles.

## Contribuir

1. Crear branch desde `develop`
2. Implementar cambios
3. Agregar tests
4. Validar con `pytest` y `mypy`
5. Crear PR

## Licencia

Ver LICENSE del proyecto principal.

