# Guía de Testing Avanzado - MCP Server

## Resumen

Utilidades avanzadas para testing del módulo MCP Server, incluyendo factories, mocks, helpers asíncronos, y assertions personalizadas.

## Componentes

### 1. TestDataFactory

Factory para crear datos de test consistentes.

```python
from mcp_server.utils.testing_advanced import TestDataFactory

# Crear manifest de recurso
manifest = TestDataFactory.create_resource_manifest(
    resource_id="test-resource",
    name="Test Resource",
    resource_type="filesystem"
)

# Crear usuario
user = TestDataFactory.create_user(
    user_id="test-user",
    email="test@example.com",
    scopes=["read", "write"]
)

# Crear request
request = TestDataFactory.create_request(
    resource_id="test-resource",
    operation="read",
    parameters={"path": "/test"}
)
```

### 2. MockHelper

Helper para crear mocks de componentes.

```python
from mcp_server.utils.testing_advanced import MockHelper

# Crear mock de conector
connector = MockHelper.create_mock_connector(
    connector_type="filesystem",
    operations=["read", "list", "write"]
)

# Crear mock de security manager
security_manager = MockHelper.create_mock_security_manager()
```

### 3. AsyncTestHelper

Helper para testing asíncrono.

```python
from mcp_server.utils.testing_advanced import AsyncTestHelper

# Ejecutar coroutine
result = AsyncTestHelper.run_async(async_function())

# Usar context manager
with AsyncTestHelper.async_context() as loop:
    result = loop.run_until_complete(async_function())
```

### 4. TestAssertions

Assertions personalizadas.

```python
from mcp_server.utils.testing_advanced import TestAssertions

# Verificar estructura de respuesta
TestAssertions.assert_response_structure(
    response,
    required_fields=["status", "data"]
)

# Verificar tiempo de respuesta
TestAssertions.assert_response_time(response_time, max_time=1.0)

# Verificar código de error
TestAssertions.assert_error_code(response, "RESOURCE_NOT_FOUND")
```

### 5. TestFixtureManager

Gestor de fixtures de testing.

```python
from mcp_server.utils.testing_advanced import TestFixtureManager

manager = TestFixtureManager()

# Registrar fixture
manager.register_fixture(
    "test_connector",
    connector,
    cleanup=lambda: connector.cleanup()
)

# Obtener fixture
connector = manager.get_fixture("test_connector")

# Limpiar todas las fixtures
manager.cleanup()
```

### 6. Decoradores

#### `@patch_config`

Parchear configuración para tests.

```python
from mcp_server.utils.testing_advanced import patch_config

@patch_config({"server.port": 8080, "server.debug": True})
def test_with_custom_config():
    # Test con configuración personalizada
    ...
```

#### `@with_timeout`

Agregar timeout a tests.

```python
from mcp_server.utils.testing_advanced import with_timeout

@with_timeout(5.0)
def test_with_timeout():
    # Test con timeout de 5 segundos
    ...
```

### 7. MockResponse

Mock de respuesta HTTP.

```python
from mcp_server.utils.testing_advanced import MockResponse

response = MockResponse(
    status_code=200,
    json_data={"status": "success"},
    headers={"Content-Type": "application/json"}
)

data = response.json()
```

## Ejemplos de Uso

### Test Completo con Factory

```python
from mcp_server.utils.testing_advanced import (
    TestDataFactory, MockHelper, TestAssertions
)

def test_resource_operation():
    # Crear datos de test
    manifest = TestDataFactory.create_resource_manifest()
    user = TestDataFactory.create_user()
    request = TestDataFactory.create_request()
    
    # Crear mocks
    connector = MockHelper.create_mock_connector()
    security_manager = MockHelper.create_mock_security_manager()
    
    # Ejecutar operación
    response = execute_operation(manifest, request, user)
    
    # Verificar respuesta
    TestAssertions.assert_response_structure(
        response,
        ["status", "data"]
    )
```

### Test Asíncrono

```python
from mcp_server.utils.testing_advanced import AsyncTestHelper

def test_async_operation():
    async def async_op():
        return await some_async_function()
    
    result = AsyncTestHelper.run_async(async_op())
    assert result is not None
```

### Test con Fixtures

```python
from mcp_server.utils.testing_advanced import TestFixtureManager

def test_with_fixtures():
    manager = TestFixtureManager()
    
    try:
        # Registrar fixtures
        connector = create_test_connector()
        manager.register_fixture("connector", connector)
        
        # Usar fixtures
        connector = manager.get_fixture("connector")
        result = connector.execute("read", {})
        
        assert result is not None
    finally:
        manager.cleanup()
```

### Test con Configuración Parcheada

```python
from mcp_server.utils.testing_advanced import patch_config

@patch_config({
    "server.port": 8080,
    "security.secret_key": "test-secret-key"
})
def test_with_custom_config():
    # Test ejecutado con configuración personalizada
    config = get_config()
    assert config["server"]["port"] == 8080
```

## Integración con pytest

```python
import pytest
from mcp_server.utils.testing_advanced import (
    TestDataFactory, MockHelper, TestFixtureManager
)

@pytest.fixture
def fixture_manager():
    manager = TestFixtureManager()
    yield manager
    manager.cleanup()

@pytest.fixture
def test_connector(fixture_manager):
    connector = MockHelper.create_mock_connector()
    fixture_manager.register_fixture("connector", connector)
    return connector

def test_resource_operation(test_connector):
    manifest = TestDataFactory.create_resource_manifest()
    # Test usando fixtures
    ...
```

## Próximos Pasos

1. Agregar más factories de datos
2. Mejorar mocks con más funcionalidades
3. Agregar más assertions
4. Integrar con frameworks de testing
5. Agregar coverage helpers

