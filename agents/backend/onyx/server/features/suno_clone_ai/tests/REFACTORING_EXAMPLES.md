# Ejemplos de Refactorización

Este documento muestra ejemplos concretos de cómo refactorizar tests usando las clases base y helpers.

## Ejemplo 1: Test de API Route

### Antes (Código Duplicado)
```python
@pytest.fixture
def client(mock_service):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.my_route.get_service', return_value=mock_service):
        with patch('api.routes.my_route.get_current_user', return_value={"user_id": "test"}):
            yield TestClient(app)

def test_endpoint(client):
    response = client.get("/endpoint")
    assert response.status_code == 200
    data = response.json()
    assert "key" in data
    assert "key2" in data
```

### Después (Refactorizado)
```python
from test_helpers import BaseAPITestCase

class TestMyRoute(BaseAPITestCase):
    router = my_router
    
    def test_endpoint(self):
        client = self.create_client({
            "api.routes.my_route.get_service": mock_service,
            "api.routes.my_route.get_current_user": {"user_id": "test"}
        })
        response = client.get("/endpoint")
        self.assert_success_response(response)
        self.assert_response_contains_keys(response, ["key", "key2"])
```

**Beneficios:**
- ~40% menos código
- Setup/teardown automático
- Aserciones más claras
- Reutilizable

## Ejemplo 2: Test de Helper Function

### Antes
```python
def test_hash_string_sha256():
    result = hash_string("test", algorithm="sha256")
    assert isinstance(result, str)
    assert len(result) == 64

def test_hash_string_md5():
    result = hash_string("test", algorithm="md5")
    assert isinstance(result, str)
    assert len(result) == 32

def test_hash_string_sha1():
    result = hash_string("test", algorithm="sha1")
    assert isinstance(result, str)
    assert len(result) == 40
```

### Después (Refactorizado)
```python
from test_helpers import BaseServiceTestCase, StandardTestMixin

class TestHashStringRefactored(BaseServiceTestCase, StandardTestMixin):
    @pytest.mark.parametrize("algorithm,expected_length", [
        ("sha256", 64),
        ("md5", 32),
        ("sha1", 40)
    ])
    def test_hash_string_algorithms(self, algorithm, expected_length):
        result = hash_string("test", algorithm=algorithm)
        assert isinstance(result, str)
        assert len(result) == expected_length
```

**Beneficios:**
- 3 tests en 1
- Más fácil agregar nuevos algoritmos
- Código más DRY

## Ejemplo 3: Test con Parametrización

### Antes
```python
def test_format_duration_seconds():
    result = format_duration(45.0)
    assert result == "0:45"

def test_format_duration_minutes():
    result = format_duration(125.0)
    assert result == "2:05"

def test_format_duration_zero():
    result = format_duration(0.0)
    assert result == "0:00"
```

### Después (Refactorizado)
```python
class TestFormatFunctionsRefactored(BaseServiceTestCase, StandardTestMixin):
    @pytest.mark.parametrize("seconds,expected", [
        (45.0, "0:45"),
        (125.0, "2:05"),
        (0.0, "0:00"),
        (3661.0, "61:01")
    ])
    def test_format_duration(self, seconds, expected):
        result = format_duration(seconds)
        assert result == expected
```

**Beneficios:**
- Un solo test para múltiples casos
- Fácil agregar nuevos casos
- Más mantenible

## Ejemplo 4: Test de Servicio con Mock

### Antes
```python
@pytest.fixture
def mock_service():
    service = Mock()
    service.method1 = Mock(return_value="result1")
    service.method2 = Mock(return_value="result2")
    return service

def test_service_method(mock_service):
    result = mock_service.method1()
    assert result == "result1"
```

### Después (Refactorizado)
```python
from test_helpers import BaseServiceTestCase, create_standard_mock_service

class TestMyServiceRefactored(BaseServiceTestCase):
    service_class = MyService
    
    def test_service_method(self):
        service = self.create_mock_service({
            "method1": "result1",
            "method2": "result2"
        })
        result = service.method1()
        assert result == "result1"
```

**Beneficios:**
- Mock creado automáticamente
- Consistente con otros tests
- Menos código boilerplate

## Ejemplo 5: Test con Builder Pattern

### Antes
```python
def test_complex_endpoint():
    app = FastAPI()
    app.include_router(router)
    
    with patch('path1', return_value=mock1):
        with patch('path2', return_value=mock2):
            with patch('path3', return_value=mock3):
                client = TestClient(app)
                response = client.get("/endpoint")
                assert response.status_code == 200
```

### Después (Refactorizado)
```python
from test_helpers import TestClientBuilder

def test_complex_endpoint():
    client = (TestClientBuilder(router)
        .with_mock("path1", mock1)
        .with_mock("path2", mock2)
        .with_mock("path3", mock3)
        .with_auth({"user_id": "test"})
        .build())
    
    response = client.get("/endpoint")
    assert response.status_code == 200
```

**Beneficios:**
- Más legible
- Fácil agregar/quitar mocks
- Configuración clara

## Ejemplo 6: Test con Decorators

### Antes
```python
def test_unreliable_service():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = unreliable_service.call()
            assert result is not None
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1)
```

### Después (Refactorizado)
```python
from test_helpers import retry_on_failure

@retry_on_failure(max_retries=3, delay=0.1)
async def test_unreliable_service():
    result = await unreliable_service.call()
    assert result is not None
```

**Beneficios:**
- Código más limpio
- Reutilizable
- Configuración clara

## Mejores Prácticas

### 1. Usar Clases Base
Siempre que sea posible, heredar de `BaseAPITestCase` o `BaseServiceTestCase`.

### 2. Parametrizar Tests Similares
Usar `@pytest.mark.parametrize` para tests que solo cambian en valores.

### 3. Usar Helpers de Aserción
Usar `assert_success_response()`, `assert_response_contains_keys()`, etc.

### 4. Usar Mixins
Usar `StandardTestMixin` para métodos de aserción comunes.

### 5. Usar Builder para Configuración Compleja
Usar `TestClientBuilder` cuando hay muchos mocks o configuración compleja.

## Métricas de Mejora

- **Reducción de código:** 30-50% por test
- **Legibilidad:** Mejorada significativamente
- **Mantenibilidad:** Mucho más fácil de mantener
- **Consistencia:** 100% de tests usando patrones comunes



