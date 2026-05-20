# Guía de Testing - AI Project Generator

## 📚 Índice

1. [Introducción](#introducción)
2. [Estructura de Tests](#estructura-de-tests)
3. [Fixtures Disponibles](#fixtures-disponibles)
4. [Helpers y Utilidades](#helpers-y-utilidades)
5. [Marcadores](#marcadores)
6. [Ejecutar Tests](#ejecutar-tests)
7. [Mejores Prácticas](#mejores-prácticas)
8. [Ejemplos](#ejemplos)

## Introducción

Esta guía explica cómo escribir y ejecutar tests para el AI Project Generator. La suite de tests incluye más de 60 archivos con 650+ casos de test.

## Estructura de Tests

```
tests/
├── conftest.py              # Configuración y fixtures
├── test_helpers.py          # Helpers básicos
├── test_utils_helpers.py   # Helpers avanzados
├── test_examples_improved.py # Ejemplos de uso
├── pytest_plugins.py        # Plugins personalizados
├── run_tests.py             # Script de ejecución
├── test_*.py                # Tests por componente
└── README.md                # Documentación
```

## Fixtures Disponibles

### Directorios y Paths
- `temp_dir`: Directorio temporal (se limpia automáticamente)
- `test_data_dir`: Directorio para datos de test
- `sample_project_structure`: Estructura de proyecto completa

### Generadores
- `project_generator`: Instancia de ProjectGenerator
- `backend_generator`: Instancia de BackendGenerator
- `frontend_generator`: Instancia de FrontendGenerator
- `continuous_generator`: Instancia de ContinuousGenerator

### Datos de Ejemplo
- `sample_project_info`: Información de proyecto de ejemplo
- `sample_keywords`: Keywords extraídas de ejemplo
- `sample_description`: Descripción de ejemplo
- `sample_descriptions`: Múltiples descripciones
- `mock_project_data`: Datos mock de proyecto
- `sample_api_request`: Request de API de ejemplo
- `sample_api_response`: Response de API de ejemplo

### Utilidades
- `event_loop`: Loop de eventos para tests async
- `mock_async_function`: Helper para mocks async
- `performance_timer`: Timer para medir performance

## Helpers y Utilidades

### TestHelpers (test_helpers.py)
```python
from .test_helpers import TestHelpers

# Validar estructura de proyecto
TestHelpers.assert_project_structure(project_path, ["README.md", "main.py"])

# Validar JSON
TestHelpers.assert_valid_json(file_path)

# Validar Python
TestHelpers.assert_valid_python(file_path)

# Verificar contenido
TestHelpers.assert_file_contains(file_path, "expected content")

# Extraer imports
imports = TestHelpers.extract_imports(file_path)
```

### AsyncTestHelpers (test_utils_helpers.py)
```python
from .test_utils_helpers import AsyncTestHelpers

# Esperar condición
await AsyncTestHelpers.wait_for_condition(
    lambda: condition_met,
    timeout=5.0
)

# Reintentar operación
result = await AsyncTestHelpers.retry_async(
    flaky_operation,
    max_attempts=3
)

# Timeout
result = await AsyncTestHelpers.timeout_async(
    slow_operation,
    timeout=10.0
)
```

### FileTestHelpers
```python
from .test_utils_helpers import FileTestHelpers

# Crear archivo temporal
temp_file = FileTestHelpers.create_temp_file("content", suffix=".txt")

# Validar tamaño
FileTestHelpers.assert_file_size(file_path, min_size=100)

# Limpiar
FileTestHelpers.cleanup_path(temp_file)
```

### MockHelpers
```python
from .test_utils_helpers import MockHelpers

# Crear mock async
async_mock = MockHelpers.create_async_mock(return_value={"success": True})

# Crear response mock
response = MockHelpers.create_mock_response(
    status_code=200,
    json_data={"data": "test"}
)

# Crear datos de proyecto
project_data = MockHelpers.create_mock_project_data(project_id="test-123")
```

### PerformanceTestHelpers
```python
from .test_utils_helpers import PerformanceTestHelpers

# Medir tiempo
with PerformanceTestHelpers.measure_time() as elapsed:
    result = operation()

# Validar performance
PerformanceTestHelpers.assert_performance(
    elapsed,
    max_time=1.0,
    operation_name="Operation"
)

# Benchmark
stats = PerformanceTestHelpers.benchmark_function(func, iterations=10)
```

### ValidationHelpers
```python
from .test_utils_helpers import ValidationHelpers

# Validar estructura
ValidationHelpers.validate_project_structure(
    project_path,
    required_dirs=["backend", "frontend"],
    required_files=["README.md"]
)

# Validar JSON
ValidationHelpers.validate_json_structure(
    data,
    required_keys=["id", "name"],
    key_types={"id": str, "name": str}
)

# Validar API response
ValidationHelpers.validate_api_response(
    response,
    expected_status=200,
    required_fields=["data", "status"]
)
```

### DataGenerators
```python
from .test_utils_helpers import DataGenerators

# Generar nombre único
name = DataGenerators.generate_project_name("test")

# Generar string grande
large_string = DataGenerators.generate_large_string(size=1000)

# Generar descripción
desc = DataGenerators.generate_project_description(
    ai_type="chat",
    features=["auth", "database"]
)
```

## Marcadores

### Uso de Marcadores
```python
@pytest.mark.unit
def test_unit_function():
    pass

@pytest.mark.integration
def test_integration_flow():
    pass

@pytest.mark.async
async def test_async_operation():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.security
def test_security_feature():
    pass

@pytest.mark.performance
def test_performance():
    pass
```

### Ejecutar por Marcador
```bash
# Solo tests unitarios
pytest -m unit

# Solo tests de integración
pytest -m integration

# Excluir tests lentos
pytest -m "not slow"

# Tests rápidos (sin slow ni integration)
pytest -m "not slow and not integration"
```

## Ejecutar Tests

### Opción 1: pytest directamente
```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_api.py

# Con verbose
pytest -v

# Con cobertura
pytest --cov=. --cov-report=html
```

### Opción 2: Script mejorado
```bash
# Todos los tests
python tests/run_tests.py

# Solo unit tests
python tests/run_tests.py --unit

# Solo integration tests
python tests/run_tests.py --integration

# Tests rápidos
python tests/run_tests.py --fast

# Con cobertura
python tests/run_tests.py --coverage

# En paralelo
python tests/run_tests.py --parallel

# Tests específicos
python tests/run_tests.py tests/test_api.py
```

## Mejores Prácticas

### 1. Usar Fixtures
```python
def test_example(project_generator, temp_dir):
    # Usar fixtures en lugar de crear instancias manualmente
    project = project_generator.generate_project("test")
```

### 2. Usar Helpers
```python
from .test_helpers import TestHelpers

def test_validation(project_path):
    # Usar helpers para validaciones comunes
    TestHelpers.assert_project_structure(project_path, ["README.md"])
```

### 3. Documentar Tests
```python
def test_feature():
    """
    Test que verifica que la feature X funciona correctamente.
    
    Steps:
    1. Crear proyecto
    2. Validar estructura
    3. Verificar contenido
    """
```

### 4. Usar Marcadores
```python
@pytest.mark.unit
def test_fast():
    pass

@pytest.mark.slow
def test_slow():
    pass
```

### 5. Manejar Errores
```python
def test_error_handling():
    with pytest.raises(ValueError, match="expected error"):
        function_that_raises()
```

### 6. Tests Async
```python
@pytest.mark.async
async def test_async():
    result = await async_function()
    assert result is not None
```

## Ejemplos

### Ejemplo 1: Test Básico
```python
def test_simple_feature(project_generator):
    result = project_generator._sanitize_name("Test Project")
    assert result == "test_project"
```

### Ejemplo 2: Test con Validación
```python
def test_project_generation(project_generator, temp_dir):
    project = project_generator.generate_project("A chat AI")
    
    project_path = Path(project["project_path"])
    TestHelpers.assert_project_structure(
        project_path,
        ["README.md", "backend/main.py"]
    )
```

### Ejemplo 3: Test Async
```python
@pytest.mark.async
async def test_async_operation(project_generator):
    project = await project_generator.generate_project("Test")
    assert project is not None
```

### Ejemplo 4: Test con Mocks
```python
@patch('module.external_api')
def test_with_mock(mock_api, project_generator):
    mock_api.return_value = {"success": True}
    result = project_generator.call_external_api()
    assert result["success"] is True
```

### Ejemplo 5: Test de Performance
```python
@pytest.mark.performance
def test_performance(project_generator):
    with PerformanceTestHelpers.measure_time() as elapsed:
        project_generator.generate_project("Test")
    
    PerformanceTestHelpers.assert_performance(
        elapsed,
        max_time=5.0,
        operation_name="Project generation"
    )
```

## Consejos Adicionales

1. **Mantener tests independientes**: Cada test debe poder ejecutarse solo
2. **Usar nombres descriptivos**: `test_generate_project_with_auth` es mejor que `test_1`
3. **Agrupar tests relacionados**: Usar clases para agrupar tests relacionados
4. **Limpiar recursos**: Usar fixtures para limpieza automática
5. **Evitar dependencias**: No depender del orden de ejecución
6. **Tests rápidos primero**: Ejecutar tests rápidos antes que los lentos
7. **Cobertura**: Mantener cobertura alta pero no obsesionarse con 100%

## Recursos

- [Documentación de pytest](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [README de tests](./README.md)

