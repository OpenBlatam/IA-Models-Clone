# Troubleshooting Guide - Testing

## 🔧 Problemas Comunes y Soluciones

### Problema: Tests fallan intermitentemente

**Causa**: Race conditions o dependencias de tiempo

**Solución**:
```python
# Usar retry para operaciones flaky
await AsyncTestHelpers.retry_async(
    operation,
    max_attempts=3,
    delay=0.5
)

# O esperar condiciones
await AsyncTestHelpers.wait_for_condition(
    lambda: condition_met,
    timeout=5.0
)
```

### Problema: Tests lentos

**Causa**: Operaciones pesadas o esperas innecesarias

**Solución**:
```python
# Marcar tests lentos
@pytest.mark.slow
def test_slow_operation():
    pass

# Ejecutar solo tests rápidos
pytest -m "not slow"
```

### Problema: Tests async no funcionan

**Causa**: Configuración incorrecta de asyncio

**Solución**:
```python
# Asegurar que pytest-asyncio está instalado
# En pytest.ini: asyncio_mode = auto

# Usar marcador
@pytest.mark.async
async def test_async():
    pass
```

### Problema: Directorios temporales no se limpian

**Causa**: Errores en teardown

**Solución**:
```python
# Usar fixtures con cleanup mejorado
@pytest.fixture
def temp_dir():
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    try:
        shutil.rmtree(temp_path, ignore_errors=True)
    except Exception:
        pass
```

### Problema: Mocks no funcionan correctamente

**Causa**: Configuración incorrecta de mocks

**Solución**:
```python
# Usar MockHelpers
from .test_utils_helpers import MockHelpers

mock = MockHelpers.create_async_mock(return_value={"success": True})

# O usar patch correctamente
@patch('module.function')
def test_with_patch(mock_func):
    mock_func.return_value = "result"
    # test code
```

### Problema: Validaciones fallan sin información clara

**Causa**: Aserciones genéricas

**Solución**:
```python
# Usar aserciones personalizadas
from .test_assertions import CustomAssertions

CustomAssertions.assert_project_exists(project_path, "expected_name")
CustomAssertions.assert_backend_structure(project_path)

# O usar TestHelpers con mensajes claros
TestHelpers.assert_project_structure(
    project_path,
    ["README.md", "backend/main.py"]
)
```

### Problema: Debugging tests es difícil

**Causa**: Falta de información de debug

**Solución**:
```python
# Usar DebugHelpers
from .debug_helpers import DebugHelpers

# En el test
DebugHelpers.print_test_info("Test Name", var1=value1, var2=value2)
DebugHelpers.print_project_structure(project_path)
DebugHelpers.print_file_content(file_path)
```

### Problema: Tests de performance son inconsistentes

**Causa**: Variabilidad en tiempos de ejecución

**Solución**:
```python
# Usar benchmarks con múltiples iteraciones
from .test_utils_helpers import PerformanceTestHelpers

stats = PerformanceTestHelpers.benchmark_function(
    function,
    iterations=10
)

# Usar rangos en lugar de valores exactos
assert stats["avg"] < max_time
```

### Problema: Cobertura no se genera correctamente

**Causa**: Configuración incorrecta

**Solución**:
```bash
# Verificar pytest-cov está instalado
pip install pytest-cov

# Ejecutar con cobertura
pytest --cov=. --cov-report=html

# Verificar pytest.ini tiene configuración de cobertura
```

### Problema: Tests fallan en CI pero no localmente

**Causa**: Diferencias de entorno

**Solución**:
```python
# Usar variables de entorno para configuración
import os

test_mode = os.getenv("TEST_MODE", "local")

if test_mode == "ci":
    # Configuración para CI
    pass
else:
    # Configuración local
    pass
```

## 🐛 Debugging Tips

### 1. Usar verbose output
```bash
pytest -v
pytest -vv  # Más verbose
```

### 2. Ejecutar test específico
```bash
pytest tests/test_specific.py::TestClass::test_method -v
```

### 3. Mostrar print statements
```bash
pytest -s
```

### 4. Usar debug helpers
```python
def test_example(debug):
    debug.print_test_info("Test", var=value)
    debug.print_project_structure(project_path)
```

### 5. Capturar excepciones
```python
try:
    operation()
except Exception as e:
    DebugHelpers.print_exception_info(e)
    raise
```

## 📊 Performance Debugging

### Identificar tests lentos
```bash
pytest --durations=10  # Mostrar 10 tests más lentos
```

### Medir tiempo específico
```python
from .test_utils_helpers import PerformanceTestHelpers

with PerformanceTestHelpers.measure_time() as elapsed:
    operation()

print(f"Operation took {elapsed:.2f}s")
```

## 🔍 Verificación de Tests

### Verificar que tests se descubren
```bash
pytest --collect-only
```

### Verificar marcadores
```bash
pytest --markers
```

### Verificar configuración
```bash
pytest --version
pytest --fixtures
```

## 💡 Consejos Adicionales

1. **Usar fixtures**: Evitar código duplicado
2. **Marcar tests apropiadamente**: Facilitar ejecución selectiva
3. **Documentar tests**: Explicar qué y por qué se prueba
4. **Mantener tests simples**: Un concepto por test
5. **Usar helpers**: Reutilizar código común
6. **Limpiar recursos**: Usar fixtures para cleanup
7. **Validar errores**: Probar tanto éxito como fallos
8. **Mantener independencia**: Tests no deben depender entre sí

## 📚 Recursos

- [pytest documentation](https://docs.pytest.org/)
- [Testing Guide](./TESTING_GUIDE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [Quick Start](./QUICK_START_TESTING.md)

