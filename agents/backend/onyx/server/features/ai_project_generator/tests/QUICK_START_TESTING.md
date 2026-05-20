# Quick Start - Testing 🚀

## Ejecutar Tests Rápidamente

### Opción 1: pytest (Recomendado)
```bash
# Todos los tests
pytest

# Solo tests rápidos
pytest -m "not slow"

# Solo unit tests
pytest -m unit

# Con cobertura
pytest --cov=. --cov-report=html
```

### Opción 2: Script Mejorado
```bash
# Todos los tests
python tests/run_tests.py

# Tests rápidos
python tests/run_tests.py --fast

# Solo unit tests
python tests/run_tests.py --unit

# Con cobertura
python tests/run_tests.py --coverage
```

## Ejemplos Rápidos

### Test Básico
```python
def test_example(project_generator):
    result = project_generator._sanitize_name("Test")
    assert result == "test"
```

### Test con Helpers
```python
from .test_helpers import TestHelpers

def test_with_helpers(project_path):
    TestHelpers.assert_project_structure(project_path, ["README.md"])
```

### Test Async
```python
@pytest.mark.async
async def test_async(project_generator):
    project = await project_generator.generate_project("Test")
    assert project is not None
```

## Fixtures Más Usadas

```python
# Directorio temporal
def test_example(temp_dir):
    file_path = temp_dir / "test.txt"
    file_path.write_text("content")

# Generador de proyectos
def test_example(project_generator):
    project = project_generator.generate_project("Test")

# Estructura de proyecto
def test_example(sample_project_structure):
    assert sample_project_structure.exists()
```

## Helpers Más Usados

```python
from .test_helpers import TestHelpers
from .test_utils_helpers import (
    AsyncTestHelpers,
    FileTestHelpers,
    MockHelpers
)

# Validar estructura
TestHelpers.assert_project_structure(path, ["README.md"])

# Validar JSON
TestHelpers.assert_valid_json(file_path)

# Esperar condición async
await AsyncTestHelpers.wait_for_condition(lambda: ready)

# Crear mock
mock = MockHelpers.create_async_mock(return_value={"success": True})
```

## Ver Más

- [Guía Completa](./TESTING_GUIDE.md)
- [README de Tests](./README.md)
- [Ejemplos Mejorados](./test_examples_improved.py)

