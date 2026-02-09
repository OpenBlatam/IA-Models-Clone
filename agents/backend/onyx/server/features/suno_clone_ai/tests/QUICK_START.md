# Quick Start - Suite de Tests Modular

## Inicio Rápido

### 1. Instalar Dependencias

```bash
pip install pytest pytest-asyncio pytest-cov
```

### 2. Ejecutar Tests

```bash
# Todos los tests
pytest tests/

# Tests específicos
pytest tests/test_api/
pytest tests/test_helpers/

# Con cobertura
pytest --cov=. --cov-report=html tests/
```

### 3. Usar el Generador de Casos de Prueba

```python
from tests.test_case_generator import generate_tests_for_function
from api.helpers import generate_song_id

# Generar tests automáticamente
test_cases, code = generate_tests_for_function(
    generate_song_id,
    num_cases=10
)

print(code)  # Ver código generado
```

### 4. Escribir un Test Nuevo

```python
import pytest
from tests.helpers.test_helpers import create_song_dict
from tests.helpers.assertion_helpers import assert_song_response_valid

class TestMyFeature:
    """Tests para mi funcionalidad"""
    
    @pytest.mark.asyncio
    async def test_my_feature(self, test_client):
        response = test_client.get("/endpoint")
        assert response.status_code == 200
```

### 5. Usar Fixtures

```python
def test_with_mock_service(mock_song_service):
    # mock_song_service ya está configurado
    result = mock_song_service.get_song("test-id")
    assert result is None
```

## Estructura de Tests

- `tests/test_api/`: Tests de endpoints API
- `tests/test_helpers/`: Tests de funciones helper
- `tests/helpers/`: Helpers reutilizables para tests
- `tests/conftest.py`: Fixtures compartidas

## Marcadores de Tests

```bash
# Ejecutar por tipo
pytest -m unit
pytest -m integration
pytest -m api
pytest -m happy_path
pytest -m error_handling
```

## Más Información

Ver `README.md` para documentación completa.

