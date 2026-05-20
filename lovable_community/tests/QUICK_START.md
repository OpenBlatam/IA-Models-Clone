# Quick Start - Suite de Tests Lovable Community

## Inicio Rápido

### 1. Instalar Dependencias

```bash
pip install pytest pytest-asyncio pytest-cov sqlalchemy
```

### 2. Ejecutar Tests

```bash
# Todos los tests
pytest tests/

# Tests específicos
pytest tests/test_schemas/
pytest tests/test_services/

# Con cobertura
pytest --cov=. --cov-report=html tests/
```

### 3. Escribir un Test Nuevo

```python
import pytest
from tests.helpers.test_helpers import create_publish_request
from tests.helpers.assertion_helpers import assert_chat_response_valid

class TestMyFeature:
    """Tests para mi funcionalidad"""
    
    def test_my_feature(self, chat_service, sample_user_id):
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        assert_chat_response_valid(chat.__dict__)
```

### 4. Usar Fixtures

```python
def test_with_db(db_session):
    # db_session ya está configurado
    # Es una base de datos en memoria para tests
    pass

def test_with_service(chat_service, sample_user_id):
    # chat_service y sample_user_id ya están disponibles
    chat = chat_service.publish_chat(...)
```

## Estructura de Tests

- `tests/test_schemas/`: Tests de validación de schemas
- `tests/test_services/`: Tests de servicios de negocio
- `tests/test_api/`: Tests de endpoints API
- `tests/helpers/`: Helpers reutilizables para tests
- `tests/conftest.py`: Fixtures compartidas

## Marcadores de Tests

```bash
# Ejecutar por tipo
pytest -m unit
pytest -m api
pytest -m validation
pytest -m happy_path
pytest -m error_handling
```

## Más Información

Ver `README.md` para documentación completa.

