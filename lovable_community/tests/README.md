# Suite de Tests Modular para Lovable Community

Suite de tests modular y extensible para el proyecto Lovable Community.

## Estructura Modular

```
tests/
├── __init__.py
├── conftest.py              # Fixtures compartidas
├── pytest.ini               # Configuración de pytest
├── README.md                 # Documentación completa
├── QUICK_START.md            # Guía rápida
├── SUMMARY.md                # Resumen del proyecto
│
├── helpers/                  # Helpers modulares
│   ├── __init__.py
│   ├── test_helpers.py       # Helpers generales
│   ├── mock_helpers.py       # Helpers para mocks
│   ├── assertion_helpers.py # Helpers de aserciones
│   ├── advanced_helpers.py   # Helpers avanzados
│   └── security_helpers.py  # Helpers de seguridad
│
├── test_schemas/             # Tests de schemas
│   └── test_schemas_validation.py
│
├── test_services/           # Tests de servicios
│   ├── test_chat_service.py
│   ├── test_chat_service_advanced.py
│   └── test_ranking_service.py
│
├── test_api/                 # Tests de API
│   └── test_routes.py
│
├── test_integration/         # Tests de integración
│   └── test_full_workflow.py
│
└── test_security/            # Tests de seguridad
    └── test_security_routes.py
```

## Características

### 1. **Modularidad**
- Cada componente tiene su propio archivo de tests
- Helpers reutilizables organizados por funcionalidad
- Fixtures compartidas en `conftest.py`

### 2. **Cobertura Completa**
- Tests de validación de schemas
- Tests de servicios de negocio
- Tests de lógica de ranking
- Tests de casos edge y error handling
- Tests de integración (flujos completos)
- Tests de seguridad (SQL injection, XSS, etc.)
- Tests de performance
- Tests de operaciones en lote

## Uso

### Ejecutar todos los tests
```bash
pytest tests/
```

### Ejecutar tests específicos
```bash
# Tests de schemas
pytest tests/test_schemas/

# Tests de servicios
pytest tests/test_services/

# Tests marcados
pytest -m unit
pytest -m validation
```

### Ejecutar con cobertura
```bash
pytest --cov=. --cov-report=html tests/
```

## Fixtures Disponibles

### Base de Datos
- `db_session`: Sesión de base de datos en memoria
- `chat_service`: Instancia de ChatService
- `ranking_service`: Instancia de RankingService

### Mocks
- `mock_db_session`: Mock de sesión de base de datos
- `mock_chat_service`: Mock del servicio de chats

### Datos de Prueba
- `sample_user_id`: ID de usuario de ejemplo
- `sample_chat_id`: ID de chat de ejemplo
- `sample_publish_request`: Request de publicación
- `sample_remix_request`: Request de remix
- `sample_vote_request`: Request de voto
- `sample_chat_data`: Datos de chat

## Helpers Disponibles

### test_helpers.py
- `generate_chat_id()`: Genera ID de chat
- `generate_user_id()`: Genera ID de usuario
- `create_chat_dict()`: Crea diccionario de chat
- `create_publish_request()`: Crea request de publicación
- `create_remix_request()`: Crea request de remix
- `create_vote_request()`: Crea request de voto
- `create_search_request()`: Crea request de búsqueda

### mock_helpers.py
- `create_mock_chat_service()`: Crea mock de ChatService
- `create_mock_ranking_service()`: Crea mock de RankingService
- `create_mock_db_session()`: Crea mock de sesión DB

### assertion_helpers.py
- `assert_chat_response_valid()`: Verifica respuesta de chat
- `assert_chat_list_valid()`: Verifica lista de chats
- `assert_pagination_valid()`: Verifica paginación
- `assert_vote_response_valid()`: Verifica respuesta de voto
- `assert_remix_response_valid()`: Verifica respuesta de remix
- `assert_stats_valid()`: Verifica estadísticas

### advanced_helpers.py
- `AsyncTestHelper`: Helper para tests asíncronos
- `PerformanceHelper`: Medición de performance
- `DataFactory`: Generación de datos de prueba
- `MockVerifier`: Verificación de mocks
- `TestDataBuilder`: Builder pattern para datos complejos
- `SecurityTestHelper`: Helpers para tests de seguridad
- `BatchTestHelper`: Helpers para operaciones en lote

### security_helpers.py
- `generate_sql_injection_payloads()`: Payloads de SQL injection
- `generate_xss_payloads()`: Payloads de XSS
- `generate_path_traversal_payloads()`: Payloads de path traversal
- `generate_large_inputs()`: Inputs grandes para DoS

## Mejores Prácticas

1. **Usar fixtures compartidas**: Reutilizar fixtures de `conftest.py`
2. **Organizar por funcionalidad**: Agrupar tests relacionados
3. **Nombres descriptivos**: Usar nombres claros para tests
4. **Usar helpers**: Aprovechar los helpers modulares
5. **Marcar tests**: Usar marcadores para categorizar tests
6. **Tests independientes**: Cada test debe ser independiente

## Extensión

Para agregar nuevos tests:

1. Crear archivo en el directorio apropiado
2. Importar fixtures y helpers necesarios
3. Usar clases para agrupar tests relacionados
4. Seguir el patrón de nombres: `test_<component>_<functionality>.py`

## Ejemplo de Test

```python
import pytest
from tests.helpers.test_helpers import create_publish_request
from tests.helpers.assertion_helpers import assert_chat_response_valid

class TestMyFeature:
    """Tests para mi funcionalidad"""
    
    def test_my_feature_success(self, chat_service, sample_user_id):
        """Test exitoso"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        assert_chat_response_valid(chat.__dict__)
```

