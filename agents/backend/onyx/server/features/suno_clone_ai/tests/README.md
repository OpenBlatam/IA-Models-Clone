# Suite de Tests Modular para Suno Clone AI

Suite de tests modular y extensible para el proyecto Suno Clone AI.

## Estructura Modular

```
tests/
├── __init__.py
├── conftest.py              # Fixtures compartidas
├── pytest.ini               # Configuración de pytest
├── test_case_generator.py   # Generador de casos de prueba
├── helpers/                 # Helpers modulares
│   ├── __init__.py
│   ├── test_helpers.py      # Helpers generales
│   ├── mock_helpers.py      # Helpers para mocks
│   └── assertion_helpers.py # Helpers de aserciones
├── test_api/                # Tests de API endpoints
│   ├── __init__.py
│   ├── test_song_api_generation.py
│   └── test_song_api_management.py
└── test_helpers/            # Tests de helpers
    └── test_api_helpers.py
```

## Características

### 1. **Modularidad**
- Cada componente tiene su propio archivo de tests
- Helpers reutilizables organizados por funcionalidad
- Fixtures compartidas en `conftest.py`

### 2. **Generador de Casos de Prueba**
El módulo `test_case_generator.py` puede analizar funciones y generar casos de prueba automáticamente:

```python
from tests.test_case_generator import generate_tests_for_function
from api.helpers import generate_song_id

# Generar casos de prueba
test_cases, code = generate_tests_for_function(
    generate_song_id,
    num_cases=10,
    output_file="test_generated.py"
)
```

### 3. **Tipos de Tests**
- **Happy Path**: Casos exitosos normales
- **Edge Cases**: Valores extremos pero válidos
- **Error Handling**: Manejo de errores
- **Boundary**: Valores límite
- **Null/Empty**: Valores None/vacíos
- **Type Validation**: Validación de tipos

## Uso

### Ejecutar todos los tests
```bash
pytest tests/
```

### Ejecutar tests específicos
```bash
# Tests de API
pytest tests/test_api/

# Tests de helpers
pytest tests/test_helpers/

# Tests marcados
pytest -m unit
pytest -m integration
pytest -m api
```

### Ejecutar con cobertura
```bash
pytest --cov=. --cov-report=html tests/
```

## Generar Tests Automáticamente

### Usando el Generador de Casos de Prueba

```python
from tests.test_case_generator import (
    TestCaseGenerator,
    TestCodeGenerator,
    FunctionAnalyzer
)

# Analizar una función
analyzer = FunctionAnalyzer()
func_info = analyzer.analyze_function(your_function)

# Generar casos de prueba
generator = TestCaseGenerator()
test_cases = generator.generate_test_cases(
    your_function,
    num_cases=10,
    include_types=[TestType.HAPPY_PATH, TestType.ERROR_HANDLING]
)

# Generar código de test
code_generator = TestCodeGenerator()
code = code_generator.generate_test_code(func_info, test_cases)
```

## Fixtures Disponibles

### Directorios
- `temp_dir`: Directorio temporal
- `temp_audio_dir`: Directorio para archivos de audio

### Mocks de Servicios
- `mock_song_service`: Mock del servicio de canciones
- `mock_music_generator`: Mock del generador de música
- `mock_chat_processor`: Mock del procesador de chat
- `mock_cache_manager`: Mock del gestor de caché
- `mock_audio_processor`: Mock del procesador de audio
- `mock_metrics_service`: Mock del servicio de métricas
- `mock_notification_service`: Mock del servicio de notificaciones

### Datos de Prueba
- `sample_audio_data`: Datos de audio de ejemplo
- `sample_song_data`: Datos de canción de ejemplo
- `sample_chat_message`: Mensaje de chat de ejemplo
- `sample_song_generation_request`: Request de generación de ejemplo

## Helpers Disponibles

### test_helpers.py
- `create_mock_audio()`: Crea audio mock
- `save_test_audio()`: Guarda audio de prueba
- `load_test_audio()`: Carga audio de prueba
- `create_song_dict()`: Crea diccionario de canción
- `generate_test_song_id()`: Genera ID de canción

### mock_helpers.py
- `create_mock_song_service()`: Crea mock de SongService
- `create_mock_music_generator()`: Crea mock de MusicGenerator
- `create_mock_audio_processor()`: Crea mock de AudioProcessor
- `create_mock_notification_service()`: Crea mock de NotificationService
- `create_mock_cache_manager()`: Crea mock de CacheManager

### assertion_helpers.py
- `assert_song_response_valid()`: Verifica respuesta de canción
- `assert_audio_files_equal()`: Compara archivos de audio
- `assert_audio_processed()`: Verifica procesamiento de audio
- `assert_song_list_valid()`: Verifica lista de canciones

## Mejores Prácticas

1. **Usar fixtures compartidas**: Reutilizar fixtures de `conftest.py`
2. **Organizar por funcionalidad**: Agrupar tests relacionados
3. **Nombres descriptivos**: Usar nombres claros para tests
4. **Usar helpers**: Aprovechar los helpers modulares
5. **Marcar tests**: Usar marcadores para categorizar tests
6. **Tests independientes**: Cada test debe ser independiente

## Extensión

Para agregar nuevos tests:

1. Crear archivo en el directorio apropiado (`test_api/`, `test_helpers/`, etc.)
2. Importar fixtures y helpers necesarios
3. Usar clases para agrupar tests relacionados
4. Seguir el patrón de nombres: `test_<component>_<functionality>.py`

## Ejemplo de Test

```python
import pytest
from tests.helpers.test_helpers import create_song_dict
from tests.helpers.assertion_helpers import assert_song_response_valid

class TestMyFeature:
    """Tests para mi funcionalidad"""
    
    @pytest.mark.asyncio
    async def test_my_feature_success(self, test_client):
        """Test exitoso"""
        response = test_client.get("/endpoint")
        assert response.status_code == 200
        assert_song_response_valid(response.json())
```

