# Guía de Testing

Guía completa para escribir y ejecutar tests.

## Test Helpers

### MockImage

Crear imágenes de prueba fácilmente.

```python
from utils.test_helpers import MockImage

# Crear imagen RGB
image = MockImage.create_rgb(width=500, height=500, color="red")

# Crear imagen como bytes
image_bytes = MockImage.create_bytes(format="PNG")
```

### TempDirectory

Directorio temporal para tests.

```python
from utils.test_helpers import TempDirectory

with TempDirectory() as temp_dir:
    # Usar temp_dir para tests
    file_path = temp_dir / "test.txt"
    file_path.write_text("test")
```

### AsyncTestHelper

Helpers para testing async.

```python
from utils.test_helpers import AsyncTestHelper

# Ejecutar coroutine
result = AsyncTestHelper.run_async(async_function())

# Crear async mock
mock = AsyncTestHelper.create_async_mock(return_value="result")
```

### MockFactory

Factory para crear mocks comunes.

```python
from utils.test_helpers import MockFactory

# Crear mocks de repositorios
storage_mock = MockFactory.create_storage_repository()
cache_mock = MockFactory.create_cache_repository()
image_processor_mock = MockFactory.create_image_processor()
ai_processor_mock = MockFactory.create_ai_processor()
metrics_mock = MockFactory.create_metrics_collector()
event_publisher_mock = MockFactory.create_event_publisher()
```

### AssertHelper

Helpers para assertions comunes.

```python
from utils.test_helpers import AssertHelper

# Validar imagen
AssertHelper.assert_image_valid(image)

# Validar path existe
AssertHelper.assert_path_exists(path)

# Validar diccionario contiene keys
AssertHelper.assert_dict_contains(data, ["key1", "key2"])
```

### FixtureHelper

Helpers para crear fixtures de prueba.

```python
from utils.test_helpers import FixtureHelper

# Crear request de visualización
request = FixtureHelper.create_visualization_request(
    surgery_type="rhinoplasty",
    intensity=0.7
)

# Crear settings de prueba
settings = FixtureHelper.create_test_settings(
    max_image_size_mb=5
)
```

## Ejemplos de Tests

### Test de Use Case

```python
import pytest
from utils.test_helpers import (
    MockFactory,
    MockImage,
    AsyncTestHelper
)
from domain.use_cases.create_visualization import CreateVisualizationUseCase

@pytest.fixture
def use_case():
    return CreateVisualizationUseCase(
        image_processor=MockFactory.create_image_processor(),
        ai_processor=MockFactory.create_ai_processor(),
        storage_repository=MockFactory.create_storage_repository(),
        cache_repository=MockFactory.create_cache_repository(),
        metrics_collector=MockFactory.create_metrics_collector()
    )

@pytest.mark.asyncio
async def test_create_visualization(use_case):
    from api.schemas.visualization import VisualizationRequest, SurgeryType
    
    request = VisualizationRequest(
        surgery_type=SurgeryType.RHINOPLASTY,
        intensity=0.5,
        image_data=MockImage.create_bytes()
    )
    
    result = await use_case.execute(request)
    
    assert result.visualization_id is not None
    assert result.surgery_type == SurgeryType.RHINOPLASTY
```

### Test con TempDirectory

```python
from utils.test_helpers import TempDirectory, AssertHelper

def test_save_file():
    with TempDirectory() as temp_dir:
        file_path = temp_dir / "test.txt"
        file_path.write_text("test content")
        
        AssertHelper.assert_path_exists(file_path)
        assert file_path.read_text() == "test content"
```

### Test de Integración

```python
from fastapi.testclient import TestClient
from main import app
from utils.test_helpers import MockImage

client = TestClient(app)

def test_create_visualization_endpoint():
    image_bytes = MockImage.create_bytes()
    
    response = client.post(
        "/api/v1/visualize",
        files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        data={
            "surgery_type": "rhinoplasty",
            "intensity": "0.5"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "visualization_id" in data
```

## Configuración de Tests

### conftest.py

```python
import pytest
from utils.test_helpers import MockFactory, TempDirectory

@pytest.fixture
def mock_services():
    return {
        "storage": MockFactory.create_storage_repository(),
        "cache": MockFactory.create_cache_repository(),
        "image_processor": MockFactory.create_image_processor(),
        "ai_processor": MockFactory.create_ai_processor(),
        "metrics": MockFactory.create_metrics_collector(),
        "events": MockFactory.create_event_publisher()
    }

@pytest.fixture
def temp_storage():
    with TempDirectory() as temp_dir:
        yield temp_dir
```

## Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_visualization.py

# Con coverage
pytest --cov=. --cov-report=html

# Verbose
pytest -v

# Con output detallado
pytest -vv -s
```

## Mejores Prácticas

1. **Usar fixtures**: Crear fixtures reutilizables
2. **Mockear dependencias**: Usar MockFactory para mocks
3. **Temp directories**: Usar TempDirectory para archivos temporales
4. **Async helpers**: Usar AsyncTestHelper para tests async
5. **Assertions claras**: Usar AssertHelper para validaciones comunes
6. **Aislar tests**: Cada test debe ser independiente
7. **Nombres descriptivos**: Nombres de tests que expliquen qué prueban

