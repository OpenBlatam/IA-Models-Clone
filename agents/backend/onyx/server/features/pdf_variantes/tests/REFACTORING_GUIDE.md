# Guía de Refactorización de Tests de Playwright

## Objetivo

Refactorizar los tests de Playwright para:
- Eliminar duplicación de código
- Centralizar fixtures comunes
- Crear clases base reutilizables
- Mejorar mantenibilidad
- Establecer patrones consistentes

## Estructura Refactorizada

```
tests/
├── fixtures_common.py          # Fixtures centralizadas
├── base_playwright_test.py      # Clases base
├── conftest_playwright.py       # Configuración Playwright
├── playwright_helpers.py         # Helpers utilitarios
└── test_playwright_*.py         # Tests específicos
```

## Fixtures Centralizadas

### Antes (Duplicado en múltiples archivos)

```python
# En test_playwright_api.py
@pytest.fixture
def api_base_url():
    return "http://localhost:8000"

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_token"}

# En test_playwright_workflows.py
@pytest.fixture
def api_base_url():
    return "http://localhost:8000"  # Duplicado!

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_token"}  # Duplicado!
```

### Después (Centralizado)

```python
# En fixtures_common.py
@pytest.fixture
def api_base_url():
    return os.getenv("API_BASE_URL", "http://localhost:8000")

@pytest.fixture
def auth_headers():
    return {
        "Authorization": "Bearer test_token_123",
        "X-User-ID": "test_user_123"
    }

# En test files - solo importar
from fixtures_common import api_base_url, auth_headers
```

## Clases Base

### Antes (Código duplicado)

```python
# En test_playwright_api.py
class TestUpload:
    def test_upload(self, page, api_base_url, auth_headers, sample_pdf):
        files = {
            "file": {
                "name": "test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        assert response.status == 200
        file_id = response.json().get("file_id")

# En test_playwright_workflows.py - código similar duplicado
class TestWorkflow:
    def test_upload(self, page, api_base_url, auth_headers, sample_pdf):
        files = {
            "file": {
                "name": "test.pdf",
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        response = page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
        assert response.status == 200
        file_id = response.json().get("file_id")
```

### Después (Usando clase base)

```python
# En base_playwright_test.py
class BaseAPITest:
    def upload_pdf(self, page, api_base_url, auth_headers, sample_pdf, filename="test.pdf"):
        files = {
            "file": {
                "name": filename,
                "mimeType": "application/pdf",
                "buffer": sample_pdf
            }
        }
        return page.request.post(
            f"{api_base_url}/pdf/upload",
            multipart=files,
            headers=auth_headers
        )
    
    def get_file_id_from_response(self, response):
        if response.status in [200, 201]:
            return response.json().get("file_id") or response.json().get("id")
        return None

# En test files
from base_playwright_test import BaseAPITest

class TestUpload(BaseAPITest):
    def test_upload(self, page, api_base_url, auth_headers, sample_pdf):
        response = self.upload_pdf(page, api_base_url, auth_headers, sample_pdf)
        self.assert_response_success(response)
        file_id = self.get_file_id_from_response(response)
        assert file_id is not None
```

## Beneficios de la Refactorización

### 1. Reducción de Duplicación
- **Antes**: Fixtures duplicadas en 20+ archivos
- **Después**: Fixtures centralizadas en 1 archivo
- **Ahorro**: ~500+ líneas de código duplicado

### 2. Mejor Mantenibilidad
- Cambios en fixtures se hacen en un solo lugar
- Patrones consistentes en todos los tests
- Más fácil de entender y mantener

### 3. Reutilización
- Helpers comunes disponibles para todos los tests
- Clases base proporcionan funcionalidad estándar
- Menos código para escribir nuevos tests

### 4. Consistencia
- Mismos métodos de aserción en todos los tests
- Mismos helpers para operaciones comunes
- Mismos patrones de error handling

## Migración Paso a Paso

### Paso 1: Identificar Duplicación
```bash
# Buscar fixtures duplicadas
grep -r "@pytest.fixture" tests/ | grep "api_base_url"
```

### Paso 2: Crear Fixtures Centralizadas
```python
# Crear fixtures_common.py con todas las fixtures comunes
```

### Paso 3: Crear Clases Base
```python
# Crear base_playwright_test.py con clases base
```

### Paso 4: Migrar Tests
```python
# Reemplazar fixtures duplicadas con imports
# Reemplazar código duplicado con métodos de clase base
```

### Paso 5: Verificar
```bash
# Ejecutar tests para asegurar que todo funciona
pytest tests/test_playwright*.py -v
```

## Ejemplos de Uso

### Ejemplo 1: Test de API Simple

```python
from base_playwright_test import BaseAPITest
from fixtures_common import api_base_url, auth_headers, sample_pdf

class TestSimpleAPI(BaseAPITest):
    def test_health(self, page, api_base_url):
        response = page.request.get(f"{api_base_url}/health")
        self.assert_response_success(response)
```

### Ejemplo 2: Test de Workflow Completo

```python
from base_playwright_test import BaseAPITest
from fixtures_common import api_base_url, auth_headers, sample_pdf

class TestWorkflow(BaseAPITest):
    def test_complete_workflow(self, page, api_base_url, auth_headers, sample_pdf):
        # Upload
        upload_response = self.upload_pdf(page, api_base_url, auth_headers, sample_pdf)
        file_id = self.get_file_id_from_response(upload_response)
        
        # Generate variant
        variant_response = self.generate_variant(
            page, api_base_url, file_id, "summary", {}, auth_headers
        )
        self.assert_response_success(variant_response, expected_status=202)
        
        # Get topics
        topics_response = self.get_topics(page, api_base_url, file_id, auth_headers)
        self.assert_response_success(topics_response)
```

### Ejemplo 3: Test de UI

```python
from base_playwright_test import BaseUITest
from fixtures_common import api_base_url

class TestUI(BaseUITest):
    def test_navigation(self, page, api_base_url):
        success = self.navigate_and_wait(page, api_base_url)
        assert success
        assert page.url.startswith(api_base_url)
```

## Checklist de Refactorización

- [ ] Crear `fixtures_common.py` con fixtures centralizadas
- [ ] Crear `base_playwright_test.py` con clases base
- [ ] Migrar fixtures duplicadas a usar imports
- [ ] Migrar tests a usar clases base
- [ ] Eliminar código duplicado
- [ ] Verificar que todos los tests pasan
- [ ] Actualizar documentación
- [ ] Crear ejemplos de uso

## Próximos Pasos

1. Migrar todos los tests existentes
2. Agregar más helpers a clases base
3. Crear más fixtures comunes según necesidad
4. Mejorar documentación de helpers
5. Crear guías de mejores prácticas



