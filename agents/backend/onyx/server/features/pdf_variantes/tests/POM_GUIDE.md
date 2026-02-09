# Guía de Page Object Model (POM)

## Introducción

El Page Object Model (POM) es un patrón de diseño que encapsula la lógica de interacción con páginas/endpoints en clases reutilizables. Esto mejora la mantenibilidad y reduce la duplicación de código.

## Estructura

### Clases Base

#### `BasePage`
Clase base para todas las páginas:
- `navigate()`: Navegar a una URL
- `get_title()`: Obtener título de la página
- `get_url()`: Obtener URL actual
- `wait_for_load()`: Esperar a que la página cargue

### Páginas Específicas

#### `HealthPage`
Operaciones de health check:
- `check_health()`: Verificar salud de la API
- `get_status()`: Obtener estado

#### `UploadPage`
Operaciones de upload:
- `upload_file()`: Subir archivo
- `get_file_id()`: Subir y obtener file_id

#### `VariantPage`
Operaciones de variantes:
- `generate_variant()`: Generar variante
- `generate_multiple_variants()`: Generar múltiples variantes

#### `TopicPage`
Operaciones de topics:
- `extract_topics()`: Extraer topics
- `get_topics_list()`: Obtener lista de topics

#### `PreviewPage`
Operaciones de preview:
- `get_preview()`: Obtener preview
- `get_multiple_pages()`: Obtener previews de múltiples páginas

#### `PDFManagementPage`
Gestión de PDFs:
- `list_pdfs()`: Listar PDFs
- `get_metadata()`: Obtener metadata
- `update_metadata()`: Actualizar metadata
- `delete_pdf()`: Eliminar PDF

#### `SearchPage`
Búsqueda:
- `search()`: Búsqueda general
- `search_by_tags()`: Búsqueda por tags
- `search_by_date_range()`: Búsqueda por rango de fechas

#### `APIPage`
Página principal que combina todas las funcionalidades:
- Acceso a todas las páginas especializadas
- `complete_workflow()`: Workflow completo

## Uso

### Ejemplo Básico

```python
from playwright_pages import APIPage

def test_health(page, api_base_url):
    api_page = APIPage(page, api_base_url)
    status = api_page.health.get_status()
    assert status == "healthy"
```

### Ejemplo de Upload

```python
def test_upload(page, api_base_url, auth_headers, sample_pdf):
    api_page = APIPage(page, api_base_url)
    file_id = api_page.upload.get_file_id("test.pdf", sample_pdf, auth_headers)
    assert file_id is not None
```

### Ejemplo de Workflow Completo

```python
def test_complete_workflow(page, api_base_url, auth_headers, sample_pdf):
    api_page = APIPage(page, api_base_url)
    
    result = api_page.complete_workflow(
        "workflow_test.pdf",
        sample_pdf,
        auth_headers
    )
    
    assert "file_id" in result
    assert "upload" in result
    assert "variant" in result
    assert "topics" in result
    assert "preview" in result
```

### Ejemplo de Variantes Múltiples

```python
def test_multiple_variants(page, api_base_url, auth_headers, sample_pdf):
    api_page = APIPage(page, api_base_url)
    
    # Upload
    file_id = api_page.upload.get_file_id("test.pdf", sample_pdf, auth_headers)
    
    # Generate multiple variants
    variant_types = ["summary", "outline", "highlights"]
    results = api_page.variant.generate_multiple_variants(
        file_id,
        variant_types,
        auth_headers
    )
    
    assert len(results) == len(variant_types)
```

## Ventajas

1. **Reutilización**: Lógica encapsulada y reutilizable
2. **Mantenibilidad**: Cambios en un solo lugar
3. **Legibilidad**: Tests más claros y fáciles de leer
4. **Separación de responsabilidades**: Lógica de interacción separada de tests
5. **Escalabilidad**: Fácil agregar nuevas páginas/operaciones

## Mejores Prácticas

1. **Una página por endpoint/feature**: Mantener páginas enfocadas
2. **Métodos descriptivos**: Nombres claros y descriptivos
3. **Retornar datos útiles**: Retornar datos que los tests necesiten
4. **Manejo de errores**: Validar respuestas dentro de los métodos
5. **Documentación**: Documentar métodos y parámetros

## Extensión

Para agregar una nueva página:

```python
class NewFeaturePage(BasePage):
    """Page object for new feature."""
    
    def new_operation(self, param1: str, headers: dict) -> dict:
        """Perform new operation."""
        response = (
            create_request_builder(self.page, self.base_url)
            .post("/new/endpoint")
            .with_headers(headers)
            .with_json({"param": param1})
            .execute()
        )
        
        return validate_response(response).assert_status(200).assert_json()
```

Luego agregar a `APIPage`:

```python
self.new_feature = NewFeaturePage(page, base_url)
```



