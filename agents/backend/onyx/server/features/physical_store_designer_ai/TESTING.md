# Guía de Testing

Esta guía describe cómo ejecutar y escribir tests para Physical Store Designer AI.

## Estructura de Tests

Los tests están organizados en el directorio `tests/`:

```
tests/
├── __init__.py
├── conftest.py              # Fixtures compartidas
├── test_storage_service.py  # Tests para StorageService
├── test_chat_service.py     # Tests para ChatService
└── test_service_base.py     # Tests para clases base
```

## Ejecutar Tests

### Ejecutar todos los tests

```bash
pytest
```

### Ejecutar tests con cobertura

```bash
pytest --cov=. --cov-report=html
```

### Ejecutar tests específicos

```bash
# Por archivo
pytest tests/test_storage_service.py

# Por clase
pytest tests/test_storage_service.py::TestStorageService

# Por método
pytest tests/test_storage_service.py::TestStorageService::test_save_design
```

### Ejecutar tests en modo verbose

```bash
pytest -v
```

### Ejecutar tests con output detallado

```bash
pytest -vv -s
```

## Fixtures Disponibles

### `temp_storage_path`
Crea un directorio temporal para almacenamiento de tests. Se limpia automáticamente después de cada test.

```python
def test_something(temp_storage_path):
    # Usar temp_storage_path como Path
    file_path = temp_storage_path / "test.json"
```

### `storage_service`
Crea una instancia de `StorageService` con un directorio temporal.

```python
def test_save(storage_service):
    # Usar storage_service para tests
    storage_service.save_design(design)
```

### `chat_service`
Crea una instancia de `ChatService` para tests.

```python
def test_chat(chat_service):
    session_id = chat_service.create_session()
```

### `store_designer_service`
Crea una instancia de `StoreDesignerService` para tests.

### `sample_store_design`
Crea un diseño de ejemplo para usar en tests.

```python
def test_design(sample_store_design):
    assert sample_store_design.store_name == "Test Store"
```

## Escribir Nuevos Tests

### Estructura de un Test

```python
"""
Tests for MyService
"""

import pytest
from ..services.my_service import MyService
from ..core.exceptions import NotFoundError


class TestMyService:
    """Test cases for MyService"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        service = MyService()
        result = service.do_something()
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling"""
        service = MyService()
        with pytest.raises(NotFoundError):
            service.get_item("non_existent")
```

### Mejores Prácticas

1. **Nombres descriptivos**: Usa nombres que describan qué se está probando
2. **Un test, una cosa**: Cada test debe verificar una funcionalidad específica
3. **Arrange-Act-Assert**: Organiza tests en tres secciones claras
4. **Usa fixtures**: Reutiliza código común con fixtures
5. **Aísla tests**: Cada test debe ser independiente

### Ejemplo Completo

```python
"""
Tests for MyService
"""

import pytest
from pathlib import Path
from ..services.my_service import MyService
from ..core.exceptions import ValidationError


class TestMyService:
    """Test cases for MyService"""
    
    def test_create_item(self):
        """Test creating an item"""
        # Arrange
        service = MyService()
        item_data = {"name": "Test Item"}
        
        # Act
        result = service.create_item(item_data)
        
        # Assert
        assert result is not None
        assert result["name"] == "Test Item"
        assert "id" in result
    
    def test_create_item_validation(self):
        """Test validation when creating item"""
        # Arrange
        service = MyService()
        invalid_data = {}
        
        # Act & Assert
        with pytest.raises(ValidationError):
            service.create_item(invalid_data)
    
    def test_get_item_not_found(self):
        """Test getting non-existent item"""
        # Arrange
        service = MyService()
        
        # Act & Assert
        with pytest.raises(NotFoundError):
            service.get_item("non_existent_id")
```

## Cobertura de Código

### Ver cobertura actual

```bash
pytest --cov=. --cov-report=term-missing
```

### Generar reporte HTML

```bash
pytest --cov=. --cov-report=html
# Abrir htmlcov/index.html en el navegador
```

### Cobertura mínima

El proyecto apunta a mantener al menos 80% de cobertura de código.

## Tests de Integración

Para tests de integración que requieren servicios externos o bases de datos:

1. Usa mocks para servicios externos
2. Usa fixtures para datos de prueba
3. Limpia después de cada test

## Troubleshooting

### Tests fallan por imports

Asegúrate de ejecutar tests desde el directorio raíz del proyecto:

```bash
cd agents/backend/onyx/server/features/physical_store_designer_ai
pytest
```

### Tests fallan por permisos

Algunos tests crean archivos temporales. Asegúrate de tener permisos de escritura.

### Tests lentos

Usa `pytest -x` para detener en el primer fallo durante desarrollo.

## Próximos Pasos

- [ ] Agregar tests para más servicios
- [ ] Tests de integración para API endpoints
- [ ] Tests de performance
- [ ] Tests de carga
- [ ] CI/CD con tests automáticos








