# Mejores Prácticas de Testing

## 📋 Principios Fundamentales

### 1. Independencia de Tests
- Cada test debe poder ejecutarse de forma independiente
- No depender del orden de ejecución
- Usar fixtures para setup/teardown

```python
# ✅ Bueno
def test_example(temp_dir):
    file = temp_dir / "test.txt"
    file.write_text("content")
    assert file.exists()

# ❌ Malo
def test_example():
    # Depende de test anterior
    assert global_state.exists()
```

### 2. Nombres Descriptivos
- Usar nombres que describan qué se está probando
- Incluir el comportamiento esperado

```python
# ✅ Bueno
def test_sanitize_name_converts_spaces_to_underscores():
    pass

def test_generate_project_creates_backend_and_frontend():
    pass

# ❌ Malo
def test_1():
    pass

def test_sanitize():
    pass
```

### 3. Arrange-Act-Assert (AAA)
- Organizar tests en tres secciones claras

```python
def test_example():
    # Arrange
    generator = ProjectGenerator()
    description = "A chat AI"
    
    # Act
    project = generator.generate_project(description)
    
    # Assert
    assert project is not None
    assert "project_id" in project
```

### 4. Un Concepto por Test
- Cada test debe verificar un solo concepto
- Evitar tests que verifiquen múltiples cosas

```python
# ✅ Bueno
def test_sanitize_name_lowercase():
    result = sanitize_name("TEST")
    assert result == "test"

def test_sanitize_name_removes_special_chars():
    result = sanitize_name("test@project")
    assert result == "test_project"

# ❌ Malo
def test_sanitize_name():
    assert sanitize_name("TEST") == "test"
    assert sanitize_name("test@project") == "test_project"
    assert sanitize_name("  test  ") == "test"
```

## 🛠️ Uso de Fixtures

### Fixtures para Datos
```python
@pytest.fixture
def sample_project_info():
    return {
        "name": "test_project",
        "description": "A test project"
    }

def test_example(sample_project_info):
    assert sample_project_info["name"] == "test_project"
```

### Fixtures para Instancias
```python
@pytest.fixture
def project_generator(temp_dir):
    return ProjectGenerator(base_output_dir=str(temp_dir))

def test_example(project_generator):
    project = project_generator.generate_project("Test")
    assert project is not None
```

### Fixtures para Limpieza
```python
@pytest.fixture
def temp_dir():
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)
```

## 🎯 Uso de Helpers

### Helpers para Validación
```python
from .test_helpers import TestHelpers

def test_example(project_path):
    TestHelpers.assert_project_structure(
        project_path,
        ["README.md", "backend/main.py"]
    )
```

### Helpers para Async
```python
from .test_utils_helpers import AsyncTestHelpers

@pytest.mark.async
async def test_example():
    await AsyncTestHelpers.wait_for_condition(
        lambda: condition_met,
        timeout=5.0
    )
```

### Helpers para Mocks
```python
from .test_utils_helpers import MockHelpers

def test_example():
    mock = MockHelpers.create_async_mock(
        return_value={"success": True}
    )
    result = await mock()
    assert result["success"] is True
```

## 📝 Documentación

### Docstrings en Tests
```python
def test_generate_project():
    """
    Test que verifica la generación de proyectos.
    
    Verifica que:
    - Se crea un proyecto con ID único
    - Se genera la estructura backend/frontend
    - Se crean los archivos esenciales
    """
    pass
```

### Comentarios Explicativos
```python
def test_complex_scenario():
    # Setup: Crear proyecto con todas las features
    project = create_project_with_all_features()
    
    # Act: Generar estructura completa
    structure = generate_structure(project)
    
    # Assert: Verificar que todo se creó correctamente
    assert structure.backend.exists()
    assert structure.frontend.exists()
```

## ⚡ Performance

### Tests Rápidos Primero
```python
@pytest.mark.unit  # Tests rápidos
def test_fast():
    pass

@pytest.mark.slow  # Tests lentos
def test_slow():
    pass
```

### Medir Performance
```python
from .test_utils_helpers import PerformanceTestHelpers

@pytest.mark.performance
def test_performance():
    with PerformanceTestHelpers.measure_time() as elapsed:
        operation()
    
    PerformanceTestHelpers.assert_performance(
        elapsed,
        max_time=1.0
    )
```

## 🔒 Seguridad

### Validar Entradas
```python
def test_security_input_validation():
    with pytest.raises(ValueError):
        sanitize_name("../../../etc/passwd")
```

### Validar Salidas
```python
def test_security_output_validation():
    result = generate_project("Test")
    assert "../" not in result["project_path"]
```

## 🎨 Organización

### Agrupar Tests Relacionados
```python
class TestProjectGenerator:
    """Tests para ProjectGenerator"""
    
    def test_init(self):
        pass
    
    def test_generate_project(self):
        pass
    
    def test_sanitize_name(self):
        pass
```

### Usar Marcadores
```python
@pytest.mark.unit
@pytest.mark.fast
def test_quick():
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_comprehensive():
    pass
```

## 🐛 Manejo de Errores

### Verificar Excepciones
```python
def test_error_handling():
    with pytest.raises(ValueError, match="expected error"):
        function_that_raises()
```

### Verificar Mensajes de Error
```python
def test_error_message():
    with pytest.raises(ValueError) as exc_info:
        function_that_raises()
    
    assert "expected message" in str(exc_info.value)
```

## 📊 Cobertura

### Mantener Alta Cobertura
- Objetivo: 95%+ de cobertura
- Enfocarse en código crítico
- No obsesionarse con 100%

### Verificar Cobertura
```bash
pytest --cov=. --cov-report=html
```

## ✅ Checklist de Calidad

Antes de commitear tests, verificar:

- [ ] Tests son independientes
- [ ] Nombres son descriptivos
- [ ] Usan fixtures apropiadas
- [ ] Tienen docstrings
- [ ] Manejan errores correctamente
- [ ] Están marcados apropiadamente
- [ ] Son rápidos o marcados como slow
- [ ] Validan tanto casos exitosos como errores
- [ ] Usan helpers cuando es apropiado
- [ ] Siguen el patrón AAA

## 📚 Recursos

- [pytest documentation](https://docs.pytest.org/)
- [Testing Guide](./TESTING_GUIDE.md)
- [Quick Start](./QUICK_START_TESTING.md)

