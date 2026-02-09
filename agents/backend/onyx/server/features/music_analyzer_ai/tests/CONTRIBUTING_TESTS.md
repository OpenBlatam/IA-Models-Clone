# Guía para Contribuir Tests

## 📝 Cómo Escribir Tests

### Estructura Básica

```python
import pytest
from unittest.mock import Mock, patch

def test_nombre_descriptivo():
    """Descripción clara de qué se está probando"""
    # Arrange - Preparar datos
    input_data = {"key": "value"}
    
    # Act - Ejecutar la función
    result = function_to_test(input_data)
    
    # Assert - Verificar resultado
    assert result is not None
    assert result["key"] == "value"
```

### Usar Fixtures Existentes

```python
def test_with_fixture(sample_audio_features):
    """Usar fixtures del conftest.py"""
    assert sample_audio_features["tempo"] == 120.0
```

### Usar Helpers

```python
def test_with_helpers(test_helpers):
    """Usar helpers para crear mocks"""
    mock_response = test_helpers.create_mock_response(
        status_code=200,
        data={"success": True}
    )
    assert mock_response.status_code == 200
```

## 🎯 Convenciones de Nombres

### Archivos de Test
- Prefijo: `test_`
- Ejemplo: `test_music_analyzer.py`

### Funciones de Test
- Prefijo: `test_`
- Descriptivo: `test_analyze_track_with_valid_data`

### Clases de Test
- Prefijo: `Test`
- Ejemplo: `TestMusicAnalyzer`

## 📋 Checklist Antes de Commit

- [ ] Test tiene nombre descriptivo
- [ ] Test tiene docstring explicando qué prueba
- [ ] Test sigue el patrón Arrange-Act-Assert
- [ ] Test es independiente (no depende de otros tests)
- [ ] Test usa fixtures cuando es apropiado
- [ ] Test usa mocks para servicios externos
- [ ] Test verifica casos edge cuando es relevante
- [ ] Test pasa localmente
- [ ] Test no es demasiado lento (marcar como `@pytest.mark.slow` si es necesario)

## 🔧 Tipos de Tests

### Tests Unitarios

```python
@pytest.mark.unit
def test_unit_function():
    """Test de una función específica"""
    result = function_to_test()
    assert result == expected
```

### Tests de Integración

```python
@pytest.mark.integration
def test_integration_flow():
    """Test de flujo completo"""
    # Test que involucra múltiples componentes
    pass
```

### Tests de API

```python
@pytest.mark.api
def test_api_endpoint(client):
    """Test de endpoint de API"""
    response = client.get("/api/endpoint")
    assert response.status_code == 200
```

## 🚫 Errores Comunes a Evitar

1. **Tests dependientes**: Cada test debe poder ejecutarse independientemente
2. **Mocks no limpiados**: Usar `@pytest.fixture(autouse=True)` o limpiar manualmente
3. **Assertions vagas**: Usar assertions específicos
4. **Tests sin propósito**: Cada test debe verificar algo específico
5. **Hardcoded values**: Usar fixtures o constantes

## ✅ Mejores Prácticas

1. **Un test, una cosa**: Cada test debe verificar un comportamiento específico
2. **Nombres descriptivos**: El nombre debe explicar qué se prueba
3. **Usar fixtures**: Reutilizar código común
4. **Mocks para externos**: No depender de servicios externos
5. **Edge cases**: Probar casos límite
6. **Documentar**: Docstrings claros

## 📚 Ejemplos

Ver `test_examples.py` para ejemplos completos de diferentes tipos de tests.

## 🎓 Recursos

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](./QUICK_START_TESTING.md#-mejores-prácticas)
- [Helpers Disponibles](./test_helpers.py)

---

**Última actualización**: 2024

