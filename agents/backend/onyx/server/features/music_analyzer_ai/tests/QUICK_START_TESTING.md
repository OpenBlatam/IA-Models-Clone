# Guía Rápida de Testing

## 🚀 Inicio Rápido

> 💡 **Tip**: Usa los scripts `MAKE_TEST_COMMANDS.sh` (Linux/Mac) o `MAKE_TEST_COMMANDS.bat` (Windows) para comandos rápidos

### Instalación de Dependencias

```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### Ejecutar Todos los Tests

```bash
# Usando script (recomendado)
./MAKE_TEST_COMMANDS.sh all          # Linux/Mac
MAKE_TEST_COMMANDS.bat all           # Windows

# O directamente con pytest
pytest

# Con output detallado
pytest -v

# Con output muy detallado
pytest -vv
```

### Ejecutar Tests Específicos

```bash
# Un archivo específico
pytest tests/test_music_analyzer.py

# Una clase específica
pytest tests/test_music_analyzer.py::TestMusicAnalyzer

# Un test específico
pytest tests/test_music_analyzer.py::TestMusicAnalyzer::test_analyze_track
```

### Ejecutar con Cobertura

```bash
# Cobertura básica
pytest --cov=music_analyzer_ai

# Cobertura con reporte HTML
pytest --cov=music_analyzer_ai --cov-report=html

# Cobertura con umbral mínimo
pytest --cov=music_analyzer_ai --cov-report=html --cov-fail-under=95
```

## 📊 Comandos Útiles

### Tests por Categoría

```bash
# Solo tests unitarios
pytest -m unit

# Solo tests de integración
pytest -m integration

# Excluir tests lentos
pytest -m "not slow"
```

### Tests en Paralelo

```bash
# Instalar pytest-xdist
pip install pytest-xdist

# Ejecutar en paralelo (auto detecta CPUs)
pytest -n auto

# Ejecutar con N workers
pytest -n 4
```

### Ver Output de Prints

```bash
# Mostrar prints en tests
pytest -s

# Mostrar prints y output detallado
pytest -sv
```

### Detener en Primer Error

```bash
# Detener en primer fallo
pytest -x

# Detener después de N fallos
pytest --maxfail=3
```

### Ejecutar Últimos Tests Fallidos

```bash
# Ejecutar solo los tests que fallaron la última vez
pytest --lf

# Ejecutar tests fallidos primero
pytest --ff
```

## 🎯 Ejemplos de Uso

### Ejemplo 1: Desarrollo Local

```bash
# Ejecutar tests relevantes mientras desarrollas
pytest tests/test_music_analyzer.py -v

# Con cobertura para ver qué falta
pytest tests/test_music_analyzer.py --cov=music_analyzer_ai.core.music_analyzer
```

### Ejemplo 2: Pre-Commit

```bash
# Ejecutar tests rápidos antes de commit
pytest -m "not slow" --maxfail=1

# Con validación de cobertura
pytest --cov --cov-fail-under=95 -m "not slow"
```

### Ejemplo 3: CI/CD

```bash
# Ejecutar todos los tests con reporte
pytest --cov=music_analyzer_ai --cov-report=xml --cov-report=html -v

# Con junit XML para CI
pytest --junitxml=test-results.xml --cov=music_analyzer_ai
```

## 🔍 Debugging

### Ejecutar con PDB

```bash
# Entrar en debugger en fallos
pytest --pdb

# Entrar en debugger en todos los tests
pytest --pdb --trace
```

### Ver Fixtures Disponibles

```bash
# Listar todas las fixtures
pytest --fixtures

# Ver fixtures de un archivo específico
pytest --fixtures tests/conftest.py
```

### Mostrar Valores de Variables

```bash
# Mostrar valores locales en fallos
pytest -l
```

## 📝 Escribir Nuevos Tests

### Estructura Básica

```python
import pytest
from unittest.mock import Mock, patch

def test_basic_functionality():
    """Test básico de funcionalidad"""
    # Arrange
    input_data = {"test": "data"}
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result is not None
    assert result["test"] == "data"
```

### Usar Fixtures

```python
def test_with_fixture(analyzer_with_mocks):
    """Test usando fixture"""
    result = analyzer_with_mocks.analyze_track({})
    assert result is not None
```

### Tests Parametrizados

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply(input, expected):
    assert input * 2 == expected
```

### Tests Asíncronos

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

## 🎨 Mejores Prácticas

1. **Nombres Descriptivos**: Usa nombres claros que describan qué se está probando
2. **Un Test, Una Aserción**: Idealmente, cada test debe verificar una cosa
3. **Arrange-Act-Assert**: Organiza tus tests en estas tres secciones
4. **Usa Fixtures**: Reutiliza código común con fixtures
5. **Mocks para Servicios Externos**: No dependas de servicios externos en tests
6. **Tests Independientes**: Cada test debe poder ejecutarse independientemente
7. **Limpieza**: Usa `teardown` o `yield` en fixtures para limpiar recursos

## 🐛 Troubleshooting

### Tests No Se Ejecutan

```bash
# Verificar que pytest encuentra los tests
pytest --collect-only

# Verificar configuración
pytest --version
```

### Import Errors

```bash
# Asegúrate de que PYTHONPATH está configurado
export PYTHONPATH="${PYTHONPATH}:."

# O usa pytest.ini con pythonpath
```

### Tests Lentos

```bash
# Identificar tests lentos
pytest --durations=10

# Ejecutar solo tests rápidos
pytest -m "not slow"
```

## 📚 Recursos Adicionales

- [Documentación Completa](./README.md)
- [Resumen de la Suite](./TEST_SUITE_SUMMARY.md)
- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

---

**Última actualización**: 2024

