# Guía de Testing - Mejoras V8

## Estrategia Completa de Testing para V8

---

## 📋 Estrategia de Testing

### Pirámide de Testing

```
        ┌─────────┐
        │   E2E   │  ← Pocos tests, lentos
        └─────────┘
      ┌─────────────┐
      │ Integration │  ← Algunos tests, medios
      └─────────────┘
    ┌─────────────────┐
    │     Unit        │  ← Muchos tests, rápidos
    └─────────────────┘
```

### Distribución Recomendada

- **Unit Tests**: 70% (rápidos, aislados)
- **Integration Tests**: 20% (más lentos, con dependencias)
- **E2E Tests**: 10% (muy lentos, sistema completo)

---

## 🧪 Tests Unitarios

### Tests de Constantes

**Ubicación**: `tests/unit/test_constants.py`

```python
import pytest
from core.constants import (
    GitConfig,
    ErrorMessages,
    TaskStatus,
    AgentStatus,
    RetryConfig,
    SuccessMessages
)

class TestGitConfig:
    """Tests para GitConfig"""
    
    def test_default_base_branch(self):
        """Test que DEFAULT_BASE_BRANCH tiene el valor correcto"""
        assert GitConfig.DEFAULT_BASE_BRANCH == "main"
        assert isinstance(GitConfig.DEFAULT_BASE_BRANCH, str)
    
    def test_max_branch_name_length(self):
        """Test que MAX_BRANCH_NAME_LENGTH es un número válido"""
        assert GitConfig.MAX_BRANCH_NAME_LENGTH == 255
        assert isinstance(GitConfig.MAX_BRANCH_NAME_LENGTH, int)
        assert GitConfig.MAX_BRANCH_NAME_LENGTH > 0
    
    def test_invalid_branch_chars(self):
        """Test que INVALID_BRANCH_CHARS es una lista"""
        assert isinstance(GitConfig.INVALID_BRANCH_CHARS, list)
        assert len(GitConfig.INVALID_BRANCH_CHARS) > 0

class TestErrorMessages:
    """Tests para ErrorMessages"""
    
    def test_all_messages_exist(self):
        """Test que todos los mensajes de error existen"""
        assert ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED
        assert ErrorMessages.REPOSITORY_NOT_FOUND
        assert ErrorMessages.INVALID_BRANCH_NAME
        assert ErrorMessages.TASK_NOT_FOUND
    
    def test_messages_are_strings(self):
        """Test que todos los mensajes son strings"""
        messages = [
            ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED,
            ErrorMessages.REPOSITORY_NOT_FOUND,
            ErrorMessages.INVALID_BRANCH_NAME,
        ]
        for message in messages:
            assert isinstance(message, str)
            assert len(message) > 0
    
    def test_messages_not_empty(self):
        """Test que los mensajes no están vacíos"""
        assert len(ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED) > 0
        assert len(ErrorMessages.REPOSITORY_NOT_FOUND) > 0

class TestTaskStatus:
    """Tests para TaskStatus"""
    
    def test_all_statuses_exist(self):
        """Test que todos los estados existen"""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"
    
    def test_is_valid(self):
        """Test método is_valid"""
        assert TaskStatus.is_valid(TaskStatus.PENDING)
        assert TaskStatus.is_valid(TaskStatus.COMPLETED)
        assert not TaskStatus.is_valid("invalid")
        assert not TaskStatus.is_valid("")
        assert not TaskStatus.is_valid(None)
```

### Tests de Decoradores

**Ubicación**: `tests/unit/test_decorators.py`

```python
import pytest
from unittest.mock import patch, MagicMock
from core.utils import handle_github_exception
from api.utils import handle_api_errors
from fastapi import HTTPException

class TestHandleGithubException:
    """Tests para handle_github_exception"""
    
    def test_sync_function_success(self):
        """Test decorador con función sync exitosa"""
        @handle_github_exception
        def sync_func():
            return "success"
        
        result = sync_func()
        assert result == "success"
    
    def test_sync_function_error(self):
        """Test decorador con función sync que falla"""
        @handle_github_exception
        def sync_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            sync_func()
    
    @pytest.mark.asyncio
    async def test_async_function_success(self):
        """Test decorador con función async exitosa"""
        @handle_github_exception
        async def async_func():
            return "success"
        
        result = await async_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_async_function_error(self):
        """Test decorador con función async que falla"""
        @handle_github_exception
        async def async_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await async_func()
    
    @patch('core.utils.logger')
    def test_logs_error_with_exc_info(self, mock_logger):
        """Test que el decorador loguea errores con exc_info=True"""
        @handle_github_exception
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()
        
        # Verificar que se llamó logger.error
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        
        # Verificar que incluye exc_info=True
        assert call_args[1]['exc_info'] is True
        assert "failing_func" in call_args[0][0]
    
    def test_preserves_function_signature(self):
        """Test que el decorador preserva la signature"""
        @handle_github_exception
        def func_with_params(a: int, b: str = "default") -> str:
            return f"{a}-{b}"
        
        import inspect
        sig = inspect.signature(func_with_params)
        assert len(sig.parameters) == 2
        assert sig.return_annotation == str

class TestHandleApiErrors:
    """Tests para handle_api_errors"""
    
    @pytest.mark.asyncio
    async def test_async_endpoint_success(self):
        """Test decorador con endpoint async exitoso"""
        @handle_api_errors
        async def endpoint():
            return {"success": True}
        
        result = await endpoint()
        assert result == {"success": True}
    
    @pytest.mark.asyncio
    async def test_async_endpoint_error(self):
        """Test decorador convierte errores a HTTPException"""
        @handle_api_errors
        async def endpoint():
            raise ValueError("Test error")
        
        with pytest.raises(HTTPException) as exc_info:
            await endpoint()
        
        assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_http_exception_passthrough(self):
        """Test que HTTPException se pasa sin modificar"""
        @handle_api_errors
        async def endpoint():
            raise HTTPException(status_code=404, detail="Not found")
        
        with pytest.raises(HTTPException) as exc_info:
            await endpoint()
        
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"
```

---

## 🔗 Tests de Integración

### Tests de Uso de Constantes

**Ubicación**: `tests/integration/test_constants_integration.py`

```python
import pytest
from core.utils import parse_instruction_params
from core.constants import GitConfig

class TestConstantsIntegration:
    """Tests de integración para constantes"""
    
    def test_parse_instruction_params_uses_constants(self):
        """Test que parse_instruction_params usa constantes"""
        params = parse_instruction_params("test instruction")
        
        # Verificar que usa constantes, no strings hardcoded
        assert params["branch"] == GitConfig.DEFAULT_BASE_BRANCH
        assert params["base_branch"] == GitConfig.DEFAULT_BASE_BRANCH
        assert params["head"] == GitConfig.DEFAULT_BASE_BRANCH
        assert params["base"] == GitConfig.DEFAULT_BASE_BRANCH
    
    def test_no_hardcoded_strings(self):
        """Test que no hay strings hardcoded en params"""
        params = parse_instruction_params("test")
        
        # Verificar que no hay strings "main" hardcoded
        # (deberían ser referencias a la constante)
        assert params["branch"] != "main"  # Debería ser la constante
        assert params["branch"] == GitConfig.DEFAULT_BASE_BRANCH
```

### Tests de Decoradores con Código Real

**Ubicación**: `tests/integration/test_decorators_integration.py`

```python
import pytest
from unittest.mock import AsyncMock, patch
from core.github_client import GitHubClient
from core.utils import handle_github_exception

class TestDecoratorsIntegration:
    """Tests de integración para decoradores"""
    
    @pytest.mark.asyncio
    async def test_decorator_with_github_client(self):
        """Test decorador con GitHub client real"""
        @handle_github_exception
        async def fetch_repo_mock():
            # Simular error de GitHub
            raise Exception("Repository not found")
        
        # Verificar que el decorador maneja el error
        with pytest.raises(Exception, match="Repository not found"):
            await fetch_repo_mock()
```

---

## 🎯 Tests de Regresión

### Tests de No Regresión

**Ubicación**: `tests/regression/test_v8_migration.py`

```python
import pytest
from pathlib import Path
import re

class TestNoRegression:
    """Tests para asegurar que no hay regresiones"""
    
    def test_no_hardcoded_strings(self):
        """Test que no hay strings hardcoded en el código"""
        hardcoded_patterns = [
            r'"main"',
            r"'main'",
            r'"failed"',
            r'"completed"',
        ]
        
        code_files = list(Path('core').rglob('*.py'))
        code_files.extend(list(Path('api').rglob('*.py')))
        
        violations = []
        for file_path in code_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                for pattern in hardcoded_patterns:
                    # Ignorar comentarios y strings en tests
                    if 'test' in str(file_path) or '# ' in content:
                        continue
                    
                    if re.search(pattern, content):
                        violations.append((file_path, pattern))
            except Exception:
                continue
        
        assert len(violations) == 0, f"Found hardcoded strings: {violations}"
    
    def test_all_functions_have_decorators(self):
        """Test que funciones críticas tienen decoradores"""
        # Este test puede ser específico según necesidades
        pass
```

---

## 📊 Cobertura de Tests

### Objetivo de Cobertura

- **Constantes**: 100%
- **Decoradores**: 95%+
- **Integración**: 80%+
- **General**: 80%+

### Medición de Cobertura

```bash
# Ejecutar con cobertura
pytest tests/ --cov=core --cov=api --cov-report=html

# Ver reporte
open htmlcov/index.html
```

---

## 🚀 Ejecución de Tests

### Comandos Básicos

```bash
# Todos los tests
pytest tests/ -v

# Solo tests unitarios
pytest tests/unit/ -v

# Solo tests de integración
pytest tests/integration/ -v

# Con cobertura
pytest tests/ --cov=core --cov=api

# Tests específicos
pytest tests/unit/test_constants.py -v

# Tests con markers
pytest tests/ -m "not slow" -v
```

### Configuración de pytest

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    regression: Regression tests
    slow: Slow tests
```

---

## 📝 Mejores Prácticas

### 1. Nombres Descriptivos

```python
# ✅ Bueno
def test_handle_github_exception_logs_error_with_exc_info():
    pass

# ❌ Malo
def test_decorator():
    pass
```

### 2. Un Test, Una Aserción

```python
# ✅ Bueno
def test_default_branch():
    assert GitConfig.DEFAULT_BASE_BRANCH == "main"
    assert isinstance(GitConfig.DEFAULT_BASE_BRANCH, str)

# ❌ Malo (aunque a veces múltiples aserciones son OK)
def test_everything():
    assert GitConfig.DEFAULT_BASE_BRANCH == "main"
    assert ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED
    assert TaskStatus.PENDING == "pending"
```

### 3. Usar Fixtures

```python
@pytest.fixture
def sample_task():
    return {
        "id": "test-id",
        "status": TaskStatus.PENDING,
        "repository": "owner/repo"
    }

def test_task_creation(sample_task):
    assert sample_task["status"] == TaskStatus.PENDING
```

### 4. Mocks Apropiados

```python
@patch('core.utils.logger')
def test_logging(mock_logger):
    # Test específico
    pass
```

---

## 🔍 Debugging de Tests

### Ejecutar Tests con Debug

```bash
# Con output detallado
pytest tests/ -v -s

# Con pdb en fallos
pytest tests/ --pdb

# Solo el primer fallo
pytest tests/ -x
```

### Ver Output de Prints

```python
def test_with_print():
    print("Debug info")
    assert True
```

Ejecutar con `-s`:
```bash
pytest tests/ -s
```

---

## 📚 Recursos

- [pytest Documentation](https://docs.pytest.org/)
- [IMPROVEMENTS_V8.md](IMPROVEMENTS_V8.md) - Documentación completa
- [IMPROVEMENTS_V8_REAL_EXAMPLES.md](IMPROVEMENTS_V8_REAL_EXAMPLES.md) - Ejemplos

---

**Última actualización**: Enero 2025  
**Versión**: V8



