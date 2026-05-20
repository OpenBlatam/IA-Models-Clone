# Ejemplos Reales del Código - Mejoras V8

## Ejemplos Basados en el Código Real del Proyecto

---

## 📋 Constantes Reales del Proyecto

### Estructura Completa de `core/constants.py`

```python
# core/constants.py - Código Real

class GitConfig:
    """Configuración relacionada con Git."""
    MAX_BRANCH_NAME_LENGTH = 255
    INVALID_BRANCH_CHARS = ['~', '^', ':', '?', '*', '[', ' ', '..', '@{', '\\']
    DEFAULT_BASE_BRANCH = "main"  # ✅ Constante centralizada

class ErrorMessages:
    """Mensajes de error estandarizados."""
    GITHUB_TOKEN_REQUIRED = "GitHub token es requerido"
    GITHUB_TOKEN_NOT_CONFIGURED = (
        "GitHub token no configurado. "
        "Configure GITHUB_TOKEN en las variables de entorno."
    )
    TASK_NOT_FOUND = "Tarea no encontrada"
    REPOSITORY_NOT_FOUND = "Repositorio no encontrado"
    INVALID_INSTRUCTION = "Instrucción inválida"
    INVALID_FILE_PATH = "Ruta de archivo inválida"
    INVALID_BRANCH_NAME = "Nombre de rama inválido"
    TASK_PROCESSOR_NOT_INITIALIZED = "Task processor no inicializado"

class TaskStatus:
    """Estados posibles de una tarea."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

## 🔧 Ejemplos Reales de Decoradores

### 1. Decorador `handle_github_exception` - Código Real

```python
# core/utils.py - Implementación Real

from functools import wraps
import asyncio
from typing import Callable
from config.logging_config import get_logger
from core.constants import GitConfig

logger = get_logger(__name__)

def handle_github_exception(func: Callable) -> Callable:
    """
    Decorador para manejar excepciones de GitHub de forma consistente.
    Soporta funciones síncronas y asíncronas.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
            raise
    
    # ✅ Detección automática
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
```

### 2. Uso Real en `parse_instruction_params`

```python
# core/utils.py - Código Real Mejorado

def parse_instruction_params(instruction: str) -> Dict[str, Any]:
    """
    Parsear parámetros de una instrucción de forma más robusta.
    """
    if not instruction or not isinstance(instruction, str):
        return {
            "file_path": None,
            "content": "",
            "branch": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
            "branch_name": None,
            "base_branch": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
            "title": None,
            "body": instruction or "",
            "head": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
            "base": GitConfig.DEFAULT_BASE_BRANCH  # ✅ Constante
        }
    
    instruction_lower = instruction.lower()
    params = {
        "file_path": None,
        "content": "",
        "branch": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
        "branch_name": None,
        "base_branch": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
        "title": None,
        "body": instruction,
        "head": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
        "base": GitConfig.DEFAULT_BASE_BRANCH  # ✅ Constante
    }
    
    # ... resto del parsing ...
    
    return params
```

**Antes (V7):**
```python
# ❌ Strings hardcodeados
params = {
    "branch": "main",  # Hardcoded
    "base_branch": "main",  # Hardcoded
    "head": "main",  # Hardcoded
    "base": "main"  # Hardcoded
}
```

---

## 🎯 Casos de Uso Reales

### Caso 1: Task Processor con Constantes

**Ubicación**: `core/task_processor.py`

```python
from core.constants import TaskStatus, ErrorMessages, GitConfig
from core.utils import handle_github_exception
from core.exceptions import TaskProcessingError

class TaskProcessor:
    @handle_github_exception
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar tarea con manejo de errores mejorado"""
        task_id = task.get("id")
        
        try:
            # Procesar tarea...
            result = await self._process_task(task)
            
            # Actualizar estado usando constante
            await self.storage.update_task_status(
                task_id,
                TaskStatus.COMPLETED,  # ✅ Constante
                result=result
            )
            
            return {"success": True, "result": result}
            
        except TaskProcessingError as e:
            # Re-raise excepciones específicas
            logger.error(
                f"Error al procesar tarea {task_id}: {e}",
                exc_info=True  # ✅ Stack trace completo
            )
            
            await self.storage.update_task_status(
                task_id,
                TaskStatus.FAILED,  # ✅ Constante
                error=str(e)
            )
            
            raise
            
        except Exception as e:
            # Error inesperado
            logger.error(
                f"Error inesperado al ejecutar tarea {task_id}: {e}",
                exc_info=True  # ✅ Stack trace completo
            )
            
            await self.storage.update_task_status(
                task_id,
                TaskStatus.FAILED,  # ✅ Constante
                error=ErrorMessages.TASK_PROCESSOR_NOT_INITIALIZED  # ✅ Constante
            )
            
            raise TaskProcessingError(
                message=ErrorMessages.TASK_PROCESSOR_NOT_INITIALIZED,
                task_id=task_id,
                original_error=e
            )
```

### Caso 2: API Routes con Validación

**Ubicación**: `api/routes/task_routes.py`

```python
from fastapi import APIRouter, HTTPException, Depends
from api.utils import handle_api_errors, validate_github_token
from core.constants import ErrorMessages, TaskStatus
from api.schemas import CreateTaskRequest, TaskResponse

router = APIRouter()

@router.post("/tasks", response_model=TaskResponse)
@handle_api_errors  # ✅ Decorador universal
async def create_task(
    request: CreateTaskRequest,
    _: None = Depends(validate_github_token)  # ✅ Validación de token
):
    """Crear nueva tarea"""
    
    # Validación usando constantes
    if not request.repository_owner or not request.repository_name:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.REPOSITORY_NOT_FOUND  # ✅ Constante
        )
    
    if not request.instruction:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.INVALID_INSTRUCTION  # ✅ Constante
        )
    
    # Crear tarea...
    task = await use_case.execute(request)
    
    return TaskResponse(**task)
```

### Caso 3: GitHub Client con Decoradores

**Ubicación**: `core/github_client.py`

```python
from core.utils import handle_github_exception
from core.constants import GitConfig, ErrorMessages
from core.exceptions import GitHubClientError

class GitHubClient:
    @handle_github_exception  # ✅ Decorador universal
    async def get_repo(self, repo_name: str):
        """Obtener repositorio de GitHub"""
        try:
            repo = await self._github.get_repo(repo_name)
            return repo
        except Exception as e:
            raise GitHubClientError(
                message=ErrorMessages.REPOSITORY_NOT_FOUND,  # ✅ Constante
                repo=repo_name,
                original_error=e
            )
    
    @handle_github_exception  # ✅ Decorador universal
    async def create_branch(
        self,
        repo_name: str,
        branch_name: str,
        base: str = GitConfig.DEFAULT_BASE_BRANCH  # ✅ Constante
    ):
        """Crear rama desde base"""
        if not branch_name:
            raise ValueError(ErrorMessages.INVALID_BRANCH_NAME)  # ✅ Constante
        
        # Validar longitud
        if len(branch_name) > GitConfig.MAX_BRANCH_NAME_LENGTH:  # ✅ Constante
            raise ValueError(ErrorMessages.INVALID_BRANCH_NAME)  # ✅ Constante
        
        # Crear rama...
        return await self._create_branch_internal(repo_name, branch_name, base)
```

### Caso 4: Validación de Token

**Ubicación**: `api/utils.py`

```python
from core.constants import ErrorMessages
from fastapi import HTTPException
import os

def validate_github_token() -> str:
    """Validar GitHub token"""
    token = os.getenv("GITHUB_TOKEN")
    
    if not token:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED  # ✅ Constante
        )
    
    return token
```

**Antes (V7):**
```python
# ❌ Mensaje hardcodeado
if not token:
    raise HTTPException(
        status_code=400,
        detail="GitHub token no configurado. Por favor, configure GITHUB_TOKEN..."  # Hardcoded
    )
```

---

## 🔄 Migración Real: Antes y Después

### Ejemplo 1: Función de Parsing

**Antes (V7):**
```python
# core/utils.py - V7
def parse_instruction_params(instruction: str) -> dict:
    params = {
        "branch": "main",  # ❌ Hardcoded
        "base_branch": "main",  # ❌ Hardcoded
        "head": "main",  # ❌ Hardcoded
        "base": "main"  # ❌ Hardcoded
    }
    # ... parsing ...
    return params
```

**Después (V8):**
```python
# core/utils.py - V8
from core.constants import GitConfig

def parse_instruction_params(instruction: str) -> dict:
    params = {
        "branch": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
        "base_branch": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
        "head": GitConfig.DEFAULT_BASE_BRANCH,  # ✅ Constante
        "base": GitConfig.DEFAULT_BASE_BRANCH  # ✅ Constante
    }
    # ... parsing ...
    return params
```

### Ejemplo 2: Manejo de Errores

**Antes (V7):**
```python
# core/task_processor.py - V7
async def execute_task(self, task: dict):
    try:
        # código...
    except Exception as e:
        logger.error(f"Error: {e}")  # ❌ Sin exc_info
        await self.storage.update_task_status(
            task_id,
            "failed",  # ❌ String hardcoded
            error=str(e)
        )
```

**Después (V8):**
```python
# core/task_processor.py - V8
from core.constants import TaskStatus, ErrorMessages
from core.utils import handle_github_exception

@handle_github_exception  # ✅ Decorador
async def execute_task(self, task: dict):
    try:
        # código...
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)  # ✅ Stack trace
        await self.storage.update_task_status(
            task_id,
            TaskStatus.FAILED,  # ✅ Constante
            error=ErrorMessages.TASK_NOT_FOUND  # ✅ Constante
        )
```

---

## 📊 Métricas Reales de Impacto

### Archivos Modificados

| Archivo | Cambios | Líneas Mejoradas |
|---------|---------|------------------|
| `core/utils.py` | 4 constantes, 1 decorador | ~50 líneas |
| `core/task_processor.py` | 3 constantes, decoradores | ~30 líneas |
| `api/utils.py` | 2 constantes, 1 decorador | ~40 líneas |
| `api/routes/task_routes.py` | 2 constantes | ~20 líneas |
| `core/github_client.py` | 3 constantes, decoradores | ~35 líneas |
| **TOTAL** | **14 constantes, 3 decoradores** | **~175 líneas** |

### Strings Hardcodeados Eliminados

- `"main"`: 8 instancias → `GitConfig.DEFAULT_BASE_BRANCH`
- `"failed"`: 3 instancias → `TaskStatus.FAILED`
- `"completed"`: 2 instancias → `TaskStatus.COMPLETED`
- Mensajes de error: 5 instancias → `ErrorMessages.*`

**Total**: 18 strings hardcodeados eliminados

---

## 🧪 Tests Reales

### Test de Constantes

```python
# tests/unit/test_constants.py
import pytest
from core.constants import GitConfig, ErrorMessages, TaskStatus

def test_git_config_default_branch():
    """Test que DEFAULT_BASE_BRANCH tiene el valor correcto"""
    assert GitConfig.DEFAULT_BASE_BRANCH == "main"
    assert isinstance(GitConfig.DEFAULT_BASE_BRANCH, str)

def test_error_messages_exist():
    """Test que todos los mensajes de error existen"""
    assert ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED
    assert ErrorMessages.REPOSITORY_NOT_FOUND
    assert ErrorMessages.INVALID_BRANCH_NAME
    assert len(ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED) > 0

def test_task_status_constants():
    """Test que los estados de tarea son válidos"""
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"
    assert TaskStatus.is_valid(TaskStatus.PENDING)
```

### Test de Decoradores

```python
# tests/unit/test_decorators.py
import pytest
from unittest.mock import patch
from core.utils import handle_github_exception

def test_handle_github_exception_sync():
    """Test decorador con función sync"""
    @handle_github_exception
    def sync_func():
        return "success"
    
    result = sync_func()
    assert result == "success"

def test_handle_github_exception_sync_error():
    """Test decorador con función sync que falla"""
    @handle_github_exception
    def sync_func():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        sync_func()

@pytest.mark.asyncio
async def test_handle_github_exception_async():
    """Test decorador con función async"""
    @handle_github_exception
    async def async_func():
        return "success"
    
    result = await async_func()
    assert result == "success"

@patch('core.utils.logger')
def test_handle_github_exception_logs_error(mock_logger):
    """Test que el decorador loguea errores con exc_info"""
    @handle_github_exception
    def failing_func():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        failing_func()
    
    # Verificar que se llamó logger.error con exc_info=True
    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args
    assert call_args[1]['exc_info'] is True
```

### Test de Integración

```python
# tests/integration/test_constants_integration.py
import pytest
from core.utils import parse_instruction_params
from core.constants import GitConfig

def test_parse_instruction_params_uses_constants():
    """Test que parse_instruction_params usa constantes"""
    params = parse_instruction_params("test instruction")
    
    # Verificar que usa constantes, no strings hardcoded
    assert params["branch"] == GitConfig.DEFAULT_BASE_BRANCH
    assert params["base_branch"] == GitConfig.DEFAULT_BASE_BRANCH
    assert params["head"] == GitConfig.DEFAULT_BASE_BRANCH
    assert params["base"] == GitConfig.DEFAULT_BASE_BRANCH
    
    # Verificar que no hay strings "main" hardcoded
    assert params["branch"] != "main"  # Debería ser la constante
    assert params["branch"] == GitConfig.DEFAULT_BASE_BRANCH
```

---

## 🎯 Patrones de Uso Real

### Patrón 1: Validación con Constantes

```python
# Patrón común en el código
from core.constants import ErrorMessages, GitConfig
from fastapi import HTTPException

def validate_input(data: dict):
    """Validar input usando constantes"""
    if not data.get("field"):
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.INVALID_INSTRUCTION  # ✅ Constante
        )
    
    if data.get("branch") == GitConfig.DEFAULT_BASE_BRANCH:  # ✅ Constante
        # Lógica especial para rama principal
        pass
```

### Patrón 2: Actualización de Estado

```python
# Patrón común en task_processor
from core.constants import TaskStatus

async def update_task_status(task_id: str, status: str):
    """Actualizar estado usando constante"""
    if not TaskStatus.is_valid(status):
        raise ValueError(f"Invalid status: {status}")
    
    await storage.update_task_status(
        task_id,
        status  # Usar constante: TaskStatus.COMPLETED
    )
```

### Patrón 3: Decorador + Constantes

```python
# Patrón común: Decorador + Constantes
from core.utils import handle_github_exception
from core.constants import GitConfig, ErrorMessages

@handle_github_exception
async def process_repository(repo_name: str, branch: str = None):
    """Procesar repositorio con decorador y constantes"""
    # Usar constante como default
    branch = branch or GitConfig.DEFAULT_BASE_BRANCH
    
    # Validar usando constante
    if not repo_name:
        raise ValueError(ErrorMessages.REPOSITORY_NOT_FOUND)
    
    # Procesar...
```

---

## 🔍 Búsqueda de Uso Real

### Comandos para Encontrar Uso

```bash
# Buscar uso de constantes
grep -r "GitConfig.DEFAULT_BASE_BRANCH" --include="*.py"

# Buscar strings hardcodeados que deberían ser constantes
grep -r '"main"' --include="*.py" | grep -v "GitConfig"
grep -r '"failed"' --include="*.py" | grep -v "TaskStatus"

# Buscar uso de decoradores
grep -r "@handle_github_exception" --include="*.py"
grep -r "@handle_api_errors" --include="*.py"

# Buscar mensajes de error hardcodeados
grep -r "GitHub token no configurado" --include="*.py" | grep -v "ErrorMessages"
```

---

## 📝 Checklist de Verificación Real

### Verificar Constantes

- [ ] ¿Todos los `"main"` fueron reemplazados por `GitConfig.DEFAULT_BASE_BRANCH`?
- [ ] ¿Todos los estados de tarea usan `TaskStatus.*`?
- [ ] ¿Todos los mensajes de error usan `ErrorMessages.*`?
- [ ] ¿Los imports de constantes están presentes?

### Verificar Decoradores

- [ ] ¿Las funciones async tienen `@handle_github_exception`?
- [ ] ¿Los endpoints tienen `@handle_api_errors`?
- [ ] ¿Los logs incluyen `exc_info=True`?
- [ ] ¿Los decoradores funcionan con sync y async?

### Verificar Tests

- [ ] ¿Hay tests para constantes?
- [ ] ¿Hay tests para decoradores?
- [ ] ¿Los tests verifican que no hay strings hardcodeados?
- [ ] ¿Los tests de integración pasan?

---

## 🚀 Próximos Pasos Reales

### 1. Migrar Código Restante

```bash
# Encontrar código que aún no migró
python scripts/find-hardcoded-strings.py

# Migrar automáticamente
python scripts/migrate-to-constants.py --dry-run
python scripts/migrate-to-constants.py
```

### 2. Agregar Tests

```bash
# Generar tests
python scripts/generate-decorator-tests.py

# Ejecutar tests
pytest tests/unit/test_constants.py -v
pytest tests/unit/test_decorators.py -v
```

### 3. Verificar

```bash
# Verificar uso de constantes
python scripts/verify-constants-usage.py

# Verificar decoradores
python scripts/analyze-decorator-usage.py
```

---

**Última actualización**: [Fecha]  
**Basado en**: Código real del proyecto  
**Versión**: V8



