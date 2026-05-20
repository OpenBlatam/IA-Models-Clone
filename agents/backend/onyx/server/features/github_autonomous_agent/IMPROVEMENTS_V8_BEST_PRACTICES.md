# Mejores Prácticas Avanzadas - Mejoras V8

## Guía Completa de Mejores Prácticas

---

## 🎯 Principios Fundamentales

### 1. DRY (Don't Repeat Yourself)

**❌ Evitar:**
```python
# Strings hardcodeados repetidos
if branch == "main":
    pass
if base == "main":
    pass
if head == "main":
    pass
```

**✅ Hacer:**
```python
from core.constants import GitConfig

if branch == GitConfig.DEFAULT_BASE_BRANCH:
    pass
if base == GitConfig.DEFAULT_BASE_BRANCH:
    pass
if head == GitConfig.DEFAULT_BASE_BRANCH:
    pass
```

---

### 2. Single Source of Truth

**❌ Evitar:**
```python
# Múltiples lugares con el mismo valor
DEFAULT_BRANCH = "main"  # En un archivo
default_branch = "main"  # En otro archivo
BRANCH = "main"  # En otro archivo
```

**✅ Hacer:**
```python
# Un solo lugar
# core/constants.py
class GitConfig:
    DEFAULT_BASE_BRANCH = "main"
```

---

### 3. Fail Fast

**❌ Evitar:**
```python
# Validación tardía
def process_branch(branch):
    # ... mucho código ...
    if branch == "main":  # Validación al final
        raise ValueError("Invalid branch")
```

**✅ Hacer:**
```python
# Validación temprana
from core.constants import GitConfig, ErrorMessages

def process_branch(branch):
    if branch == GitConfig.DEFAULT_BASE_BRANCH:
        raise ValueError(ErrorMessages.INVALID_BRANCH_NAME)
    # ... resto del código ...
```

---

## 📝 Convenciones de Código

### Nombres de Constantes

**✅ Buenas prácticas:**
```python
# UPPER_CASE para constantes
class GitConfig:
    DEFAULT_BASE_BRANCH = "main"  # ✅ Claro y descriptivo
    MAX_BRANCH_NAME_LENGTH = 255  # ✅ Específico
    
# Nombres descriptivos
class ErrorMessages:
    GITHUB_TOKEN_NOT_CONFIGURED = "..."  # ✅ Específico
    REPOSITORY_NOT_FOUND = "..."  # ✅ Claro
```

**❌ Evitar:**
```python
# Nombres genéricos
class Config:
    BRANCH = "main"  # ❌ Muy genérico
    MSG = "Error"  # ❌ Muy corto
```

---

### Organización de Constantes

**✅ Estructura recomendada:**
```python
# core/constants.py

# 1. Estados (primero, más usados)
class TaskStatus:
    PENDING = "pending"
    COMPLETED = "completed"
    # ...

# 2. Configuración
class GitConfig:
    DEFAULT_BASE_BRANCH = "main"
    # ...

# 3. Mensajes
class ErrorMessages:
    # ...
class SuccessMessages:
    # ...

# 4. Configuración avanzada
class RetryConfig:
    # ...
```

---

## 🔧 Uso de Decoradores

### Cuándo Usar Decoradores

**✅ Usar decoradores para:**
- Funciones que interactúan con APIs externas
- Endpoints de API
- Funciones que pueden fallar y necesitan logging
- Funciones que necesitan manejo de errores consistente

**❌ No usar decoradores para:**
- Funciones muy simples (getters/setters)
- Funciones de utilidad pura sin riesgo de error
- Funciones que ya manejan errores específicamente

---

### Orden de Decoradores

**✅ Orden correcto:**
```python
# 1. Retry primero (más externo)
# 2. Error handling después
@retry_async_on_github_error(max_attempts=3)
@handle_github_exception
async def my_function():
    pass
```

**Explicación**: El retry debe estar más externo para capturar errores del decorador interno.

---

### Decoradores Múltiples

**✅ Buen uso:**
```python
# Combinar decoradores cuando tiene sentido
@rate_limit(max_calls=10, period=60)
@handle_github_exception
@cache_result(ttl=300)
async def expensive_operation():
    pass
```

**❌ Evitar:**
```python
# Demasiados decoradores hacen el código difícil de leer
@decorator1
@decorator2
@decorator3
@decorator4
@decorator5
async def my_function():
    pass  # ❌ Demasiado complejo
```

---

## 📊 Logging

### Niveles de Logging

**✅ Usar niveles apropiados:**
```python
# DEBUG: Información detallada para debugging
logger.debug("Processing task", extra={"task_id": task_id})

# INFO: Eventos normales
logger.info("Task completed", extra={"task_id": task_id})

# WARNING: Algo inesperado pero manejable
logger.warning("Rate limit approaching", extra={"remaining": 10})

# ERROR: Errores que requieren atención
logger.error("Task failed", exc_info=True, extra={"task_id": task_id})

# CRITICAL: Errores críticos
logger.critical("System failure", exc_info=True)
```

---

### Logging Estructurado

**✅ Buen logging:**
```python
logger.error(
    f"Error en {func.__name__}: {e}",
    exc_info=True,  # ✅ Stack trace
    extra={  # ✅ Contexto estructurado
        "function": func.__name__,
        "error_type": type(e).__name__,
        "task_id": task_id,
        "repository": repo_name
    }
)
```

**❌ Logging pobre:**
```python
logger.error(f"Error: {e}")  # ❌ Sin contexto ni stack trace
```

---

## 🎨 Type Hints

### Type Hints Completos

**✅ Buenos type hints:**
```python
from typing import Callable, TypeVar, Optional, Dict, Any
from core.constants import GitConfig

T = TypeVar('T')

def process_branch(
    branch: str = GitConfig.DEFAULT_BASE_BRANCH,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Procesar rama con type hints completos"""
    pass
```

**❌ Type hints incompletos:**
```python
def process_branch(branch, metadata=None):  # ❌ Sin type hints
    pass
```

---

### Type Hints en Decoradores

**✅ Decorador con type hints:**
```python
from typing import Callable, TypeVar, Awaitable

T = TypeVar('T')

def handle_github_exception(
    func: Callable[..., T]
) -> Callable[..., T | Awaitable[T]]:
    """Decorador con type hints completos"""
    # ...
```

---

## 🔒 Seguridad

### Validación de Inputs

**✅ Validar siempre:**
```python
from core.constants import ErrorMessages, GitConfig
from fastapi import HTTPException

def validate_branch(branch: str) -> str:
    """Validar rama con constantes"""
    if not branch:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.INVALID_BRANCH_NAME
        )
    
    if len(branch) > GitConfig.MAX_BRANCH_NAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.INVALID_BRANCH_NAME
        )
    
    return branch
```

---

### No Exponer Información Sensible

**❌ Nunca hacer:**
```python
# ❌ NUNCA loguear tokens
logger.debug(f"Token: {token}")

# ❌ NUNCA exponer en mensajes de error
raise HTTPException(
    detail=f"Error with token: {token[:10]}..."
)
```

**✅ Hacer:**
```python
# ✅ Loguear solo metadata
logger.debug(
    "Token configured",
    extra={"token_length": len(token) if token else 0}
)

# ✅ Mensajes genéricos
raise HTTPException(
    status_code=400,
    detail=ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED
)
```

---

## ⚡ Performance

### Lazy Evaluation

**✅ Usar lazy evaluation:**
```python
# ✅ Solo evalúa si DEBUG está habilitado
logger.debug("Processing %s", lambda: expensive_operation())

# ❌ Evalúa siempre, incluso si DEBUG está deshabilitado
logger.debug(f"Processing {expensive_operation()}")
```

---

### Cache de Constantes

**✅ Las constantes son eficientes:**
```python
# ✅ Referencia directa, sin overhead
branch = GitConfig.DEFAULT_BASE_BRANCH

# ✅ Mismo performance que string literal
if branch == "main":  # Comparación rápida
    pass
```

---

## 🧪 Testing

### Tests de Constantes

**✅ Tests exhaustivos:**
```python
def test_git_config():
    """Test completo de GitConfig"""
    # Test valores
    assert GitConfig.DEFAULT_BASE_BRANCH == "main"
    
    # Test tipos
    assert isinstance(GitConfig.DEFAULT_BASE_BRANCH, str)
    
    # Test inmutabilidad (si aplica)
    # ...
```

---

### Tests de Decoradores

**✅ Tests completos:**
```python
@pytest.mark.asyncio
async def test_decorator_async():
    """Test decorador con función async"""
    @handle_github_exception
    async def test_func():
        return "success"
    
    result = await test_func()
    assert result == "success"

@patch('core.utils.logger')
def test_decorator_logging(mock_logger):
    """Test que decorador loguea correctamente"""
    @handle_github_exception
    def failing_func():
        raise ValueError("Test")
    
    with pytest.raises(ValueError):
        failing_func()
    
    mock_logger.error.assert_called_once()
    assert mock_logger.error.call_args[1]['exc_info'] is True
```

---

## 📚 Documentación

### Docstrings

**✅ Buen docstring:**
```python
def process_branch(
    branch: str = GitConfig.DEFAULT_BASE_BRANCH
) -> Dict[str, Any]:
    """
    Procesar rama de Git.
    
    Args:
        branch: Nombre de la rama (default: GitConfig.DEFAULT_BASE_BRANCH)
    
    Returns:
        Diccionario con información de la rama procesada
    
    Raises:
        ValueError: Si la rama es inválida
        HTTPException: Si hay error en la API
    
    Example:
        >>> result = process_branch("feature/new-feature")
        >>> assert result["success"] is True
    """
    pass
```

---

### Comentarios

**✅ Comentarios útiles:**
```python
# Usar constante para mantener consistencia
branch = GitConfig.DEFAULT_BASE_BRANCH

# El decorador maneja errores y logging automáticamente
@handle_github_exception
async def fetch_repo():
    pass
```

**❌ Comentarios innecesarios:**
```python
# Asignar variable
branch = GitConfig.DEFAULT_BASE_BRANCH  # ❌ Obvio

# Función async
async def fetch_repo():  # ❌ Ya se ve en el código
    pass
```

---

## 🔄 Refactoring

### Migración Incremental

**✅ Estrategia recomendada:**
1. Migrar un módulo a la vez
2. Verificar tests después de cada cambio
3. Commit frecuente
4. Code review después de cada módulo

---

### Verificación Post-Migración

**✅ Checklist:**
- [ ] Todos los strings hardcodeados migrados
- [ ] Todos los imports agregados
- [ ] Tests pasan
- [ ] No hay regresiones
- [ ] Documentación actualizada

---

## 🎯 Code Review

### Checklist para Reviewers

**Constantes:**
- [ ] ¿Se usan constantes en lugar de strings?
- [ ] ¿Las constantes están importadas?
- [ ] ¿No hay strings hardcodeados equivalentes?

**Decoradores:**
- [ ] ¿Las funciones críticas tienen decoradores?
- [ ] ¿El orden de decoradores es correcto?
- [ ] ¿Los logs incluyen `exc_info=True`?

**Type Hints:**
- [ ] ¿Los type hints están completos?
- [ ] ¿Pasa mypy sin errores?

**Logging:**
- [ ] ¿Los logs incluyen contexto?
- [ ] ¿Los logs incluyen stack traces?
- [ ] ¿No se expone información sensible?

---

## 📊 Métricas de Calidad

### Objetivos

- **Strings hardcodeados**: 0
- **Cobertura de constantes**: 100%
- **Cobertura de decoradores**: 95%+
- **Type hints completos**: 95%+
- **Cobertura de tests**: 80%+

---

## 🚀 Optimizaciones

### Preferencias de Performance

1. **Constantes**: Sin overhead, usar siempre
2. **Decoradores**: Overhead mínimo, beneficios grandes
3. **Logging**: Lazy evaluation cuando sea posible
4. **Type hints**: Sin overhead en runtime

---

## 📝 Checklist Final

### Antes de Commit

- [ ] ¿Usé constantes en lugar de strings?
- [ ] ¿Agregué decoradores apropiados?
- [ ] ¿Los type hints están completos?
- [ ] ¿Los logs incluyen `exc_info=True`?
- [ ] ¿Los tests pasan?
- [ ] ¿La documentación está actualizada?

---

## 🔗 Recursos

- [IMPROVEMENTS_V8.md](IMPROVEMENTS_V8.md) - Documentación completa
- [IMPROVEMENTS_V8_REAL_EXAMPLES.md](IMPROVEMENTS_V8_REAL_EXAMPLES.md) - Ejemplos
- [IMPROVEMENTS_V8_FAQ.md](IMPROVEMENTS_V8_FAQ.md) - FAQ

---

**Última actualización**: Enero 2025  
**Versión**: V8



