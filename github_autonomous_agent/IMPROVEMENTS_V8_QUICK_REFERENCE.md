# Quick Reference - Mejoras V8

## Cheat Sheet Rápido

---

## 🎯 Constantes Principales

```python
from core.constants import GitConfig, ErrorMessages

# Git
GitConfig.DEFAULT_BASE_BRANCH  # "main"

# Errores
ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED
ErrorMessages.REPOSITORY_NOT_FOUND
ErrorMessages.INVALID_BRANCH_NAME
```

---

## 🔧 Decoradores

### handle_github_exception

```python
from core.utils import handle_github_exception

# Sync
@handle_github_exception
def sync_func():
    pass

# Async
@handle_github_exception
async def async_func():
    pass
```

### handle_api_errors

```python
from api.utils import handle_api_errors

@handle_api_errors
async def api_endpoint():
    pass
```

---

## ✅ Checklist Rápido

### Antes de Commit
- [ ] Constantes usadas (no strings hardcodeados)
- [ ] Decoradores aplicados
- [ ] Type hints completos
- [ ] `exc_info=True` en logs de error
- [ ] Tests pasan

### Code Review
- [ ] ¿Strings hardcodeados? → Usar constantes
- [ ] ¿Funciones async sin decorador? → Agregar
- [ ] ¿Logs sin stack trace? → Agregar `exc_info=True`
- [ ] ¿Type hints faltantes? → Agregar

---

## 🚨 Errores Comunes

| Error | Solución |
|-------|----------|
| `NameError: name 'GitConfig' is not defined` | `from core.constants import GitConfig` |
| Decorador no funciona con async | Usar versión V8 (detección automática) |
| Sin stack trace en logs | Agregar `exc_info=True` |
| Type error en mypy | Usar `TypeVar` y `Callable` |

---

## 📝 Templates Rápidos

### Nueva Función
```python
@handle_github_exception
async def my_func(param: str = GitConfig.DEFAULT_BASE_BRANCH):
    pass
```

### Nuevo Endpoint
```python
@router.post("/endpoint")
@handle_api_errors
async def endpoint(data: Schema):
    if not data.field:
        raise HTTPException(400, detail=ErrorMessages.FIELD_REQUIRED)
    return {"success": True}
```

---

## 🔍 Comandos Útiles

```bash
# Verificar constantes
python scripts/verify-constants-usage.py

# Buscar hardcoded
python scripts/find-hardcoded-strings.py

# Analizar decoradores
python scripts/analyze-decorator-usage.py

# Migrar código
python scripts/migrate-to-constants.py --dry-run
```

---

**Versión**: V8  
**Última actualización**: [Fecha]



