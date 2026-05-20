# Workflows y Procesos - Mejoras V8

## Procesos de Desarrollo Mejorados

---

## 🔄 Workflow de Desarrollo

### 1. Desarrollo de Nueva Feature

```
┌─────────────────────┐
│  Crear Feature      │
│  Branch             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Implementar        │
│  Código             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Usar Constantes    │
│  (no hardcode)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Agregar Decoradores│
│  (si aplica)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Verificar con      │
│  Scripts            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Tests              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Code Review        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Merge              │
└─────────────────────┘
```

### Checklist de Desarrollo

**Antes de Commit:**
- [ ] ¿Usé constantes en lugar de strings hardcodeados?
- [ ] ¿Agregué decoradores apropiados a funciones?
- [ ] ¿Los type hints están completos?
- [ ] ¿Los logs incluyen `exc_info=True`?
- [ ] ¿Ejecuté los scripts de verificación?
- [ ] ¿Los tests pasan?

---

## 🧪 Workflow de Testing

### Estrategia de Tests

**1. Tests Unitarios**
```python
# tests/unit/test_constants.py
def test_git_config_default_branch():
    from core.constants import GitConfig
    assert GitConfig.DEFAULT_BASE_BRANCH == "main"

# tests/unit/test_decorators.py
@pytest.mark.asyncio
async def test_handle_github_exception_async():
    @handle_github_exception
    async def test_func():
        raise ValueError("Test")
    
    with pytest.raises(ValueError):
        await test_func()
```

**2. Tests de Integración**
```python
# tests/integration/test_decorator_integration.py
@pytest.mark.asyncio
async def test_decorator_with_github_client():
    @handle_github_exception
    async def fetch_repo():
        client = GitHubClient()
        return await client.get_repo("owner/repo")
    
    # Test real con GitHub API
```

**3. Tests de Regresión**
```python
# tests/regression/test_v8_migration.py
def test_no_hardcoded_strings():
    """Verificar que no hay strings hardcodeados"""
    import re
    from pathlib import Path
    
    for file in Path('core').rglob('*.py'):
        content = file.read_text()
        assert '"main"' not in content, f"Hardcoded 'main' en {file}"
```

---

## 📋 Workflow de Code Review

### Checklist para Reviewers

**Constantes:**
- [ ] ¿Se usan constantes en lugar de strings?
- [ ] ¿Las constantes están importadas correctamente?
- [ ] ¿No hay strings hardcodeados equivalentes?

**Decoradores:**
- [ ] ¿Las funciones async tienen decoradores compatibles?
- [ ] ¿Los decoradores incluyen logging con `exc_info=True`?
- [ ] ¿El orden de decoradores es correcto?

**Manejo de Errores:**
- [ ] ¿Los errores se manejan apropiadamente?
- [ ] ¿Los mensajes de error usan constantes?
- [ ] ¿Los logs incluyen stack traces?

**Type Hints:**
- [ ] ¿Los type hints están completos?
- [ ] ¿Pasa mypy sin errores?

### Comentarios de Review

**Template para comentarios:**

```
❌ Problema encontrado:
- Ubicación: archivo.py:123
- Problema: String hardcodeado "main"
- Solución: Usar GitConfig.DEFAULT_BASE_BRANCH

✅ Buen trabajo:
- Uso correcto de constantes
- Decorador aplicado apropiadamente
```

---

## 🔍 Workflow de Debugging

### Proceso de Debugging

**1. Identificar el Error**
```bash
# Ver logs con stack traces
tail -f logs/app.log | grep ERROR
```

**2. Analizar Stack Trace**
```
ERROR: Error en fetch_repo: Repository not found
Traceback (most recent call last):
  File "core/github_client.py", line 45, in fetch_repo
    repo = await client.get_repo(repo_name)
  ...
```

**3. Verificar Constantes**
```python
# Verificar valor de constante
from core.constants import GitConfig
print(GitConfig.DEFAULT_BASE_BRANCH)
```

**4. Verificar Decoradores**
```python
# Verificar que decorador está aplicado
import inspect
print(inspect.getsource(func))
```

**5. Reproducir y Fix**
```python
# Reproducir error
# Aplicar fix
# Verificar con tests
```

---

## 🚀 Workflow de Deployment

### Pre-Deployment Checklist

- [ ] Todos los strings hardcodeados migrados
- [ ] Decoradores aplicados correctamente
- [ ] Tests pasan (100% cobertura de nuevas features)
- [ ] Linting pasa
- [ ] Type checking pasa
- [ ] Documentación actualizada
- [ ] Changelog actualizado

### Deployment Steps

**1. Pre-deployment**
```bash
# Verificar código
make check-constants
make lint
make type-check
make test

# Verificar migraciones
python scripts/verify-constants-usage.py
```

**2. Build**
```bash
# Build Docker image
docker build -t github-autonomous-agent:v8 .

# Test image
docker run --rm github-autonomous-agent:v8 python -m pytest
```

**3. Deploy**
```bash
# Deploy a staging
kubectl apply -f k8s/staging/

# Verificar
kubectl logs -f deployment/github-autonomous-agent
```

**4. Post-deployment**
```bash
# Health check
curl https://api.example.com/health

# Verificar logs
kubectl logs -f deployment/github-autonomous-agent | grep ERROR
```

---

## 📊 Workflow de Monitoreo

### Métricas a Monitorear

**1. Errores**
- Tasa de errores por función
- Tipos de errores más comunes
- Stack traces en logs

**2. Performance**
- Tiempo de ejecución de funciones decoradas
- Overhead de decoradores
- Tiempo de respuesta de API

**3. Uso de Constantes**
- Número de constantes usadas
- Strings hardcodeados restantes (debería ser 0)

### Dashboard Recomendado

```python
# Métricas Prometheus
github_agent_errors_total{function="fetch_repo", error_type="ValueError"}
github_agent_execution_seconds{function="fetch_repo"}
github_agent_constants_usage{constant="GitConfig.DEFAULT_BASE_BRANCH"}
```

---

## 🔄 Workflow de Migración V7 → V8

### Paso a Paso

**Fase 1: Preparación (1 día)**
1. Backup del código
2. Crear branch `v8-migration`
3. Ejecutar scripts de análisis
4. Crear plan de migración

**Fase 2: Migración Automática (1 día)**
1. Ejecutar `migrate-to-constants.py --dry-run`
2. Revisar cambios propuestos
3. Ejecutar `migrate-to-constants.py`
4. Ejecutar `auto-refactor-v8.py`

**Fase 3: Migración Manual (2-3 días)**
1. Revisar cada archivo modificado
2. Ajustar imports
3. Verificar lógica
4. Agregar tests

**Fase 4: Testing (1 día)**
1. Ejecutar suite completa de tests
2. Tests de integración
3. Tests de regresión
4. Code review

**Fase 5: Deployment (1 día)**
1. Deploy a staging
2. Verificar funcionamiento
3. Deploy a producción
4. Monitoreo

**Total estimado: 5-7 días**

---

## 🎯 Best Practices por Tipo de Código

### Funciones Async

**✅ Hacer:**
```python
@handle_github_exception
async def my_async_function():
    # código limpio
    pass
```

**❌ Evitar:**
```python
async def my_async_function():
    try:
        # código
    except Exception as e:
        logger.error(f"Error: {e}")  # Sin exc_info
        raise
```

### Funciones Sync

**✅ Hacer:**
```python
@handle_github_exception
def my_sync_function():
    # código limpio
    pass
```

**❌ Evitar:**
```python
def my_sync_function():
    try:
        # código
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
```

### Validaciones

**✅ Hacer:**
```python
from core.constants import ErrorMessages

if not token:
    raise HTTPException(
        status_code=400,
        detail=ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED
    )
```

**❌ Evitar:**
```python
if not token:
    raise HTTPException(
        status_code=400,
        detail="GitHub token no configurado..."
    )
```

### Valores por Defecto

**✅ Hacer:**
```python
from core.constants import GitConfig

def create_branch(name: str, base: str = GitConfig.DEFAULT_BASE_BRANCH):
    pass
```

**❌ Evitar:**
```python
def create_branch(name: str, base: str = "main"):
    pass
```

---

## 📝 Templates de Código

### Template: Nueva Función con Decorador

```python
from core.utils import handle_github_exception
from core.constants import GitConfig, ErrorMessages
from typing import Optional

@handle_github_exception
async def new_function(
    param1: str,
    param2: Optional[str] = None,
    branch: str = GitConfig.DEFAULT_BASE_BRANCH
) -> dict:
    """
    Descripción de la función.
    
    Args:
        param1: Descripción
        param2: Descripción opcional
        branch: Rama base (default: GitConfig.DEFAULT_BASE_BRANCH)
    
    Returns:
        Diccionario con resultado
    
    Raises:
        ValueError: Si param1 es inválido
    """
    # Validación
    if not param1:
        raise ValueError(ErrorMessages.INVALID_PARAMETER)
    
    # Lógica
    # ...
    
    return {"success": True}
```

### Template: Endpoint con Validación

```python
from fastapi import APIRouter, HTTPException, Depends
from api.utils import handle_api_errors, validate_github_token
from core.constants import ErrorMessages
from api.schemas import MySchema

router = APIRouter()

@router.post("/endpoint")
@handle_api_errors
async def create_endpoint(
    data: MySchema,
    _: None = Depends(validate_github_token)
):
    """Crear endpoint con validación"""
    # Validación
    if not data.field:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.FIELD_REQUIRED
        )
    
    # Procesar
    # ...
    
    return {"success": True}
```

---

## 🔗 Integración con CI/CD

### GitHub Actions Workflow

```yaml
name: V8 Quality Checks

on: [push, pull_request]

jobs:
  check-constants:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: |
          pip install -r requirements-dev.txt
          python scripts/verify-constants-usage.py
  
  check-decorators:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: |
          pip install -r requirements-dev.txt
          python scripts/analyze-decorator-usage.py
  
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: |
          pip install -r requirements-dev.txt
          pytest tests/ -v --cov=core --cov=api
```

---

## 📚 Recursos Adicionales

### Documentación Interna
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Scripts Disponibles
- `scripts/find-hardcoded-strings.py`
- `scripts/migrate-to-constants.py`
- `scripts/verify-constants-usage.py`
- `scripts/analyze-decorator-usage.py`
- `scripts/generate-decorator-tests.py`
- `scripts/auto-refactor-v8.py`
- `scripts/generate-constants-docs.py`

---

**Última actualización**: [Fecha]



