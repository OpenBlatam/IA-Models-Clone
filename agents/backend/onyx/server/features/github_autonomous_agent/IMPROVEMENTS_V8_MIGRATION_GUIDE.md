# Guía de Migración Paso a Paso - V7 a V8

## Migración Completa de Código Legacy

---

## 📋 Pre-requisitos

### Antes de Empezar

- [ ] Backup del código actual
- [ ] Branch de migración creado (`v8-migration`)
- [ ] Scripts de automatización descargados
- [ ] Documentación leída
- [ ] Equipo informado

### Herramientas Necesarias

- Python 3.10+
- Git
- Editor de código (VS Code recomendado)
- Scripts de automatización (ver IMPROVEMENTS_V8_SCRIPTS.md)

---

## 🗺️ Mapa de Migración

```
┌─────────────────────────────────────────┐
│  Fase 1: Análisis y Preparación         │
│  ─────────────────────────────────────   │
│  • Identificar strings hardcodeados      │
│  • Mapear constantes necesarias         │
│  • Crear plan de migración               │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Fase 2: Migración Automática          │
│  ─────────────────────────────────────   │
│  • Ejecutar scripts de migración        │
│  • Revisar cambios propuestos           │
│  • Aplicar migraciones                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Fase 3: Migración Manual               │
│  ─────────────────────────────────────   │
│  • Revisar cada archivo                 │
│  • Ajustar imports                      │
│  • Verificar lógica                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Fase 4: Agregar Decoradores            │
│  ─────────────────────────────────────   │
│  • Identificar funciones que necesitan   │
│  • Agregar decoradores apropiados       │
│  • Verificar funcionamiento             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Fase 5: Testing                        │
│  ─────────────────────────────────────   │
│  • Tests unitarios                      │
│  • Tests de integración                 │
│  • Tests de regresión                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Fase 6: Code Review y Merge            │
│  ─────────────────────────────────────   │
│  • Code review completo                 │
│  • Ajustes finales                     │
│  • Merge a main                        │
└─────────────────────────────────────────┘
```

---

## 📝 Fase 1: Análisis y Preparación

### Paso 1.1: Identificar Strings Hardcodeados

```bash
# Ejecutar script de búsqueda
python scripts/find-hardcoded-strings.py

# Output esperado:
# ⚠️  Se encontraron 18 strings hardcodeados:
# 
# 📄 core/utils.py:101
#    Patrón: "main"
#    Sugerencia: Usar GitConfig.DEFAULT_BASE_BRANCH
```

### Paso 1.2: Crear Inventario

Crear archivo `MIGRATION_INVENTORY.md`:

```markdown
# Inventario de Migración

## Strings Hardcodeados Encontrados

### "main" (8 instancias)
- core/utils.py:101
- core/utils.py:102
- core/utils.py:103
- core/task_processor.py:45
- ...

### "failed" (3 instancias)
- core/task_processor.py:120
- ...

### Mensajes de error (5 instancias)
- api/utils.py:25
- ...
```

### Paso 1.3: Verificar Constantes Existentes

```bash
# Verificar que core/constants.py existe
cat core/constants.py

# Verificar constantes disponibles
grep -r "class.*Config\|class.*Messages" core/constants.py
```

---

## 🔄 Fase 2: Migración Automática

### Paso 2.1: Dry Run (Ver Cambios Sin Aplicar)

```bash
# Ver qué cambios se harían
python scripts/migrate-to-constants.py --dry-run

# Output esperado:
# 🔍 Dry run: 18 reemplazos encontrados
# Ejecuta sin --dry-run para aplicar cambios
```

### Paso 2.2: Revisar Cambios Propuestos

```bash
# Guardar output para revisión
python scripts/migrate-to-constants.py --dry-run > migration-changes.txt

# Revisar cambios
cat migration-changes.txt
```

### Paso 2.3: Aplicar Migración Automática

```bash
# Aplicar cambios
python scripts/migrate-to-constants.py

# Output esperado:
# ✅ Migrado core/utils.py: 4 reemplazos
# ✅ Migrado core/task_processor.py: 3 reemplazos
# ...
# ✅ Migración completa: 18 reemplazos realizados
```

### Paso 2.4: Verificar Migración

```bash
# Verificar que no quedan strings hardcodeados
python scripts/verify-constants-usage.py

# Output esperado:
# ✅ Archivos correctos: 15
# ⚠️  Archivos con problemas: 0
```

---

## ✏️ Fase 3: Migración Manual

### Paso 3.1: Revisar Cada Archivo Modificado

```bash
# Ver archivos modificados
git status

# Revisar cambios
git diff core/utils.py
```

### Paso 3.2: Ajustar Imports

**Ejemplo: core/utils.py**

**Antes:**
```python
# Sin imports de constantes
def parse_instruction_params(instruction: str) -> dict:
    params = {
        "branch": "main",  # Hardcoded
    }
```

**Después:**
```python
# ✅ Agregar import
from core.constants import GitConfig

def parse_instruction_params(instruction: str) -> dict:
    params = {
        "branch": GitConfig.DEFAULT_BASE_BRANCH,  # Constante
    }
```

### Paso 3.3: Verificar Lógica

**Checklist por archivo:**

- [ ] ¿Los imports están correctos?
- [ ] ¿Las constantes se usan correctamente?
- [ ] ¿La lógica sigue funcionando igual?
- [ ] ¿No hay errores de sintaxis?

### Paso 3.4: Casos Especiales

#### Caso 1: Strings en Comentarios

```python
# ❌ No cambiar strings en comentarios
# Esta función usa "main" como rama por defecto

# ✅ Mantener comentarios como están
```

#### Caso 2: Strings en Tests

```python
# En tests, a veces necesitas strings literales
def test_default_branch():
    assert GitConfig.DEFAULT_BASE_BRANCH == "main"  # ✅ OK en tests
```

#### Caso 3: Strings en Configuración Externa

```python
# Si el string viene de configuración externa, no cambiar
config_value = os.getenv("BRANCH", "main")  # ✅ OK, viene de env
```

---

## 🎨 Fase 4: Agregar Decoradores

### Paso 4.1: Identificar Funciones que Necesitan Decoradores

```bash
# Analizar uso de decoradores
python scripts/analyze-decorator-usage.py

# Output esperado:
# ⚠️  Funciones que podrían necesitar decoradores: 5
#   core/github_client.py:45 - async fetch_repo()
#   ...
```

### Paso 4.2: Agregar Decoradores a Funciones Async

**Ejemplo: core/github_client.py**

**Antes:**
```python
async def get_repo(self, repo_name: str):
    try:
        repo = await self._github.get_repo(repo_name)
        return repo
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
```

**Después:**
```python
from core.utils import handle_github_exception

@handle_github_exception  # ✅ Decorador universal
async def get_repo(self, repo_name: str):
    repo = await self._github.get_repo(repo_name)
    return repo  # ✅ Código más limpio, decorador maneja errores
```

### Paso 4.3: Agregar Decoradores a Endpoints

**Ejemplo: api/routes/task_routes.py**

**Antes:**
```python
@router.post("/tasks")
async def create_task(request: CreateTaskRequest):
    try:
        # código...
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Después:**
```python
from api.utils import handle_api_errors

@router.post("/tasks")
@handle_api_errors  # ✅ Decorador universal
async def create_task(request: CreateTaskRequest):
    # código...  # ✅ Decorador maneja errores automáticamente
```

### Paso 4.4: Verificar Decoradores

```bash
# Verificar que decoradores están aplicados
grep -r "@handle_github_exception\|@handle_api_errors" --include="*.py"

# Verificar que funciones async tienen decoradores
python scripts/analyze-decorator-usage.py
```

---

## 🧪 Fase 5: Testing

### Paso 5.1: Tests Unitarios

```bash
# Ejecutar tests de constantes
pytest tests/unit/test_constants.py -v

# Output esperado:
# test_git_config_default_branch ... PASSED
# test_error_messages_exist ... PASSED
# ...
```

### Paso 5.2: Tests de Decoradores

```bash
# Ejecutar tests de decoradores
pytest tests/unit/test_decorators.py -v

# Output esperado:
# test_handle_github_exception_sync ... PASSED
# test_handle_github_exception_async ... PASSED
# ...
```

### Paso 5.3: Tests de Integración

```bash
# Ejecutar tests de integración
pytest tests/integration/ -v

# Verificar que no hay regresiones
```

### Paso 5.4: Tests de Regresión

```bash
# Verificar que no hay strings hardcodeados
python scripts/verify-constants-usage.py

# Verificar que decoradores están aplicados
python scripts/analyze-decorator-usage.py
```

---

## 👀 Fase 6: Code Review y Merge

### Paso 6.1: Preparar PR

```bash
# Commit cambios
git add .
git commit -m "feat: Migrate to V8 - Use constants and improved decorators"

# Push a branch
git push origin v8-migration
```

### Paso 6.2: Checklist de Code Review

**Para el Autor:**

- [ ] ¿Todos los strings hardcodeados fueron migrados?
- [ ] ¿Los imports están correctos?
- [ ] ¿Los decoradores están aplicados?
- [ ] ¿Los tests pasan?
- [ ] ¿La documentación está actualizada?

**Para el Reviewer:**

- [ ] ¿No hay strings hardcodeados restantes?
- [ ] ¿Las constantes se usan correctamente?
- [ ] ¿Los decoradores están en el orden correcto?
- [ ] ¿Los logs incluyen `exc_info=True`?
- [ ] ¿Los type hints están completos?

### Paso 6.3: Resolver Comentarios

**Ejemplo de comentario de review:**

```
❌ Problema encontrado:
- Ubicación: core/utils.py:105
- Problema: String hardcodeado "main"
- Solución: Usar GitConfig.DEFAULT_BASE_BRANCH

✅ Buen trabajo:
- Uso correcto de constantes en parse_instruction_params
- Decorador aplicado apropiadamente
```

### Paso 6.4: Merge

```bash
# Después de aprobación
git checkout main
git merge v8-migration
git push origin main
```

---

## 🔍 Troubleshooting

### Problema 1: Scripts No Funcionan

**Síntoma**: `python scripts/migrate-to-constants.py` falla

**Solución**:
```bash
# Verificar que estás en el directorio correcto
pwd  # Debería ser: /path/to/github_autonomous_agent

# Verificar que el script existe
ls scripts/migrate-to-constants.py

# Verificar Python
python --version  # Debería ser 3.10+
```

### Problema 2: Imports Faltantes

**Síntoma**: `NameError: name 'GitConfig' is not defined`

**Solución**:
```python
# Agregar import al inicio del archivo
from core.constants import GitConfig
```

### Problema 3: Decorador No Funciona

**Síntoma**: Decorador no captura errores

**Solución**:
```python
# Verificar que el decorador está antes de la función
@handle_github_exception  # ✅ Correcto
async def my_func():
    pass

# NO así:
async def my_func():  # ❌ Incorrecto
    @handle_github_exception
    pass
```

---

## 📊 Checklist Final

### Antes de Merge

- [ ] Todos los strings hardcodeados migrados
- [ ] Todos los imports agregados
- [ ] Decoradores aplicados correctamente
- [ ] Tests pasan (100%)
- [ ] Code review aprobado
- [ ] Documentación actualizada
- [ ] Changelog actualizado

### Post-Merge

- [ ] Verificar que no hay errores en producción
- [ ] Monitorear logs
- [ ] Verificar métricas
- [ ] Documentar lecciones aprendidas

---

## 🎯 Timeline Estimado

| Fase | Tiempo | Responsable |
|------|--------|-------------|
| Fase 1: Análisis | 4 horas | Dev |
| Fase 2: Migración Automática | 2 horas | Dev |
| Fase 3: Migración Manual | 8 horas | Dev |
| Fase 4: Decoradores | 4 horas | Dev |
| Fase 5: Testing | 4 horas | Dev |
| Fase 6: Code Review | 2 horas | Team |
| **TOTAL** | **24 horas (3 días)** | |

---

## 📚 Recursos

- [IMPROVEMENTS_V8.md](IMPROVEMENTS_V8.md) - Documentación completa
- [IMPROVEMENTS_V8_SCRIPTS.md](IMPROVEMENTS_V8_SCRIPTS.md) - Scripts de automatización
- [IMPROVEMENTS_V8_REAL_EXAMPLES.md](IMPROVEMENTS_V8_REAL_EXAMPLES.md) - Ejemplos reales

---

**Última actualización**: [Fecha]  
**Versión**: V8



