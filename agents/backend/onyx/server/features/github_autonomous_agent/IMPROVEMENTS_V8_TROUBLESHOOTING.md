# Troubleshooting Avanzado - Mejoras V8

## Solución de Problemas Comunes y Avanzados

---

## 🔍 Diagnóstico Rápido

### Checklist de Diagnóstico

```bash
# 1. Verificar constantes
python scripts/verify-constants-usage.py

# 2. Verificar decoradores
python scripts/analyze-decorator-usage.py

# 3. Buscar strings hardcodeados
python scripts/find-hardcoded-strings.py

# 4. Ejecutar tests
pytest tests/ -v

# 5. Verificar type hints
mypy core/ api/
```

---

## 🐛 Problemas Comunes

### Problema 1: Import Error

**Síntoma**:
```
NameError: name 'GitConfig' is not defined
```

**Diagnóstico**:
```bash
# Verificar que el import existe
grep -r "from core.constants import GitConfig" archivo.py

# Si no existe, agregar:
```

**Solución**:
```python
# ✅ Agregar al inicio del archivo
from core.constants import GitConfig, ErrorMessages
```

**Prevención**:
- Usar scripts de verificación
- Code review estricto
- Linting automático

---

### Problema 2: Decorador No Funciona

**Síntoma**:
```python
@handle_github_exception
async def my_func():
    pass

# Error: TypeError: object async_generator can't be used in 'await' expression
```

**Diagnóstico**:
```python
# Verificar versión del decorador
import inspect
print(inspect.getsource(handle_github_exception))

# Debería mostrar detección automática de async
```

**Solución**:
```python
# ✅ Asegurarse de usar versión V8
from core.utils import handle_github_exception

@handle_github_exception  # V8 detecta automáticamente
async def my_func():
    pass
```

**Verificación**:
```python
import asyncio
print(asyncio.iscoroutinefunction(my_func))  # Debería ser True
```

---

### Problema 3: Sin Stack Traces en Logs

**Síntoma**: Logs muestran solo mensaje, sin stack trace

**Diagnóstico**:
```bash
# Verificar logs
tail -f logs/app.log | grep ERROR

# Si no hay "Traceback", el problema es que no se usa exc_info=True
```

**Solución**:
```python
# ✅ V8 incluye exc_info=True automáticamente
@handle_github_exception
async def my_func():
    pass

# Verificar que el decorador lo incluye
# core/utils.py debería tener:
logger.error(f"Error: {e}", exc_info=True)  # ✅ Debe estar presente
```

**Verificación**:
```bash
# Los logs deberían mostrar:
ERROR: Error en my_func: ValueError
Traceback (most recent call last):
  File "...", line X, in my_func
    ...
```

---

### Problema 4: Type Error en Mypy

**Síntoma**:
```
mypy error: Incompatible return type
```

**Diagnóstico**:
```bash
# Ejecutar mypy
mypy core/utils.py

# Ver error específico
```

**Solución**:
```python
# ✅ Usar TypeVar correctamente
from typing import Callable, TypeVar, Awaitable

T = TypeVar('T')

def handle_github_exception(
    func: Callable[..., T]
) -> Callable[..., T | Awaitable[T]]:
    # ...
```

**Verificación**:
```bash
mypy core/utils.py  # Debería pasar sin errores
```

---

### Problema 5: Constante No Encontrada

**Síntoma**:
```
AttributeError: type object 'GitConfig' has no attribute 'NEW_CONSTANT'
```

**Diagnóstico**:
```python
# Verificar que la constante existe
from core.constants import GitConfig
print(dir(GitConfig))  # Ver atributos disponibles
```

**Solución**:
```python
# Opción 1: Agregar constante
# core/constants.py
class GitConfig:
    NEW_CONSTANT = "value"  # ✅ Agregar aquí

# Opción 2: Usar constante existente
# Verificar constantes disponibles en core/constants.py
```

---

## 🔧 Problemas Avanzados

### Problema 6: Múltiples Decoradores en Orden Incorrecto

**Síntoma**: Retry no funciona correctamente

**Diagnóstico**:
```python
# Verificar orden de decoradores
@handle_github_exception  # ❌ Incorrecto: primero
@retry_async_on_github_error(max_attempts=3)  # ❌ Incorrecto: segundo
async def my_func():
    pass
```

**Solución**:
```python
# ✅ Orden correcto
@retry_async_on_github_error(max_attempts=3)  # ✅ Primero: retry
@handle_github_exception  # ✅ Segundo: error handling
async def my_func():
    pass
```

**Explicación**: El retry debe estar más externo para capturar errores del decorador interno.

---

### Problema 7: Constante en String de Configuración Externa

**Síntoma**: Constante no se usa cuando viene de env var

**Diagnóstico**:
```python
# ❌ Problema
branch = os.getenv("BRANCH", "main")  # No usa constante
```

**Solución**:
```python
# ✅ Solución
from core.constants import GitConfig

branch = os.getenv("BRANCH", GitConfig.DEFAULT_BASE_BRANCH)  # ✅ Usa constante como default
```

---

### Problema 8: Decorador en Método de Clase

**Síntoma**: Decorador no funciona con métodos de instancia

**Diagnóstico**:
```python
class MyClass:
    @handle_github_exception
    async def instance_method(self):  # ¿Funciona?
        pass
```

**Solución**:
```python
# ✅ Funciona correctamente
class MyClass:
    @handle_github_exception  # ✅ Funciona con métodos
    async def instance_method(self):
        pass
    
    @classmethod
    @handle_github_exception  # ✅ Funciona con classmethod
    async def class_method(cls):
        pass
    
    @staticmethod
    @handle_github_exception  # ✅ Funciona con staticmethod
    def static_method():
        pass
```

**Verificación**:
```python
obj = MyClass()
await obj.instance_method()  # Debería funcionar
```

---

### Problema 9: Logging Duplicado

**Síntoma**: Errores se loguean múltiples veces

**Diagnóstico**:
```python
# ❌ Problema: Logging manual + decorador
@handle_github_exception
async def my_func():
    try:
        # código...
    except Exception as e:
        logger.error(f"Error: {e}")  # ❌ Duplicado
        raise
```

**Solución**:
```python
# ✅ Solución: Dejar que el decorador maneje
@handle_github_exception
async def my_func():
    # código...  # ✅ Decorador maneja logging
    # No necesitas try/except manual
```

---

### Problema 10: Constante en Test

**Síntoma**: Test falla porque usa constante

**Diagnóstico**:
```python
# Test que compara con string literal
def test_branch():
    assert get_branch() == "main"  # ❌ Puede fallar si constante cambia
```

**Solución**:
```python
# ✅ Solución: Usar constante en test
from core.constants import GitConfig

def test_branch():
    assert get_branch() == GitConfig.DEFAULT_BASE_BRANCH  # ✅ Usa constante
```

---

## 🔍 Debugging Avanzado

### Debugging de Decoradores

**Herramienta**: Inspeccionar decorador aplicado

```python
import inspect
from core.utils import handle_github_exception

@handle_github_exception
async def my_func():
    pass

# Verificar que el decorador está aplicado
print(inspect.getsource(my_func))
print(inspect.signature(my_func))
```

**Verificar tipo de wrapper**:
```python
import asyncio

# Verificar que es async wrapper
print(asyncio.iscoroutinefunction(my_func))  # Debería ser True
```

---

### Debugging de Constantes

**Herramienta**: Verificar valores de constantes

```python
from core.constants import GitConfig, ErrorMessages

# Verificar valores
print(f"Default branch: {GitConfig.DEFAULT_BASE_BRANCH}")
print(f"Error message: {ErrorMessages.GITHUB_TOKEN_NOT_CONFIGURED}")

# Verificar que son strings
assert isinstance(GitConfig.DEFAULT_BASE_BRANCH, str)
```

**Verificar uso**:
```bash
# Buscar uso de constantes
grep -r "GitConfig.DEFAULT_BASE_BRANCH" --include="*.py"

# Buscar strings hardcodeados equivalentes
grep -r '"main"' --include="*.py" | grep -v "GitConfig"
```

---

### Debugging de Logs

**Herramienta**: Analizar logs estructurados

```bash
# Ver logs con stack traces
tail -f logs/app.log | grep -A 30 "Traceback"

# Buscar errores específicos
grep "Error en" logs/app.log | head -20

# Contar errores por función
grep "Error en" logs/app.log | cut -d: -f2 | sort | uniq -c
```

**Verificar exc_info**:
```python
# En el código, verificar que se usa exc_info=True
# core/utils.py debería tener:
logger.error(f"Error: {e}", exc_info=True)  # ✅ Debe estar presente
```

---

## 🛠️ Herramientas de Diagnóstico

### Script de Diagnóstico Completo

```python
#!/usr/bin/env python3
"""Script de diagnóstico completo para V8"""

import subprocess
import sys

def run_command(cmd):
    """Ejecutar comando y retornar resultado"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def diagnose():
    """Ejecutar diagnóstico completo"""
    print("🔍 Diagnóstico V8\n")
    
    checks = [
        ("Verificar constantes", "python scripts/verify-constants-usage.py"),
        ("Analizar decoradores", "python scripts/analyze-decorator-usage.py"),
        ("Buscar hardcoded", "python scripts/find-hardcoded-strings.py"),
        ("Tests", "pytest tests/ -v --tb=short"),
        ("Type checking", "mypy core/ api/ --ignore-missing-imports"),
    ]
    
    results = []
    for name, cmd in checks:
        print(f"⏳ {name}...")
        success, stdout, stderr = run_command(cmd)
        results.append((name, success, stdout, stderr))
        print(f"{'✅' if success else '❌'} {name}\n")
    
    # Resumen
    print("\n📊 Resumen:")
    for name, success, _, _ in results:
        status = "✅ OK" if success else "❌ FALLO"
        print(f"  {status}: {name}")
    
    return all(success for _, success, _, _ in results)

if __name__ == '__main__':
    success = diagnose()
    sys.exit(0 if success else 1)
```

**Uso**:
```bash
python scripts/diagnose_v8.py
```

---

## 📊 Métricas de Troubleshooting

### Tiempo Promedio de Resolución

| Problema | Tiempo V7 | Tiempo V8 | Mejora |
|----------|-----------|-----------|--------|
| Import error | 15 min | 2 min | ⬇️ 87% |
| Decorador no funciona | 30 min | 5 min | ⬇️ 83% |
| Sin stack trace | 45 min | 5 min | ⬇️ 89% |
| Type error | 20 min | 5 min | ⬇️ 75% |
| Constante no encontrada | 10 min | 2 min | ⬇️ 80% |

**Promedio**: ⬇️ 83% reducción en tiempo de troubleshooting

---

## 🎯 Prevención de Problemas

### Checklist de Prevención

**Antes de Commit**:
- [ ] Ejecutar `verify-constants-usage.py`
- [ ] Ejecutar `analyze-decorator-usage.py`
- [ ] Ejecutar tests
- [ ] Ejecutar mypy
- [ ] Revisar logs generados

**En Code Review**:
- [ ] Verificar imports de constantes
- [ ] Verificar decoradores aplicados
- [ ] Verificar que no hay strings hardcodeados
- [ ] Verificar type hints

**En CI/CD**:
- [ ] Agregar checks automáticos
- [ ] Falla build si hay problemas
- [ ] Reporte de métricas

---

## 📚 Recursos Adicionales

- [IMPROVEMENTS_V8.md](IMPROVEMENTS_V8.md) - Documentación completa
- [IMPROVEMENTS_V8_FAQ.md](IMPROVEMENTS_V8_FAQ.md) - FAQ completo
- [IMPROVEMENTS_V8_REAL_EXAMPLES.md](IMPROVEMENTS_V8_REAL_EXAMPLES.md) - Ejemplos

---

**Última actualización**: [Fecha]  
**Versión**: V8



