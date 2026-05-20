# Guía de Contribución - Mejoras V8

## Cómo Contribuir a las Mejoras V8

---

## 🎯 Tipos de Contribuciones

### 1. Reportar Bugs

**Proceso:**
1. Verificar que no existe un issue similar
2. Crear issue con:
   - Descripción clara del problema
   - Pasos para reproducir
   - Comportamiento esperado vs actual
   - Versión y entorno

**Template:**
```markdown
## Descripción
[Descripción clara del bug]

## Pasos para Reproducir
1. ...
2. ...
3. ...

## Comportamiento Esperado
[Qué debería pasar]

## Comportamiento Actual
[Qué pasa actualmente]

## Entorno
- Versión: V8.0.0
- Python: 3.10+
- OS: [tu OS]
```

---

### 2. Sugerir Mejoras

**Proceso:**
1. Crear issue con etiqueta "enhancement"
2. Describir la mejora propuesta
3. Explicar el beneficio
4. Proponer implementación (opcional)

**Template:**
```markdown
## Descripción
[Descripción de la mejora propuesta]

## Beneficio
[Por qué es útil]

## Implementación Propuesta
[Cómo se podría implementar]

## Alternativas Consideradas
[Otras opciones]
```

---

### 3. Contribuir Código

**Proceso:**
1. Fork del repositorio
2. Crear branch para tu feature
3. Implementar cambios siguiendo estándares V8
4. Agregar tests
5. Actualizar documentación
6. Crear PR

---

## 🔧 Estándares de Código

### Constantes

**✅ Requisitos:**
- Usar constantes en lugar de strings hardcodeados
- Agregar nuevas constantes a `core/constants.py`
- Seguir convenciones de nombres (UPPER_CASE)
- Documentar constantes nuevas

**Ejemplo:**
```python
# core/constants.py
class NewConfig:
    """Nueva configuración"""
    NEW_CONSTANT = "value"  # ✅ Documentar propósito
```

---

### Decoradores

**✅ Requisitos:**
- Soportar funciones sync y async
- Incluir `exc_info=True` en logging
- Preservar type hints
- Documentar comportamiento

**Ejemplo:**
```python
def new_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """
    Nuevo decorador.
    
    Soporta funciones sync y async automáticamente.
    """
    # Implementación...
```

---

### Type Hints

**✅ Requisitos:**
- Type hints completos en todas las funciones
- Usar TypeVar para decoradores
- Type hints en parámetros y retornos

**Ejemplo:**
```python
from typing import Callable, TypeVar, Optional

T = TypeVar('T')

def new_function(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """Función con type hints completos"""
    pass
```

---

### Logging

**✅ Requisitos:**
- Usar `exc_info=True` en logs de error
- Incluir contexto estructurado
- No exponer información sensible

**Ejemplo:**
```python
logger.error(
    f"Error en {func.__name__}: {e}",
    exc_info=True,
    extra={
        "function": func.__name__,
        "error_type": type(e).__name__
    }
)
```

---

## 🧪 Testing

### Requisitos de Tests

**✅ Obligatorio:**
- Tests para nuevas constantes
- Tests para nuevos decoradores
- Tests de integración si aplica
- Cobertura mínima: 80%

**Ejemplo:**
```python
def test_new_constant():
    """Test nueva constante"""
    from core.constants import NewConfig
    assert NewConfig.NEW_CONSTANT == "value"
```

---

## 📚 Documentación

### Requisitos de Documentación

**✅ Obligatorio:**
- Actualizar documentación relevante
- Agregar ejemplos si es necesario
- Actualizar changelog si hay cambios significativos

**Archivos a actualizar:**
- `IMPROVEMENTS_V8.md` (si es relevante)
- `IMPROVEMENTS_V8_REAL_EXAMPLES.md` (si hay ejemplos)
- `IMPROVEMENTS_V8_CHANGELOG.md` (para cambios significativos)

---

## 🔄 Proceso de PR

### 1. Preparar PR

**Checklist:**
- [ ] Código sigue estándares V8
- [ ] Tests agregados y pasan
- [ ] Documentación actualizada
- [ ] No hay strings hardcodeados
- [ ] Type hints completos
- [ ] Logging mejorado

---

### 2. Crear PR

**Template:**
```markdown
## Descripción
[Descripción de los cambios]

## Tipo de Cambio
- [ ] Bug fix
- [ ] Nueva feature
- [ ] Mejora
- [ ] Documentación

## Cambios
- [Cambio 1]
- [Cambio 2]

## Tests
- [ ] Tests unitarios agregados
- [ ] Tests de integración agregados
- [ ] Todos los tests pasan

## Checklist
- [ ] Código sigue estándares V8
- [ ] Documentación actualizada
- [ ] Changelog actualizado (si aplica)
```

---

### 3. Code Review

**Proceso:**
1. Esperar review de al menos 1 maintainer
2. Responder a comentarios
3. Hacer cambios si es necesario
4. Esperar aprobación

**Criterios de Aprobación:**
- ✅ Código sigue estándares
- ✅ Tests pasan
- ✅ Documentación actualizada
- ✅ No hay regresiones

---

## 📋 Checklist de Contribución

### Antes de Empezar

- [ ] Leer documentación completa
- [ ] Verificar que no existe trabajo similar
- [ ] Crear issue si es necesario

### Durante Desarrollo

- [ ] Seguir estándares V8
- [ ] Escribir tests
- [ ] Actualizar documentación
- [ ] Commit frecuente con mensajes claros

### Antes de PR

- [ ] Todos los tests pasan
- [ ] Linting pasa
- [ ] Type checking pasa
- [ ] Documentación actualizada
- [ ] Changelog actualizado (si aplica)

---

## 🎓 Recursos para Contribuidores

### Documentación

- [IMPROVEMENTS_V8.md](IMPROVEMENTS_V8.md) - Documentación completa
- [IMPROVEMENTS_V8_BEST_PRACTICES.md](IMPROVEMENTS_V8_BEST_PRACTICES.md) - Mejores prácticas
- [IMPROVEMENTS_V8_REAL_EXAMPLES.md](IMPROVEMENTS_V8_REAL_EXAMPLES.md) - Ejemplos

### Scripts Útiles

```bash
# Verificar constantes
python scripts/verify-constants-usage.py

# Verificar decoradores
python scripts/analyze-decorator-usage.py

# Ejecutar tests
pytest tests/ -v

# Type checking
mypy core/ api/
```

---

## 🤝 Código de Conducta

### Principios

- **Respeto**: Tratar a todos con respeto
- **Colaboración**: Trabajar juntos constructivamente
- **Apertura**: Aceptar feedback de forma positiva
- **Profesionalismo**: Mantener estándares profesionales

---

## 📞 Contacto

### Preguntas

- **Documentación**: Revisar [IMPROVEMENTS_V8_FAQ.md](IMPROVEMENTS_V8_FAQ.md)
- **Issues**: Crear issue en GitHub
- **Discusiones**: Usar GitHub Discussions

---

## 🏆 Reconocimiento

### Contribuidores

Los contribuidores serán reconocidos en:
- Changelog
- README (si aplica)
- Release notes

---

**Última actualización**: Enero 2025  
**Versión**: V8  
**Mantenido por**: Equipo de Desarrollo



