# 🤝 Guía de Contribución - Blatam Academy Features

## 📋 Tabla de Contenidos

1. [Cómo Contribuir](#cómo-contribuir)
2. [Estándares de Código](#estándares)
3. [Testing](#testing)
4. [Documentación](#documentación)
5. [Pull Requests](#pull-requests)

## 🚀 Cómo Contribuir

### Proceso de Contribución

1. **Fork el repositorio**
2. **Crear branch de feature**
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```

3. **Hacer cambios**
4. **Agregar tests**
5. **Actualizar documentación**
6. **Crear Pull Request**

### Tipos de Contribuciones

- 🐛 **Bug Fixes**: Corregir errores existentes
- ✨ **Nuevas Features**: Agregar funcionalidad nueva
- 📚 **Documentación**: Mejorar documentación
- ⚡ **Optimización**: Mejorar rendimiento
- 🔧 **Refactoring**: Mejorar código existente

## 📝 Estándares de Código

### Python Style Guide

Seguir **PEP 8** con estas extensiones:

```python
# ✅ Bueno
async def process_request(
    request: Dict[str, Any],
    session_id: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process request with caching.
    
    Args:
        request: Request dictionary
        session_id: Optional session ID
    
    Returns:
        Tuple of (result, metadata)
    """
    ...

# ❌ Evitar
def process_request(request, session_id=None):
    ...
```

### Type Hints

**Siempre usar type hints:**

```python
from typing import Dict, List, Optional, Tuple, Any

def function(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    ...
```

### Docstrings

**Usar Google-style docstrings:**

```python
def function(param: str) -> str:
    """Short description.
    
    Longer description if needed.
    
    Args:
        param: Parameter description
    
    Returns:
        Return value description
    
    Raises:
        ValueError: When param is invalid
    """
    ...
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Usar niveles apropiados
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)
```

## 🧪 Testing

### Estructura de Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestKVCache:
    """Test suite for KV Cache."""
    
    @pytest.fixture
    def engine(self):
        """Create test engine."""
        return create_test_engine()
    
    @pytest.mark.asyncio
    async def test_process_request(self, engine):
        """Test request processing."""
        result = await engine.process_request(test_request)
        assert result is not None
        assert 'response' in result
```

### Cobertura de Tests

- Objetivo: >80% cobertura
- Tests unitarios para todas las funciones
- Tests de integración para flujos completos
- Tests de carga para rendimiento

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_kv_cache.py

# Con cobertura
pytest --cov=bulk --cov-report=html

# Tests de carga
pytest tests/load/ -v
```

## 📚 Documentación

### Actualizar READMEs

Cuando agregues nueva funcionalidad:

1. **Actualizar README principal** si es feature principal
2. **Actualizar README del módulo** correspondiente
3. **Agregar ejemplos** en EXAMPLES.md
4. **Actualizar DOCUMENTATION_INDEX.md**

### Formato de Documentación

- Usar Markdown
- Incluir ejemplos de código
- Agregar diagramas cuando sea útil
- Mantener índice actualizado

## 🔄 Pull Requests

### Título del PR

- Usar formato: `[Tipo] Descripción breve`
- Tipos: `Feature`, `Fix`, `Docs`, `Refactor`, `Performance`

Ejemplos:
- `[Feature] Add multi-tenant support to KV Cache`
- `[Fix] Resolve memory leak in cache engine`
- `[Docs] Update API reference documentation`

### Descripción del PR

```markdown
## Descripción
Breve descripción de los cambios

## Tipo de Cambio
- [ ] Bug fix
- [ ] Nueva feature
- [ ] Breaking change
- [ ] Documentación

## Testing
- [ ] Tests unitarios agregados
- [ ] Tests de integración pasan
- [ ] Tests manuales realizados

## Checklist
- [ ] Código sigue estándares
- [ ] Tests agregados/actualizados
- [ ] Documentación actualizada
- [ ] No hay warnings
- [ ] CHANGELOG actualizado
```

### Review Process

1. PR será revisado por maintainers
2. Los comentarios deben ser atendidos
3. Una vez aprobado, se mergea

## 🎯 Áreas Prioritarias para Contribuciones

### Alta Prioridad
- 🔧 Optimizaciones de rendimiento
- 🐛 Bug fixes críticos
- 📚 Mejoras de documentación
- 🧪 Aumentar cobertura de tests

### Media Prioridad
- ✨ Nuevas features menores
- 🔄 Refactoring
- 📊 Mejoras de monitoring
- 🔒 Mejoras de seguridad

### Baja Prioridad
- 🎨 Mejoras de UI/UX
- 📝 Correcciones menores de documentación
- 🔧 Herramientas de desarrollo

## 📋 Checklist Antes de Enviar PR

- [ ] Código sigue PEP 8
- [ ] Type hints agregados
- [ ] Docstrings completos
- [ ] Tests agregados y pasan
- [ ] Documentación actualizada
- [ ] No hay linter errors
- [ ] No hay warnings
- [ ] Commits descriptivos
- [ ] Branch actualizado con main

## 🔍 Code Review Guidelines

### Para Reviewers

- Ser constructivo y respetuoso
- Explicar razones de cambios solicitados
- Sugerir mejoras, no solo criticar
- Aprobar cuando esté bien

### Para Contributors

- Responder a todos los comentarios
- Hacer cambios solicitados
- Preguntar si algo no está claro
- Agradecer feedback

## 📖 Recursos Adicionales

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

## 💬 Comunicación

- **Issues**: Para reportar bugs o sugerir features
- **Discussions**: Para preguntas y discusiones
- **Pull Requests**: Para código y cambios

---

**¡Gracias por contribuir!** 🎉



