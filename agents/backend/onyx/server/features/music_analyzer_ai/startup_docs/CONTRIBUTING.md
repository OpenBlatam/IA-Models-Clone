# 🤝 Guía de Contribución - Music Analyzer AI

¡Gracias por tu interés en contribuir a Music Analyzer AI! Esta guía te ayudará a empezar.

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [Cómo Contribuir](#cómo-contribuir)
- [Proceso de Desarrollo](#proceso-de-desarrollo)
- [Estándares de Código](#estándares-de-código)
- [Testing](#testing)
- [Documentación](#documentación)
- [Pull Requests](#pull-requests)

## 📜 Código de Conducta

### Nuestro Compromiso

- Ser respetuoso y inclusivo
- Aceptar críticas constructivas
- Enfocarse en lo mejor para la comunidad
- Mostrar empatía hacia otros miembros

## 🚀 Cómo Contribuir

### Reportar Bugs

Si encuentras un bug:

1. **Verifica que no esté reportado**: Busca en los issues existentes
2. **Crea un nuevo issue** con:
   - Título descriptivo
   - Descripción clara del problema
   - Pasos para reproducir
   - Comportamiento esperado vs actual
   - Versión del sistema
   - Logs relevantes (sin información sensible)

### Sugerir Features

Para sugerir una nueva feature:

1. **Verifica que no esté sugerida**: Busca en los issues
2. **Crea un issue** con:
   - Descripción clara de la feature
   - Casos de uso
   - Beneficios
   - Posible implementación (si tienes ideas)

### Contribuir Código

1. **Fork el repositorio**
2. **Crea una rama** para tu feature/fix:
   ```bash
   git checkout -b feature/mi-nueva-feature
   # o
   git checkout -b fix/correccion-bug
   ```
3. **Haz tus cambios**
4. **Escribe tests**
5. **Asegúrate que todo pase**: `pytest`
6. **Commit tus cambios**: Usa mensajes descriptivos
7. **Push a tu fork**
8. **Crea un Pull Request**

## 🔧 Proceso de Desarrollo

### Setup del Entorno

```bash
# 1. Fork y clonar
git clone https://github.com/tu-usuario/music_analyzer_ai.git
cd music_analyzer_ai

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configurar .env
cp .env.example .env
# Editar .env con tus credenciales
```

### Estructura de Ramas

- `main` - Código de producción estable
- `develop` - Código de desarrollo
- `feature/*` - Nuevas features
- `fix/*` - Correcciones de bugs
- `docs/*` - Mejoras de documentación

### Flujo de Trabajo

```bash
# 1. Actualizar tu fork
git checkout main
git pull upstream main

# 2. Crear rama de feature
git checkout -b feature/mi-feature

# 3. Hacer cambios y commits
git add .
git commit -m "feat: agregar nueva feature X"

# 4. Push
git push origin feature/mi-feature

# 5. Crear Pull Request en GitHub
```

## 📝 Estándares de Código

### Convenciones de Nombres

- **Funciones**: `snake_case`
- **Clases**: `PascalCase`
- **Constantes**: `UPPER_SNAKE_CASE`
- **Variables**: `snake_case`

### Type Hints

✅ **Bien:**
```python
async def analyze_track(
    track_id: str,
    include_coaching: bool = False
) -> AnalysisResponse:
    pass
```

### Docstrings

✅ **Bien:**
```python
def analyze_track(track_id: str) -> Analysis:
    """
    Analiza una canción de Spotify.
    
    Args:
        track_id: ID de la canción en Spotify
        
    Returns:
        Analysis: Objeto con el análisis completo
        
    Raises:
        TrackNotFoundException: Si la canción no existe
    """
    pass
```

### Formato de Código

Usamos:
- **Black** para formateo
- **isort** para imports
- **flake8** para linting

```bash
# Formatear código
black .

# Ordenar imports
isort .

# Verificar linting
flake8 .
```

### Mensajes de Commit

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: agregar endpoint de recomendaciones
fix: corregir error en análisis de tempo
docs: actualizar README
refactor: mejorar estructura de use cases
test: agregar tests para análisis
chore: actualizar dependencias
```

## 🧪 Testing

### Escribir Tests

```python
# tests/test_analysis.py
import pytest
from application.use_cases.analysis import AnalyzeTrackUseCase

@pytest.mark.asyncio
async def test_analyze_track_success():
    # Arrange
    use_case = AnalyzeTrackUseCase(...)
    
    # Act
    result = await use_case.execute("track_id")
    
    # Assert
    assert result.track_id == "track_id"
    assert result.analysis is not None
```

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_analysis.py

# Con cobertura
pytest --cov=api --cov=application --cov-report=html

# Solo tests rápidos
pytest -m "not slow"
```

### Cobertura Mínima

- Nuevo código: **80%+ cobertura**
- Tests deben ser rápidos y determinísticos
- Usa mocks para dependencias externas

## 📚 Documentación

### Documentar Código

- Docstrings en todas las funciones públicas
- Type hints en todas las funciones
- Comentarios para lógica compleja

### Documentar Features

Si agregas una nueva feature:

1. Actualiza `README.md` si es necesario
2. Agrega ejemplos en `startup_docs/EXAMPLES.md`
3. Actualiza `CHANGELOG.md`
4. Documenta en la docstring del código

## 🔀 Pull Requests

### Antes de Crear un PR

- [ ] Código sigue los estándares
- [ ] Tests pasan (`pytest`)
- [ ] Cobertura de tests adecuada
- [ ] Documentación actualizada
- [ ] CHANGELOG actualizado
- [ ] Sin conflictos con `main`

### Template de PR

```markdown
## Descripción
Breve descripción de los cambios

## Tipo de Cambio
- [ ] Bug fix
- [ ] Nueva feature
- [ ] Breaking change
- [ ] Documentación

## Testing
Cómo se probaron los cambios

## Checklist
- [ ] Tests agregados/actualizados
- [ ] Documentación actualizada
- [ ] Código sigue estándares
- [ ] Sin warnings de linting
```

### Revisión de Código

- Los PRs serán revisados por maintainers
- Puede haber solicitudes de cambios
- Responde a comentarios de manera constructiva

## 🎯 Áreas donde Necesitamos Ayuda

### Prioridad Alta

- 🐛 Corrección de bugs
- 📚 Mejora de documentación
- 🧪 Aumentar cobertura de tests
- ⚡ Optimizaciones de rendimiento

### Prioridad Media

- 🎨 Mejoras de UI/UX (si hay frontend)
- 🔧 Nuevas features
- 🌐 Traducciones
- 📊 Mejoras de analytics

### Prioridad Baja

- 🎨 Mejoras estéticas
- 📝 Mejoras menores de código
- 🔍 Refactorizaciones menores

## 📞 Contacto

- **Issues**: Para bugs y features
- **Discussions**: Para preguntas y discusiones
- **Email**: (si está disponible)

## 🙏 Reconocimientos

Todas las contribuciones son valiosas. Los contribuidores serán reconocidos en:
- README.md
- CHANGELOG.md
- Releases

---

**Gracias por contribuir a Music Analyzer AI!** 🎵

---

**Última actualización**: 2025  
**Versión**: 2.21.0






