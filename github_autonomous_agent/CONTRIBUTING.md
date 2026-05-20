# 🤝 Guía de Contribución - GitHub Autonomous Agent

> Guía completa para contribuir al proyecto

¡Gracias por tu interés en contribuir! 🎉 Tu ayuda hace que este proyecto sea mejor para todos.

## 📋 Tabla de Contenidos

- [Cómo Contribuir](#-cómo-contribuir)
- [Convenciones de Código](#-convenciones-de-código)
- [Testing](#-testing)
- [Documentación](#-documentación)
- [Code Review](#-code-review)
- [Reportar Bugs](#-reportar-bugs)
- [Sugerir Funcionalidades](#-sugerir-funcionalidades)
- [Preguntas Frecuentes](#-preguntas-frecuentes)

---

## 🚀 Cómo Contribuir

### Paso 1: Fork y Clone

```bash
# 1. Fork el repositorio en GitHub (botón "Fork" en la esquina superior derecha)

# 2. Clona tu fork
git clone https://github.com/tu-usuario/github-autonomous-agent.git
cd github-autonomous-agent

# 3. Agrega el repositorio original como upstream
git remote add upstream https://github.com/original-owner/github-autonomous-agent.git
```

### Paso 2: Setup de Desarrollo

```bash
# Opción 1: Setup automático (recomendado)
./scripts/setup.sh --dev

# Opción 2: Manual
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# o
.\venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configurar .env
cp .env.example .env
# Editar .env con tus credenciales

# Validar setup
python scripts/validate-env.py
python scripts/check-dependencies.py
```

### Paso 3: Crear Branch

```bash
# Actualizar main
git checkout main
git pull upstream main

# Crear branch para tu feature/fix
git checkout -b feature/mi-nueva-funcionalidad
# o
git checkout -b fix/correccion-de-bug
# o
git checkout -b docs/mejora-documentacion
```

**Convenciones de nombres de branches:**
- `feature/` - Nueva funcionalidad
- `fix/` - Corrección de bug
- `docs/` - Documentación
- `refactor/` - Refactorización
- `test/` - Tests
- `chore/` - Mantenimiento

### Paso 4: Desarrollar

**Mejores Prácticas:**
- ✅ Escribe código limpio y bien documentado
- ✅ Sigue las convenciones del proyecto (ver abajo)
- ✅ Escribe tests para nuevas funcionalidades
- ✅ Asegúrate de que todos los tests pasen
- ✅ Actualiza documentación si es necesario
- ✅ Mantén commits pequeños y atómicos

### Paso 5: Pre-commit Hooks

```bash
# Instalar hooks (si están configurados)
pre-commit install

# Ejecutar manualmente
pre-commit run --all-files

# O ejecutar verificaciones manualmente
make lint
make format
make type-check
```

### Paso 6: Commit

Usa [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Formato: <tipo>(<ámbito>): <descripción>

# Ejemplos:
git commit -m "feat(api): agregar endpoint para listar tareas"
git commit -m "fix(core): corregir bug en procesamiento de tareas"
git commit -m "docs(readme): actualizar sección de instalación"
git commit -m "test(github): agregar tests para GitHub client"
git commit -m "refactor(utils): simplificar función de validación"
git commit -m "chore(deps): actualizar dependencias"
```

**Tipos de commits:**
- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Documentación
- `style`: Formato (sin cambios de código)
- `refactor`: Refactorización
- `test`: Tests
- `chore`: Mantenimiento
- `perf`: Mejoras de performance
- `ci`: Cambios en CI/CD

### Paso 7: Push y Pull Request

```bash
# Push a tu fork
git push origin feature/mi-nueva-funcionalidad

# Si necesitas actualizar tu branch con cambios de main:
git fetch upstream
git rebase upstream/main
git push origin feature/mi-nueva-funcionalidad --force-with-lease
```

**Luego crea un Pull Request en GitHub con:**
- ✅ Descripción clara del cambio
- ✅ Referencia a issues relacionados (si aplica)
- ✅ Screenshots o ejemplos (si aplica)
- ✅ Checklist completado

---

## 📝 Convenciones de Código

### Python

#### Estilo
- **PEP 8** - Seguir estilo de código Python estándar
- **Type Hints** - Siempre usar type hints
- **Docstrings** - Documentar todas las funciones y clases
- **Line Length** - Máximo 100 caracteres
- **Imports** - Organizados (stdlib, third-party, local)

#### Formato
```python
# Usar black para formateo automático
black .

# Usar isort para ordenar imports
isort .
```

#### Ejemplo de Código

```python
from typing import Dict, Any, Optional
from core.constants import GitConfig
from core.exceptions import GitHubAPIError

def process_repository(
    owner: str,
    repo: str,
    branch: Optional[str] = None
) -> Dict[str, Any]:
    """
    Procesa un repositorio de GitHub.
    
    Args:
        owner: Propietario del repositorio
        repo: Nombre del repositorio
        branch: Rama a procesar (default: main)
        
    Returns:
        Diccionario con información del repositorio procesado
        
    Raises:
        GitHubAPIError: Si hay error al acceder a GitHub API
        
    Example:
        >>> result = process_repository("owner", "repo", "main")
        >>> print(result["name"])
        'repo'
    """
    if not branch:
        branch = GitConfig.DEFAULT_BASE_BRANCH
    
    # Implementación...
    return {"name": repo, "branch": branch}
```

### Nombres

- **Funciones**: `snake_case` - `process_task()`
- **Clases**: `PascalCase` - `GitHubClient`
- **Constantes**: `UPPER_SNAKE_CASE` - `DEFAULT_BASE_BRANCH`
- **Archivos**: `snake_case.py` - `github_client.py`
- **Variables**: `snake_case` - `task_id`
- **Privadas**: `_leading_underscore` - `_internal_method()`

### Imports

```python
# 1. Standard library
import os
import sys
from typing import Dict, List

# 2. Third-party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 3. Local
from core.github_client import GitHubClient
from core.constants import GitConfig
```

---

## 🧪 Testing

### Escribir Tests

**Estructura:**
```python
# tests/unit/test_github_client.py
import pytest
from unittest.mock import Mock, patch
from core.github_client import GitHubClient
from core.exceptions import GitHubAPIError

class TestGitHubClient:
    """Tests para GitHubClient."""
    
    def test_initialization(self):
        """Test inicialización del cliente."""
        client = GitHubClient(token="test_token")
        assert client is not None
        assert client.token == "test_token"
    
    @pytest.mark.asyncio
    async def test_get_repo_async(self):
        """Test obtener repositorio de forma asíncrona."""
        client = GitHubClient(token="test_token")
        # Mock de GitHub API
        with patch('core.github_client.Github') as mock_github:
            # Test implementation
            pass
    
    def test_get_repo_error_handling(self):
        """Test manejo de errores."""
        client = GitHubClient(token="invalid")
        with pytest.raises(GitHubAPIError):
            client.get_repo("owner", "repo")
```

### Ejecutar Tests

```bash
# Todos los tests
make test
# o
pytest

# Con coverage
make test-cov
# o
pytest --cov=. --cov-report=html

# Tests específicos
pytest tests/unit/test_github_client.py
pytest tests/unit/test_github_client.py::TestGitHubClient::test_initialization

# Tests con verbose
pytest -v

# Tests con output de prints
pytest -s
```

### Cobertura Mínima

- **Objetivo**: >80% de cobertura
- **Crítico**: >90% para código core
- **Verificar**: `pytest --cov=. --cov-report=term-missing`

---

## 📚 Documentación

### Actualizar Documentación

**Cuando actualizar:**
- ✅ Nueva funcionalidad agregada
- ✅ Cambios en API
- ✅ Cambios en configuración
- ✅ Cambios en arquitectura
- ✅ Nuevos scripts o herramientas

**Archivos a actualizar:**
- `README.md` - Si afecta uso general
- `API_GUIDE.md` - Si cambia API
- `ARCHITECTURE.md` - Si cambia arquitectura
- `CHANGELOG.md` - Cambios importantes
- Docstrings en código - Siempre

### Docstrings

**Formato Google Style:**

```python
def process_task(
    task_id: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Procesa una tarea específica.
    
    Esta función procesa una tarea identificada por su ID, aplicando
    las opciones especificadas. El procesamiento incluye validación,
    ejecución y almacenamiento de resultados.
    
    Args:
        task_id: ID único de la tarea a procesar. Debe existir en la
            base de datos.
        options: Diccionario opcional con opciones de procesamiento.
            Keys soportados:
            - 'priority': Nivel de prioridad ('low', 'medium', 'high')
            - 'timeout': Timeout en segundos (default: 300)
            - 'retry': Número de reintentos (default: 3)
        
    Returns:
        Diccionario con los siguientes keys:
        - 'status': Estado final ('completed', 'failed', 'cancelled')
        - 'result': Resultado del procesamiento
        - 'duration': Duración en segundos
        - 'logs': Lista de logs generados
        
    Raises:
        TaskNotFoundError: Si la tarea con el ID especificado no existe.
        TaskProcessingError: Si ocurre un error durante el procesamiento.
        ValidationError: Si las opciones proporcionadas son inválidas.
        
    Example:
        >>> options = {'priority': 'high', 'timeout': 600}
        >>> result = process_task('task-123', options)
        >>> print(result['status'])
        'completed'
        
    Note:
        Esta función es asíncrona y debe ser llamada con await en
        contextos async.
    """
    # Implementación...
    pass
```

---

## 🔍 Code Review

### Checklist Antes de Enviar PR

#### Código
- [ ] Código sigue convenciones (PEP 8, type hints, docstrings)
- [ ] No hay código comentado o dead code
- [ ] No hay secretos o credenciales hardcodeadas
- [ ] Imports organizados correctamente
- [ ] Nombres descriptivos y consistentes

#### Testing
- [ ] Tests pasan (`make test`)
- [ ] Coverage adecuado (>80%)
- [ ] Tests para casos edge
- [ ] Tests para errores

#### Documentación
- [ ] Docstrings en funciones/clases nuevas
- [ ] README actualizado si es necesario
- [ ] CHANGELOG.md actualizado
- [ ] Comentarios donde el código es complejo

#### Pre-commit
- [ ] Pre-commit hooks pasan
- [ ] Linting pasa (`make lint`)
- [ ] Type checking pasa (`make type-check`)
- [ ] Formato correcto (`make format`)

### Checklist de Pull Request

#### Descripción
- [ ] Descripción clara del cambio
- [ ] Tipo de cambio identificado (feat, fix, docs, etc.)
- [ ] Referencia a issues relacionados
- [ ] Breaking changes documentados (si aplica)

#### Testing
- [ ] Tests agregados/actualizados
- [ ] Todos los tests pasan
- [ ] Coverage mantenido o mejorado

#### Documentación
- [ ] Documentación actualizada
- [ ] Ejemplos de uso (si nueva funcionalidad)
- [ ] Screenshots o GIFs (si cambios de UI)

#### Compatibilidad
- [ ] Sin breaking changes (o documentados)
- [ ] Backward compatibility mantenida
- [ ] Migraciones documentadas (si aplica)

### Proceso de Review

1. **Automático**: CI/CD ejecuta tests y linting
2. **Review de código**: Al menos un mantenedor revisa
3. **Feedback**: Se proporciona feedback constructivo
4. **Cambios**: Se hacen cambios si es necesario
5. **Aprobación**: Se aprueba cuando está listo
6. **Merge**: Se mergea a main

---

## 🐛 Reportar Bugs

### Template de Issue

```markdown
## Descripción del Bug
Descripción clara y concisa del problema.

## Pasos para Reproducir
1. Ir a '...'
2. Hacer clic en '...'
3. Scroll hasta '...'
4. Ver error

## Comportamiento Esperado
Descripción clara de qué debería pasar.

## Comportamiento Actual
Descripción clara de qué pasa actualmente.

## Screenshots
Si aplica, agregar screenshots.

## Entorno
- OS: [ej: Ubuntu 22.04, Windows 11, macOS 14]
- Python: [ej: 3.11.5]
- Versión del proyecto: [ej: 1.0.0]
- Versión de dependencias: [ej: FastAPI 0.115.0]

## Logs
```
Pegar logs relevantes aquí
```

## Contexto Adicional
Cualquier otra información que pueda ser útil.
```

### Severidad de Bugs

- **Crítico**: Sistema no funciona, data loss
- **Alto**: Funcionalidad importante no funciona
- **Medio**: Funcionalidad menor afectada
- **Bajo**: Problemas menores, mejoras

---

## 💡 Sugerir Funcionalidades

### Template de Feature Request

```markdown
## Descripción de la Funcionalidad
Descripción clara de la funcionalidad que quieres.

## Problema que Resuelve
Qué problema resuelve esta funcionalidad. ¿Es un problema
que experimentas? ¿Es una mejora de UX?

## Solución Propuesta
Cómo imaginas que funcionaría esta funcionalidad.

## Alternativas Consideradas
Otras soluciones que consideraste y por qué las descartaste.

## Impacto
- ¿Afecta a muchos usuarios?
- ¿Es fácil de implementar?
- ¿Requiere cambios breaking?

## Contexto Adicional
Cualquier otra información relevante, mockups, ejemplos, etc.
```

---

## ❓ Preguntas Frecuentes

### ¿Cómo empiezo a contribuir?

1. Lee el [README.md](README.md)
2. Lee [DEVELOPMENT.md](DEVELOPMENT.md)
3. Busca issues con etiqueta "good first issue"
4. Fork, clone, y crea tu primer PR

### ¿Qué tipo de contribuciones se aceptan?

- ✅ Código (features, fixes, refactoring)
- ✅ Documentación
- ✅ Tests
- ✅ Ejemplos
- ✅ Mejoras de UI/UX
- ✅ Reportes de bugs
- ✅ Sugerencias de features

### ¿Cómo sé qué trabajar?

- Revisa issues con etiqueta "help wanted"
- Busca "good first issue" para empezar
- Propón nuevas features si tienes ideas
- Mejora documentación existente

### ¿Qué pasa si mi PR es rechazado?

- No te desanimes, es parte del proceso
- Revisa el feedback cuidadosamente
- Haz los cambios sugeridos
- Vuelve a enviar el PR

### ¿Cómo contacto a los mantenedores?

- Abre un issue con etiqueta "question"
- Revisa la documentación primero
- Busca en issues existentes

---

## 📖 Recursos Adicionales

### Documentación del Proyecto
- [README.md](README.md) - Documentación principal
- [DEVELOPMENT.md](DEVELOPMENT.md) - Guía de desarrollo
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura
- [API_GUIDE.md](API_GUIDE.md) - Guía de API
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Código de conducta
- [CHANGELOG.md](CHANGELOG.md) - Historial de cambios

### Recursos Externos
- [Conventional Commits](https://www.conventionalcommits.org/)
- [PEP 8](https://pep8.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

---

## 🎯 Tipos de Contribuciones

### 🐛 Bug Fixes
- Identifica el bug claramente
- Crea test que reproduce el bug
- Implementa la fix
- Verifica que el test pasa
- Actualiza documentación si es necesario

### ✨ Features
- Discute la feature en un issue primero
- Implementa la feature
- Agrega tests completos
- Actualiza documentación
- Agrega ejemplos de uso

### 📚 Documentación
- Identifica qué necesita documentación
- Escribe de forma clara y concisa
- Incluye ejemplos
- Verifica que los enlaces funcionan

### 🧪 Tests
- Mejora coverage
- Agrega tests para casos edge
- Tests de integración
- Tests de performance

### 🔧 Refactoring
- Mejora código sin cambiar funcionalidad
- Mantiene tests pasando
- Mejora legibilidad
- Optimiza performance

---

## ✅ Checklist Final

Antes de enviar tu PR, asegúrate de:

- [ ] Código sigue convenciones
- [ ] Tests pasan y coverage adecuado
- [ ] Documentación actualizada
- [ ] CHANGELOG actualizado (si aplica)
- [ ] Pre-commit hooks pasan
- [ ] PR tiene descripción clara
- [ ] No hay secretos hardcodeados
- [ ] Imports organizados
- [ ] Type hints completos
- [ ] Docstrings en funciones nuevas

---

## 🙏 Agradecimientos

¡Gracias por contribuir! Cada contribución, grande o pequeña, hace que este proyecto sea mejor.

**Contribuidores destacados:**
- Ver [CONTRIBUTORS.md](CONTRIBUTORS.md) (si existe)

---

**¿Tienes preguntas?** Abre un issue con etiqueta "question" o consulta la [documentación completa](DOCUMENTATION_INDEX.md).

**¡Feliz contribución! 🎉**
