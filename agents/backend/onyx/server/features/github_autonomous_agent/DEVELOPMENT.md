# 💻 Guía de Desarrollo - GitHub Autonomous Agent

> Guía completa para desarrolladores del proyecto

## 📋 Tabla de Contenidos

- [Inicio Rápido](#-inicio-rápido)
- [Configuración](#-configuración)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Ejecutar la Aplicación](#-ejecutar-la-aplicación)
- [Testing](#-testing)
- [Code Quality](#-code-quality)
- [Debugging](#-debugging)
- [Convenciones](#-convenciones)
- [Flujo de Trabajo](#-flujo-de-trabajo)
- [Recursos](#-recursos)

---

## 🚀 Inicio Rápido

### Prerrequisitos

**Requisitos Mínimos:**
- Python 3.10 o superior
- pip (gestor de paquetes Python)
- Git
- Redis 6+ (para Celery, opcional en desarrollo)
- GitHub Personal Access Token

**Requisitos Recomendados:**
- PostgreSQL 13+ (opcional, SQLite funciona para desarrollo)
- Docker y Docker Compose (para desarrollo con contenedores)
- VS Code o PyCharm (IDE recomendado)

### Instalación Automática

#### Linux/macOS

```bash
# Dar permisos de ejecución
chmod +x scripts/setup.sh

# Ejecutar setup para desarrollo
./scripts/setup.sh --dev

# O para producción
./scripts/setup.sh --prod
```

#### Windows

```powershell
# Ejecutar setup para desarrollo
.\scripts\setup.ps1 -Dev

# O para producción
.\scripts\setup.ps1 -Prod
```

**El script automático:**
- ✅ Crea entorno virtual
- ✅ Instala dependencias
- ✅ Configura pre-commit hooks
- ✅ Valida configuración
- ✅ Verifica dependencias

### Instalación Manual

```bash
# 1. Crear entorno virtual
python3 -m venv venv

# 2. Activar entorno virtual
# Linux/macOS:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat

# 3. Actualizar pip
pip install --upgrade pip setuptools wheel

# 4. Instalar dependencias base
pip install -r requirements.txt

# 5. Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# 6. Configurar .env
cp .env.example .env
# Editar .env con tus credenciales

# 7. Validar configuración
python scripts/validate-env.py

# 8. Verificar dependencias
python scripts/check-dependencies.py
```

---

## 🔧 Configuración

### Variables de Entorno

Copia `.env.example` a `.env` y configura las siguientes variables:

#### Obligatorias

```bash
# GitHub Token (requerido)
GITHUB_TOKEN=ghp_tu_token_aqui

# Secret Key para JWT y encriptación
SECRET_KEY=genera_una_clave_segura_de_32_caracteres
```

#### Opcionales pero Recomendadas

```bash
# Base de datos
DATABASE_URL=sqlite+aiosqlite:///./storage/github_agent.db
# O PostgreSQL:
# DATABASE_URL=postgresql+asyncpg://user:pass@localhost/github_agent

# Redis (para Celery)
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8030

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
DEBUG=false

# OpenRouter (para LLM Service)
OPENROUTER_API_KEY=tu_api_key_aqui
```

#### Generar SECRET_KEY

```bash
# Opción 1: Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Opción 2: Script incluido
python scripts/generate-secret.py
```

### Validar Configuración

```bash
# Validar todas las variables
python scripts/validate-env.py

# Verificar dependencias instaladas
python scripts/check-dependencies.py

# Health check completo
python scripts/health-check.py
```

### Configuración de IDE

#### VS Code

Crear `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. Seleccionar el entorno virtual: `venv/bin/python`
3. Habilitar inspections: Settings → Editor → Inspections → Python

---

## 📦 Estructura del Proyecto

```
github-autonomous-agent/
├── api/                      # Capa de API
│   ├── __init__.py
│   ├── routes/              # Routers de FastAPI
│   │   ├── __init__.py
│   │   ├── agent.py        # Endpoints de agent
│   │   ├── github.py       # Endpoints de GitHub
│   │   └── tasks.py        # Endpoints de tareas
│   └── schemas/            # Pydantic models
│       ├── __init__.py
│       ├── agent.py
│       ├── github.py
│       └── tasks.py
│
├── application/             # Capa de aplicación
│   ├── __init__.py
│   └── use_cases/          # Casos de uso
│       ├── __init__.py
│       ├── agent_use_cases.py
│       └── task_use_cases.py
│
├── core/                    # Lógica de negocio core
│   ├── __init__.py
│   ├── di/                 # Dependency Injection
│   │   ├── __init__.py
│   │   └── container.py
│   ├── services/           # Servicios core
│   │   ├── __init__.py
│   │   ├── github_client.py
│   │   ├── task_processor.py
│   │   └── llm_service.py
│   ├── worker.py          # Worker manager
│   ├── storage.py         # Persistencia
│   ├── constants.py       # Constantes
│   └── exceptions.py      # Excepciones custom
│
├── config/                 # Configuración
│   ├── __init__.py
│   ├── settings.py        # Settings de Pydantic
│   └── logging.py         # Configuración de logging
│
├── scripts/                # Scripts de utilidad
│   ├── setup.sh           # Setup automático (Linux/macOS)
│   ├── setup.ps1          # Setup automático (Windows)
│   ├── check-dependencies.py
│   ├── validate-env.py
│   ├── health-check.py
│   ├── migrate-db.py
│   └── generate-secret.py
│
├── storage/                # Almacenamiento local
│   ├── tasks/             # Tareas procesadas
│   ├── logs/              # Logs de aplicación
│   ├── cache/             # Cache local
│   └── backups/           # Backups
│
├── tests/                  # Tests
│   ├── __init__.py
│   ├── conftest.py        # Configuración compartida
│   ├── unit/              # Tests unitarios
│   │   ├── test_github_client.py
│   │   ├── test_task_processor.py
│   │   └── test_worker.py
│   ├── integration/       # Tests de integración
│   │   ├── test_api.py
│   │   └── test_workflows.py
│   └── fixtures/         # Fixtures de test
│
├── main.py                # Punto de entrada
├── requirements.txt        # Dependencias base
├── requirements-dev.txt   # Dependencias desarrollo
├── requirements-prod.txt  # Dependencias producción
├── pyproject.toml         # Configuración del proyecto
├── Makefile              # Comandos útiles
├── Dockerfile            # Docker image
├── docker-compose.yml    # Docker Compose
├── .env.example          # Template de variables
└── README.md             # Documentación principal
```

### Convenciones de Estructura

- **`api/`**: Solo endpoints, validación y serialización
- **`application/`**: Casos de uso y orquestación
- **`core/`**: Lógica de negocio pura, sin dependencias de framework
- **`config/`**: Configuración y settings
- **`scripts/`**: Scripts de utilidad y automatización
- **`tests/`**: Tests organizados por tipo (unit, integration)

---

## 🏃 Ejecutar la Aplicación

### Modo Desarrollo

#### Opción 1: Python Directo

```bash
# Activar entorno virtual
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate.ps1  # Windows

# Ejecutar servidor con auto-reload
python main.py

# O con uvicorn directamente
uvicorn main:app --reload --host 0.0.0.0 --port 8030
```

#### Opción 2: Makefile

```bash
# Desarrollo con auto-reload
make run-dev

# Desarrollo con workers múltiples
make run-dev-workers
```

#### Opción 3: Docker Compose

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f app

# Detener
docker-compose down
```

### Modo Producción

```bash
# Con uvicorn (múltiples workers)
uvicorn main:app --host 0.0.0.0 --port 8030 --workers 4

# Con gunicorn (recomendado para producción)
gunicorn main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8030 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -

# O con Makefile
make run-prod
```

### Celery Worker

```bash
# Iniciar worker
celery -A core.worker worker --loglevel=info

# Con múltiples workers
celery -A core.worker worker --loglevel=info --concurrency=4

# Con beat scheduler
celery -A core.worker beat --loglevel=info
```

### Verificar que Funciona

```bash
# Health check
curl http://localhost:8030/health

# API docs
open http://localhost:8030/docs  # macOS
xdg-open http://localhost:8030/docs  # Linux
start http://localhost:8030/docs  # Windows
```

---

## 🧪 Testing

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Con coverage
pytest --cov=. --cov-report=html --cov-report=term

# Tests específicos
pytest tests/unit/test_github_client.py
pytest tests/unit/test_github_client.py::TestGitHubClient::test_get_repo

# Modo verbose
pytest -v

# Con output de prints
pytest -s

# Solo tests que fallaron anteriormente
pytest --lf

# Parar en primer error
pytest -x

# Con markers
pytest -m "not slow"
```

### Estructura de Tests

```python
# tests/unit/test_github_client.py
import pytest
from unittest.mock import Mock, patch
from core.services.github_client import GitHubClient
from core.exceptions import GitHubAPIError

class TestGitHubClient:
    """Tests para GitHubClient."""
    
    @pytest.fixture
    def client(self):
        """Fixture para crear cliente."""
        return GitHubClient(token="test_token")
    
    def test_initialization(self, client):
        """Test inicialización del cliente."""
        assert client is not None
        assert client.token == "test_token"
    
    @pytest.mark.asyncio
    async def test_get_repo_async(self, client):
        """Test obtener repositorio de forma asíncrona."""
        with patch('core.services.github_client.Github') as mock_github:
            # Mock implementation
            mock_repo = Mock()
            mock_repo.name = "test-repo"
            mock_github.return_value.get_repo.return_value = mock_repo
            
            repo = await client.get_repo_async("owner", "test-repo")
            assert repo.name == "test-repo"
    
    def test_get_repo_error_handling(self, client):
        """Test manejo de errores."""
        with pytest.raises(GitHubAPIError):
            client.get_repo("owner", "nonexistent")
```

### Fixtures

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Fixture para cliente de test."""
    return TestClient(app)

@pytest.fixture
def mock_github_token(monkeypatch):
    """Fixture para mock de GitHub token."""
    monkeypatch.setenv("GITHUB_TOKEN", "test_token")
```

### Coverage

**Objetivo:** >80% de cobertura

```bash
# Generar reporte HTML
pytest --cov=. --cov-report=html

# Ver reporte
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## 📝 Code Quality

### Formateo de Código

```bash
# Con black (recomendado)
black .

# Verificar sin cambiar
black --check .

# Con ruff (más rápido)
ruff format .

# Verificar con ruff
ruff format --check .
```

### Linting

```bash
# Con ruff (recomendado, más rápido)
ruff check .

# Auto-fix con ruff
ruff check --fix .

# Con pylint
pylint api/ core/ application/

# Con mypy (type checking)
mypy .

# Con flake8
flake8 .
```

### Pre-commit Hooks

```bash
# Instalar hooks
pre-commit install

# Ejecutar manualmente en todos los archivos
pre-commit run --all-files

# Ejecutar en archivos staged
pre-commit run

# Actualizar hooks
pre-commit autoupdate
```

### Makefile Commands

```bash
# Formatear código
make format

# Linting
make lint

# Type checking
make type-check

# Todo junto
make quality
```

---

## 🐛 Debugging

### Debug con VS Code

Crear `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Python: Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "${file}",
                "-v"
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

### Debug con ipdb

```python
# Instalar
pip install ipdb

# Usar en código
import ipdb; ipdb.set_trace()

# O con breakpoint() (Python 3.7+)
breakpoint()
```

### Debug con pdb

```python
import pdb; pdb.set_trace()
```

### Logging

```python
import logging

# Obtener logger
logger = logging.getLogger(__name__)

# Usar diferentes niveles
logger.debug("Debug message - información detallada")
logger.info("Info message - información general")
logger.warning("Warning message - advertencia")
logger.error("Error message - error")
logger.critical("Critical message - error crítico")

# Con contexto
logger.info("Processing task", extra={
    "task_id": task_id,
    "user_id": user_id
})
```

### Ver Logs

```bash
# Logs en tiempo real
tail -f storage/logs/app.log

# Últimas 100 líneas
tail -n 100 storage/logs/app.log

# Buscar en logs
grep "ERROR" storage/logs/app.log

# Con Docker
docker-compose logs -f app
```

---

## 📚 Convenciones

### Commits

Usar [Conventional Commits](https://www.conventionalcommits.org/):

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

**Tipos:**
- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Documentación
- `style`: Formato (sin cambios de código)
- `refactor`: Refactorización
- `test`: Tests
- `chore`: Mantenimiento
- `perf`: Mejoras de performance
- `ci`: Cambios en CI/CD

### Nombres

- **Funciones**: `snake_case` - `process_task()`
- **Clases**: `PascalCase` - `GitHubClient`
- **Constantes**: `UPPER_SNAKE_CASE` - `DEFAULT_BASE_BRANCH`
- **Archivos**: `snake_case.py` - `github_client.py`
- **Variables**: `snake_case` - `task_id`
- **Privadas**: `_leading_underscore` - `_internal_method()`

### Type Hints

Siempre usar type hints:

```python
from typing import Dict, List, Optional, Any
from datetime import datetime

def process_task(
    task_id: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Procesa una tarea."""
    ...

# Con async
async def get_repo_async(
    owner: str,
    repo: str
) -> Dict[str, Any]:
    """Obtiene un repositorio."""
    ...
```

### Docstrings

Usar formato Google Style:

```python
def process_task(task_id: str) -> Dict[str, Any]:
    """
    Procesa una tarea específica.
    
    Args:
        task_id: ID de la tarea a procesar
        
    Returns:
        Diccionario con resultado del procesamiento
        
    Raises:
        TaskNotFoundError: Si la tarea no existe
    """
    ...
```

---

## 🔄 Flujo de Trabajo

### 1. Crear Branch

```bash
# Actualizar main
git checkout main
git pull upstream main

# Crear branch
git checkout -b feature/nueva-funcionalidad
# o
git checkout -b fix/correccion-de-bug
```

### 2. Desarrollar

```bash
# Activar entorno virtual
source venv/bin/activate

# Escribir código
# Escribir tests
# Ejecutar tests
pytest

# Verificar calidad
make quality
```

### 3. Commit

```bash
# Agregar cambios
git add .

# Commit con mensaje convencional
git commit -m "feat(api): agregar endpoint para listar tareas"

# O commit interactivo
git commit
```

### 4. Push y PR

```bash
# Push a tu fork
git push origin feature/nueva-funcionalidad

# Crear Pull Request en GitHub
# - Descripción clara
# - Referencia a issues
# - Screenshots si aplica
```

### 5. Code Review

- Revisar feedback
- Hacer cambios si es necesario
- Actualizar PR

### 6. Merge

- Esperar aprobación
- Merge a main
- Eliminar branch

---

## 📖 Recursos

### Documentación del Proyecto

- [README.md](README.md) - Documentación principal
- [QUICK_START.md](QUICK_START.md) - Inicio rápido
- [API_GUIDE.md](API_GUIDE.md) - Guía de API
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura
- [CONTRIBUTING.md](CONTRIBUTING.md) - Guía de contribución
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Troubleshooting

### Documentación Externa

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [GitHub API Documentation](https://docs.github.com/en/rest)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

### Herramientas

- [Black](https://black.readthedocs.io/) - Formateo de código
- [Ruff](https://docs.astral.sh/ruff/) - Linter rápido
- [Pytest](https://docs.pytest.org/) - Framework de testing
- [MyPy](https://mypy.readthedocs.io/) - Type checking

---

## ❓ Troubleshooting

### Error: "Module not found"

```bash
# Verificar que estás en el entorno virtual
which python  # Debe mostrar venv/bin/python

# Reinstalar dependencias
pip install -r requirements.txt -r requirements-dev.txt
```

### Error: "Redis connection failed"

```bash
# Verificar que Redis está corriendo
redis-cli ping  # Debe responder PONG

# Iniciar Redis
redis-server
# O con Docker:
docker run -d -p 6379:6379 redis:alpine
```

### Error: "GitHub API rate limit"

- Usa autenticación con token
- Implementa cache
- Reduce frecuencia de requests

### Error: "Pre-commit hooks failed"

```bash
# Ejecutar manualmente para ver errores
pre-commit run --all-files

# Auto-fix si es posible
black .
ruff check --fix .
```

---

**¿Necesitas más ayuda?** Consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md) o abre un issue en GitHub.

**Última actualización:** Diciembre 2024
