# 🤖 GitHub Autonomous Agent

> Agente autónomo inteligente que se conecta a repositorios de GitHub y ejecuta instrucciones de forma continua desde el frontend, funcionando incluso con la computadora apagada hasta que el usuario le indique detenerse.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)
[![Dependencies](https://img.shields.io/badge/Dependencies-Updated-brightgreen.svg)](requirements.txt)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com)

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Instalación](#️-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [API](#-api)
- [Docker](#-docker)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Documentación](#-documentación)
- [Contribuir](#-contribuir)

## 🚀 Características

### Core Features
- ✅ **Conexión Universal**: Conecta a cualquier repositorio de GitHub (público o privado)
- ✅ **Ejecución Continua**: Procesa tareas de forma autónoma y persistente
- ✅ **Control Remoto**: Inicio/parada desde el frontend en tiempo real
- ✅ **Persistencia de Estado**: Funciona incluso después de reinicios del sistema
- ✅ **Dashboard en Tiempo Real**: Monitoreo completo del estado de tareas

### Advanced Features
- 🤖 **Servicio LLM con OpenRouter**: Acceso a múltiples modelos de IA (GPT-4, Claude, Gemini, etc.)
- ⚡ **Ejecución Paralela de Modelos**: Ejecuta múltiples modelos simultáneamente para comparar respuestas
- 📊 **A/B Testing Framework**: Compara modelos y prompts para optimización continua
- 🔔 **Webhooks y Notificaciones**: Sistema completo de notificaciones para eventos LLM
- 📝 **Prompt Versioning**: Versionado y gestión de prompts con rollback
- 🧪 **LLM Testing Framework**: Framework completo para testing automatizado de modelos
- 🧠 **Semantic Caching**: Cache inteligente basado en embeddings para respuestas similares
- 🚦 **Advanced Rate Limiting**: Rate limiting sofisticado con múltiples estrategias
- 📊 **Dashboard & Analytics**: Dashboard completo con métricas y analytics
- 🔄 **Request Queue System**: Sistema de cola con priorización y gestión de timeouts
- ⚖️ **Load Balancer**: Balanceo de carga inteligente con health checks y failover automático
- 🔁 **Adaptive Retry System**: Sistema de retry inteligente con backoff adaptativo y análisis de errores
- 🎯 **Prompt Optimizer**: Optimización automática de prompts para mejorar claridad y eficiencia
- 🛡️ **Model Fallback System**: Sistema de fallback automático cuando un modelo falla
- ⚡ **Performance Optimizer**: Optimización automática de rendimiento con auto-tuning
- 🔄 **Sistema de Cola Robusto**: Celery + Redis para tareas asíncronas
- 📊 **Logs y Monitoreo**: Logging estructurado con métricas Prometheus
- 🔒 **Seguridad**: Autenticación JWT, rate limiting, validación de datos
- 🐳 **Docker Ready**: Contenedores optimizados para desarrollo y producción
- 🔌 **API RESTful**: API completa con documentación automática (Swagger/OpenAPI)
- 🧪 **Testing Completo**: Suite de tests con alta cobertura
- 📝 **Validación Automática**: Validación de configuración y dependencias

### Technical Highlights
- **Arquitectura Modular**: Clean Architecture con separación de responsabilidades
- **Dependency Injection**: Sistema DI para mejor testabilidad
- **Async/Await**: Operaciones asíncronas para mejor performance
- **Type Safety**: Type hints completos con Pydantic
- **Error Handling**: Manejo robusto de errores con retry logic
- **Gestión de Dependencias**: Requirements.txt mejorado con versiones actualizadas y documentación completa

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React/Vue)                     │
│                  Control y Monitoreo en Tiempo Real           │
└──────────────────────┬────────────────────────────────────────┘
                       │ HTTP/REST API
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Agent Routes │  │ GitHub Routes│  │ Task Routes │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└──────────────────────┬────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ GitHub Client│ │Task Processor│ │   Worker     │
│  (PyGithub)  │ │   (Core)     │ │  (Celery)    │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
        ┌──────────────────────────┐
        │   Storage & Queue        │
        │  ┌──────┐  ┌──────────┐ │
        │  │ SQLite│  │  Redis   │ │
        │  │ / PG  │  │ (Celery) │ │
        │  └──────┘  └──────────┘ │
        └──────────────────────────┘
```

### Componentes Principales

- **API Layer** (`api/`): Endpoints REST, validación, middleware
- **Application Layer** (`application/`): Casos de uso y lógica de negocio
- **Core Layer** (`core/`): Clientes, procesadores, workers, storage
- **Config Layer** (`config/`): Settings, DI, logging

## 🛠️ Instalación

### Prerrequisitos

- Python 3.10 o superior
- Git
- Redis (para cola de tareas)
- Token de GitHub con permisos apropiados

### Opción 1: Instalación Automática (Recomendado)

#### Linux/macOS
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh --dev
```

#### Windows (PowerShell)
```powershell
.\scripts\setup.ps1 -Dev
```

### Opción 2: Instalación Manual

#### Desarrollo Local

```bash
# 1. Clonar repositorio (si aplica)
git clone <repository-url>
cd github_autonomous_agent

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# o
.\venv\Scripts\Activate.ps1  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de GitHub

# 5. Validar configuración
python scripts/validate-env.py

# 6. Verificar dependencias
python scripts/check-dependencies.py

# 7. Iniciar servicios (Redis, etc.)
./scripts/start-services.sh  # Linux/macOS

# 8. Ejecutar migraciones de BD (si aplica)
python scripts/migrate-db.py upgrade

# 9. Iniciar aplicación
python main.py
# O con Make:
make run-dev
```

#### Producción

```bash
# Instalar dependencias de producción
pip install -r requirements.txt
pip install -r requirements-prod.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con configuración de producción

# Iniciar con uvicorn (recomendado)
uvicorn main:app --host 0.0.0.0 --port 8030 --workers 4

# O con gunicorn (alternativa)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8030
```

### Opción 3: Docker (Recomendado para Producción)

```bash
# Desarrollo
docker-compose up -d

# Ver logs
docker-compose logs -f app

# Producción
docker build -t github-autonomous-agent:latest .
docker run -d -p 8030:8030 --env-file .env github-autonomous-agent:latest
```

📚 **Ver [QUICK_START.md](QUICK_START.md) para guía rápida de 5 minutos**  
📚 **Ver [REQUIREMENTS_GUIDE.md](REQUIREMENTS_GUIDE.md) para detalles sobre dependencias**  
📚 **Ver [DEVELOPMENT.md](DEVELOPMENT.md) para guía completa de desarrollo**

> 💡 **Nota**: El archivo `requirements.txt` ha sido mejorado con versiones actualizadas, mejor documentación, y guías de seguridad. Ver el archivo para más detalles.

## ⚙️ Configuración

### Variables de Entorno

Copia `.env.example` a `.env` y configura las siguientes variables:

#### Obligatorias

```bash
# GitHub API Token (requiere permisos: repo, workflow, admin:repo_hook)
GITHUB_TOKEN=ghp_tu_token_aqui

# Secret Key para JWT (genera una clave segura de 32+ caracteres)
SECRET_KEY=tu_clave_secreta_super_segura_aqui
```

#### Opcionales pero Recomendadas

```bash
# Base de datos (SQLite por defecto, PostgreSQL para producción)
DATABASE_URL=sqlite+aiosqlite:///./github_agent.db
# O para PostgreSQL:
# DATABASE_URL=postgresql+asyncpg://user:password@localhost/github_agent

# Redis para cola de tareas
REDIS_URL=redis://localhost:6379/0

# CORS (orígenes permitidos, separados por coma)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # json o text

# API
API_HOST=0.0.0.0
API_PORT=8030
```

#### Avanzadas

```bash
# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60

# Task Queue
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Monitoring
PROMETHEUS_ENABLED=false

# OpenRouter / LLM (Opcional)
OPENROUTER_API_KEY=sk-or-v1-...
LLM_DEFAULT_MODELS=openai/gpt-4o-mini,anthropic/claude-3.5-sonnet,google/gemini-pro-1.5
LLM_ENABLED=true
LLM_TIMEOUT=60
LLM_MAX_PARALLEL_REQUESTS=10
```

### Validar Configuración

```bash
# Validar todas las variables de entorno
python scripts/validate-env.py

# Verificar dependencias instaladas
python scripts/check-dependencies.py

# Health check
python scripts/health-check.py
```

Ver `config/settings.py` para todas las opciones de configuración disponibles.

## 📖 Uso

### Flujo Básico

1. **Conectar Repositorio**: Desde el frontend, conecta tu repositorio de GitHub
2. **Enviar Instrucciones**: Define las tareas que quieres que el agente ejecute
3. **Monitorear Progreso**: Observa el dashboard en tiempo real
4. **Controlar Agente**: Inicia, pausa o detén el agente cuando lo necesites

### Ejemplo de Uso desde API

```bash
# 1. Conectar repositorio
curl -X POST http://localhost:8030/api/v1/github/connect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "repo_owner": "usuario",
    "repo_name": "repositorio",
    "branch": "main"
  }'

# 2. Crear tarea
curl -X POST http://localhost:8030/api/v1/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "instruction": "Analizar código y generar documentación",
    "priority": "high"
  }'

# 3. Ver estado de tareas
curl http://localhost:8030/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN"

# 4. Detener agente
curl -X POST http://localhost:8030/api/v1/agent/stop \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Documentación Interactiva de API

Una vez iniciado el servidor, visita:
- **Swagger UI**: http://localhost:8030/docs
- **ReDoc**: http://localhost:8030/redoc
- **OpenAPI JSON**: http://localhost:8030/openapi.json

## 🔧 Comandos Útiles

### Con Make (Recomendado)

```bash
make help              # Ver todos los comandos disponibles
make setup             # Setup completo del proyecto
make install-dev       # Instalar dependencias de desarrollo
make install-prod      # Instalar dependencias de producción
make test              # Ejecutar todos los tests
make test-cov          # Tests con coverage
make lint              # Ejecutar linters (ruff, mypy, etc.)
make format            # Formatear código (black, isort)
make type-check        # Verificar tipos con mypy
make run-dev           # Ejecutar en modo desarrollo
make run-prod          # Ejecutar en modo producción
make docker-up         # Iniciar con Docker Compose
make docker-down       # Detener Docker Compose
make clean             # Limpiar archivos temporales
make migrate           # Ejecutar migraciones de BD
```

### Scripts Directos

```bash
# Setup automático
./scripts/setup.sh --dev          # Linux/macOS
.\scripts\setup.ps1 -Dev          # Windows

# Verificar dependencias
python scripts/check-dependencies.py

# Validar configuración
python scripts/validate-env.py

# Health check
python scripts/health-check.py

# Migraciones de BD
python scripts/migrate-db.py upgrade
python scripts/migrate-db.py downgrade

# Generar secret key
python scripts/generate-secret.py

# Security check
python scripts/security-check.py

# Backup
python scripts/backup.py

# Cleanup
python scripts/cleanup.py
```

## 🐳 Docker

### Desarrollo

```bash
# Iniciar todos los servicios (app, redis, etc.)
docker-compose up -d

# Ver logs
docker-compose logs -f app

# Rebuild después de cambios
docker-compose up -d --build

# Detener servicios
docker-compose down
```

### Producción

```bash
# Build imagen
docker build -t github-autonomous-agent:latest .

# Run contenedor
docker run -d \
  --name github-agent \
  -p 8030:8030 \
  --env-file .env \
  --restart unless-stopped \
  github-autonomous-agent:latest

# Ver logs
docker logs -f github-agent
```

Ver `Dockerfile` y `docker-compose.yml` para más detalles.

## 🧪 Testing

```bash
# Todos los tests
pytest

# Con coverage
pytest --cov=. --cov-report=html --cov-report=term

# Tests específicos
pytest tests/unit/
pytest tests/integration/

# Con Make
make test
make test-cov

# Ver coverage en navegador
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 🔍 Troubleshooting

> 📖 **Para una guía completa de troubleshooting, ver [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

### Problemas Comunes Rápidos

#### 1. Error de conexión a GitHub
```bash
# Verificar token
python scripts/validate-env.py

# Verificar permisos del token
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user
```

#### 2. Redis no disponible
```bash
# Verificar que Redis está corriendo
redis-cli ping

# Iniciar Redis
redis-server
# O con Docker:
docker run -d -p 6379:6379 redis:alpine
```

#### 3. Dependencias faltantes
```bash
# Verificar dependencias
python scripts/check-dependencies.py

# Reinstalar
pip install -r requirements.txt
```

#### 4. Error de base de datos
```bash
# Verificar conexión
python scripts/health-check.py

# Ejecutar migraciones
python scripts/migrate-db.py upgrade
```

#### 5. Puerto ya en uso
```bash
# Cambiar puerto en .env
API_PORT=8031

# O matar proceso
lsof -ti:8030 | xargs kill  # Linux/macOS
```

### Logs

```bash
# Ver logs en tiempo real
tail -f storage/logs/app.log

# Con Docker
docker-compose logs -f app

# Logs estructurados (JSON)
cat storage/logs/app.log | jq
```

### Más Ayuda

- 📖 **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Guía completa de troubleshooting
- 🔍 **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Índice de documentación
- 💬 Abre un issue en GitHub si el problema persiste

## 📚 Documentación

### Documentación Principal

- **[QUICK_START.md](QUICK_START.md)** - Inicio rápido en 5 minutos ⚡
- **[LLM_SERVICE.md](LLM_SERVICE.md)** - Servicio LLM con OpenRouter y ejecución paralela 🤖
- **[LLM_ARCHITECTURE.md](LLM_ARCHITECTURE.md)** - Arquitectura modular del servicio LLM 🏗️
- **[API_GUIDE.md](API_GUIDE.md)** - Guía completa de la API REST 📡
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Arquitectura del sistema 🏗️
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Guía de resolución de problemas 🔧
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Guía completa de desarrollo 💻
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Guía de deployment y producción 🚀
- **[REQUIREMENTS_GUIDE.md](REQUIREMENTS_GUIDE.md)** - Guía detallada de dependencias 📦
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guía de contribución 🤝
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Índice completo de documentación 📚

### Mejoras y Refactorizaciones

- **[IMPROVEMENTS_V8.md](IMPROVEMENTS_V8.md)** - Mejoras V8: Constantes y manejo de errores
- **[IMPROVEMENTS_V9.md](IMPROVEMENTS_V9.md)** - Mejoras V9: [Ver archivo para detalles]
- **[IMPROVEMENTS_V10.md](IMPROVEMENTS_V10.md)** - Mejoras V10: [Ver archivo para detalles]
- **[IMPROVEMENTS_V11.md](IMPROVEMENTS_V11.md)** - Mejoras V11: [Ver archivo para detalles]
- **[REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md)** - Resumen de refactorizaciones completadas

### Documentación Adicional

- **[examples/README.md](examples/README.md)** - Ejemplos de uso
- **[templates/README.md](templates/README.md)** - Templates de código
- **[scripts/README.md](scripts/README.md)** - Documentación de scripts

### API Documentation

- **Swagger UI**: http://localhost:8030/docs (cuando el servidor está corriendo)
- **ReDoc**: http://localhost:8030/redoc
- **OpenAPI Schema**: http://localhost:8030/openapi.json

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor lee [CONTRIBUTING.md](CONTRIBUTING.md) para detalles sobre:

- Código de conducta
- Proceso de contribución
- Estándares de código
- Cómo reportar bugs
- Cómo sugerir features

## 📁 Estructura del Proyecto

```
github_autonomous_agent/
├── api/                      # API endpoints (FastAPI)
│   ├── routes/              # Routers de FastAPI
│   │   ├── agent_routes.py  # Rutas del agente
│   │   ├── github_routes.py # Rutas de GitHub
│   │   ├── task_routes.py   # Rutas de tareas
│   │   └── llm_routes.py    # Rutas de LLM (OpenRouter)
│   ├── schemas.py           # Pydantic models
│   ├── dependencies.py      # Dependencias de FastAPI
│   ├── middleware.py        # Middleware personalizado
│   └── validators.py        # Validadores
├── application/             # Capa de aplicación
│   └── use_cases/          # Casos de uso
│       ├── github_use_cases.py
│       └── task_use_cases.py
├── core/                     # Lógica de negocio core
│   ├── github_client.py     # Cliente de GitHub API
│   ├── task_processor.py    # Procesador de tareas
│   ├── worker.py            # Worker para ejecución continua
│   ├── storage.py           # Persistencia de estado
│   ├── di/                  # Dependency Injection
│   │   └── container.py
│   └── services/            # Servicios core
│       ├── cache_service.py
│       ├── llm_service.py  # Servicio LLM (OpenRouter)
│       └── llm/            # Componentes modulares LLM
│           ├── prompt_templates.py
│           ├── token_manager.py
│           ├── batch_processor.py
│           ├── response_validator.py
│           ├── model_registry.py
│           ├── experiment_tracker.py
│           ├── config_manager.py
│           └── evaluation_metrics.py
├── config/                   # Configuración
│   ├── settings.py          # Settings de Pydantic
│   ├── di_setup.py          # Setup de DI
│   ├── logging_config.py    # Configuración de logging
│   └── llm/                 # Configuraciones LLM (YAML/JSON)
│       ├── models.yaml.example
│       ├── prompts.yaml.example
│       └── optimization.yaml.example
├── scripts/                  # Scripts de utilidad
│   ├── setup.sh             # Setup automático (Linux/macOS)
│   ├── setup.ps1            # Setup automático (Windows)
│   ├── check-dependencies.py
│   ├── validate-env.py
│   ├── migrate-db.py
│   └── health-check.py
├── storage/                  # Almacenamiento local
│   ├── tasks/               # Tareas
│   ├── logs/                # Logs
│   ├── cache/               # Cache
│   └── experiments/         # Experimentos LLM
├── tests/                    # Tests
│   ├── unit/                # Tests unitarios
│   └── integration/         # Tests de integración
├── examples/                 # Ejemplos de uso
├── templates/                # Templates de código
├── main.py                   # Punto de entrada
├── Dockerfile                # Imagen Docker
├── docker-compose.yml        # Orquestación Docker
├── Makefile                  # Comandos útiles
├── pyproject.toml           # Configuración del proyecto
├── requirements.txt          # Dependencias base
├── requirements-dev.txt      # Dependencias desarrollo
├── requirements-prod.txt     # Dependencias producción
├── .env.example             # Template de variables de entorno
└── README.md                 # Este archivo
```

## 🔄 CI/CD

El proyecto incluye GitHub Actions para CI/CD automático:

- ✅ Tests automáticos en cada push
- ✅ Linting y type checking
- ✅ Security scanning
- ✅ Build de Docker images
- ✅ Deployment automático (opcional)

Ver `.github/workflows/ci.yml` para más detalles.

## 📝 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

## 🆕 Últimas Mejoras

### Diciembre 2024

- ✅ **Funcionalidades Avanzadas LLM**: Sistema completo de features enterprise-grade
  - **A/B Testing Framework**: Compara modelos y prompts para optimización continua
  - **Webhooks y Notificaciones**: Sistema completo de notificaciones para eventos LLM
  - **Prompt Versioning**: Versionado y gestión de prompts con rollback y comparación
  - **LLM Testing Framework**: Framework completo para testing automatizado de modelos
  - **Semantic Caching**: Cache inteligente basado en embeddings para respuestas similares
  - **Advanced Rate Limiting**: Rate limiting sofisticado con múltiples estrategias (fixed/sliding window, token/leaky bucket)
  - **Dashboard & Analytics**: Dashboard completo con métricas y analytics detallados
- ✅ **Arquitectura Modular LLM**: Sistema completo de componentes modulares siguiendo principios de deep learning
  - Prompt Templates: Sistema de templates reutilizables
  - Token Manager: Gestión avanzada de tokens con validación
  - Batch Processor: Procesamiento eficiente por lotes
  - Response Validator: Validación y evaluación de respuestas
  - Model Registry: Gestión centralizada de modelos y configuraciones
  - Experiment Tracker: Tracking completo de experimentos (estilo wandb/tensorboard)
  - Config Manager: Gestión de configuraciones YAML/JSON
  - Evaluation Metrics: Métricas de evaluación (ROUGE-L, Jaccard, etc.)
  - Data Pipeline: Pipeline funcional de procesamiento de datos
  - Checkpoint Manager: Gestión de checkpoints y estados
  - Performance Profiler: Profiling y análisis de performance
  - Model Selector: Selección inteligente de modelos
  - Cost Optimizer: Optimización de costos
- ✅ **Requirements.txt Mejorado**: Versiones actualizadas, mejor documentación, y guías de seguridad
- ✅ **Gestión de Dependencias**: Rangos de versiones optimizados para permitir actualizaciones seguras
- ✅ **Documentación de Dependencias**: Comentarios detallados sobre cada dependencia y su uso
- ✅ **Herramientas de Seguridad**: Guías para pip-audit, safety check, y mejores prácticas

Ver [LLM_ARCHITECTURE.md](LLM_ARCHITECTURE.md) para detalles sobre la arquitectura modular.
Ver [LLM_ADVANCED_FEATURES.md](LLM_ADVANCED_FEATURES.md) para detalles sobre las funcionalidades avanzadas.

## 🙏 Agradecimientos

- FastAPI por el excelente framework
- PyGithub por la integración con GitHub
- Celery por el sistema de cola de tareas
- Todos los contribuidores del proyecto

---

## 🚀 Quick Start

```bash
# 1. Setup automático
./scripts/setup.sh --dev

# 2. Configurar .env
cp .env.example .env
# Editar .env con tus credenciales

# 3. Validar
python scripts/validate-env.py

# 4. Ejecutar
make run-dev
```

📚 **Para más detalles, ver [QUICK_START.md](QUICK_START.md)**

¡Listo! 🎉

---

**¿Necesitas ayuda?** Abre un issue o consulta la [documentación completa](DOCUMENTATION_INDEX.md).
