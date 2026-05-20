# 🚀 Funcionalidades Avanzadas

## ✨ Nuevas Características Implementadas

### 1. 🧪 Generación Automática de Tests

El sistema ahora genera tests automáticos para backend y frontend:

#### Backend Tests
- Tests para endpoints principales
- Tests para endpoints de IA
- Configuración de pytest
- Fixtures y helpers

#### Frontend Tests
- Tests con React Testing Library
- Setup de Jest
- Tests de componentes

**Ejemplo de uso:**
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

### 2. 🔄 CI/CD Pipelines Automáticos

Genera pipelines de CI/CD para GitHub Actions y GitLab CI:

#### GitHub Actions
- **Backend CI**: Tests automáticos en cada push
- **Frontend CI**: Tests y build automáticos
- **Docker Build**: Build y push de imágenes Docker

#### GitLab CI
- Pipeline completo con stages: test, build, deploy
- Soporte para Docker registry

**Archivos generados:**
- `.github/workflows/backend-ci.yml`
- `.github/workflows/frontend-ci.yml`
- `.github/workflows/docker-build.yml`
- `.gitlab-ci.yml`

### 3. 🐙 Integración con GitHub

Crea repositorios en GitHub automáticamente y hace push del código:

#### Funcionalidades
- Crear repositorio en GitHub
- Push automático del código generado
- Soporte para repositorios privados/públicos

**Endpoints:**
- `POST /api/v1/github/create` - Crear repositorio
- `POST /api/v1/github/push` - Push a GitHub

**Ejemplo:**
```python
# Crear repositorio
POST /api/v1/github/create
{
  "project_name": "mi_proyecto",
  "description": "Mi proyecto de IA",
  "github_token": "ghp_...",
  "private": false
}

# Push del código
POST /api/v1/github/push
{
  "project_path": "/path/to/project",
  "repo_url": "https://github.com/user/repo.git",
  "branch": "main"
}
```

### 4. 📋 Listado de Templates

Endpoint para ver templates disponibles:

**Endpoint:**
- `GET /api/v1/templates`

**Respuesta:**
```json
{
  "backend_frameworks": ["fastapi", "flask", "django"],
  "frontend_frameworks": ["react", "vue", "nextjs"],
  "ai_types": [
    "chat", "vision", "audio", "nlp", "video",
    "recommendation", "analytics", "generation",
    "classification", "summarization", "qa"
  ]
}
```

## 🎯 Flujo Completo de Generación

1. **Recibir descripción** → Analizar y extraer características
2. **Generar backend** → FastAPI con estructura modular
3. **Generar frontend** → React con TypeScript
4. **Generar tests** → Tests automáticos para ambos
5. **Generar CI/CD** → Pipelines de GitHub Actions/GitLab
6. **Crear GitHub repo** (opcional) → Repositorio automático
7. **Push a GitHub** (opcional) → Código subido automáticamente

## 📦 Estructura Completa Generada

```
proyecto/
├── backend/
│   ├── app/
│   ├── tests/              # ✨ Tests automáticos
│   │   ├── test_main.py
│   │   ├── test_ai_endpoints.py
│   │   └── conftest.py
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── src/__tests__/      # ✨ Tests automáticos
│   │   ├── App.test.tsx
│   │   └── setupTests.ts
│   └── package.json
├── .github/
│   └── workflows/          # ✨ CI/CD Pipelines
│       ├── backend-ci.yml
│       ├── frontend-ci.yml
│       └── docker-build.yml
├── .gitlab-ci.yml          # ✨ GitLab CI
└── README.md
```

## 🔧 Configuración

### Variables de Entorno

```bash
# GitHub Integration
GITHUB_TOKEN=ghp_tu_token_aqui

# Docker Registry (para CI/CD)
DOCKER_REGISTRY=registry.example.com
```

### Opciones de Generación

Al crear un proyecto, puedes especificar:

```json
{
  "description": "...",
  "generate_tests": true,        # Generar tests
  "include_cicd": true,          # Generar CI/CD
  "create_github_repo": false,    # Crear repo en GitHub
  "github_token": "...",          # Token de GitHub
  "github_private": false         # Repo privado
}
```

## 🚀 Ejemplo Completo

```python
import requests

# Generar proyecto con todas las características
response = requests.post(
    "http://localhost:8020/api/v1/generate",
    json={
        "description": "Un sistema de chat con IA",
        "project_name": "chat_ai",
        "generate_tests": True,
        "include_cicd": True,
        "create_github_repo": True,
        "github_token": "ghp_...",
        "github_private": False
    }
)

project_id = response.json()["project_id"]

# El proyecto se genera con:
# ✅ Backend completo
# ✅ Frontend completo
# ✅ Tests automáticos
# ✅ CI/CD pipelines
# ✅ Repositorio en GitHub
# ✅ Código subido a GitHub
```

## 📊 Estadísticas

Puedes ver estadísticas del generador:

```bash
GET /api/v1/stats
```

Respuesta:
```json
{
  "total_processed": 50,
  "total_completed": 48,
  "total_failed": 2,
  "total_pending": 3,
  "average_processing_time_seconds": 12.5,
  "success_rate": 96.0
}
```

## 🎉 Beneficios

- ✅ **Tests desde el inicio**: Cobertura de tests desde el día 1
- ✅ **CI/CD listo**: Pipelines configurados automáticamente
- ✅ **GitHub integrado**: Repositorios creados automáticamente
- ✅ **Producción ready**: Todo listo para desplegar
- ✅ **Mejores prácticas**: Estructura profesional desde el inicio


