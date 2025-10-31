#!/usr/bin/env python3
"""
Script de Implementación Rápida de Mejores Prácticas
Quick Implementation Script for Best Practices
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command: str, description: str, check: bool = True):
    """Ejecutar comando y mostrar resultado"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            if check:
                sys.exit(1)
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        if check:
            sys.exit(1)

def create_file(file_path: str, content: str):
    """Crear archivo con contenido"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created {file_path}")
    except Exception as e:
        print(f"❌ Failed to create {file_path}: {str(e)}")

def main():
    """Función principal de implementación"""
    print("🚀 AI History Comparison System - Best Practices Implementation")
    print("=" * 70)
    
    # Cambiar al directorio del proyecto
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # 1. Crear archivos de configuración
    print("\n📁 Creating configuration files...")
    
    # .env.example
    env_example_content = """# AI History Comparison System - Environment Variables
# Copia este archivo a .env y configura tus valores

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ai_history
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=20

# Security
SECRET_KEY=your-secret-key-here-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# LLM Services
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30

# Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
SENTRY_DSN=your-sentry-dsn-here

# Environment
ENVIRONMENT=development
DEBUG=true
"""
    create_file(".env.example", env_example_content)
    
    # pyproject.toml
    pyproject_content = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-history-comparison"
version = "1.0.0"
description = "AI History Comparison System"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "sqlalchemy>=2.0.23",
    "alembic>=1.13.0",
    "redis>=5.0.1",
    "openai>=1.3.7",
    "anthropic>=0.7.8",
    "google-generativeai>=0.3.2",
    "transformers>=4.36.2",
    "sentence-transformers>=2.2.2",
    "numpy>=1.24.3",
    "pandas>=2.1.4",
    "scikit-learn>=1.3.2",
    "loguru>=0.7.2",
    "prometheus-client>=0.19.0",
    "structlog>=23.2.0",
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "black>=23.11.0",
    "flake8>=6.1.0",
    "mypy>=1.7.1",
    "isort>=5.12.0",
]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --strict-markers"
"""
    create_file("pyproject.toml", pyproject_content)
    
    # .pre-commit-config.yaml
    precommit_content = """repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.8

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
"""
    create_file(".pre-commit-config.yaml", precommit_content)
    
    # docker-compose.yml
    docker_compose_content = """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ai_history
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=development
    depends_on:
      - db
      - redis
    volumes:
      - .:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: ai_history
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app

volumes:
  postgres_data:
  redis_data:
"""
    create_file("docker-compose.yml", docker_compose_content)
    
    # Dockerfile
    dockerfile_content = """FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependencias de build
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage de producción
FROM python:3.11-slim

WORKDIR /app

# Crear usuario no-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiar dependencias del builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar código de la aplicación
COPY . .

# Cambiar ownership
RUN chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Configurar PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    create_file("Dockerfile", dockerfile_content)
    
    # 2. Crear directorios necesarios
    print("\n📁 Creating necessary directories...")
    directories = [
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/load",
        "docs",
        "scripts",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # 3. Crear archivos de test
    print("\n🧪 Creating test files...")
    
    # conftest.py
    conftest_content = """import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from models import Base

# Base de datos de test
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
"""
    create_file("tests/conftest.py", conftest_content)
    
    # test_main.py
    test_main_content = """import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_analyze_endpoint(async_client):
    """Test analyze endpoint"""
    response = await async_client.post("/api/v1/analyze", json={
        "content": "Test content",
        "analysis_type": "comprehensive"
    })
    assert response.status_code in [200, 422]  # 422 if validation fails
"""
    create_file("tests/test_main.py", test_main_content)
    
    # 4. Crear scripts útiles
    print("\n📜 Creating utility scripts...")
    
    # scripts/setup_dev.py
    setup_dev_content = """#!/usr/bin/env python3
\"\"\"
Script de configuración para desarrollo
Development setup script
\"\"\"

import subprocess
import sys
import os

def run_command(command: str, description: str):
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} - SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED: {e}")
        sys.exit(1)

def main():
    print("🚀 Setting up development environment...")
    
    # Instalar pre-commit hooks
    run_command("pre-commit install", "Installing pre-commit hooks")
    
    # Instalar dependencias de desarrollo
    run_command("pip install -e .", "Installing package in development mode")
    
    # Ejecutar tests
    run_command("pytest", "Running tests")
    
    print("✅ Development environment setup complete!")

if __name__ == "__main__":
    main()
"""
    create_file("scripts/setup_dev.py", setup_dev_content)
    
    # scripts/run_tests.py
    run_tests_content = """#!/usr/bin/env python3
\"\"\"
Script para ejecutar tests
Test runner script
\"\"\"

import subprocess
import sys

def run_command(command: str, description: str):
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED: {e}")
        return False

def main():
    print("🧪 Running test suite...")
    
    # Tests unitarios
    if not run_command("pytest tests/unit -v", "Running unit tests"):
        sys.exit(1)
    
    # Tests de integración
    if not run_command("pytest tests/integration -v", "Running integration tests"):
        sys.exit(1)
    
    # Tests de carga (opcional)
    if os.path.exists("tests/load"):
        run_command("pytest tests/load -v", "Running load tests")
    
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    main()
"""
    create_file("scripts/run_tests.py", run_tests_content)
    
    # 5. Crear documentación
    print("\n📚 Creating documentation...")
    
    # README.md
    readme_content = """# 🚀 AI History Comparison System

Sistema completo para análisis, comparación y seguimiento de salidas de modelos de IA a lo largo del tiempo.

## ✨ Características

- **Análisis de Contenido** - Análisis completo de calidad, legibilidad, sentimiento
- **Comparación Histórica** - Comparación entre diferentes períodos y versiones
- **Análisis de Tendencias** - Identificación de patrones y tendencias
- **Reportes de Calidad** - Generación de reportes comprensivos
- **API REST** - API moderna con documentación automática
- **LLM Integration** - Integración con múltiples proveedores de LLM

## 🚀 Quick Start

### Prerrequisitos

- Python 3.8+
- PostgreSQL 13+
- Redis 6+
- Docker (opcional)

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/your-org/ai-history-comparison.git
cd ai-history-comparison

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# Ejecutar migraciones
alembic upgrade head

# Iniciar aplicación
uvicorn main:app --reload
```

### Docker

```bash
# Construir y ejecutar con Docker Compose
docker-compose up -d

# Ver logs
docker-compose logs -f app
```

## 📖 Uso

### API Endpoints

#### Análisis de Contenido
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "content": "Tu contenido aquí",
    "analysis_type": "comprehensive"
  }'
```

#### Comparación de Contenidos
```bash
curl -X POST "http://localhost:8000/api/v1/compare" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "content1": "Primer contenido",
    "content2": "Segundo contenido"
  }'
```

## 🧪 Testing

```bash
# Ejecutar tests
pytest

# Con coverage
pytest --cov=app --cov-report=html

# Tests de carga
pytest tests/load_tests.py
```

## 📊 Monitoreo

- **Health Check**: `GET /health`
- **Métricas**: `GET /metrics`
- **Documentación**: `GET /docs`

## 🤝 Contribución

1. Fork el proyecto
2. Crear feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Distribuido bajo la licencia MIT. Ver `LICENSE` para más información.

## 📞 Soporte

- Email: support@ai-history.com
- Issues: [GitHub Issues](https://github.com/your-org/ai-history-comparison/issues)
- Documentación: [API Docs](https://docs.ai-history.com)
"""
    create_file("README.md", readme_content)
    
    # 6. Instalar dependencias
    print("\n📦 Installing dependencies...")
    run_command("pip install pre-commit", "Installing pre-commit")
    run_command("pip install -r requirements.txt", "Installing project dependencies")
    
    # 7. Configurar pre-commit
    print("\n🔧 Setting up pre-commit hooks...")
    run_command("pre-commit install", "Installing pre-commit hooks")
    
    # 8. Ejecutar formateo inicial
    print("\n🎨 Running initial code formatting...")
    run_command("black .", "Formatting code with Black")
    run_command("isort .", "Sorting imports with isort")
    
    print("\n🎉 Best practices implementation completed!")
    print("\n📋 Next steps:")
    print("1. Copy .env.example to .env and configure your settings")
    print("2. Set up your database and Redis")
    print("3. Configure your LLM API keys")
    print("4. Run: python main.py")
    print("5. Test: curl http://localhost:8000/health")
    print("6. View docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main()







