#!/usr/bin/env python3
"""
Setup Modular Structure Script
Script para crear la estructura modular completa del sistema
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Crear estructura de directorios modular"""
    
    # Estructura base
    structure = {
        "ai_history_comparison": {
            "core": {},
            "domain": {
                "entities": {},
                "services": {},
                "repositories": {},
                "events": {}
            },
            "infrastructure": {
                "database": {
                    "repositories": {},
                    "migrations": {}
                },
                "cache": {},
                "external": {
                    "llm": {},
                    "storage": {},
                    "monitoring": {}
                },
                "messaging": {}
            },
            "application": {
                "use_cases": {},
                "handlers": {},
                "dto": {},
                "validators": {}
            },
            "presentation": {
                "api": {
                    "v1": {},
                    "v2": {},
                    "websocket": {}
                },
                "cli": {},
                "web": {
                    "templates": {},
                    "static": {}
                }
            },
            "plugins": {
                "analyzers": {},
                "exporters": {},
                "integrations": {}
            },
            "tests": {
                "unit": {
                    "domain": {},
                    "application": {},
                    "infrastructure": {},
                    "presentation": {}
                },
                "integration": {
                    "api": {},
                    "database": {},
                    "external": {}
                },
                "e2e": {
                    "scenarios": {}
                },
                "fixtures": {}
            },
            "scripts": {},
            "docs": {
                "api": {},
                "architecture": {},
                "deployment": {},
                "development": {}
            },
            "config": {},
            "docker": {},
            "k8s": {}
        }
    }
    
    def create_dirs(base_path: Path, structure: dict):
        """Crear directorios recursivamente"""
        for name, children in structure.items():
            dir_path = base_path / name
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Module initialization"""\n')
            
            # Crear subdirectorios
            if children:
                create_dirs(dir_path, children)
    
    # Crear estructura
    base_path = Path.cwd()
    create_dirs(base_path, structure)
    
    print("✅ Estructura de directorios creada")

def create_core_files():
    """Crear archivos del módulo core"""
    
    # core/exceptions.py
    exceptions_content = '''"""
Core Exceptions - Excepciones del Sistema
Excepciones personalizadas para el sistema
"""

class AIHistoryException(Exception):
    """Excepción base del sistema"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ValidationError(AIHistoryException):
    """Error de validación de datos"""
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message, "VALIDATION_ERROR")

class NotFoundError(AIHistoryException):
    """Recurso no encontrado"""
    def __init__(self, resource: str, resource_id: str = None):
        message = f"{resource} not found"
        if resource_id:
            message += f" with id: {resource_id}"
        super().__init__(message, "NOT_FOUND")

class ExternalServiceError(AIHistoryException):
    """Error de servicio externo"""
    def __init__(self, service: str, message: str):
        self.service = service
        super().__init__(f"{service} error: {message}", "EXTERNAL_SERVICE_ERROR")

class CacheError(AIHistoryException):
    """Error de caché"""
    def __init__(self, message: str):
        super().__init__(f"Cache error: {message}", "CACHE_ERROR")

class DatabaseError(AIHistoryException):
    """Error de base de datos"""
    def __init__(self, message: str):
        super().__init__(f"Database error: {message}", "DATABASE_ERROR")

class AuthenticationError(AIHistoryException):
    """Error de autenticación"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_ERROR")

class AuthorizationError(AIHistoryException):
    """Error de autorización"""
    def __init__(self, message: str = "Authorization failed"):
        super().__init__(message, "AUTHORIZATION_ERROR")

class RateLimitError(AIHistoryException):
    """Error de rate limiting"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "RATE_LIMIT_ERROR")
'''
    
    with open("ai_history_comparison/core/exceptions.py", "w", encoding="utf-8") as f:
        f.write(exceptions_content)
    
    # core/utils.py
    utils_content = '''"""
Core Utils - Utilidades Comunes
Utilidades y funciones auxiliares del sistema
"""

import uuid
import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

def generate_id(prefix: str = "") -> str:
    """Generar ID único"""
    if prefix:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    return uuid.uuid4().hex

def format_timestamp(dt: datetime = None) -> str:
    """Formatear timestamp"""
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat()

def validate_content(content: str, max_length: int = 100000) -> bool:
    """Validar contenido"""
    if not content or not content.strip():
        return False
    if len(content) > max_length:
        return False
    return True

def sanitize_input(text: str) -> str:
    """Sanitizar entrada de usuario"""
    if not text:
        return ""
    
    # Remover caracteres de control
    text = re.sub(r'[\\x00-\\x1f\\x7f-\\x9f]', '', text)
    
    # Limpiar espacios extra
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text

def calculate_content_hash(content: str) -> str:
    """Calcular hash del contenido"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extraer palabras clave del texto"""
    # Implementación simple - en producción usar NLP
    words = re.findall(r'\\b\\w+\\b', text.lower())
    
    # Filtrar palabras comunes
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Contar frecuencia
    word_count = {}
    for word in keywords:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Ordenar por frecuencia
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, count in sorted_words[:max_keywords]]

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncar texto"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionar diccionarios"""
    result = {}
    for d in dicts:
        result.update(d)
    return result

def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Obtener valor de diccionario de forma segura"""
    keys = key.split('.')
    value = data
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Dividir lista en chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
'''
    
    with open("ai_history_comparison/core/utils.py", "w", encoding="utf-8") as f:
        f.write(utils_content)
    
    print("✅ Archivos del módulo core creados")

def create_domain_files():
    """Crear archivos del módulo domain"""
    
    # domain/entities/__init__.py
    entities_init = '''"""
Domain Entities - Entidades de Dominio
Entidades principales del sistema
"""

from .content import Content, ContentType, ContentStatus
from .analysis import Analysis, AnalysisType, AnalysisStatus
from .comparison import Comparison, ComparisonType, ComparisonStatus
from .report import Report, ReportType, ReportStatus
from .trend import Trend, TrendType, TrendStatus

__all__ = [
    # Content
    "Content",
    "ContentType", 
    "ContentStatus",
    
    # Analysis
    "Analysis",
    "AnalysisType",
    "AnalysisStatus",
    
    # Comparison
    "Comparison",
    "ComparisonType",
    "ComparisonStatus",
    
    # Report
    "Report",
    "ReportType",
    "ReportStatus",
    
    # Trend
    "Trend",
    "TrendType",
    "TrendStatus"
]
'''
    
    with open("ai_history_comparison/domain/entities/__init__.py", "w", encoding="utf-8") as f:
        f.write(entities_init)
    
    # domain/services/__init__.py
    services_init = '''"""
Domain Services - Servicios de Dominio
Servicios de lógica de negocio
"""

from .content_service import ContentService
from .analysis_service import AnalysisService
from .comparison_service import ComparisonService
from .report_service import ReportService

__all__ = [
    "ContentService",
    "AnalysisService",
    "ComparisonService",
    "ReportService"
]
'''
    
    with open("ai_history_comparison/domain/services/__init__.py", "w", encoding="utf-8") as f:
        f.write(services_init)
    
    print("✅ Archivos del módulo domain creados")

def create_application_files():
    """Crear archivos del módulo application"""
    
    # application/dto/__init__.py
    dto_init = '''"""
Application DTOs - Data Transfer Objects
Objetos de transferencia de datos
"""

from .content_dto import ContentDTO
from .analysis_dto import AnalysisDTO
from .comparison_dto import ComparisonDTO
from .report_dto import ReportDTO
from .trend_dto import TrendDTO

__all__ = [
    "ContentDTO",
    "AnalysisDTO",
    "ComparisonDTO",
    "ReportDTO",
    "TrendDTO"
]
'''
    
    with open("ai_history_comparison/application/dto/__init__.py", "w", encoding="utf-8") as f:
        f.write(dto_init)
    
    # application/validators/__init__.py
    validators_init = '''"""
Application Validators - Validadores
Validadores de datos de entrada
"""

from .content_validator import ContentValidator
from .analysis_validator import AnalysisValidator
from .comparison_validator import ComparisonValidator
from .report_validator import ReportValidator

__all__ = [
    "ContentValidator",
    "AnalysisValidator",
    "ComparisonValidator",
    "ReportValidator"
]
'''
    
    with open("ai_history_comparison/application/validators/__init__.py", "w", encoding="utf-8") as f:
        f.write(validators_init)
    
    print("✅ Archivos del módulo application creados")

def create_presentation_files():
    """Crear archivos del módulo presentation"""
    
    # presentation/api/__init__.py
    api_init = '''"""
Presentation API - API de Presentación
API REST y WebSocket
"""

from .factory import create_app, create_router, setup_middleware, setup_routes

__all__ = [
    "create_app",
    "create_router",
    "setup_middleware",
    "setup_routes"
]
'''
    
    with open("ai_history_comparison/presentation/api/__init__.py", "w", encoding="utf-8") as f:
        f.write(api_init)
    
    # presentation/api/factory.py
    factory_content = '''"""
API Factory - Factory de API
Factory para crear aplicación FastAPI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from ...core.config import get_settings
from ...core.middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    SecurityMiddleware
)

def create_app() -> FastAPI:
    """Crear aplicación FastAPI"""
    settings = get_settings()
    
    app = FastAPI(
        title="AI History Comparison System",
        description="Sistema modular de comparación de historial de IA",
        version="1.0.0",
        docs_url="/docs" if settings.is_development() else None,
        redoc_url="/redoc" if settings.is_development() else None,
        openapi_url="/openapi.json" if settings.is_development() else None
    )
    
    # Configurar middleware
    setup_middleware(app)
    
    # Configurar rutas
    setup_routes(app)
    
    return app

def setup_middleware(app: FastAPI):
    """Configurar middleware"""
    settings = get_settings()
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=settings.security.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"]
    )
    
    # Compresión
    if settings.performance.enable_compression:
        app.add_middleware(
            GZipMiddleware,
            minimum_size=settings.performance.compression_min_size
        )
    
    # Middleware personalizado
    app.middleware("http")(LoggingMiddleware())
    app.middleware("http")(MetricsMiddleware())
    app.middleware("http")(RateLimitMiddleware())
    app.middleware("http")(SecurityMiddleware())

def setup_routes(app: FastAPI):
    """Configurar rutas"""
    # Importar routers
    from .v1 import (
        content_router,
        analysis_router,
        comparison_router,
        report_router,
        system_router
    )
    
    # Incluir routers
    app.include_router(content_router, prefix="/api/v1")
    app.include_router(analysis_router, prefix="/api/v1")
    app.include_router(comparison_router, prefix="/api/v1")
    app.include_router(report_router, prefix="/api/v1")
    app.include_router(system_router, prefix="/api/v1")

def create_router(prefix: str = "/api/v1") -> APIRouter:
    """Crear router base"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix=prefix)
    return router
'''
    
    with open("ai_history_comparison/presentation/api/factory.py", "w", encoding="utf-8") as f:
        f.write(factory_content)
    
    print("✅ Archivos del módulo presentation creados")

def create_test_files():
    """Crear archivos de testing"""
    
    # tests/conftest.py
    conftest_content = '''"""
Test Configuration - Configuración de Tests
Configuración común para todos los tests
"""

import pytest
import asyncio
from typing import AsyncGenerator
from fastapi.testclient import TestClient

from ai_history_comparison.presentation.api.factory import create_app
from ai_history_comparison.core.config import get_settings

@pytest.fixture(scope="session")
def event_loop():
    """Crear event loop para tests asíncronos"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def app():
    """Crear aplicación de test"""
    return create_app()

@pytest.fixture
def client(app):
    """Crear cliente de test"""
    return TestClient(app)

@pytest.fixture
def settings():
    """Obtener configuración de test"""
    return get_settings()

@pytest.fixture
async def db_session():
    """Crear sesión de base de datos de test"""
    # Implementar sesión de test
    yield None

@pytest.fixture
async def cache_service():
    """Crear servicio de caché de test"""
    # Implementar caché de test
    yield None
'''
    
    with open("ai_history_comparison/tests/conftest.py", "w", encoding="utf-8") as f:
        f.write(conftest_content)
    
    print("✅ Archivos de testing creados")

def create_config_files():
    """Crear archivos de configuración"""
    
    # config/development.yaml
    dev_config = '''# Development Configuration
database:
  url: "sqlite:///./dev.db"
  echo: true
  pool_size: 5

cache:
  redis_url: "redis://localhost:6379"
  default_ttl: 3600

llm:
  default_model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 4000

security:
  secret_key: "dev-secret-key"
  cors_origins: ["*"]
  rate_limit_requests: 1000

monitoring:
  log_level: "DEBUG"
  enable_metrics: true

performance:
  workers: 1
  enable_compression: true
'''
    
    with open("ai_history_comparison/config/development.yaml", "w", encoding="utf-8") as f:
        f.write(dev_config)
    
    # config/production.yaml
    prod_config = '''# Production Configuration
database:
  url: "${DATABASE_URL}"
  echo: false
  pool_size: 20

cache:
  redis_url: "${REDIS_URL}"
  default_ttl: 3600

llm:
  default_model: "gpt-4"
  temperature: 0.3
  max_tokens: 4000

security:
  secret_key: "${SECRET_KEY}"
  cors_origins: ["${ALLOWED_ORIGINS}"]
  rate_limit_requests: 100

monitoring:
  log_level: "WARNING"
  enable_metrics: true

performance:
  workers: 8
  enable_compression: true
'''
    
    with open("ai_history_comparison/config/production.yaml", "w", encoding="utf-8") as f:
        f.write(prod_config)
    
    print("✅ Archivos de configuración creados")

def create_docker_files():
    """Crear archivos de Docker"""
    
    # docker/Dockerfile
    dockerfile_content = '''# Dockerfile para AI History Comparison System
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Crear usuario no-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open("ai_history_comparison/docker/Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    
    # docker/docker-compose.yml
    compose_content = '''# Docker Compose para desarrollo
version: '3.8'

services:
  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://user:password@db:5432/ai_history
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ..:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=ai_history
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
'''
    
    with open("ai_history_comparison/docker/docker-compose.yml", "w", encoding="utf-8") as f:
        f.write(compose_content)
    
    print("✅ Archivos de Docker creados")

def create_main_file():
    """Crear archivo main.py"""
    
    main_content = '''"""
Main Entry Point - Punto de Entrada Principal
Punto de entrada para la aplicación modular
"""

import uvicorn
from ai_history_comparison.presentation.api.factory import create_app
from ai_history_comparison.core.config import get_settings

def main():
    """Función principal"""
    settings = get_settings()
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=settings.performance.workers,
        log_level=settings.monitoring.log_level.lower(),
        reload=settings.is_development()
    )

if __name__ == "__main__":
    main()
'''
    
    with open("ai_history_comparison/main.py", "w", encoding="utf-8") as f:
        f.write(main_content)
    
    print("✅ Archivo main.py creado")

def create_requirements():
    """Crear archivos de requirements"""
    
    # requirements.txt
    requirements = '''# AI History Comparison System - Requirements
# Dependencias principales

# Core Framework
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0

# Database
sqlalchemy>=2.0.23
asyncpg>=0.29.0
alembic>=1.13.0

# Cache
redis>=5.0.1
aioredis>=2.0.1

# HTTP Clients
httpx>=0.25.2
aiohttp>=3.9.1

# LLM Services
openai>=1.3.0
anthropic>=0.7.0
google-generativeai>=0.3.0

# AI/ML
numpy>=1.24.3
pandas>=2.1.4
scikit-learn>=1.3.2
transformers>=4.35.0
sentence-transformers>=2.2.2

# Utilities
python-dotenv>=1.0.0
click>=8.1.7
rich>=13.7.0
loguru>=0.7.2

# Performance
uvloop>=0.19.0
httptools>=0.6.0

# Security
passlib[bcrypt]>=1.7.4
python-jose[cryptography]>=3.3.0
python-multipart>=0.0.6

# Monitoring
prometheus-client>=0.19.0
sentry-sdk[fastapi]>=1.38.0
'''
    
    with open("ai_history_comparison/requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    # requirements-dev.txt
    dev_requirements = '''# Development Requirements
# Dependencias de desarrollo

# Testing
pytest>=7.4.3
pytest-asyncio>=0.21.1
pytest-cov>=4.1.0
pytest-mock>=3.12.0
httpx>=0.25.2

# Code Quality
black>=23.11.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.7.0
pre-commit>=3.5.0

# Documentation
mkdocs>=1.5.3
mkdocs-material>=9.4.0
mkdocstrings[python]>=0.24.0

# Development Tools
ipython>=8.17.0
jupyter>=1.0.0
watchdog>=3.0.0
'''
    
    with open("ai_history_comparison/requirements-dev.txt", "w", encoding="utf-8") as f:
        f.write(dev_requirements)
    
    print("✅ Archivos de requirements creados")

def main():
    """Función principal del script"""
    print("🏗️ AI History Comparison System - Setup Modular Structure")
    print("=" * 60)
    
    try:
        # Crear estructura de directorios
        create_directory_structure()
        
        # Crear archivos del core
        create_core_files()
        
        # Crear archivos del domain
        create_domain_files()
        
        # Crear archivos del application
        create_application_files()
        
        # Crear archivos del presentation
        create_presentation_files()
        
        # Crear archivos de testing
        create_test_files()
        
        # Crear archivos de configuración
        create_config_files()
        
        # Crear archivos de Docker
        create_docker_files()
        
        # Crear archivo main.py
        create_main_file()
        
        # Crear archivos de requirements
        create_requirements()
        
        print("\n🎉 ESTRUCTURA MODULAR COMPLETADA!")
        print("\n📋 Archivos creados:")
        print("  ✅ Estructura de directorios completa")
        print("  ✅ Módulo core (config, exceptions, utils)")
        print("  ✅ Módulo domain (entities, services, repositories)")
        print("  ✅ Módulo application (use cases, DTOs, validators)")
        print("  ✅ Módulo presentation (API, CLI, web)")
        print("  ✅ Módulo infrastructure (database, cache, external)")
        print("  ✅ Módulo plugins (analyzers, exporters, integrations)")
        print("  ✅ Tests (unit, integration, e2e)")
        print("  ✅ Configuración (development, production)")
        print("  ✅ Docker (Dockerfile, docker-compose)")
        print("  ✅ Requirements (main, dev)")
        print("  ✅ main.py")
        
        print("\n🚀 Próximos pasos:")
        print("  1. cd ai_history_comparison")
        print("  2. pip install -r requirements.txt")
        print("  3. pip install -r requirements-dev.txt")
        print("  4. python main.py")
        
        print("\n📚 Documentación:")
        print("  📖 MODULAR_ARCHITECTURE.md - Arquitectura completa")
        print("  📖 MODULAR_IMPLEMENTATION_GUIDE.md - Guía de implementación")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()







