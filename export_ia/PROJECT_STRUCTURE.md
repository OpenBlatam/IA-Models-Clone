# Export IA - Project Structure

## 📁 **Organización del Proyecto**

```
export_ia/
├── 📁 app/                          # Aplicación principal
│   ├── 📁 core/                     # Lógica de negocio
│   │   ├── 📄 engine.py             # Motor principal
│   │   ├── 📄 models.py             # Modelos de datos
│   │   ├── 📄 config.py             # Configuración
│   │   ├── 📄 task_manager.py       # Gestión de tareas
│   │   └── 📄 quality_manager.py    # Gestión de calidad
│   ├── 📁 api/                      # API REST
│   │   ├── 📄 main.py               # Aplicación FastAPI
│   │   ├── 📄 routes/               # Rutas de la API
│   │   │   ├── 📄 export.py         # Rutas de exportación
│   │   │   ├── 📄 tasks.py          # Rutas de tareas
│   │   │   └── 📄 health.py         # Rutas de salud
│   │   └── 📄 middleware/           # Middleware
│   │       └── 📄 cors.py           # CORS middleware
│   ├── 📁 exporters/                # Exportadores
│   │   ├── 📄 base.py               # Clase base
│   │   ├── 📄 pdf.py                # Exportador PDF
│   │   ├── 📄 docx.py               # Exportador DOCX
│   │   ├── 📄 html.py               # Exportador HTML
│   │   ├── 📄 markdown.py           # Exportador Markdown
│   │   └── 📄 factory.py            # Factory de exportadores
│   ├── 📁 services/                 # Servicios
│   │   ├── 📄 export_service.py     # Servicio de exportación
│   │   ├── 📄 validation_service.py # Servicio de validación
│   │   └── 📄 file_service.py       # Servicio de archivos
│   └── 📁 utils/                    # Utilidades
│       ├── 📄 helpers.py            # Funciones auxiliares
│       ├── 📄 validators.py         # Validadores
│       └── 📄 formatters.py         # Formateadores
├── 📁 config/                       # Configuración
│   ├── 📄 settings.py               # Configuración principal
│   ├── 📄 database.py               # Configuración de BD
│   └── 📄 logging.py                # Configuración de logs
├── 📁 database/                     # Base de datos
│   ├── 📄 models.py                 # Modelos de BD
│   ├── 📄 migrations/               # Migraciones
│   └── 📄 connection.py             # Conexión a BD
├── 📁 tests/                        # Pruebas
│   ├── 📁 unit/                     # Pruebas unitarias
│   ├── 📁 integration/              # Pruebas de integración
│   └── 📁 fixtures/                 # Datos de prueba
├── 📁 docs/                         # Documentación
│   ├── 📄 README.md                 # Documentación principal
│   ├── 📄 API.md                    # Documentación de API
│   ├── 📄 DEPLOYMENT.md             # Guía de despliegue
│   └── 📄 DEVELOPMENT.md            # Guía de desarrollo
├── 📁 scripts/                      # Scripts
│   ├── 📄 setup.py                  # Script de configuración
│   ├── 📄 migrate.py                # Script de migración
│   └── 📄 deploy.py                 # Script de despliegue
├── 📁 docker/                       # Docker
│   ├── 📄 Dockerfile                # Dockerfile principal
│   ├── 📄 docker-compose.yml        # Docker Compose
│   └── 📄 docker-compose.dev.yml    # Docker Compose desarrollo
├── 📁 examples/                     # Ejemplos
│   ├── 📄 basic_usage.py            # Uso básico
│   ├── 📄 api_examples.py           # Ejemplos de API
│   └── 📄 advanced_usage.py         # Uso avanzado
├── 📁 logs/                         # Logs (generados)
├── 📁 exports/                      # Archivos exportados
├── 📄 requirements.txt              # Dependencias
├── 📄 requirements-dev.txt          # Dependencias desarrollo
├── 📄 .env.example                  # Variables de entorno ejemplo
├── 📄 .gitignore                    # Git ignore
├── 📄 pyproject.toml                # Configuración del proyecto
└── 📄 README.md                     # README principal
```

## 🎯 **Descripción de Directorios**

### **📁 app/** - Aplicación Principal
- **core/**: Lógica de negocio central
- **api/**: API REST con FastAPI
- **exporters/**: Exportadores de diferentes formatos
- **services/**: Servicios de aplicación
- **utils/**: Utilidades y funciones auxiliares

### **📁 config/** - Configuración
- Configuración centralizada del sistema
- Variables de entorno
- Configuración de base de datos y logging

### **📁 database/** - Base de Datos
- Modelos de base de datos
- Migraciones
- Conexión y configuración

### **📁 tests/** - Pruebas
- Pruebas unitarias
- Pruebas de integración
- Datos de prueba (fixtures)

### **📁 docs/** - Documentación
- Documentación completa del proyecto
- Guías de uso y desarrollo
- Documentación de API

### **📁 scripts/** - Scripts
- Scripts de configuración
- Scripts de migración
- Scripts de despliegue

### **📁 docker/** - Docker
- Configuración de contenedores
- Docker Compose para diferentes entornos

### **📁 examples/** - Ejemplos
- Ejemplos de uso
- Ejemplos de API
- Casos de uso avanzados

## 🔧 **Convenciones de Naming**

### **Archivos Python**
- **snake_case**: `export_service.py`
- **Descriptivo**: `pdf_exporter.py`
- **Claro**: `task_manager.py`

### **Directorios**
- **snake_case**: `export_services/`
- **Singular**: `exporter/` (no `exporters/`)
- **Descriptivo**: `api_routes/`

### **Clases**
- **PascalCase**: `ExportService`
- **Descriptivo**: `PDFExporter`
- **Sufijo apropiado**: `TaskManager`

### **Funciones y Variables**
- **snake_case**: `export_document()`
- **Descriptivo**: `get_task_status()`
- **Claro**: `validate_content()`

## 📋 **Estructura de Archivos**

### **Archivo Principal (app/api/main.py)**
```python
"""
Export IA API - Aplicación principal FastAPI
"""

from fastapi import FastAPI
from app.api.routes import export, tasks, health
from app.core.engine import get_export_engine

def create_app() -> FastAPI:
    """Crear aplicación FastAPI."""
    app = FastAPI(
        title="Export IA API",
        description="API para exportación de documentos",
        version="2.0.0"
    )
    
    # Incluir rutas
    app.include_router(export.router, prefix="/api/v1")
    app.include_router(tasks.router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")
    
    return app

app = create_app()
```

### **Configuración (config/settings.py)**
```python
"""
Configuración principal de la aplicación
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # API
    api_title: str = "Export IA API"
    api_version: str = "2.0.0"
    debug: bool = False
    
    # Base de datos
    database_url: str = "sqlite:///./export_ia.db"
    
    # Archivos
    exports_dir: str = "./exports"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/export_ia.log"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## 🚀 **Getting Started**

### **1. Configuración Inicial**
```bash
# Clonar repositorio
git clone https://github.com/your-org/export-ia.git
cd export-ia

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Para desarrollo
```

### **2. Configuración de Variables**
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar variables de entorno
nano .env
```

### **3. Ejecutar Aplicación**
```bash
# Desarrollo
python -m app.api.main

# Producción
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### **4. Con Docker**
```bash
# Desarrollo
docker-compose -f docker/docker-compose.dev.yml up --build

# Producción
docker-compose -f docker/docker-compose.yml up --build
```

## 📚 **Documentación**

### **Documentos Principales**
- **README.md**: Introducción y configuración rápida
- **docs/API.md**: Documentación completa de la API
- **docs/DEPLOYMENT.md**: Guía de despliegue
- **docs/DEVELOPMENT.md**: Guía de desarrollo

### **Ejemplos**
- **examples/basic_usage.py**: Uso básico del SDK
- **examples/api_examples.py**: Ejemplos de uso de la API
- **examples/advanced_usage.py**: Casos de uso avanzados

## 🎯 **Beneficios de esta Organización**

### **✅ Ventajas**
- **Clara separación de responsabilidades**
- **Fácil navegación y comprensión**
- **Escalabilidad y mantenibilidad**
- **Convenciones consistentes**
- **Documentación organizada**
- **Scripts de automatización**
- **Configuración centralizada**

### **🔧 Mantenimiento**
- **Fácil localización de archivos**
- **Cambios aislados por módulo**
- **Pruebas organizadas**
- **Documentación actualizada**
- **Scripts de automatización**

**¡Esta organización hace que el proyecto sea mucho más profesional, mantenible y fácil de trabajar!** 🚀




