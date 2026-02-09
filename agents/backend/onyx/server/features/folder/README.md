# Folder Management System

## 📋 Descripción

Sistema para gestión de carpetas con modelos, esquemas, servicios y API RESTful.

## 🚀 Características Principales

- **Gestión de Carpetas**: Creación y gestión de carpetas
- **Modelos de Datos**: Modelos bien definidos
- **Esquemas Pydantic**: Validación de datos
- **Servicios**: Servicios de negocio
- **API RESTful**: Interfaz API para integración

## 📁 Estructura

```
folder/
├── models.py              # Modelos de datos
├── schemas.py            # Esquemas Pydantic
├── service.py            # Servicios de negocio
├── api.py                # Endpoints de API
└── test_models.py       # Tests de modelos
```

## 🔧 Instalación

Este módulo requiere las dependencias del sistema principal. No requiere instalación separada.

## 💻 Uso Básico

```python
from folder.service import FolderService
from folder.schemas import FolderCreate

# Inicializar servicio
service = FolderService()

# Crear carpeta
folder = service.create(FolderCreate(
    name="Mi Carpeta",
    parent_id=None
))
```

## 🔗 Integración

Este módulo se integra con:
- **Integration System**: Para orquestación
- **Document Set**: Para gestión de documentos
- **Document Workflow Chain**: Para flujos de trabajo
- Otros módulos que requieren organización por carpetas



