# Document Set Management

## 📋 Descripción

Sistema para gestión de conjuntos de documentos con modelos, repositorios, servicios y esquemas bien definidos.

## 🚀 Características Principales

- **Gestión de Conjuntos**: Creación y gestión de conjuntos de documentos
- **Repositorios**: Sistema de repositorios para acceso a datos
- **Servicios**: Servicios de negocio para operaciones
- **Esquemas**: Esquemas Pydantic para validación
- **API RESTful**: Interfaz API para integración

## 📁 Estructura

```
document_set/
├── models.py              # Modelos de datos
├── repositories.py        # Repositorios de datos
├── schemas.py             # Esquemas Pydantic
├── service.py             # Servicios de negocio
├── api.py                 # Endpoints de API
└── router.py              # Rutas de API
```

## 🔧 Instalación

Este módulo requiere las dependencias del sistema principal. No requiere instalación separada.

## 💻 Uso Básico

```python
from document_set.service import DocumentSetService
from document_set.schemas import DocumentSetCreate

# Inicializar servicio
service = DocumentSetService()

# Crear conjunto de documentos
document_set = service.create(DocumentSetCreate(
    name="Mi Conjunto",
    description="Descripción del conjunto"
))
```

## 🔗 Integración

Este módulo se integra con:
- **Integration System**: Para orquestación
- **Document Workflow Chain**: Para flujos de trabajo
- **AI Document Processor**: Para procesamiento con IA
- **Export IA**: Para exportación de conjuntos

