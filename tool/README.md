# Tools Module

## 📋 Descripción

Módulo de herramientas generales con modelos, esquemas, servicios y API RESTful.

## 🚀 Características Principales

- **Herramientas Generales**: Sistema de herramientas compartidas
- **Modelos de Datos**: Modelos bien definidos
- **Esquemas Pydantic**: Validación de datos
- **Servicios**: Servicios de negocio
- **API RESTful**: Interfaz API para integración

## 📁 Estructura

```
tool/
├── models.py              # Modelos de datos
├── schemas.py            # Esquemas Pydantic
├── service.py            # Servicios de negocio
└── api.py                # Endpoints de API
```

## 🔧 Instalación

Este módulo requiere las dependencias del sistema principal.

## 💻 Uso

```python
from tool.service import ToolService
from tool.schemas import ToolCreate

# Inicializar servicio
service = ToolService()

# Crear herramienta
tool = service.create(ToolCreate(
    name="Herramienta de análisis",
    type="analytics",
    config={}
))
```

## 🔗 Integración

Este módulo se integra con:
- **Integration System**: Para orquestación
- Otros módulos que requieren herramientas compartidas

