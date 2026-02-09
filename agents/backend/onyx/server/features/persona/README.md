# Persona Management System

## 📋 Descripción

Sistema para gestión de personas/perfiles con modelos, esquemas, servicios y API RESTful.

## 🚀 Características Principales

- **Gestión de Personas**: Creación y gestión de perfiles/personas
- **Modelos de Datos**: Modelos bien definidos
- **Esquemas Pydantic**: Validación de datos
- **Servicios**: Servicios de negocio
- **API RESTful**: Interfaz API para integración

## 📁 Estructura

```
persona/
├── models.py              # Modelos de datos
├── schemas.py            # Esquemas Pydantic
├── service.py            # Servicios de negocio
└── api.py                # Endpoints de API
```

## 🔧 Instalación

Este módulo requiere las dependencias del sistema principal.

## 💻 Uso

```python
from persona.service import PersonaService
from persona.schemas import PersonaCreate

# Inicializar servicio
service = PersonaService()

# Crear persona
persona = service.create(PersonaCreate(
    name="Juan Pérez",
    email="juan@example.com",
    role="content_creator"
))
```

## 🔗 Integración

Este módulo se integra con:
- **Brand Voice**: Para gestión de voces de marca
- **Business Agents**: Para agentes personalizados
- **Integration System**: Para orquestación



