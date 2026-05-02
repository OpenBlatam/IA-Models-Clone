# Persona Management System

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

System for persona/profile management with models, schemas, services, and RESTful API.

## 🚀 Key Features

- **Persona Management**: Creation and management of profiles/personas
- **Data Models**: Well-defined data models
- **Pydantic Schemas**: Data validation
- **Services**: Business services
- **RESTful API**: API interface for integration

## 📁 Structure

```
persona/
├── models.py              # Data models
├── schemas.py             # Pydantic schemas
├── service.py             # Business services
└── api.py                 # API endpoints
```

## 🔧 Installation

This module requires the main system dependencies.

## 💻 Usage

```python
from persona.service import PersonaService
from persona.schemas import PersonaCreate

# Initialize service
service = PersonaService()

# Create persona
persona = service.create(PersonaCreate(
    name="John Doe",
    email="john@example.com",
    role="content_creator"
))
```

## 🔗 Integration

This module integrates with:
- **Brand Voice**: For brand voice management
- **Business Agents**: For custom agents
- **Integration System**: For orchestration

---

[← Back to Main README](../README.md)
