# Folder Management System

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

Folder management system with models, schemas, services, and RESTful API.

## 🚀 Key Features

- **Folder Management**: Creation and management of folders
- **Data Models**: Well-defined models
- **Pydantic Schemas**: Data validation
- **Services**: Business services
- **RESTful API**: API interface for integration

## 📁 Structure

```
folder/
├── models.py              # Data models
├── schemas.py             # Pydantic schemas
├── service.py             # Business services
├── api.py                 # API endpoints
└── test_models.py         # Model tests
```

## 🔧 Installation

This module requires the main system dependencies. No separate installation required.

## 💻 Basic Usage

```python
from folder.service import FolderService
from folder.schemas import FolderCreate

# Initialize service
service = FolderService()

# Create folder
folder = service.create(FolderCreate(
    name="My Folder",
    parent_id=None
))
```

## 🔗 Integration

This module integrates with:
- **Integration System**: For orchestration
- **Document Set**: For document management
- **Document Workflow Chain**: For workflows
- Other modules requiring folder organization

---

[← Back to Main README](../README.md)
