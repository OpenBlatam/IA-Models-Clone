# Input Prompt Management

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

System for input prompt management with models, schemas, services, and RESTful API.

## 🚀 Key Features

- **Prompt Management**: Creation and management of input prompts
- **Data Models**: Well-defined models
- **Pydantic Schemas**: Data validation
- **Services**: Business services
- **RESTful API**: API interface for integration

## 📁 Structure

```
input_prompt/
├── models.py              # Data models
├── schemas.py             # Pydantic schemas
├── service.py             # Business services
└── api.py                 # API endpoints
```

## 🔧 Installation

This module requires the main system dependencies. No separate installation required.

## 💻 Basic Usage

```python
from input_prompt.service import InputPromptService
from input_prompt.schemas import InputPromptCreate

# Initialize service
service = InputPromptService()

# Create prompt
prompt = service.create(InputPromptCreate(
    name="Example Prompt",
    content="Generate content about...",
    category="marketing"
))
```

## 🔗 Integration

This module integrates with:
- **Integration System**: For orchestration
- **AI Document Processor**: For AI processing
- **Business Agents**: For automation
- All modules requiring prompt management

---

[← Back to Main README](../README.md)
