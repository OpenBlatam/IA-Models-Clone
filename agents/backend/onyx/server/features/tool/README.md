# General Tools Module

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Type](https://img.shields.io/badge/module-shared--utilities-blue.svg)
![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Shared utilities and common tool abstractions for the Onyx Server ecosystem, providing unified models, schemas, and RESTful services.**

[Overview](#-overview) •
[Features](#-key-features) •
[Structure](#-structure) •
[Installation](#-installation) •
[Usage](#-usage) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

**General Tools** is a foundational module that provides reusable component logic and shared abstractions across the entire server repository. It ensures that common tasks—from data transformation to utility orchestration—follow a standardized pattern consisting of strict Pydantic validation and clean service separation.

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Shared Utilities** | Centralized repository for common system-wide functions. |
| **RESTful Interface** | Uniform API endpoints for tool management and execution. |
| **Logic Decoupling** | Clean separation of business logic from infrastructure. |
| **Schema Validation** | Enforced data integrity via strictly typed Pydantic models. |

## 📁 Structure

```
tool/
├── api.py                # REST API endpoints for tool access
├── models.py             # Shared data models (ORM)
├── schemas.py           # Pydantic validation schemas
└── service.py           # Business logic and tool implementations
```

## ⚡ Usage

```python
from tool.service import ToolService
from tool.schemas import ToolCreate

# Initialize the shared tool service
service = ToolService()

# Register or execute a new utility tool
tool = service.create(ToolCreate(
    name="Log Analytics Utility",
    type="analytics_engine",
    config={"threshold": 0.8}
))
```

## 🔗 Integration

This module is a mandatory dependency for:
- **Integration System**: For common tool discovery.
- **Cross-Component Utilities**: Any feature requiring standardized helper functions.

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
