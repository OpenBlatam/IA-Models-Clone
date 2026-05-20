# Core - Core System Components

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

Central system components including application factories, declarative routes, functional components, and lifespan and middleware examples.

## 🚀 Key Features

- **App Factory**: Factory for application creation
- **Declarative Routes**: Declarative routing system
- **Functional Components**: Functional components
- **Lifespan Management**: Lifecycle management
- **Middleware Examples**: Middleware examples
- **Sync/Async Examples**: Sync/Async examples

## 📁 Structure

```
core/
├── app_factory.py           # Application factory
├── declarative_routes.py     # Declarative routes
├── example_declarative_app.py # Declarative app example
├── functional_components.py  # Functional components
├── functional_endpoints.py  # Functional endpoints
├── lifespan_examples.py     # Lifespan examples
├── middleware_examples.py   # Middleware examples
└── sync_async_example.py    # Sync/async example
```

## 🔧 Installation

This module is installed with the main system.

## 💻 Usage

```python
from core.app_factory import create_app
from core.declarative_routes import DeclarativeRoutes

# Create application
app = create_app()

# Use declarative routes
routes = DeclarativeRoutes()
routes.register(app)
```

## 🔗 Integration

This module provides base components for:
- **Integration System**: Main system
- All modules requiring core components

---

[← Back to Main README](../README.md)
