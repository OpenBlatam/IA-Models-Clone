# Core - Core System Components

## 📋 Descripción

Componentes centrales del sistema incluyendo factories de aplicaciones, rutas declarativas, componentes funcionales, y ejemplos de lifespan y middleware.

## 🚀 Características Principales

- **App Factory**: Factory para creación de aplicaciones
- **Declarative Routes**: Sistema de rutas declarativas
- **Functional Components**: Componentes funcionales
- **Lifespan Management**: Gestión de ciclo de vida
- **Middleware Examples**: Ejemplos de middleware
- **Sync/Async Examples**: Ejemplos de sincronización/asíncrono

## 📁 Estructura

```
core/
├── app_factory.py           # Factory de aplicaciones
├── declarative_routes.py     # Rutas declarativas
├── example_declarative_app.py # Ejemplo de app declarativa
├── functional_components.py  # Componentes funcionales
├── functional_endpoints.py  # Endpoints funcionales
├── lifespan_examples.py     # Ejemplos de lifespan
├── middleware_examples.py   # Ejemplos de middleware
└── sync_async_example.py    # Ejemplo sync/async
```

## 🔧 Instalación

Este módulo se instala con el sistema principal.

## 💻 Uso

```python
from core.app_factory import create_app
from core.declarative_routes import DeclarativeRoutes

# Crear aplicación
app = create_app()

# Usar rutas declarativas
routes = DeclarativeRoutes()
routes.register(app)
```

## 🔗 Integración

Este módulo proporciona componentes base para:
- **Integration System**: Sistema principal
- Todos los módulos que requieren componentes centrales

