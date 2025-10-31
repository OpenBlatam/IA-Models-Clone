# Utils - Shared Utilities

## 📋 Descripción

Utilidades compartidas para todo el ecosistema Blatam Academy, incluyendo sistemas de caché, lazy loading, dependencias, serialización, y optimizaciones de rendimiento.

## 🚀 Características Principales

- **Sistema de Caché**: Sistema avanzado de caché
- **Lazy Loading**: Sistema de carga perezosa
- **Dependency Injection**: Sistema de inyección de dependencias
- **Serialización**: Optimizaciones de serialización
- **Performance Optimization**: Optimizaciones de rendimiento
- **Async Operations**: Operaciones asíncronas
- **Middleware System**: Sistema de middleware
- **Redis Integration**: Integración con Redis
- **Pydantic Schemas**: Sistema de esquemas Pydantic

## 📁 Estructura

```
utils/
├── brand_kit/             # Brand kit utilities
└── tests/                  # Tests
```

## 🔧 Instalación

Este módulo se instala con el sistema principal.

## 💻 Uso

```python
from utils.cache_manager import CacheManager
from utils.lazy_loading_system import LazyLoadingSystem
from utils.dependency_injection_system import DependencyInjection

# Sistema de caché
cache = CacheManager()
value = cache.get("key")

# Lazy loading
loader = LazyLoadingSystem()
module = loader.load_module("module_name")

# Dependency injection
di = DependencyInjection()
service = di.get_service("ServiceName")
```

## 📚 Documentación

- [Caching System Summary](CACHING_SYSTEM_SUMMARY.md)
- [Lazy Loading Summary](LAZY_LOADING_SUMMARY.md)
- [Dependency Injection Summary](DEPENDENCY_INJECTION_SUMMARY.md)
- [Performance Optimization Summary](PERFORMANCE_OPTIMIZATION_SUMMARY.md)

## 🔗 Integración

Este módulo es utilizado por:
- Todos los módulos del sistema
- **Integration System**: Para utilidades compartidas

