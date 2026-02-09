# Arquitectura Modular V7.0 - Plugin System

## Resumen

Arquitectura completamente modular con sistema de plugins, carga dinámica de módulos y factory pattern para dependency injection.

## Componentes Principales

### 1. Plugin System (`core/plugin_system.py`)

Sistema de plugins que permite:
- Registro dinámico de plugins
- Carga automática desde directorios
- Inicialización y shutdown controlados
- Validación de configuración
- Tipos de plugins: middleware, router, service, database, cache, etc.

**Uso:**
```python
from core.plugin_system import BasePlugin, PluginMetadata, PluginType, get_plugin_registry

class MyPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-plugin",
            version="1.0.0",
            plugin_type=PluginType.SERVICE,
            description="My custom plugin"
        )
    
    async def initialize(self, config=None):
        # Initialize plugin
        pass
    
    async def shutdown(self):
        # Cleanup
        pass

# Register plugin
registry = get_plugin_registry()
registry.register(MyPlugin())

# Initialize
await registry.initialize_plugin("service:my-plugin", {"enabled": True})
```

### 2. Module Loader (`core/module_loader.py`)

Carga dinámica de módulos con:
- Lazy loading para optimizar cold start
- Carga desde archivos
- Descubrimiento automático de módulos
- Cache de módulos cargados

**Uso:**
```python
from core.module_loader import get_module_loader

loader = get_module_loader()

# Lazy load (optimized)
router = loader.load_module("api.routers.analysis_router", lazy=True)

# Direct load
module = loader.load_module("services.image_processor", lazy=False)

# Discover modules
modules = loader.discover_modules("api/routers", recursive=True)
```

### 3. Service Factory (`core/service_factory.py`)

Factory pattern para dependency injection:
- Singleton, Request, Transient scopes
- Resolución automática de dependencias
- Factory functions support
- Lifecycle management

**Uso:**
```python
from core.service_factory import get_service_factory, ServiceScope

factory = get_service_factory()

# Register service
factory.register(
    "image_processor",
    ImageProcessor,
    scope=ServiceScope.SINGLETON,
    dependencies=["cache", "database"]
)

# Create instance (dependencies auto-resolved)
processor = await factory.create("image_processor")
```

## Estructura Modular

```
dermatology_ai/
├── core/                    # Core modular components
│   ├── plugin_system.py     # Plugin registry and management
│   ├── module_loader.py     # Dynamic module loading
│   └── service_factory.py   # Dependency injection factory
│
├── plugins/                  # Plugin directory
│   ├── __init__.py
│   └── example_plugin.py    # Example plugin template
│
├── api/                      # API layer (modular routers)
│   └── routers/             # Individual router modules
│
├── services/                 # Business logic (modular services)
│   └── *.py                 # Individual service modules
│
├── utils/                    # Utilities (modular utilities)
│   └── *.py                 # Individual utility modules
│
└── main.py                   # Application entry point
```

## Ventajas de la Arquitectura Modular

### 1. Separación de Responsabilidades
- Cada módulo tiene una responsabilidad única
- Fácil de entender y mantener
- Testing aislado

### 2. Extensibilidad
- Plugins para agregar funcionalidad sin modificar core
- Módulos intercambiables
- Hot-swappable components

### 3. Performance
- Lazy loading reduce cold start
- Carga solo lo necesario
- Cache inteligente

### 4. Testabilidad
- Módulos pueden ser mockeados fácilmente
- Testing aislado por módulo
- Dependency injection facilita testing

### 5. Escalabilidad
- Módulos pueden escalarse independientemente
- Fácil migrar a microservicios
- Load balancing por módulo

## Ejemplos de Uso

### Crear un Plugin

```python
# plugins/custom_auth_plugin.py
from core.plugin_system import BasePlugin, PluginMetadata, PluginType

class CustomAuthPlugin(BasePlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom-auth",
            version="1.0.0",
            plugin_type=PluginType.AUTHENTICATION,
            description="Custom authentication plugin"
        )
    
    async def initialize(self, config=None):
        # Setup authentication
        pass
    
    async def shutdown(self):
        # Cleanup
        pass
```

### Registrar y Usar Plugin

```python
from core.plugin_system import get_plugin_registry

registry = get_plugin_registry()

# Auto-discover plugins
registry.load_from_directory("plugins")

# Initialize all authentication plugins
await registry.initialize_all(PluginType.AUTHENTICATION)
```

### Crear Servicio Modular

```python
# services/modular_service.py
from core.service_factory import get_service_factory, ServiceScope

class ModularService:
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database

# Register
factory = get_service_factory()
factory.register(
    "modular_service",
    ModularService,
    scope=ServiceScope.SINGLETON,
    dependencies=["cache", "database"]
)
```

## Mejores Prácticas

### 1. Plugin Design
- Un plugin = una responsabilidad
- Configuración validada
- Cleanup apropiado en shutdown
- Documentación clara

### 2. Module Loading
- Usar lazy loading cuando sea posible
- Preload solo módulos críticos
- Clear cache cuando sea necesario

### 3. Service Registration
- Usar scopes apropiados
- Declarar dependencias explícitamente
- Evitar dependencias circulares

### 4. Testing
- Testear plugins aisladamente
- Mockear dependencias
- Testear lifecycle (init/shutdown)

## Migración desde V6.x

### Paso 1: Identificar Módulos
```python
# Identificar componentes que pueden ser plugins
# - Middlewares
# - Routers
# - Services
# - Database adapters
```

### Paso 2: Convertir a Plugins
```python
# Antes
class MyService:
    pass

# Después
class MyServicePlugin(BasePlugin):
    @property
    def metadata(self):
        return PluginMetadata(...)
```

### Paso 3: Registrar
```python
# En main.py o startup
registry = get_plugin_registry()
registry.register(MyServicePlugin())
await registry.initialize_all()
```

## Conclusión

La arquitectura V7.0 proporciona:

- ✅ **Máxima Modularidad**: Sistema de plugins completo
- ✅ **Extensibilidad**: Fácil agregar nuevas funcionalidades
- ✅ **Performance**: Lazy loading y optimizaciones
- ✅ **Mantenibilidad**: Separación clara de responsabilidades
- ✅ **Testabilidad**: Componentes aislados y mockeables
- ✅ **Escalabilidad**: Módulos independientes y escalables

El sistema está ahora completamente modular y listo para crecimiento y extensión.















