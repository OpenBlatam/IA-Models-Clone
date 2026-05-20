# Ultra Modular Architecture

## 🏗️ Arquitectura Ultra Modular

Sistema completamente modular con módulos independientes, intercambiables y pluggables.

## 📁 Estructura de Módulos

```
addiction_recovery_ai/
├── core/                          # Core abstracciones
│   ├── interfaces.py             # Interfaces base
│   ├── module_registry.py        # Registry de módulos
│   ├── module_loader.py          # Cargador dinámico
│   └── service_container.py      # DI Container
│
├── modules/                       # Módulos de features
│   ├── base_module.py           # Clase base para módulos
│   ├── storage_module.py        # Módulo de storage
│   ├── cache_module.py          # Módulo de cache
│   ├── observability_module.py  # Módulo de observabilidad
│   ├── security_module.py       # Módulo de seguridad
│   ├── messaging_module.py      # Módulo de messaging
│   ├── api_module.py            # Módulo base de API
│   └── recovery_api_module.py   # Módulo de API de recovery
│
├── infrastructure/               # Implementaciones
├── handlers/                     # Handlers
├── api/                          # Endpoints
└── services/                     # Business logic
```

## 🔌 Sistema de Módulos

### Module Registry

Sistema centralizado para gestionar módulos:

```python
from core.module_registry import get_registry

registry = get_registry()

# Registrar módulo
from modules.storage_module import StorageModule
registry.register(StorageModule())

# Inicializar todos los módulos
registry.initialize_all()
```

### Base Module

Todos los módulos heredan de `BaseModule`:

```python
from modules.base_module import BaseModule

class MyModule(BaseModule):
    def __init__(self):
        super().__init__("my_module", "1.0.0")
    
    def get_dependencies(self) -> List[str]:
        return ["storage", "cache"]
    
    def _on_initialize(self) -> None:
        # Lógica de inicialización
        pass
    
    def _on_shutdown(self) -> None:
        # Lógica de shutdown
        pass
```

## 📦 Módulos Disponibles

### 1. Storage Module

**Responsabilidad**: Gestión de almacenamiento de datos

```python
from modules.storage_module import StorageModule

module = StorageModule()
module.initialize()
storage = module.get_storage_service()
```

**Dependencias**: Ninguna

### 2. Cache Module

**Responsabilidad**: Gestión de caché

```python
from modules.cache_module import CacheModule

module = CacheModule()
module.initialize()
cache = module.get_cache_service()
```

**Dependencias**: Ninguna

### 3. Observability Module

**Responsabilidad**: Métricas y tracing

```python
from modules.observability_module import ObservabilityModule

module = ObservabilityModule()
module.initialize()
metrics = module.get_metrics_service()
tracing = module.get_tracing_service()
```

**Dependencias**: Ninguna

### 4. Security Module

**Responsabilidad**: Autenticación y autorización

```python
from modules.security_module import SecurityModule

module = SecurityModule()
module.initialize()
auth = module.get_authentication_service()
```

**Dependencias**: Ninguna

### 5. Messaging Module

**Responsabilidad**: Colas y notificaciones

```python
from modules.messaging_module import MessagingModule

module = MessagingModule()
module.initialize()
queue = module.get_message_queue_service()
notification = module.get_notification_service()
```

**Dependencias**: Ninguna

### 6. API Module

**Responsabilidad**: Endpoints de API

```python
from modules.api_module import APIModule

module = APIModule()
module.initialize()
router = module.get_router()
```

**Dependencias**: storage, cache, security

### 7. Recovery API Module

**Responsabilidad**: Endpoints específicos de recovery

```python
from modules.recovery_api_module import RecoveryAPIModule

module = RecoveryAPIModule()
module.initialize()
router = module.get_router()
```

**Dependencias**: api, storage, cache, security, observability

## 🔄 Module Loader

### Carga Automática

```python
from core.module_loader import get_loader

loader = get_loader()

# Cargar módulos por defecto
loader.load_default_modules()

# O desde configuración
config = {
    "enabled_modules": ["storage", "cache", "security"],
    "module_paths": {
        "storage": "modules.storage_module.StorageModule"
    }
}
loader.load_from_config(config)

# O desde variables de entorno
# ENABLED_MODULES=storage,cache,security
loader.load_from_environment()

# Inicializar todos
loader.initialize_all()
```

### Carga Dinámica

```python
# Cargar módulo específico
loader.load_module("modules.custom_module.CustomModule")

# Descubrir módulos en un package
discovered = loader.load_all_from_package("modules")
```

## 🔗 Dependency Resolution

El sistema resuelve automáticamente las dependencias:

```python
# Módulo A depende de B y C
# Módulo B depende de C
# Orden de carga: C -> B -> A

registry = get_registry()
registry.register(ModuleA())  # Depende de B, C
registry.register(ModuleB())  # Depende de C
registry.register(ModuleC())  # Sin dependencias

registry.initialize_all()  # Inicializa en orden correcto
```

## 🎯 Crear Nuevo Módulo

### 1. Crear Clase de Módulo

```python
# modules/my_feature_module.py
from modules.base_module import BaseModule
from typing import List

class MyFeatureModule(BaseModule):
    def __init__(self):
        super().__init__("my_feature", "1.0.0")
    
    def get_dependencies(self) -> List[str]:
        return ["storage", "cache"]
    
    def _on_initialize(self) -> None:
        # Inicializar servicios
        from core.service_container import get_container
        container = get_container()
        # Registrar servicios
        pass
    
    def _on_shutdown(self) -> None:
        # Limpiar recursos
        pass
```

### 2. Registrar Módulo

```python
from modules.my_feature_module import MyFeatureModule
from core.module_registry import get_registry

registry = get_registry()
registry.register(MyFeatureModule())
```

### 3. Usar Módulo

```python
module = registry.get_module("my_feature")
if module and module.is_initialized():
    # Usar módulo
    pass
```

## 🔧 Configuración

### Por Archivo

```python
# config/modules.yaml
enabled_modules:
  - storage
  - cache
  - security
  - api

module_paths:
  storage: modules.storage_module.StorageModule
  cache: modules.cache_module.CacheModule
```

### Por Variables de Entorno

```bash
# Habilitar módulos específicos
export ENABLED_MODULES=storage,cache,security

# O cargar todos los módulos por defecto
# (no definir ENABLED_MODULES)
```

## 📊 Ventajas de la Arquitectura Modular

### 1. Independencia

- Cada módulo es independiente
- Puede desarrollarse y testearse por separado
- Fácil de reemplazar o actualizar

### 2. Pluggable

- Módulos intercambiables
- Fácil agregar/quitar features
- Configuración flexible

### 3. Escalabilidad

- Agregar nuevos módulos sin afectar existentes
- Módulos pueden escalarse independientemente
- Soporte para microservicios

### 4. Testabilidad

- Mockear módulos fácilmente
- Tests aislados por módulo
- Dependency injection nativo

### 5. Mantenibilidad

- Código organizado por feature
- Responsabilidades claras
- Fácil de entender y modificar

## 🚀 Uso en Producción

### Inicialización Completa

```python
from core.module_loader import get_loader
from fastapi import FastAPI

app = FastAPI()

# Cargar e inicializar módulos
loader = get_loader()
loader.load_default_modules()
loader.initialize_all()

# Incluir routers de módulos API
registry = get_registry()
for module_name in ["api", "recovery_api"]:
    module = registry.get_module(module_name)
    if module and hasattr(module, "get_router"):
        app.include_router(module.get_router())
```

### Shutdown Graceful

```python
import atexit

loader = get_loader()

# Registrar shutdown
atexit.register(loader.shutdown_all)
```

## 🧪 Testing

### Mockear Módulos

```python
from unittest.mock import Mock
from modules.base_module import BaseModule

class MockModule(BaseModule):
    def __init__(self):
        super().__init__("mock", "1.0.0")
    
    def _on_initialize(self):
        pass

# Usar en tests
registry = get_registry()
registry.register(MockModule())
```

### Tests Aislados

```python
# Test de un módulo específico
def test_storage_module():
    module = StorageModule()
    module.initialize()
    
    storage = module.get_storage_service()
    # Test storage...
    
    module.shutdown()
```

## 📝 Ejemplo Completo

```python
from fastapi import FastAPI
from core.module_loader import get_loader
from core.module_registry import get_registry

# Crear app
app = FastAPI(title="Addiction Recovery AI")

# Cargar módulos
loader = get_loader()
loader.load_default_modules()
loader.initialize_all()

# Incluir routers
registry = get_registry()
api_module = registry.get_module("recovery_api")
if api_module:
    app.include_router(api_module.get_router())

# Shutdown handler
@app.on_event("shutdown")
async def shutdown():
    loader.shutdown_all()
```

## ✅ Checklist de Modularidad

- [x] Sistema de registro de módulos
- [x] Base class para módulos
- [x] Dependency resolution automático
- [x] Module loader dinámico
- [x] Lifecycle management
- [x] Módulos independientes
- [x] Configuración flexible
- [x] Shutdown graceful
- [x] Documentación completa

---

**Arquitectura ultra modular completada** ✅

Sistema completamente modular, pluggable y escalable con módulos independientes.
