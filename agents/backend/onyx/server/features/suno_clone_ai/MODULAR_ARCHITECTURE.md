# Arquitectura Modular - Suno Clone AI

Este documento describe la arquitectura modular del sistema y cómo extenderlo.

## 🏗️ Principios de Diseño

### 1. Separación de Responsabilidades
Cada módulo tiene una responsabilidad única y bien definida.

### 2. Inversión de Dependencias
Los módulos dependen de abstracciones (interfaces), no de implementaciones concretas.

### 3. Inyección de Dependencias
Las dependencias se inyectan en lugar de crearse internamente.

### 4. Sistema de Eventos
Comunicación desacoplada mediante eventos.

### 5. Sistema de Plugins
Extensibilidad mediante plugins.

## 📦 Estructura Modular

```
core/
├── interfaces.py          # Interfaces y contratos
├── factories.py           # Factories para crear servicios
├── dependency_injection.py # Sistema de DI
├── events.py              # Sistema de eventos
├── storage/               # Backends de almacenamiento
│   ├── local_storage.py
│   └── __init__.py
├── plugins/               # Sistema de plugins
│   ├── plugin_manager.py
│   ├── base_plugin.py
│   └── __init__.py
└── modules/               # Sistema de módulos
    ├── module_registry.py
    ├── base_module.py
    └── __init__.py
```

## 🔌 Interfaces

### IMusicGenerator
```python
from core.interfaces import IMusicGenerator

class MyGenerator(IMusicGenerator):
    async def generate(self, prompt: str, **kwargs):
        # Implementación
        pass
```

### ICacheManager
```python
from core.interfaces import ICacheManager

class MyCache(ICacheManager):
    async def get(self, key: str):
        # Implementación
        pass
```

## 🏭 Factories

### MusicGeneratorFactory
```python
from core.factories import MusicGeneratorFactory

# Crear generador por tipo
generator = MusicGeneratorFactory.create_generator(
    generator_type="fast"
)
```

### CacheFactory
```python
from core.factories import CacheFactory

# Crear caché por tipo
cache = CacheFactory.create_cache(
    cache_type="distributed"
)
```

## 💉 Dependency Injection

### Registro de Servicios
```python
from core.dependency_injection import get_container

container = get_container()
container.register("music_generator", generator, singleton=True)
```

### Resolución de Dependencias
```python
from core.dependency_injection import resolve_dependency

generator = resolve_dependency("music_generator")
```

### Decorador @inject
```python
from core.dependency_injection import inject

@inject("music_generator")
async def my_function(music_generator):
    result = await music_generator.generate("prompt")
```

## 📡 Sistema de Eventos

### Publicar Eventos
```python
from core.events import get_event_bus, Event, EventType

bus = get_event_bus()
event = Event(
    event_type=EventType.MUSIC_GENERATED,
    data={"song_id": "123"}
)
await bus.publish(event)
```

### Suscribirse a Eventos
```python
from core.events import event_handler, EventType

@event_handler(EventType.MUSIC_GENERATED)
async def handle_music_generated(event):
    print(f"Music generated: {event.data['song_id']}")
```

## 🔌 Sistema de Plugins

### Crear un Plugin
```python
from core.plugins import BasePlugin

class MyPlugin(BasePlugin):
    def __init__(self):
        super().__init__("my_plugin", "1.0.0")
    
    async def _on_initialize(self, config):
        # Inicialización
        return True
    
    async def _on_shutdown(self):
        # Limpieza
        pass
```

### Registrar Plugin
```python
from core.plugins import get_plugin_manager

manager = get_plugin_manager()
plugin = MyPlugin()
manager.register_plugin(plugin, config={"key": "value"})
await manager.initialize_plugin("my_plugin")
```

## 📦 Sistema de Módulos

### Crear un Módulo
```python
from core.modules import BaseModule

class MyModule(BaseModule):
    def __init__(self):
        super().__init__("my_module", "1.0.0")
    
    async def _on_initialize(self, config):
        # Inicialización
        return True
```

### Registrar Módulo
```python
from core.modules import get_module_registry

registry = get_module_registry()
module = MyModule()
registry.register_module(
    "my_module",
    module,
    config={"key": "value"},
    dependencies=["other_module"]
)
await registry.initialize_module("my_module")
```

## 🗄️ Storage Backends

### LocalStorage
```python
from core.storage import LocalStorage

storage = LocalStorage(base_path="storage")
await storage.save("song.wav", audio_data)
data = await storage.load("song.wav")
```

### Implementar Nuevo Backend
```python
from core.interfaces import IStorageBackend

class S3Storage(IStorageBackend):
    async def save(self, path: str, data: bytes):
        # Implementación S3
        pass
```

## 🎯 Ejemplo Completo

```python
# 1. Crear servicio con DI
from core.dependency_injection import get_container
from core.factories import MusicGeneratorFactory

container = get_container()
generator = MusicGeneratorFactory.create_generator("fast")
container.register("music_generator", generator)

# 2. Suscribirse a eventos
from core.events import event_handler, EventType

@event_handler(EventType.MUSIC_GENERATED)
async def on_music_generated(event):
    print(f"New music: {event.data}")

# 3. Usar servicio con DI
from core.dependency_injection import resolve_dependency

generator = resolve_dependency("music_generator")
result = await generator.generate("happy music")
```

## 📈 Beneficios

1. **Testabilidad**: Fácil mockear dependencias
2. **Extensibilidad**: Agregar funcionalidad sin modificar código existente
3. **Mantenibilidad**: Código más limpio y organizado
4. **Reutilización**: Componentes reutilizables
5. **Desacoplamiento**: Componentes independientes

## 🚀 Próximos Pasos

- Implementar más backends de storage (S3, GCS, Azure)
- Agregar más factories para otros servicios
- Crear módulos predefinidos para funcionalidades comunes
- Sistema de configuración modular
- Health checks por módulo

