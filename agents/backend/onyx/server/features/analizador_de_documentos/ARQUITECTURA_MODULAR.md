# Arquitectura Modular - Document Analyzer

## Resumen

Sistema modular completo para el Document Analyzer con mejor organización, separación de responsabilidades y extensibilidad.

---

## Componentes Modulares

### 1. Module Manager

**Gestor de módulos con gestión de dependencias.**

```python
from analizador_de_documentos.core.modular import ModuleManager

manager = ModuleManager()

# Registrar módulo
manager.register_module(
    module_id="analysis",
    name="Analysis Module",
    factory=lambda: AnalysisModule(),
    dependencies=["core"],
    description="Módulo de análisis"
)

# Cargar módulo
analysis_module = await manager.load_module("analysis")

# Cargar todos los módulos en orden
loaded_modules = await manager.load_all_modules()
```

**Características:**
- ✅ Gestión de dependencias automática
- ✅ Orden de carga topológico
- ✅ Estado de módulos
- ✅ Factory pattern

### 2. Plugin System

**Sistema de plugins extensible.**

```python
from analizador_de_documentos.core.modular import PluginSystem, PluginInterface

class CustomPlugin(PluginInterface):
    def get_plugin_info(self):
        return {
            "id": "custom_plugin",
            "name": "Custom Plugin",
            "version": "1.0.0"
        }
    
    async def initialize(self, context):
        # Inicialización
        pass
    
    async def execute(self, *args, **kwargs):
        # Ejecución
        return result
    
    async def cleanup(self):
        # Limpieza
        pass

# Registrar plugin
plugin_system = PluginSystem()
plugin_system.register_plugin(
    plugin_id="custom_plugin",
    name="Custom Plugin",
    version="1.0.0",
    description="Plugin personalizado",
    author="Developer",
    plugin_type="analysis",
    plugin_class=CustomPlugin
)

# Inicializar
await plugin_system.initialize_all_plugins(context={"analyzer": analyzer})

# Ejecutar
result = await plugin_system.execute_plugin("custom_plugin", arg1, arg2)
```

**Características:**
- ✅ Interfaz de plugin estándar
- ✅ Inicialización automática
- ✅ Gestión de ciclo de vida
- ✅ Tipos de plugins

### 3. Service Locator

**Patrón Service Locator para inyección de dependencias.**

```python
from analizador_de_documentos.core.modular import ServiceLocator

locator = ServiceLocator()

# Registrar servicio (singleton)
locator.register(
    service_id="cache",
    service_type=CacheService,
    instance=CacheService(),
    singleton=True
)

# Registrar servicio con factory
locator.register(
    service_id="analyzer",
    service_type=DocumentAnalyzer,
    factory=lambda: DocumentAnalyzer(),
    singleton=True
)

# Obtener servicio
cache = locator.get("cache")
analyzer = locator.get_by_type(DocumentAnalyzer)
```

**Características:**
- ✅ Inyección de dependencias
- ✅ Singleton pattern
- ✅ Factory pattern
- ✅ Búsqueda por tipo

### 4. Event Bus

**Sistema de eventos para comunicación desacoplada.**

```python
from analizador_de_documentos.core.modular import EventBus, Event, EventType

bus = EventBus()

# Suscribirse a evento
async def on_document_analyzed(event: Event):
    print(f"Documento analizado: {event.payload['document_id']}")

bus.subscribe(EventType.DOCUMENT_ANALYZED, on_document_analyzed)

# Publicar evento
await bus.publish_sync(
    EventType.DOCUMENT_ANALYZED,
    payload={"document_id": "doc_123", "score": 0.95},
    source="analyzer"
)

# Obtener historial
history = bus.get_event_history(EventType.DOCUMENT_ANALYZED, limit=10)
```

**Características:**
- ✅ Comunicación desacoplada
- ✅ Múltiples suscriptores
- ✅ Historial de eventos
- ✅ Eventos asíncronos

---

## Integración en DocumentAnalyzer

```python
from analizador_de_documentos.core.document_analyzer import DocumentAnalyzer
from analizador_de_documentos.core.modular import ModuleManager, PluginSystem, ServiceLocator, EventBus

class ModularDocumentAnalyzer(DocumentAnalyzer):
    def __init__(self):
        super().__init__()
        
        # Inicializar componentes modulares
        self.module_manager = ModuleManager()
        self.plugin_system = PluginSystem()
        self.service_locator = ServiceLocator()
        self.event_bus = EventBus()
        
        # Registrar servicios
        self.service_locator.register(
            "analyzer",
            DocumentAnalyzer,
            instance=self,
            singleton=True
        )
        
        # Suscribirse a eventos
        self.event_bus.subscribe(
            EventType.DOCUMENT_ANALYZED,
            self._on_document_analyzed
        )
    
    async def _on_document_analyzed(self, event: Event):
        # Manejar evento
        pass
```

---

## Beneficios de la Arquitectura Modular

### 1. Separación de Responsabilidades
- Cada módulo tiene una responsabilidad clara
- Fácil de entender y mantener
- Cambios aislados

### 2. Extensibilidad
- Plugins fáciles de agregar
- Módulos intercambiables
- Configuración flexible

### 3. Testabilidad
- Módulos testables independientemente
- Mocks fáciles con Service Locator
- Eventos para testing

### 4. Escalabilidad
- Carga bajo demanda
- Módulos opcionales
- Gestión de recursos

### 5. Mantenibilidad
- Código organizado
- Dependencias claras
- Documentación automática

---

## Ejemplo Completo

```python
from analizador_de_documentos.core.modular import (
    ModuleManager, PluginSystem, ServiceLocator, EventBus, EventType
)

# Inicializar sistema modular
manager = ModuleManager()
plugins = PluginSystem()
services = ServiceLocator()
events = EventBus()

# Registrar módulos
manager.register_module("core", "Core Module", lambda: CoreModule())
manager.register_module("analysis", "Analysis Module", lambda: AnalysisModule(), dependencies=["core"])

# Registrar servicios
services.register("cache", CacheService, instance=CacheService())

# Registrar plugins
plugins.register_plugin(
    "custom_analyzer",
    "Custom Analyzer",
    "1.0.0",
    "Análisis personalizado",
    "Developer",
    "analysis",
    CustomAnalyzerPlugin
)

# Cargar módulos
await manager.load_all_modules()

# Inicializar plugins
await plugins.initialize_all_plugins(context={"services": services})

# Suscribirse a eventos
events.subscribe(EventType.DOCUMENT_ANALYZED, handle_analysis)

# Usar sistema
analyzer = services.get("analyzer")
result = await analyzer.analyze_document("content...")

# Publicar evento
await events.publish_sync(
    EventType.DOCUMENT_ANALYZED,
    {"result": result}
)
```

---

## Archivos Creados

1. **`core/modular/__init__.py`**: Punto de entrada modular
2. **`core/modular/module_manager.py`**: Gestor de módulos
3. **`core/modular/plugin_system.py`**: Sistema de plugins
4. **`core/modular/service_locator.py`**: Localizador de servicios
5. **`core/modular/event_bus.py`**: Bus de eventos

---

## Resumen

El sistema ahora es completamente modular con:

- ✅ **Gestión de módulos** con dependencias
- ✅ **Sistema de plugins** extensible
- ✅ **Service Locator** para DI
- ✅ **Event Bus** para comunicación
- ✅ **Arquitectura limpia** y mantenible

**Sistema modular, extensible y listo para escalar.**


