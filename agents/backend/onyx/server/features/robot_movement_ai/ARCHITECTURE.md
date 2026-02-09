# Arquitectura Mejorada del Sistema de Routing

## 🏗️ Visión General

La nueva arquitectura sigue principios de **Clean Architecture** y **Domain-Driven Design**, con separación clara de responsabilidades y patrones de diseño modernos.

## 📐 Capas de Arquitectura

```
┌─────────────────────────────────────┐
│     Presentation Layer (API)        │
├─────────────────────────────────────┤
│     Application Layer (Services)    │
├─────────────────────────────────────┤
│     Domain Layer (Entities)         │
├─────────────────────────────────────┤
│  Infrastructure Layer (Repos, Impl) │
└─────────────────────────────────────┘
```

### 1. Domain Layer (`domain.py`)

**Responsabilidad**: Entidades de dominio y value objects.

- `Route`: Entidad principal de ruta
- `RouteMetrics`: Value object para métricas
- `Node`, `Edge`, `Graph`: Entidades del grafo
- `RouteStatus`: Enum para estados

**Principios**:
- Sin dependencias externas
- Lógica de negocio pura
- Inmutabilidad donde sea posible

### 2. Application Layer (`services.py`)

**Responsabilidad**: Lógica de aplicación y orquestación.

- `RouteService`: Servicio principal de routing
- `ModelService`: Gestión de modelos
- `TrainingService`: Servicio de entrenamiento
- `InferenceService`: Servicio de inferencia

**Principios**:
- Usa interfaces (Protocols)
- Orquesta llamadas a repositorios y estrategias
- Maneja eventos

### 3. Infrastructure Layer

#### Repositories (`repositories.py`)

**Responsabilidad**: Persistencia de datos.

- `RouteRepository`: Persistencia de rutas
- `ModelRepository`: Persistencia de modelos
- `DataRepository`: Persistencia de datasets

**Principios**:
- Implementa interfaces
- Abstrae detalles de persistencia
- Fácil intercambiar implementaciones (memoria, DB, etc.)

## 🎨 Patrones de Diseño

### 1. Interface Segregation (`interfaces.py`)

**Protocols** para definir contratos:

```python
@runtime_checkable
class IRouteModel(Protocol):
    def forward(self, x: Any) -> Any: ...
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]: ...
```

**Ventajas**:
- Type safety
- Duck typing
- Fácil testing con mocks

### 2. Factory Pattern (`factories.py`)

**Factories** para crear instancias:

```python
RouteModelFactory.register("mlp", MLPModel)
model = RouteModelFactory.create("mlp", config={...})
```

**Ventajas**:
- Desacoplamiento
- Centralización de creación
- Fácil extensión

### 3. Builder Pattern (`builders.py`)

**Builders** para construcción compleja:

```python
service = (RouteServiceBuilder()
    .add_strategy("shortest", ShortestPathStrategy())
    .with_repository(repository)
    .with_event_bus(event_bus)
    .build())
```

**Ventajas**:
- Construcción flexible
- Validación en build
- Fluent interface

### 4. Repository Pattern (`repositories.py`)

**Repositories** para abstraer persistencia:

```python
repository = RouteRepository()
route_id = repository.save_route(response)
route = repository.get_route(route_id)
```

**Ventajas**:
- Abstracción de datos
- Fácil testing
- Intercambiable (memoria, DB, cache)

### 5. Event-Driven Architecture (`events.py`)

**Event Bus** para comunicación desacoplada:

```python
event_bus = EventBus()
event_bus.subscribe("route_found", handler)
event_bus.emit("route_found", data={...})
```

**Ventajas**:
- Desacoplamiento
- Extensibilidad
- Observabilidad

### 6. Plugin System (`plugins.py`)

**Plugin Manager** para extensibilidad:

```python
manager = PluginManager()
manager.register_plugin(MyStrategyPlugin())
strategy = manager.get_strategy_plugin("my_strategy").create_strategy()
```

**Ventajas**:
- Extensibilidad sin modificar código
- Carga dinámica
- Aislamiento

### 7. Dependency Injection (`dependency_injection.py`)

**Container** para gestión de dependencias:

```python
container = Container()
container.register(IRouteService, RouteService(...))
service = container.resolve(IRouteService)
```

**Ventajas**:
- Inversión de control
- Fácil testing
- Gestión de ciclo de vida

## 🔄 Flujo de Datos

### Encontrar Ruta

```
Request → RouteService → Strategy → Graph → Response
                ↓
         Repository (save)
                ↓
         EventBus (emit)
```

### Entrenar Modelo

```
Data → TrainingService → Pipeline → Model → Results
            ↓
     EventBus (training_started, completed)
```

## 📦 Estructura de Módulos

```
core/architecture/
├── __init__.py              # Exports principales
├── interfaces.py            # Protocols e interfaces
├── domain.py                # Entidades de dominio
├── services.py              # Servicios de aplicación
├── repositories.py          # Repositorios
├── factories.py             # Factories
├── builders.py              # Builders
├── plugins.py               # Plugin system
├── events.py                # Event bus
└── dependency_injection.py  # DI container
```

## 🚀 Uso Básico

### 1. Crear Servicio de Routing

```python
from core.architecture import (
    RouteServiceBuilder,
    RouteRepository,
    EventBus
)
from core.routing_strategies import ShortestPathStrategy

# Crear componentes
repository = RouteRepository()
event_bus = EventBus()
strategy = ShortestPathStrategy()

# Construir servicio
service = (RouteServiceBuilder()
    .add_strategy("shortest", strategy)
    .with_repository(repository)
    .with_event_bus(event_bus)
    .build())

# Usar servicio
from core.architecture import RouteRequest
request = RouteRequest(
    start_node="A",
    end_node="B",
    strategy="shortest"
)
response = service.find_route(request)
```

### 2. Usar Dependency Injection

```python
from core.architecture import (
    Container,
    register_service,
    resolve_service
)

# Registrar servicios
register_service(IRouteService, RouteService(...))
register_service(IRouteRepository, RouteRepository())

# Resolver
service = resolve_service(IRouteService)
repository = resolve_service(IRouteRepository)
```

### 3. Usar Event Bus

```python
from core.architecture import EventBus, EventHandler

event_bus = EventBus()

def on_route_found(event):
    print(f"Ruta encontrada: {event.data}")

handler = EventHandler(on_route_found)
event_bus.subscribe("route_found", handler)

# Emitir evento
event_bus.emit("route_found", {"route": ["A", "B", "C"]})
```

### 4. Usar Plugins

```python
from core.architecture import PluginManager, RouteStrategyPlugin

class MyStrategyPlugin(RouteStrategyPlugin):
    def get_name(self): return "my_strategy"
    def get_version(self): return "1.0.0"
    def initialize(self, config): pass
    def create_strategy(self): return MyStrategy()

manager = PluginManager()
manager.register_plugin(MyStrategyPlugin())
strategy = manager.get_strategy_plugin("my_strategy").create_strategy()
```

## ✅ Ventajas de la Nueva Arquitectura

1. **Separación de Responsabilidades**: Cada capa tiene una responsabilidad clara
2. **Testabilidad**: Fácil mockear interfaces y testear en aislamiento
3. **Extensibilidad**: Plugins y factories permiten extensión sin modificar código
4. **Mantenibilidad**: Código organizado y fácil de entender
5. **Escalabilidad**: Fácil agregar nuevas funcionalidades
6. **Type Safety**: Protocols y type hints en toda la arquitectura
7. **Desacoplamiento**: Event bus y DI reducen acoplamiento
8. **Flexibilidad**: Builders y factories permiten configuración flexible

## 🔧 Integración con Código Existente

La nueva arquitectura se integra con el código existente:

```python
# Usar modelos existentes con nueva arquitectura
from core.routing_models import MLPRoutePredictor
from core.architecture import RouteModelFactory

RouteModelFactory.register("mlp", MLPRoutePredictor)
model = RouteModelFactory.create("mlp", config={...})
```

## 📚 Próximos Pasos

1. Implementar repositorios con base de datos real
2. Agregar más plugins (estrategias, modelos)
3. Implementar pipeline pattern para procesamiento
4. Agregar validación con Pydantic
5. Implementar logging estructurado
6. Agregar testing utilities
7. Crear CLI para gestión de plugins

## 🎯 Mejores Prácticas

1. **Siempre usar interfaces**: Define Protocols para contratos
2. **Inyección de dependencias**: Usa DI container para servicios
3. **Event-driven**: Usa event bus para comunicación desacoplada
4. **Factory pattern**: Usa factories para crear instancias
5. **Repository pattern**: Abstrae persistencia con repositories
6. **Type hints**: Usa type hints en todo el código
7. **Error handling**: Maneja errores apropiadamente
8. **Logging**: Usa logging estructurado

