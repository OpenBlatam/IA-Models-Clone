# 🏗️ Arquitectura Modular Ultimate - TruthGPT Optimization Core

Este documento describe la arquitectura modular definitiva del sistema, implementando patrones avanzados de diseño para máxima modularidad, extensibilidad y mantenibilidad.

## 🎯 Principios de Arquitectura

### 1. **Separación de Responsabilidades (SoC)**
Cada módulo tiene una responsabilidad única y bien definida.

### 2. **Dependency Injection (DI)**
Componentes reciben sus dependencias desde el exterior, no las crean internamente.

### 3. **Observer Pattern (Event System)**
Comunicación desacoplada entre módulos mediante eventos.

### 4. **Plugin Architecture**
Sistema extensible mediante plugins dinámicamente cargables.

### 5. **Factory Pattern con Registro Dinámico**
Creación de objetos mediante factories con registro automático.

## 📦 Componentes Principales

### Core Module (`core/`)

#### 1. Service Registry (`service_registry.py`)
Sistema central de registro de servicios con inyección de dependencias.

**Características:**
- Registro de servicios (singleton o factory)
- Dependency injection automático
- Service container para gestión de ciclo de vida

**Ejemplo:**
```python
from core.service_registry import register_service, get_service

@register_service("model_manager", singleton=True)
class ModelManager:
    def load_model(self, name):
        return AutoModelForCausalLM.from_pretrained(name)

# Usar servicio
manager = get_service("model_manager")
model = manager.load_model("gpt2")
```

#### 2. Event System (`event_system.py`)
Sistema de eventos para comunicación desacoplada.

**Características:**
- Event types definidos
- Emisión y suscripción de eventos
- Handlers globales y específicos

**Ejemplo:**
```python
from core.event_system import EventType, on_event, emit_event

# Suscribirse a eventos
def on_training_step(event):
    print(f"Step {event.data['step']}: loss={event.data['loss']}")

on_event(EventType.TRAINING_STEP, on_training_step)

# Emitir evento
emit_event(EventType.TRAINING_STEP, {
    "step": 100,
    "loss": 0.5
})
```

#### 3. Plugin System (`plugin_system.py`)
Sistema de plugins para extensibilidad.

**Características:**
- Carga dinámica de plugins
- Gestión de dependencias entre plugins
- Activación/desactivación de plugins

**Ejemplo:**
```python
from core.plugin_system import Plugin, PluginManager

class MyPlugin(Plugin):
    @property
    def name(self):
        return "my_plugin"
    
    @property
    def version(self):
        return "1.0.0"
    
    def initialize(self, registry):
        registry.register("my_service", MyService)

# Cargar plugin
manager = get_plugin_manager()
manager.register_plugin(MyPlugin())
```

#### 4. Dynamic Factory (`dynamic_factory.py`)
Factory con registro dinámico y auto-discovery.

**Características:**
- Registro automático de componentes
- Auto-discovery desde módulos
- Validación de tipos base

**Ejemplo:**
```python
from core.dynamic_factory import DynamicFactory, register_component

# Crear factory
factory = DynamicFactory(base_class=BaseOptimizer)

# Registrar componente
@register_component("adamw")
class AdamWOptimizer(BaseOptimizer):
    pass

# Auto-registro desde módulo
factory.auto_register_from_module(optimizers_module)

# Crear instancia
optimizer = factory.create("adamw", lr=0.001)
```

### Services Layer (`core/services/`)

Servicios de alto nivel que orquestan operaciones complejas.

#### 1. ModelService
Gestiona el ciclo de vida completo de modelos.

```python
from core.services import ModelService

service = ModelService()
service.initialize()

# Cargar modelo
model = service.load_model("gpt2", {
    "torch_dtype": torch.bfloat16,
    "gradient_checkpointing": True,
})

# Optimizar
optimized = service.optimize_model(model, ["torch_compile", "quantize"])
```

#### 2. TrainingService
Orquesta entrenamiento completo.

```python
from core.services import TrainingService

service = TrainingService()
service.configure(
    config=training_config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device,
    output_dir="runs/experiment",
)

# Entrenar
for epoch in range(epochs):
    metrics = service.train_epoch(
        model, train_loader, optimizer, scheduler, scaler, epoch
    )
    
    # Evaluar
    val_metrics = service.evaluate(model, val_loader, device)
    
    # Guardar checkpoint
    if epoch % 10 == 0:
        service.save_checkpoint(model, optimizer, scheduler, scaler, epoch, path)
```

#### 3. InferenceService
Gestiona inferencia con caching y batching.

```python
from core.services import InferenceService

service = InferenceService()
service.configure(
    model=model,
    tokenizer=tokenizer,
    config={
        "use_cache": True,
        "cache_dir": "cache",
        "max_batch_size": 8,
    }
)

# Generar
text = service.generate("The future of AI is", {
    "max_new_tokens": 64,
    "temperature": 0.8,
})

# Batch generation
texts = service.generate(["Prompt 1", "Prompt 2"])
```

## 🔄 Flujo de Trabajo Modular

### 1. Inicialización

```python
from core.service_registry import ServiceRegistry, ServiceContainer
from core.event_system import EventEmitter
from core.plugin_system import PluginManager

# Crear contenedor de servicios
container = ServiceContainer()

# Configurar servicios
container.register("model_service", ModelService, singleton=True)
container.register("training_service", TrainingService, singleton=True)
container.register("inference_service", InferenceService, singleton=True)

# Inicializar plugins
plugin_manager = get_plugin_manager()
plugin_manager.load_plugins_from_directory("plugins/")

# Configurar event system
emitter = get_event_emitter()
emitter.on(EventType.TRAINING_STEP, log_training_step)
```

### 2. Entrenamiento Completo

```python
# Obtener servicios
model_service = container.get("model_service")
training_service = container.get("training_service")
config_manager = ConfigManager.load_config("configs/train.yaml")

# Cargar modelo
model = model_service.load_model(
    config_manager.model.name_or_path,
    config_manager.model.__dict__
)

# Cargar datos (usando data module)
from data import DatasetManager, DataLoaderFactory

train_texts, val_texts = DatasetManager.load_dataset(
    source=config_manager.data.source,
    dataset_name=config_manager.data.dataset,
)

train_loader = DataLoaderFactory.create_train_loader(...)
val_loader = DataLoaderFactory.create_val_loader(...)

# Configurar training
training_service.configure(
    config=config_manager.training.__dict__,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    ...
)

# Entrenar
training_service.emit(EventType.TRAINING_STARTED, {})
for epoch in range(config_manager.training.epochs):
    training_service.train_epoch(...)
    metrics = training_service.evaluate(...)
training_service.finish()
```

### 3. Inferencia con Caching

```python
# Configurar inference
inference_service = container.get("inference_service")
inference_service.configure(model, tokenizer, {
    "use_cache": True,
    "cache_dir": "cache",
})

# Generar con caching automático
result = inference_service.generate("prompt")
# Segunda llamada usa cache
result = inference_service.generate("prompt")  # From cache
```

## 🧩 Extensión mediante Plugins

### Crear un Plugin Personalizado

```python
from core.plugin_system import Plugin
from core.service_registry import ServiceRegistry

class CustomOptimizerPlugin(Plugin):
    @property
    def name(self):
        return "custom_optimizer"
    
    @property
    def version(self):
        return "1.0.0"
    
    def initialize(self, registry: ServiceRegistry):
        from optimizers.custom import CustomOptimizer
        
        # Registrar optimizador
        registry.register("custom_optimizer", CustomOptimizer)
        
        # Registrar en factory
        from core.dynamic_factory import get_factory
        factory = get_factory("optimizer_factory")
        factory.register("custom", CustomOptimizer)
```

## 📊 Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  (train_llm.py, demo_gradio_llm.py, CLI)               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   Services Layer                         │
│  ModelService | TrainingService | InferenceService     │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│                   Core Infrastructure                    │
│  ServiceRegistry | EventSystem | PluginSystem          │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│                   Domain Modules                         │
│  Models | Data | Training | Inference | Optimization   │
└─────────────────────────────────────────────────────────┘
```

## ✅ Ventajas de la Arquitectura

1. **Máxima Modularidad**: Cada componente es independiente
2. **Desacoplamiento**: Componentes se comunican mediante eventos
3. **Extensibilidad**: Plugins permiten agregar funcionalidad sin modificar código
4. **Testabilidad**: Cada componente puede testearse aisladamente
5. **Mantenibilidad**: Código organizado y claro
6. **Reutilización**: Componentes reutilizables en diferentes contextos
7. **Type Safety**: Interfaces claras con type hints

## 🚀 Ejemplo Completo

```python
"""
Ejemplo completo usando la arquitectura modular.
"""
from core.service_registry import ServiceContainer
from core.config import ConfigManager
from core.event_system import EventType, on_event
from core.services import ModelService, TrainingService
from data import DatasetManager, DataLoaderFactory

# Configurar event handlers
def on_step(event):
    print(f"Step {event.data['step']}: loss={event.data['loss']:.4f}")

on_event(EventType.TRAINING_STEP, on_step)

# Crear contenedor
container = ServiceContainer()

# Registrar servicios
container.register("model_service", ModelService, singleton=True)
container.register("training_service", TrainingService, singleton=True)

# Cargar configuración
config = ConfigManager.load_config("configs/train.yaml")

# Obtener servicios
model_service = container.get("model_service")
training_service = container.get("training_service")

# Cargar modelo
model = model_service.load_model(
    config.model.name_or_path,
    config.model.__dict__
)

# Cargar datos
train_texts, val_texts = DatasetManager.load_dataset(
    source=config.data.source,
    dataset_name=config.data.dataset,
)

tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

train_loader = DataLoaderFactory.create_train_loader(
    texts=train_texts,
    tokenizer=tokenizer,
    max_length=config.data.max_seq_len,
    batch_size=config.training.train_batch_size,
)

val_loader = DataLoaderFactory.create_val_loader(
    texts=val_texts,
    tokenizer=tokenizer,
    max_length=config.data.max_seq_len,
    batch_size=config.training.eval_batch_size,
)

# Configurar entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
scheduler = get_cosine_schedule_with_warmup(...)
scaler = GradScaler()

training_service.configure(
    config=config.training.__dict__,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device,
    output_dir=config.output_dir,
)

# Entrenar
for epoch in range(config.training.epochs):
    training_service.train_epoch(
        model, train_loader, optimizer, scheduler, scaler, epoch
    )
    
    if epoch % config.training.eval_interval == 0:
        metrics = training_service.evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: {metrics}")

training_service.finish()
```

## 📚 Referencias

- [Dependency Injection Pattern](https://martinfowler.com/articles/injection.html)
- [Observer Pattern](https://refactoring.guru/design-patterns/observer)
- [Plugin Architecture](https://en.wikipedia.org/wiki/Plugin_pattern)
- [Factory Pattern](https://refactoring.guru/design-patterns/factory-method)

---

**Estado**: ✅ Arquitectura modular completa y lista para producción


