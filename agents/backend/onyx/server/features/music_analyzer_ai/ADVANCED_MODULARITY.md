# Advanced Modularity - Music Analyzer AI

## Nuevos Componentes Modulares

### 1. **Dependency Injection Container** (`core/dependency_injection.py`)
Sistema de inyección de dependencias para desacoplar componentes:

```python
from music_analyzer_ai import register_service, get_service

# Registrar servicio
register_service("analyzer", DeepMusicAnalyzer, singleton=True)

# Obtener servicio
analyzer = get_service("analyzer", device="cuda")
```

### 2. **Event System** (`core/event_system.py`)
Sistema de eventos publish/subscribe para comunicación desacoplada:

```python
from music_analyzer_ai import subscribe, publish

# Suscribirse a eventos
def on_training_complete(event):
    print(f"Training completed: {event.data}")

subscribe("training.complete", on_training_complete)

# Publicar evento
publish("training.complete", {"epoch": 10, "loss": 0.5})
```

### 3. **Plugin System** (`plugins/`)
Sistema extensible de plugins:

```python
from music_analyzer_ai import BasePlugin, register_plugin

class MyPlugin(BasePlugin):
    def __init__(self):
        super().__init__("my_plugin", "1.0.0")
    
    def execute(self, data):
        # Procesar datos
        return processed_data

# Registrar plugin
plugin = MyPlugin()
register_plugin(plugin, config={"param": "value"})
```

### 4. **Validator System** (`validators/`)
Sistema de validación modular:

```python
from music_analyzer_ai import ModelInputValidator, ValidationResult

# Crear validador
validator = ModelInputValidator(expected_shape=(1, 169))

# Validar
result = validator.validate(tensor)
if result:
    print("Valid!")
else:
    print(f"Errors: {result.errors}")
```

### 5. **Service Interfaces** (`interfaces/service_interface.py`)
Interfaces para servicios:
- `IService` - Base para todos los servicios
- `IFeatureService` - Servicios de extracción de features
- `IAnalysisService` - Servicios de análisis
- `IRecommendationService` - Servicios de recomendación

### 6. **Plugin Interfaces** (`interfaces/plugin_interface.py`)
Interfaces para plugins:
- `IPlugin` - Base para plugins
- `IPreprocessingPlugin` - Plugins de preprocesamiento
- `IPostprocessingPlugin` - Plugins de postprocesamiento
- `IAnalysisPlugin` - Plugins de análisis

## Arquitectura Completa

```
music_analyzer_ai/
├── interfaces/          # Contratos (ABC)
│   ├── model_interface.py
│   ├── trainer_interface.py
│   ├── analyzer_interface.py
│   ├── inference_interface.py
│   ├── service_interface.py
│   └── plugin_interface.py
├── factories/          # Creación de instancias
│   ├── model_factory.py
│   ├── trainer_factory.py
│   ├── analyzer_factory.py
│   └── config_factory.py
├── base/               # Implementaciones base
│   ├── base_model.py
│   ├── base_trainer.py
│   └── base_analyzer.py
├── core/               # Componentes core
│   ├── dependency_injection.py
│   └── event_system.py
├── plugins/            # Sistema de plugins
│   ├── plugin_manager.py
│   └── base_plugin.py
└── validators/         # Sistema de validación
    ├── validator.py
    ├── model_validator.py
    └── data_validator.py
```

## Ejemplo de Uso Completo

```python
from music_analyzer_ai import (
    create_model,
    create_trainer,
    register_service,
    get_service,
    subscribe,
    publish,
    register_plugin,
    ModelInputValidator
)

# 1. Crear modelo usando factory
model = create_model("music_classifier", config={...})

# 2. Registrar servicio usando DI
register_service("model", lambda: model, singleton=True)
model = get_service("model")

# 3. Suscribirse a eventos
def on_epoch_end(event):
    print(f"Epoch {event.data['epoch']} completed")

subscribe("training.epoch_end", on_epoch_end)

# 4. Registrar plugin
class FeatureNormalizer(BasePlugin):
    def execute(self, data):
        return (data - data.mean()) / data.std()

register_plugin(FeatureNormalizer())

# 5. Validar inputs
validator = ModelInputValidator(expected_shape=(1, 169))
result = validator.validate(input_tensor)
```

## Beneficios

1. **Desacoplamiento**: Componentes independientes
2. **Extensibilidad**: Fácil agregar nuevos componentes
3. **Testabilidad**: Mocking fácil con interfaces
4. **Mantenibilidad**: Código organizado y claro
5. **Reutilización**: Componentes reutilizables
6. **Flexibilidad**: Múltiples formas de usar el sistema

## Principios de Diseño Aplicados

- **SOLID**: Todos los principios aplicados
- **Dependency Injection**: Container para DI
- **Observer Pattern**: Event system
- **Strategy Pattern**: Plugins
- **Factory Pattern**: Factories
- **Template Method**: Base classes
- **Chain of Responsibility**: Validators








