# Arquitectura Modular Ultra-Avanzada

## Estructura Completa

```
llm_trainer/
├── __init__.py                    # API pública unificada
├── trainer.py                     # Implementación principal
│
├── core/                          # Componentes core
│   ├── __init__.py
│   ├── base_trainer.py           # Interface abstracta
│   ├── trainer_factory.py        # Factory pattern
│   └── config_builder.py         # Builder pattern
│
├── plugins/                       # Sistema de plugins
│   ├── __init__.py
│   ├── base_plugin.py            # Base para plugins
│   ├── callback_plugin.py        # Plugin para callbacks
│   └── metric_plugin.py          # Plugin para métricas
│
├── data/                          # Componentes de datos
│   ├── __init__.py
│   ├── validators.py             # Validadores
│   └── processors.py             # Procesadores
│
├── models/                        # Componentes de modelos
│   ├── __init__.py
│   ├── model_factory.py          # Factory de modelos
│   └── model_config.py          # Configuración de modelos
│
├── utils/                         # Utilidades
│   ├── __init__.py
│   └── helpers.py               # Funciones helper
│
├── examples/                      # Ejemplos
│   ├── basic_usage.py
│   ├── factory_usage.py
│   └── builder_usage.py
│
├── device_manager.py              # Gestión de dispositivos
├── dataset_loader.py              # Carga de datasets
├── tokenizer_utils.py             # Utilidades de tokenización
├── model_loader.py                # Carga de modelos
├── config.py                      # Configuración
├── callbacks.py                   # Callbacks
├── metrics.py                     # Métricas
│
└── docs/                          # Documentación
    ├── README.md
    ├── ARCHITECTURE.md
    ├── MODULAR_ARCHITECTURE.md
    ├── QUICK_START.md
    └── CHANGELOG.md
```

## Principios de Modularidad

### 1. Separación por Responsabilidad

Cada directorio tiene un propósito específico:

- **core/**: Interfaces y patrones de diseño fundamentales
- **plugins/**: Sistema extensible de plugins
- **data/**: Todo relacionado con datos
- **models/**: Todo relacionado con modelos
- **utils/**: Funciones de utilidad general

### 2. Sistema de Plugins

Permite extender funcionalidad sin modificar código core:

```python
from llm_trainer import BasePlugin, PluginRegistry

class MyCustomPlugin(BasePlugin):
    def __init__(self):
        super().__init__("my_plugin", "1.0.0")
    
    def on_train_begin(self, trainer, **kwargs):
        print("Custom plugin activated!")

# Registrar plugin
registry = PluginRegistry()
registry.register(MyCustomPlugin())
```

### 3. Componentes de Datos Separados

- **Validators**: Validación de formato y estructura
- **Processors**: Procesamiento y transformación
- **Loaders**: Carga de diferentes formatos

### 4. Factory Patterns

Múltiples factories para diferentes componentes:

- `TrainerFactory`: Crear trainers
- `ModelFactory`: Crear modelos
- Futuras: `DatasetFactory`, `TokenizerFactory`

### 5. Configuración Modular

- `ModelConfig`: Configuración de modelos
- `TrainingConfig`: Configuración de entrenamiento
- `DatasetConfig`: Configuración de datasets (futuro)

## Ventajas de esta Arquitectura

### 1. Extensibilidad Máxima

```python
# Agregar nuevo plugin
from llm_trainer import CallbackPlugin

class MyPlugin(CallbackPlugin):
    def handle_log(self, args, state, control, logs, **kwargs):
        # Tu lógica personalizada
        pass
```

### 2. Testabilidad

Cada componente puede testearse independientemente:

```python
# Test de validadores
from llm_trainer.data import DatasetValidator

validator = DatasetValidator()
is_valid, errors = validator.validate_structure(data)
assert is_valid
```

### 3. Reutilización

Componentes pueden usarse en otros proyectos:

```python
# Solo usar validadores
from llm_trainer.data import DatasetValidator

validator = DatasetValidator()
# Usar sin cargar todo el trainer
```

### 4. Mantenibilidad

Código organizado por responsabilidad facilita mantenimiento.

### 5. Escalabilidad

Fácil agregar nuevos módulos sin afectar existentes.

## Ejemplos de Uso Modular

### Usar Solo Validadores

```python
from llm_trainer.data import DatasetValidator
import json

validator = DatasetValidator()
with open("data.json") as f:
    data = json.load(f)

is_valid, errors, cleaned_data = validator.validate_file("data.json")
```

### Usar Solo Model Factory

```python
from llm_trainer.models import ModelFactory
from llm_trainer import DeviceManager

device_manager = DeviceManager()
factory = ModelFactory(device_manager)

model = factory.create_causal_model("gpt2")
```

### Crear Plugin Personalizado

```python
from llm_trainer import CallbackPlugin

class CustomLoggingPlugin(CallbackPlugin):
    def __init__(self):
        super().__init__("custom_logging", "1.0.0")
    
    def handle_log(self, args, state, control, logs, **kwargs):
        if self.enabled and logs:
            # Enviar a sistema externo
            send_to_external_system(logs)
```

### Usar Processors

```python
from llm_trainer.data import DatasetProcessor

processor = DatasetProcessor()
cleaned = processor.clean_dataset(raw_data)
filtered = processor.filter_by_length(
    cleaned,
    min_response_length=20,
    max_response_length=500
)
```

## Extensión Futura

La arquitectura permite fácilmente agregar:

- Nuevos tipos de plugins
- Nuevos formatos de datos
- Nuevos tipos de modelos
- Nuevos procesadores
- Nuevos validadores
- Nuevos factories

## Dependencias entre Módulos

```
CustomLLMTrainer
    ├── core/ (interfaces)
    ├── plugins/ (extensiones)
    ├── data/ (validación y procesamiento)
    ├── models/ (creación de modelos)
    ├── utils/ (helpers)
    └── [módulos base] (device, dataset, tokenizer, etc.)

Todos los módulos son independientes y pueden usarse solos.
```

