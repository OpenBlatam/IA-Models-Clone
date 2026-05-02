# Arquitectura Modular del LLM Trainer

## Estructura de Directorios

```
llm_trainer/
├── __init__.py              # API pública principal
├── trainer.py               # Implementación principal CustomLLMTrainer
│
├── core/                    # Componentes core (interfaces, factories)
│   ├── __init__.py
│   ├── base_trainer.py      # Interface abstracta BaseLLMTrainer
│   ├── trainer_factory.py   # Factory pattern para crear trainers
│   └── config_builder.py    # Builder pattern para configuraciones
│
├── utils/                   # Utilidades y helpers
│   ├── __init__.py
│   └── helpers.py          # Funciones de utilidad (validación, formato, etc.)
│
├── examples/                # Ejemplos de uso
│   ├── basic_usage.py
│   ├── factory_usage.py
│   └── builder_usage.py
│
├── device_manager.py        # Gestión de dispositivos (GPU/TPU/CPU)
├── dataset_loader.py        # Carga y validación de datasets
├── tokenizer_utils.py       # Utilidades de tokenización
├── model_loader.py          # Carga de modelos
├── config.py                # Configuración de entrenamiento
├── callbacks.py             # Callbacks personalizados
├── metrics.py              # Métricas de evaluación
│
├── README.md                # Documentación principal
├── ARCHITECTURE.md          # Este archivo
└── CHANGELOG.md             # Registro de cambios
```

## Principios de Diseño

### 1. Separación de Responsabilidades

Cada módulo tiene una responsabilidad única y bien definida:

- **device_manager**: Solo gestión de dispositivos
- **dataset_loader**: Solo carga y validación de datasets
- **tokenizer_utils**: Solo tokenización
- **model_loader**: Solo carga de modelos
- **config**: Solo configuración
- **callbacks**: Solo callbacks
- **metrics**: Solo métricas

### 2. Interfaces y Abstracciones

El módulo `core/base_trainer.py` define una interfaz abstracta que permite:

- Extensibilidad: Crear nuevos tipos de trainers
- Testabilidad: Mockear fácilmente para tests
- Polimorfismo: Usar diferentes implementaciones

### 3. Factory Pattern

El módulo `core/trainer_factory.py` proporciona:

- Presets comunes: `create_basic_trainer()`, `create_advanced_trainer()`, etc.
- Configuración centralizada
- Facilita creación de trainers con configuraciones específicas

### 4. Builder Pattern

El módulo `core/config_builder.py` permite:

- API fluida para construir configuraciones
- Validación paso a paso
- Configuraciones complejas de forma legible

### 5. Utilidades Separadas

El módulo `utils/helpers.py` contiene:

- Funciones de validación
- Funciones de formato
- Estimaciones y cálculos
- Sin dependencias de otros módulos complejos

## Flujo de Uso

### Opción 1: Uso Directo

```python
from llm_trainer import CustomLLMTrainer

trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data.json",
    output_dir="./checkpoints"
)
trainer.train()
```

### Opción 2: Factory Pattern

```python
from llm_trainer import TrainerFactory

factory = TrainerFactory()
trainer = factory.create_advanced_trainer(
    model_name="gpt2",
    dataset_path="data.json"
)
trainer.train()
```

### Opción 3: Builder Pattern

```python
from llm_trainer import ConfigBuilder, CustomLLMTrainer

config = (ConfigBuilder()
    .with_model("gpt2")
    .with_dataset("data.json")
    .with_learning_rate(3e-5)
    .with_epochs(3)
    .with_batch_size(8)
    .build())

trainer = CustomLLMTrainer(**config)
trainer.train()
```

## Extensiones

### Crear un Trainer Personalizado

```python
from llm_trainer import BaseLLMTrainer

class MyCustomTrainer(BaseLLMTrainer):
    def train(self, resume_from_checkpoint=None):
        # Tu implementación
        pass
    
    def evaluate(self, eval_dataset=None):
        # Tu implementación
        pass
    
    # ... otros métodos requeridos
```

### Agregar Nuevos Callbacks

```python
from transformers import TrainerCallback

class MyCustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Tu lógica personalizada
        pass
```

### Agregar Nuevas Métricas

```python
from llm_trainer.metrics import compute_metrics

def my_custom_metric(eval_pred):
    # Tu cálculo de métrica
    return {"my_metric": value}

# Combinar con métricas existentes
def combined_metrics(eval_pred):
    base_metrics = compute_metrics(eval_pred)
    custom_metrics = my_custom_metric(eval_pred)
    return {**base_metrics, **custom_metrics}
```

## Ventajas de esta Arquitectura

1. **Modularidad**: Cada componente es independiente y reutilizable
2. **Extensibilidad**: Fácil agregar nuevas funcionalidades
3. **Testabilidad**: Componentes aislados son fáciles de testear
4. **Mantenibilidad**: Código organizado y fácil de entender
5. **Flexibilidad**: Múltiples formas de usar el sistema
6. **Escalabilidad**: Fácil agregar nuevos módulos sin afectar existentes

## Dependencias entre Módulos

```
CustomLLMTrainer
    ├── DeviceManager (independiente)
    ├── DatasetLoader (independiente)
    ├── TokenizerUtils (independiente)
    ├── ModelLoader (depende de DeviceManager)
    ├── TrainingConfig (depende de DeviceManager)
    ├── Callbacks (independiente)
    └── Metrics (independiente)

Core/
    ├── BaseLLMTrainer (interfaz, sin dependencias)
    ├── TrainerFactory (depende de CustomLLMTrainer)
    └── ConfigBuilder (independiente)

Utils/
    └── Helpers (independiente)
```

## Mejores Prácticas

1. **Usar interfaces**: Para componentes que pueden tener múltiples implementaciones
2. **Factory para presets**: Para configuraciones comunes
3. **Builder para complejidad**: Para configuraciones complejas
4. **Validación temprana**: Validar inputs lo antes posible
5. **Logging consistente**: Usar logging apropiado en todos los módulos
6. **Documentación**: Docstrings completos en todos los componentes

