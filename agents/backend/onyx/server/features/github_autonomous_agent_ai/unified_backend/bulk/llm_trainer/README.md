# LLM Trainer - Modular Custom Trainer for Language Models

## Descripción

Este es un módulo modular y extensible para entrenar modelos de lenguaje (LLMs) usando Hugging Face Transformers. La arquitectura está dividida en módulos independientes que pueden ser reutilizados y extendidos fácilmente.

## Estructura Modular Ultra-Avanzada (v2.1.0)

```
llm_trainer/
├── __init__.py                    # API pública unificada
├── trainer.py                     # Implementación principal CustomLLMTrainer
│
├── core/                          # Componentes core (interfaces, factories)
│   ├── __init__.py
│   ├── base_trainer.py           # Interface abstracta BaseLLMTrainer
│   ├── trainer_factory.py        # Factory pattern para crear trainers
│   └── config_builder.py         # Builder pattern para configuraciones
│
├── plugins/                       # 🆕 Sistema de plugins
│   ├── __init__.py
│   ├── base_plugin.py            # Base para todos los plugins
│   ├── callback_plugin.py        # Plugin para callbacks
│   └── metric_plugin.py          # Plugin para métricas
│
├── data/                          # 🆕 Componentes de datos
│   ├── __init__.py
│   ├── validators.py             # Validadores de formato y estructura
│   ├── processors.py             # Procesadores de datos
│   └── formats.py                # 🆕 Soporte multi-formato (JSON/CSV/Parquet)
│
├── training/                      # 🆕 Componentes de entrenamiento
│   ├── __init__.py
│   ├── checkpoint_manager.py     # Gestión avanzada de checkpoints
│   └── resume_manager.py         # Gestión de resume
│
├── models/                        # 🆕 Componentes de modelos
│   ├── __init__.py
│   ├── model_factory.py          # Factory de modelos
│   └── model_config.py          # Configuración de modelos
│
├── utils/                         # Utilidades y helpers
│   ├── __init__.py
│   └── helpers.py               # Funciones de utilidad
│
├── examples/                      # Ejemplos de uso
│   ├── basic_usage.py
│   ├── factory_usage.py
│   ├── builder_usage.py
│   ├── plugin_example.py        # 🆕 Ejemplo de plugins
│   ├── data_processing_example.py # 🆕 Procesamiento de datos
│   ├── modular_usage.py         # 🆕 Uso modular completo
│   └── advanced_usage.py        # 🆕 Ejemplo avanzado completo
│
├── device_manager.py              # Gestión de dispositivos (GPU/TPU/CPU)
├── dataset_loader.py              # Carga de datasets
├── tokenizer_utils.py             # Utilidades de tokenización
├── model_loader.py                # Carga de modelos
├── config.py                      # Configuración de entrenamiento
├── callbacks.py                   # Callbacks personalizados
├── metrics.py                     # Métricas de evaluación
│
└── docs/                          # Documentación
    ├── README.md
    ├── ARCHITECTURE.md
    ├── MODULAR_ARCHITECTURE.md   # 🆕 Arquitectura modular
    ├── QUICK_START.md
    └── CHANGELOG.md
```

## Nuevos Módulos (v2.0.0)

### Sistema de Plugins (`plugins/`)
- **BasePlugin**: Clase base para crear plugins personalizados
- **PluginRegistry**: Sistema de registro y gestión de plugins
- **CallbackPlugin**: Plugin para callbacks personalizados
- **MetricPlugin**: Plugin para métricas personalizadas

### Componentes de Datos (`data/`)
- **DatasetValidator**: Validación independiente de datasets
- **FormatValidator**: Validación de formatos (extensible)
- **DatasetProcessor**: Procesamiento y transformación de datos
- **TextProcessor**: Procesamiento de texto personalizable
- **DatasetFormatLoader**: Soporte multi-formato (JSON/CSV/Parquet)

### Componentes de Modelos (`models/`)
- **ModelFactory**: Factory para crear modelos con diferentes configuraciones
- **ModelConfig**: Configuración de modelos usando dataclass

### Componentes de Entrenamiento (`training/`)
- **CheckpointManager**: Gestión avanzada de checkpoints
- **ResumeManager**: Gestión inteligente de resume

## Módulos Base

### 1. DeviceManager (`device_manager.py`)
- Detecta y configura GPU, TPU, Apple Silicon (MPS) o CPU
- Proporciona información sobre capacidades del dispositivo
- Detecta soporte para BF16
- Calcula tamaños de batch recomendados

### 2. DatasetLoader (`dataset_loader.py`)
- Carga datasets JSON con formato prompt-response
- Valida formato y contenido
- Divide automáticamente en train/validation
- Proporciona estadísticas del dataset

### 3. TokenizerUtils (`tokenizer_utils.py`)
- Carga y configura tokenizers pre-entrenados
- Maneja padding tokens y special tokens
- Tokeniza para modelos causales y seq2seq
- Proporciona funciones de tokenización para inferencia

### 4. ModelLoader (`model_loader.py`)
- Carga modelos causales (GPT-2, GPT-Neo, etc.)
- Carga modelos seq2seq (T5, BART, etc.)
- Ajusta embeddings de tokens si es necesario
- Maneja diferentes tipos de datos (float16, float32)

### 5. TrainingConfig (`config.py`)
- Configura TrainingArguments de Hugging Face
- Calcula automáticamente warmup steps
- Configura precisión mixta basada en el dispositivo
- Ajusta parámetros según capacidades del hardware

### 6. TrainingProgressCallback (`callbacks.py`)
- Callback personalizado para logging
- Registra métricas de entrenamiento
- Información de inicio y fin de entrenamiento
- Logs de pérdida por época

### 7. CustomLLMTrainer (`trainer.py`)
- Clase principal que integra todos los módulos
- Hereda de `transformers.Trainer`
- Proporciona interfaz simplificada
- Métodos: `train()`, `evaluate()`, `predict()`, `save_model()`

## Formas de Uso

### Opción 1: Uso Directo (Clásico)

```python
from llm_trainer import CustomLLMTrainer

trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    output_dir="./checkpoints",
    learning_rate=3e-5,
    num_train_epochs=3,
    batch_size=8
)

trainer.train()
```

### Opción 2: Factory Pattern (Presets)

```python
from llm_trainer import TrainerFactory

factory = TrainerFactory()

# Trainer básico
trainer = factory.create_basic_trainer(
    model_name="gpt2",
    dataset_path="data/training.json"
)

# Trainer avanzado con evaluación y early stopping
trainer = factory.create_advanced_trainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    enable_early_stopping=True
)

# Trainer optimizado para memoria
trainer = factory.create_memory_efficient_trainer(
    model_name="gpt2",
    dataset_path="data/training.json"
)

trainer.train()
```

### Opción 3: Builder Pattern (Configuración Compleja)

```python
from llm_trainer import ConfigBuilder, CustomLLMTrainer

# API fluida para construir configuración
config = (ConfigBuilder()
    .with_model("gpt2", model_type="causal")
    .with_dataset("data/training.json")
    .with_output_dir("./checkpoints")
    .with_learning_rate(3e-5)
    .with_epochs(3)
    .with_batch_size(8)
    .with_early_stopping(patience=3)
    .with_gradient_checkpointing(True)
    .with_mixed_precision(fp16=True)
    .build())

trainer = CustomLLMTrainer(**config)
trainer.train()
```

## Uso de Módulos Individuales

### DeviceManager

```python
from llm_trainer import DeviceManager

manager = DeviceManager()
device = manager.get_device()
info = manager.get_device_info()
print(f"Using: {device}, Info: {info}")
```

### DatasetLoader

```python
from llm_trainer import DatasetLoader

loader = DatasetLoader("data.json")
data = loader.load()
stats = loader.get_statistics(data)
dataset = loader.prepare_dataset(data)
```

### TokenizerUtils

```python
from llm_trainer import TokenizerUtils

utils = TokenizerUtils("gpt2", model_type="causal")
tokenizer = utils.get_tokenizer()
tokenized = utils.tokenize_examples({"prompt": ["Hello"], "response": ["Hi"]})
```

### ModelLoader

```python
from llm_trainer import ModelLoader, DeviceManager

device_manager = DeviceManager()
loader = ModelLoader("gpt2", "causal", device_manager)
model = loader.load()
info = loader.get_model_info()
```

### TrainingConfig

```python
from llm_trainer import TrainingConfig, DeviceManager

device_manager = DeviceManager()
config = TrainingConfig(
    output_dir="./checkpoints",
    learning_rate=3e-5,
    num_train_epochs=3,
    batch_size=8,
    device_manager=device_manager
)
args = config.get_training_args()
```

## Ventajas de la Arquitectura Modular

1. **Reutilización**: Cada módulo puede usarse independientemente
2. **Testabilidad**: Fácil de testear cada componente por separado
3. **Extensibilidad**: Fácil agregar nuevas funcionalidades
4. **Mantenibilidad**: Código organizado y fácil de entender
5. **Flexibilidad**: Puedes usar solo los módulos que necesites

## Nuevas Funcionalidades (v1.1.0)

### Validación de Calidad del Dataset
```python
from llm_trainer import DatasetLoader

loader = DatasetLoader("data.json")
data = loader.load()
quality = loader.validate_dataset_quality(data)
print(f"Quality score: {quality['quality_score']}/100")
print(f"Warnings: {quality['warnings']}")
```

### Estadísticas Mejoradas
```python
stats = loader.get_statistics(data)
print(f"Percentiles: p50={stats['prompt_length']['p50']}, "
      f"p95={stats['prompt_length']['p95']}")
```

### Dispositivo Preferido
```python
from llm_trainer import DeviceManager

# Forzar uso de CPU
manager = DeviceManager(preferred_device="cpu")

# O forzar CUDA
manager = DeviceManager(preferred_device="cuda")
```

### Utilidades de Tokenización
```python
from llm_trainer import TokenizerUtils

utils = TokenizerUtils("gpt2")
token_count = utils.get_token_count("Hello world")
truncated = utils.truncate_text(long_text, max_tokens=100)
```

### Resumen de Entrenamiento
```python
trainer = CustomLLMTrainer(...)
summary = trainer.get_training_summary()
print(summary)  # Información completa del entrenamiento
```

### Generación Mejorada
```python
# Con más opciones de control
responses = trainer.predict(
    prompts=["What is AI?"],
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)
```

## Extensibilidad

### Crear Trainer Personalizado (Implementando Interface)

```python
from llm_trainer import BaseLLMTrainer

class MyCustomTrainer(BaseLLMTrainer):
    def train(self, resume_from_checkpoint=None):
        # Tu implementación personalizada
        pass
    
    def evaluate(self, eval_dataset=None):
        # Tu implementación
        pass
    
    def predict(self, prompts, **kwargs):
        # Tu implementación
        pass
    
    # ... implementar todos los métodos requeridos
```

### Usar Componentes Individuales

```python
from llm_trainer import (
    DeviceManager, DatasetLoader, TokenizerUtils,
    ModelLoader, TrainingConfig
)

# Usar módulos de forma independiente
device_manager = DeviceManager()
dataset_loader = DatasetLoader("data.json")
tokenizer_utils = TokenizerUtils("gpt2")
# ... construir tu propio trainer
```

### Sistema de Plugins

```python
from llm_trainer import CallbackPlugin, CustomLLMTrainer

class MyCustomPlugin(CallbackPlugin):
    def __init__(self):
        super().__init__("my_plugin", "1.0.0")
    
    def handle_log(self, args, state, control, logs, **kwargs):
        if self.enabled:
            # Tu lógica personalizada
            print(f"Custom log: {logs}")

# Usar plugin
trainer = CustomLLMTrainer(...)
plugin = MyCustomPlugin()
trainer.add_callback(plugin)
```

### Usar Componentes de Datos Independientemente

```python
from llm_trainer.data import DatasetValidator, DatasetProcessor

# Validar dataset
validator = DatasetValidator()
is_valid, errors, data = validator.validate_file("data.json")
quality = validator.validate_quality(data)

# Procesar dataset
processor = DatasetProcessor()
cleaned = processor.clean_dataset(data)
filtered = processor.filter_by_length(
    cleaned,
    min_response_length=20
)
```

### Usar Model Factory

```python
from llm_trainer.models import ModelFactory
from llm_trainer import DeviceManager

device_manager = DeviceManager()
factory = ModelFactory(device_manager)

# Crear diferentes tipos de modelos
causal_model = factory.create_causal_model("gpt2")
seq2seq_model = factory.create_seq2seq_model("t5-small")
```

### Soporte Multi-Formato

```python
from llm_trainer import CustomLLMTrainer

# Auto-detecta formato por extensión
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data.csv",  # CSV, JSON, o Parquet
    output_dir="./checkpoints"
)
```

### Gestión de Checkpoints

```python
from llm_trainer import CustomLLMTrainer, CheckpointManager

trainer = CustomLLMTrainer(...)

# Obtener info de checkpoints
checkpoint_info = trainer.get_checkpoint_info()
print(f"Total checkpoints: {checkpoint_info['total_checkpoints']}")
print(f"Best checkpoint: {checkpoint_info['best_checkpoint']}")

# Limpiar checkpoints antiguos
deleted = trainer.cleanup_checkpoints(keep=3)

# Resumir desde último checkpoint
trainer.resume_from_latest()
```

## Requisitos

- Python >= 3.8
- transformers >= 4.35.0
- torch >= 2.1.0
- datasets >= 2.14.0

## Instalación

```bash
pip install transformers torch datasets
```

Para soporte TPU (opcional):
```bash
pip install torch-xla
```

## Documentación Adicional

Ver `README_CUSTOM_LLM_TRAINER.md` para documentación completa sobre uso y características.

## Autor

BUL System - 2024

