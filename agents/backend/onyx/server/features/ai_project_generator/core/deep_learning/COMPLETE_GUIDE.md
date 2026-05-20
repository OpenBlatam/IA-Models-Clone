# Guía Completa - Sistema Modular de Deep Learning

## 🎯 Visión General

Este sistema modular de deep learning está completamente alineado con las mejores prácticas de PyTorch, Transformers, Diffusers y Gradio. Proporciona una arquitectura limpia, modular y extensible para proyectos de deep learning.

## 📦 Módulos Principales

### 1. Models (`models/`)

#### Modelos Base
- **BaseModel**: Clase base abstracta con checkpointing, device management
- **TransformerModel**: Implementación completa de transformer
- **CNNModel**: Arquitectura CNN configurable con bloques residuales
- **RNNModel**: RNN/LSTM/GRU con atención opcional

#### Integraciones
- **TransformersModelWrapper**: Integración con Hugging Face Transformers
- **DiffusionModelWrapper**: Integración con Hugging Face Diffusers

```python
from core.deep_learning.models import (
    TransformerModel, CNNModel, RNNModel,
    create_transformers_model, create_diffusion_model
)

# Modelo personalizado
model = TransformerModel(vocab_size=10000, d_model=512)

# Modelo pre-entrenado con LoRA
model = create_transformers_model(
    "bert-base-uncased",
    task="classification",
    num_labels=2,
    use_lora=True
)

# Modelo de difusión
diffusion = create_diffusion_model("runwayml/stable-diffusion-v1-5")
images = diffusion.generate("A beautiful landscape")
```

### 2. Data (`data/`)

#### Datasets
- **TextDataset**: Dataset para texto con tokenización
- **ImageDataset**: Dataset para imágenes con transformaciones

#### Utilidades
- **create_dataloader()**: DataLoader optimizado para GPU
- **train_val_test_split()**: División de datasets
- **get_image_augmentation()**: Pipelines de augmentación
- **Mixup/CutMix**: Augmentación avanzada

```python
from core.deep_learning.data import (
    TextDataset, ImageDataset,
    create_dataloader, train_val_test_split,
    get_image_augmentation, Mixup
)

# Crear dataset
dataset = TextDataset(texts, labels, tokenizer=tokenizer)

# Dividir dataset
train_ds, val_ds, test_ds = train_val_test_split(dataset)

# Crear DataLoader optimizado
train_loader = create_dataloader(
    train_ds,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

# Augmentación
transform = get_image_augmentation('standard', image_size=224)
mixup = Mixup(alpha=0.2)
```

### 3. Training (`training/`)

#### Entrenamiento
- **Trainer**: Loop de entrenamiento completo con mejores prácticas
- **TrainingConfig**: Configuración completa de entrenamiento
- **EarlyStopping**: Early stopping callback

#### Optimización
- **create_optimizer()**: Factory para optimizadores
- **create_scheduler()**: Factory para schedulers

#### Callbacks
- **ModelCheckpoint**: Guardado automático
- **MetricsLogger**: Logging de métricas
- **CallbackList**: Sistema de callbacks extensible

#### Distributed Training
- **setup_distributed()**: Configuración de entrenamiento distribuido
- **wrap_model_for_distributed()**: Wrapping para DDP/DP
- **DistributedSamplerWrapper**: Sampler distribuido

```python
from core.deep_learning.training import (
    Trainer, TrainingConfig, EarlyStopping,
    create_optimizer, create_scheduler,
    ModelCheckpoint, MetricsLogger, CallbackList,
    setup_distributed, wrap_model_for_distributed
)

# Configuración
config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    use_mixed_precision=True,
    gradient_accumulation_steps=2,
    early_stopping=EarlyStopping(patience=5)
)

# Optimizador y scheduler
optimizer = create_optimizer(model, 'adamw', lr=1e-4)
scheduler = create_scheduler(optimizer, 'cosine', num_epochs=10)

# Callbacks
callbacks = CallbackList([
    ModelCheckpoint(save_dir="checkpoints", monitor="val_loss"),
    MetricsLogger(tracker=tracker)
])

# Entrenar
trainer = Trainer(model, config, optimizer, scheduler)
history = trainer.train(train_loader, val_loader)
```

### 4. Evaluation (`evaluation/`)

- **Metrics**: Contenedor de métricas
- **compute_classification_metrics()**: Métricas de clasificación
- **compute_regression_metrics()**: Métricas de regresión
- **evaluate_model()**: Evaluación completa

```python
from core.deep_learning.evaluation import (
    evaluate_model, compute_classification_metrics
)

# Evaluar modelo
metrics, info = evaluate_model(
    model, test_loader, device,
    task_type='classification',
    num_classes=10
)

print(metrics.to_dict())
```

### 5. Inference (`inference/`)

- **InferenceEngine**: Motor de inferencia con error handling
- **create_gradio_app()**: Creación de apps Gradio
- **create_model_comparison_app()**: Comparación de modelos
- **create_interactive_training_app()**: Entrenamiento interactivo

```python
from core.deep_learning.inference import (
    InferenceEngine, create_gradio_app,
    create_model_comparison_app
)

# Inferencia
engine = InferenceEngine(model, device=device)
predictions = engine.predict(inputs, return_probabilities=True)

# Gradio app
app = create_gradio_app(model, inference_fn=custom_fn)
app.launch(share=True)
```

### 6. Config (`config/`)

- **ConfigManager**: Gestión de configuración YAML/JSON
- **load_config()**: Cargar configuración
- **save_config()**: Guardar configuración

```python
from core.deep_learning.config import ConfigManager

config_manager = ConfigManager()
config = config_manager.load(Path("config.yaml"))
lr = config_manager.get('training.learning_rate')
```

### 7. Utils (`utils/`)

#### Device Management
- **get_device()**: Detección automática de dispositivo
- **set_seed()**: Reproducibilidad
- **enable_anomaly_detection()**: Debugging

#### Experiment Tracking
- **ExperimentTracker**: TensorBoard y W&B

#### Profiling
- **profile_model()**: Profiling de modelo
- **check_for_bottlenecks()**: Análisis de cuellos de botella

#### Validation
- **validate_model_inputs()**: Validación de inputs
- **check_gradients()**: Verificación de gradientes
- **validate_data_loader()**: Validación de DataLoader

```python
from core.deep_learning.utils import (
    get_device, set_seed, ExperimentTracker,
    profile_model, validate_model_inputs,
    check_gradients
)

# Device y seed
device = get_device()
set_seed(42)

# Tracking
tracker = ExperimentTracker("experiment", use_tensorboard=True)

# Profiling
results = profile_model(model, (32, 3, 224, 224))

# Validación
is_valid, error = validate_model_inputs(model, sample_input)
grad_stats = check_gradients(model)
```

## 🚀 Flujo de Trabajo Completo

### 1. Configuración Inicial

```python
from core.deep_learning.utils import get_device, set_seed
from core.deep_learning.config import ConfigManager

# Reproducibilidad
set_seed(42)
device = get_device()

# Configuración
config_manager = ConfigManager()
config = config_manager.load("config.yaml")
```

### 2. Crear Modelo

```python
from core.deep_learning.models import create_model

model = create_model(
    model_type='transformer',
    config={
        'vocab_size': 10000,
        'd_model': 512,
        'num_heads': 8
    }
)
model = model.to(device)
```

### 3. Preparar Datos

```python
from core.deep_learning.data import (
    TextDataset, create_dataloader, train_val_test_split
)

dataset = TextDataset(texts, labels, tokenizer=tokenizer)
train_ds, val_ds, test_ds = train_val_test_split(dataset)

train_loader = create_dataloader(train_ds, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_ds, batch_size=32, shuffle=False)
```

### 4. Configurar Entrenamiento

```python
from core.deep_learning.training import (
    Trainer, TrainingConfig, EarlyStopping,
    create_optimizer, create_scheduler,
    ModelCheckpoint, MetricsLogger
)
from core.deep_learning.utils import ExperimentTracker

# Tracking
tracker = ExperimentTracker("my_experiment", use_tensorboard=True)

# Configuración
config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    use_mixed_precision=True,
    early_stopping=EarlyStopping(patience=5)
)

# Optimizador
optimizer = create_optimizer(model, 'adamw', lr=1e-4)
scheduler = create_scheduler(optimizer, 'cosine', num_epochs=10)

# Callbacks
callbacks = CallbackList([
    ModelCheckpoint(save_dir="checkpoints"),
    MetricsLogger(tracker=tracker)
])
```

### 5. Entrenar

```python
trainer = Trainer(model, config, optimizer, scheduler)
history = trainer.train(train_loader, val_loader)
```

### 6. Evaluar

```python
from core.deep_learning.evaluation import evaluate_model

metrics, info = evaluate_model(
    model, test_loader, device,
    task_type='classification'
)
```

### 7. Inferencia y Demo

```python
from core.deep_learning.inference import InferenceEngine, create_gradio_app

# Inferencia
engine = InferenceEngine(model, device=device)
predictions = engine.predict(inputs)

# Gradio app
app = create_gradio_app(model)
app.launch()
```

## 🎯 Mejores Prácticas Implementadas

✅ **Object-Oriented para Modelos**: Todos los modelos heredan de `BaseModel`
✅ **Functional para Data**: Pipelines funcionales para procesamiento
✅ **GPU Optimization**: pin_memory, prefetch_factor, num_workers
✅ **Mixed Precision**: AMP automático con GradScaler
✅ **Gradient Management**: Accumulation, clipping, zero_grad
✅ **Error Handling**: Try-except, NaN detection, logging
✅ **Reproducibility**: Random seeds, deterministic operations
✅ **Experiment Tracking**: TensorBoard y W&B
✅ **Profiling**: Performance analysis y bottleneck detection
✅ **Validation**: Input validation, gradient checks
✅ **Distributed Training**: DDP y DataParallel support
✅ **Modularity**: Componentes independientes y reutilizables

## 📚 Ejemplos Completos

Ver `examples/complete_example.py` para un ejemplo completo de uso.

## 🔧 Dependencias

Todas las dependencias están en `requirements.txt`:
- torch, torchvision, torchaudio
- transformers, diffusers
- gradio
- tensorboard, wandb
- numpy, pandas, pillow
- tqdm, scikit-learn

## 🎨 Extensibilidad

El sistema está diseñado para ser fácilmente extensible:

1. **Nuevos Modelos**: Heredar de `BaseModel`
2. **Nuevos Callbacks**: Heredar de `Callback`
3. **Nuevos Datasets**: Heredar de `BaseDataset`
4. **Nuevas Métricas**: Agregar a `evaluation/metrics.py`

## 📖 Documentación Adicional

- `MODULAR_ARCHITECTURE.md`: Arquitectura detallada
- `IMPROVEMENTS.md`: Lista de mejoras implementadas
- `examples/`: Ejemplos de uso



