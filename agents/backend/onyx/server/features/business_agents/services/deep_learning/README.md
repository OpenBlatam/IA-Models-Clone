# Deep Learning Service - Modular Architecture v2.0

## ✅ **Refactorización Completa - Todas las Mejores Prácticas Implementadas**

## 📁 Estructura Modular

```
deep_learning/
├── __init__.py                 # Main exports
├── service.py                  # Main service class (refactorizado)
├── config/
│   ├── __init__.py
│   ├── config_loader.py        # ✅ YAML configuration loader
│   └── default_config.yaml     # ✅ Default configuration
├── models/
│   ├── __init__.py
│   ├── base_model.py          # ✅ Base model class con utilidades
│   ├── cnn.py                 # ✅ CNN model
│   ├── lstm.py                # ✅ LSTM model
│   └── transformer.py         # ✅ Transformer model
├── data/
│   ├── __init__.py
│   ├── datasets.py            # ✅ Dataset classes (Simple, Text, Image)
│   └── dataloader.py          # ✅ DataLoader utilities eficientes
├── training/
│   ├── __init__.py
│   ├── trainer.py             # ✅ Training manager completo
│   ├── early_stopping.py      # ✅ Early stopping callback
│   ├── checkpoint.py          # ✅ Checkpointing utilities
│   └── lora.py                # ✅ LoRA fine-tuning support
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # ✅ Evaluation metrics
│   └── evaluator.py           # ✅ Evaluation manager
├── utils/
│   ├── __init__.py
│   ├── distributed.py         # ✅ DistributedDataParallel support
│   └── profiling.py           # ✅ Performance profiling
└── gradio_apps/
    ├── __init__.py
    └── model_demo.py          # ✅ Gradio interface
```

## 🚀 Características Implementadas

### ✅ **Arquitectura Modular**
- ✅ Separación completa de responsabilidades
- ✅ Modelos orientados a objetos (OOP)
- ✅ Pipelines funcionales para datos
- ✅ Módulos independientes y reutilizables

### ✅ **Configuración YAML**
- ✅ `ConfigLoader` para cargar configuraciones
- ✅ `default_config.yaml` con valores por defecto
- ✅ Soporte para modelos, datos, entrenamiento y evaluación
- ✅ Validación y actualización de configuraciones

### ✅ **Modelos**
- ✅ `BaseModel` con inicialización de pesos, conteo de parámetros
- ✅ `SimpleCNN` para clasificación de imágenes
- ✅ `LSTMTextClassifier` para clasificación de texto
- ✅ `TransformerEncoder` con positional encoding

### ✅ **Data Loading**
- ✅ `SimpleDataset`, `TextDataset`, `ImageDataset`
- ✅ `create_dataloader` con optimizaciones (pin_memory, persistent_workers)
- ✅ `split_dataset` para train/val/test splits
- ✅ Configuración flexible de workers y prefetch

### ✅ **Training**
- ✅ `TrainingManager` con mixed precision (AMP)
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ Learning rate scheduling (cosine, step, reduce_on_plateau, etc.)
- ✅ Early stopping con restauración de mejores pesos
- ✅ Checkpointing automático
- ✅ Detección de NaN/Inf
- ✅ Experiment tracking (TensorBoard/W&B)

### ✅ **Fine-tuning Eficiente**
- ✅ Soporte LoRA (Low-Rank Adaptation)
- ✅ Integración con PEFT
- ✅ Reducción de parámetros entrenables

### ✅ **Distributed Training**
- ✅ `DistributedDataParallel` support
- ✅ Setup y cleanup de procesos distribuidos
- ✅ Soporte para multi-GPU

### ✅ **Evaluación**
- ✅ Métricas completas (accuracy, precision, recall, F1)
- ✅ Confusion matrix
- ✅ ROC curve y AUC
- ✅ Precision-Recall curve
- ✅ `ModelEvaluator` para evaluación completa

### ✅ **Utilidades**
- ✅ Profiling de entrenamiento e inferencia
- ✅ Distributed training utilities
- ✅ Performance optimization

### ✅ **Gradio Integration**
- ✅ Interfaces interactivas para modelos
- ✅ `create_transformers_demo` - Demo para modelos HuggingFace
- ✅ `create_diffusion_demo` - Demo para generación de imágenes
- ✅ Manejo de errores y validación de inputs
- ✅ Visualización de resultados

### ✅ **HuggingFace Transformers**
- ✅ `HuggingFaceModel` - Wrapper para modelos pre-entrenados
- ✅ Soporte para BERT, GPT, T5, y otros
- ✅ Clasificación de texto
- ✅ Generación de texto
- ✅ Sequence-to-sequence tasks
- ✅ `CLIPTextEncoder` - Encoder CLIP para multi-modal

### ✅ **Diffusion Models**
- ✅ `DiffusionModel` - Wrapper para modelos de difusión
- ✅ Stable Diffusion v1.5
- ✅ Stable Diffusion XL
- ✅ Generación de imágenes desde prompts
- ✅ `DDPMTrainer` - Trainer para modelos DDPM

### ✅ **Utilidades Adicionales**
- ✅ `set_seed` - Reproducibilidad
- ✅ `get_device` - Selección automática de dispositivo
- ✅ `count_parameters` - Conteo de parámetros
- ✅ `get_model_size` - Información de tamaño
- ✅ `format_size` - Formateo de tamaños
- ✅ `save_model_summary` - Guardado de resúmenes

### ✅ **Optimizaciones de Rendimiento**
- ✅ `ModelOptimizer` - Compilación de modelos (torch.compile)
- ✅ `MemoryOptimizer` - Gestión de memoria GPU
- ✅ `InferenceOptimizer` - Optimización de inferencia
- ✅ `BatchProcessor` - Procesamiento por lotes optimizado
- ✅ `InferenceCache` - Caché para inferencia repetida
- ✅ `OptimizedInference` - Wrapper completo para inferencia optimizada
- ✅ `OptimizedTrainingManager` - Trainer con optimizaciones adicionales
- ✅ `create_optimized_dataloader` - DataLoader con auto-optimización
- ✅ `OptimizedTransformerEncoder` - Transformer con flash attention

## 📝 Uso

### Ejemplo Básico

```python
from deep_learning import DeepLearningService, ConfigLoader
import numpy as np

# Cargar configuración
config_loader = ConfigLoader("config/training.yaml")
service = DeepLearningService(config_path="config/training.yaml")

# Crear modelo
model = service.create_model("transformer", model_id="my_model")

# Crear dataset
data = np.random.randn(1000, 512)
labels = np.random.randint(0, 2, 1000)
dataset = service.create_dataset(data, labels, dataset_type="simple")

# Crear dataloaders
from deep_learning.data import create_dataloader, split_dataset
train_ds, val_ds, test_ds = split_dataset(dataset, 0.7, 0.15, 0.15)
train_loader = create_dataloader(train_ds, batch_size=32, num_workers=4)
val_loader = create_dataloader(val_ds, batch_size=32, num_workers=4)

# Entrenar
history = service.train_model(model, train_loader, val_loader, model_id="my_model")

# Evaluar
metrics = service.evaluate_model(model, val_loader)

# Guardar modelo
service.save_model(model, "checkpoints/best_model.pt")
```

### Con LoRA

```python
# Configurar LoRA en YAML o código
service.training_config.use_lora = True
service.training_config.lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["query", "key", "value"]
}

model = service.create_model("transformer")
# LoRA se aplica automáticamente
```

### Con HuggingFace Models

```python
from deep_learning.models.transformers_models import HuggingFaceModel

# Crear modelo HuggingFace
model = HuggingFaceModel(
    model_name="bert-base-uncased",
    task_type="classification",
    num_labels=2
)

# O usar el servicio
model = service.create_model("huggingface", model_id="bert_model")

# Hacer predicciones
results = model.predict(["This is great!", "This is terrible!"])
```

### Con Diffusion Models

```python
from deep_learning.models.diffusion_models import DiffusionModel

# Crear modelo de difusión
model = DiffusionModel(
    model_name="runwayml/stable-diffusion-v1-5",
    model_type="stable-diffusion"
)

# Generar imágenes
result = model.generate(
    prompt="A beautiful sunset over mountains",
    num_inference_steps=50,
    guidance_scale=7.5
)
```

### Interfaces Gradio

```python
from deep_learning.gradio_apps import create_transformers_demo, create_diffusion_demo

# Demo para Transformers
demo = create_transformers_demo(model, title="BERT Classifier")
demo.launch(share=True)

# Demo para Diffusion
demo = create_diffusion_demo(diffusion_model, title="Image Generator")
demo.launch(share=True)
```

### Optimizaciones de Rendimiento

```python
from deep_learning.utils.optimization import optimize_model_for_production
from deep_learning.utils.batch_optimization import OptimizedInference
from deep_learning.data import create_optimized_dataloader

# Optimizar modelo
model = optimize_model_for_production(
    model,
    device=device,
    compile_model=True
)

# Crear inferencia optimizada con caché
inference = OptimizedInference(
    model=model,
    device=device,
    batch_size=64,
    use_cache=True
)

# Usar DataLoader optimizado
train_loader = create_optimized_dataloader(
    dataset,
    batch_size=32,
    prefetch_factor=4
)

# Entrenar con trainer optimizado
history = service.train_model(
    model,
    train_loader,
    val_loader,
    use_optimized=True  # Usa OptimizedTrainingManager
)
```

### Distributed Training

```python
from deep_learning.utils.distributed import setup_ddp, wrap_model_ddp

# Setup DDP
setup_ddp(rank=0, world_size=2)

# Crear y entrenar modelo
model = service.create_model("transformer")
# El modelo se envuelve automáticamente con DDP si está configurado
history = service.train_model(model, train_loader, val_loader)
```

## 📋 Mejores Prácticas Implementadas

1. ✅ **Código Modular**: Separación clara de modelos, datos, entrenamiento y evaluación
2. ✅ **Archivos de Configuración**: YAML para hiperparámetros y settings
3. ✅ **Experiment Tracking**: Integración con TensorBoard y W&B
4. ✅ **Checkpointing**: Guardado automático de modelos
5. ✅ **Mixed Precision**: AMP para entrenamiento más rápido
6. ✅ **Distributed Training**: Soporte para multi-GPU
7. ✅ **Manejo de Errores**: Try/except comprehensivo
8. ✅ **Logging**: Structured logging con structlog
9. ✅ **Weight Initialization**: Métodos apropiados por tipo de capa
10. ✅ **Gradient Handling**: Clipping y accumulation
11. ✅ **Early Stopping**: Prevención de overfitting
12. ✅ **Learning Rate Scheduling**: Múltiples estrategias
13. ✅ **Fine-tuning Eficiente**: LoRA para modelos grandes
14. ✅ **Profiling**: Análisis de rendimiento
15. ✅ **Gradio Integration**: Interfaces interactivas
16. ✅ **HuggingFace Integration**: Modelos pre-entrenados
17. ✅ **Diffusion Models**: Generación de imágenes
18. ✅ **Helper Utilities**: Funciones de utilidad comunes
19. ✅ **Ejemplos Completos**: Ejemplos de uso para todos los casos
20. ✅ **Performance Optimization**: Optimizaciones avanzadas de rendimiento
21. ✅ **Model Compilation**: torch.compile para máxima velocidad
22. ✅ **Inference Caching**: Caché para inferencia repetida
23. ✅ **Batch Optimization**: Procesamiento optimizado por lotes
24. ✅ **Memory Management**: Gestión avanzada de memoria GPU

## 🎯 Dependencias

- `torch` - PyTorch framework
- `transformers` - HuggingFace transformers (opcional)
- `diffusers` - Diffusion models (opcional)
- `gradio` - Interactive interfaces
- `numpy` - Numerical operations
- `tqdm` - Progress bars
- `tensorboard` - Experiment tracking (opcional)
- `wandb` - Weights & Biases (opcional)
- `peft` - LoRA support (opcional)
- `structlog` - Structured logging (opcional)
- `orjson` - Fast JSON (opcional)
- `pyyaml` - YAML parsing

## 📚 Documentación

Cada módulo tiene documentación completa con:
- Docstrings detallados
- Ejemplos de uso
- Type hints
- Manejo de errores

## 🔧 Configuración

Ver `config/default_config.yaml` para todas las opciones de configuración disponibles.

