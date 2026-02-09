# Mejoras V49: Integración de Mejores Librerías

## Resumen

Se han integrado las mejores librerías y herramientas para deep learning, transformers, diffusion models y LLM development, siguiendo las mejores prácticas de la industria.

## Nuevas Integraciones

### 1. Experiment Tracking (`core/dl_experiments/`)

**Sistemas Soportados:**
- **TensorBoard**: Tracking con `torch.utils.tensorboard.SummaryWriter`
- **Weights & Biases**: Tracking completo con `wandb`
- **Tracker Base**: Interfaz común para diferentes sistemas

**Características:**
- Logging de métricas en tiempo real
- Logging de hiperparámetros
- Logging de modelos como artifacts
- Logging de imágenes
- Interfaz unificada para múltiples sistemas

**Ejemplo:**
```python
from core.dl_experiments import create_tracker, TrackerType

# TensorBoard
tracker = create_tracker(TrackerType.TENSORBOARD, "experiment_1", "robot_ai")
tracker.init(config)
tracker.log_metric("train_loss", 0.5, step=1)
tracker.log_metrics({"val_loss": 0.4, "accuracy": 0.9}, step=1)
tracker.finish()

# Weights & Biases
tracker = create_tracker(TrackerType.WANDB, "experiment_1", "robot_ai")
tracker.init(config)
tracker.log_model(model, "best_model")
tracker.finish()
```

### 2. LoRA (Low-Rank Adaptation) (`core/dl_utils/lora.py`)

**Características:**
- Fine-tuning eficiente con adaptación de bajo rango
- Configurable rank y alpha
- Soporte para dropout
- Aplicación selectiva a módulos específicos

**Ejemplo:**
```python
from core.dl_utils import apply_lora

# Aplicar LoRA a capas específicas
model = apply_lora(
    model,
    target_modules=["layer1", "layer2"],
    rank=8,
    alpha=16.0,
    dropout=0.1
)
```

### 3. P-tuning (`core/dl_utils/ptuning.py`)

**Características:**
- Fine-tuning eficiente de LLMs con prompts aprendibles
- Encoder LSTM para generar embeddings de prefijo
- Configuración flexible

**Ejemplo:**
```python
from core.dl_utils import apply_ptuning, PTuningConfig

config = PTuningConfig(
    prefix_length=10,
    hidden_size=512,
    num_layers=2
)

model = apply_ptuning(model, config)
```

### 4. Diffusion Models (`core/diffusion_models.py`)

**Modelos Soportados:**
- Stable Diffusion
- Stable Diffusion XL
- Schedulers: DDIM, DDPM

**Características:**
- Carga de modelos desde HuggingFace
- Generación de imágenes desde texto
- Soporte para FP16
- Configuración flexible de parámetros

**Ejemplo:**
```python
from core.diffusion_models import get_diffusion_manager, DiffusionModelType

manager = get_diffusion_manager()
pipeline_id = manager.load_pipeline(
    DiffusionModelType.STABLE_DIFFUSION,
    "runwayml/stable-diffusion-v1-5"
)

image = manager.generate(
    pipeline_id,
    prompt="A robot moving in a factory",
    num_inference_steps=50,
    guidance_scale=7.5
)
```

### 5. Progress Bars (`core/dl_utils/progress.py`)

**Características:**
- Wrapper para `tqdm`
- Progress bar especializado para entrenamiento
- Muestra métricas en tiempo real

**Ejemplo:**
```python
from core.dl_utils import TrainingProgressBar

with TrainingProgressBar(total_epochs=100) as pbar:
    for epoch in range(100):
        # ... entrenamiento ...
        pbar.update_epoch(epoch, train_loss=0.5, val_loss=0.4, lr=0.001)
```

### 6. Profiling (`core/dl_utils/profiling.py`)

**Características:**
- Profiling con `torch.profiler`
- Exportación a Chrome trace format
- Profiling de forward pass de modelos
- Context managers para profiling simple

**Ejemplo:**
```python
from core.dl_utils import Profiler, profile_function

# Profiling avanzado
profiler = Profiler(use_cuda=True)
with profiler.profile(record_shapes=True, profile_memory=True) as prof:
    # ... código a perfilar ...
    pass
profiler.export_chrome_trace(prof, "trace.json")

# Profiling simple
with profile_function("my_function"):
    # ... código ...
    pass

# Profiling de modelo
stats = profile_model_forward(model, (1, 6), device="cuda")
```

### 7. Checkpoint Management (`core/dl_utils/checkpointing.py`)

**Características:**
- Gestión completa de checkpoints
- Guardado de modelo, optimizer y scheduler
- Carga de checkpoints
- Búsqueda del mejor checkpoint según métrica

**Ejemplo:**
```python
from core.dl_utils import CheckpointManager

manager = CheckpointManager("./checkpoints")

# Guardar
checkpoint_id = manager.save(
    model, epoch=10, loss=0.5,
    metrics={"val_loss": 0.4, "accuracy": 0.9},
    optimizer=optimizer,
    scheduler=scheduler
)

# Cargar
checkpoint = manager.load(checkpoint_id, model, optimizer, scheduler)

# Mejor checkpoint
best_id = manager.get_best_checkpoint("val_loss", minimize=True)
```

## Dependencias Recomendadas

```python
# Core
torch>=2.0.0
transformers>=4.30.0
diffusers>=0.21.0

# Experiment Tracking
tensorboard>=2.13.0
wandb>=0.15.0

# Utilities
tqdm>=4.65.0
numpy>=1.24.0
gradio>=3.40.0

# Optional
accelerate>=0.20.0  # Para distributed training
peft>=0.4.0         # Para LoRA avanzado
```

## Flujo de Trabajo Completo

```python
# 1. Crear modelo
from core.dl_models import ModelFactory, ModelType
model = ModelFactory.create_model(ModelType.TRAJECTORY_PREDICTOR, config)

# 2. Aplicar LoRA (opcional)
from core.dl_utils import apply_lora
model = apply_lora(model, ["network.0", "network.2"], rank=8)

# 3. Preparar datos
from core.dl_data import create_train_val_loaders
train_loader, val_loader = create_train_val_loaders(inputs, targets)

# 4. Inicializar tracker
from core.dl_experiments import create_tracker, TrackerType
tracker = create_tracker(TrackerType.WANDB, "exp1", "robot_ai")
tracker.init(config)

# 5. Entrenar con progress bar
from core.dl_training import ModelTrainer, TrainingConfig
from core.dl_utils import TrainingProgressBar, CheckpointManager

trainer = ModelTrainer()
checkpoint_manager = CheckpointManager("./checkpoints")

with TrainingProgressBar(total_epochs=100) as pbar:
    for epoch in range(100):
        # ... entrenamiento ...
        
        # Loggear métricas
        tracker.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)
        
        # Guardar checkpoint
        if epoch % 10 == 0:
            checkpoint_manager.save(
                model, epoch, train_loss,
                {"val_loss": val_loss},
                optimizer, scheduler
            )
        
        # Actualizar progress bar
        pbar.update_epoch(epoch, train_loss, val_loss, lr)

# 6. Evaluar
from core.dl_evaluation import ModelEvaluator
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, val_loader)

# 7. Finalizar
tracker.finish()
```

## Mejoras Técnicas

### Experiment Tracking
- **TensorBoard**: Integración nativa con PyTorch
- **Weights & Biases**: Tracking completo con artifacts
- **Interfaz Unificada**: Mismo código para diferentes sistemas

### Fine-tuning Eficiente
- **LoRA**: Reducción de parámetros entrenables
- **P-tuning**: Prompts aprendibles para LLMs
- **Configuración Flexible**: Parámetros ajustables

### Diffusion Models
- **Stable Diffusion**: Modelos pre-entrenados
- **Schedulers**: DDIM, DDPM
- **Generación de Imágenes**: Desde texto a imagen

### Utilidades
- **Progress Bars**: Feedback visual durante entrenamiento
- **Profiling**: Identificación de cuellos de botella
- **Checkpointing**: Gestión robusta de checkpoints

## Compatibilidad

Todas las nuevas funcionalidades son opcionales y el sistema funciona sin ellas (con funcionalidad limitada). Las librerías se detectan automáticamente.

## Próximos Pasos

- Integrar más técnicas de fine-tuning (QLoRA, AdaLoRA)
- Agregar más modelos de difusión
- Mejorar experiment tracking con más métricas
- Optimización automática de hiperparámetros

## Estado

✅ **Completado y listo para producción**

Todas las integraciones están completas, documentadas y listas para uso en producción.

