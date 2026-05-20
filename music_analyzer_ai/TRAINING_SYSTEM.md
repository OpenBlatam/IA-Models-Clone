# Training System - Music Analyzer AI v2.2.0

## Resumen

Se ha implementado un sistema completo de entrenamiento con capacidades avanzadas de deep learning, fine-tuning, experiment tracking y visualización interactiva.

## Componentes del Sistema de Entrenamiento

### 1. Data Loader Optimizado (`training/data_loader.py`)

Sistema de carga de datos eficiente:

- ✅ **MusicDataset**: Dataset PyTorch con caching
- ✅ **MusicDataLoader**: DataLoader optimizado
- ✅ **Caching**: Cache de features para acelerar entrenamiento
- ✅ **Augmentation**: Data augmentation para mejor generalización
- ✅ **Weighted Sampling**: Para datasets desbalanceados
- ✅ **Train/Val/Test Split**: División automática

**Características**:
```python
from training.data_loader import MusicDataset, MusicDataLoader, create_train_val_test_split

# Create splits
train, val, test = create_train_val_test_split(samples, train_ratio=0.7)

# Create dataset
train_dataset = MusicDataset(train, cache_dir="./cache", augment=True)

# Create loader
loader = MusicDataLoader(batch_size=32, num_workers=4, pin_memory=True)
train_loader = loader.create_loader(train_dataset)
```

### 2. Training System (`training/trainer.py`)

Sistema completo de entrenamiento:

- ✅ **Optimized Training Loop**: Loop de entrenamiento optimizado
- ✅ **Mixed Precision**: Entrenamiento con float16
- ✅ **Learning Rate Scheduling**: Cosine, Step, Plateau
- ✅ **Early Stopping**: Parada temprana automática
- ✅ **Gradient Clipping**: Prevención de exploding gradients
- ✅ **Checkpointing**: Guardado automático de modelos
- ✅ **Multiple Optimizers**: Adam, AdamW, SGD

**Características**:
```python
from training.trainer import MusicModelTrainer, TrainingConfig
from training.experiment_tracker import ExperimentTracker

# Config
config = TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    use_mixed_precision=True,
    early_stopping_patience=10
)

# Tracker
tracker = ExperimentTracker("experiment_001", use_wandb=True)

# Trainer
trainer = MusicModelTrainer(model, config, device="cuda", experiment_tracker=tracker)

# Train
history = trainer.train(train_loader, val_loader, criterion)
```

### 3. Experiment Tracking (`training/experiment_tracker.py`)

Sistema de tracking de experimentos:

- ✅ **Weights & Biases**: Integración con wandb
- ✅ **TensorBoard**: Integración con TensorBoard
- ✅ **JSON Logging**: Logging custom en JSONL
- ✅ **Hyperparameter Logging**: Tracking de hiperparámetros
- ✅ **Model Architecture Logging**: Visualización de modelos

**Características**:
```python
from training.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(
    experiment_name="genre_classifier_v1",
    use_wandb=True,
    use_tensorboard=True,
    project_name="music_analyzer_ai"
)

# Log metrics
tracker.log({
    "train_loss": 0.5,
    "val_loss": 0.4,
    "accuracy": 0.85
}, step=epoch)

# Log hyperparameters
tracker.log_hyperparameters({
    "learning_rate": 0.001,
    "batch_size": 32,
    "model": "DeepGenreClassifier"
})
```

### 4. LoRA Fine-tuning (`training/lora_finetuning.py`)

Fine-tuning eficiente con LoRA:

- ✅ **Low-Rank Adaptation**: Fine-tuning con mínimos parámetros
- ✅ **PEFT Integration**: Integración con PEFT library
- ✅ **Parameter Efficiency**: Solo ~1% de parámetros entrenables
- ✅ **Memory Efficient**: Requiere menos memoria que fine-tuning completo

**Características**:
```python
from training.lora_finetuning import LoRATransformerFineTuner

# Fine-tune transformer with LoRA
fine_tuner = LoRATransformerFineTuner(
    model=transformer_model,
    rank=8,
    alpha=16.0,
    target_modules=["q_proj", "v_proj"]
)

# Get LoRA model
lora_model = fine_tuner.get_model()

# Save only LoRA weights (much smaller)
fine_tuner.save_lora_weights("./lora_weights")
```

### 5. Gradio Interface (`gradio/music_analyzer_ui.py`)

Interfaz web interactiva:

- ✅ **Audio Upload**: Subida de archivos de audio
- ✅ **Spotify Integration**: Análisis de tracks de Spotify
- ✅ **Real-time Visualization**: Visualizaciones en tiempo real
- ✅ **Model Inference**: Inferencia de modelos
- ✅ **Results Display**: Visualización de resultados

**Características**:
```python
from gradio.music_analyzer_ui import MusicAnalyzerGradioUI

ui = MusicAnalyzerGradioUI()
ui.launch(server_port=7860)
```

## Flujo de Entrenamiento Completo

```
1. Data Preparation
   ├─ Load audio files
   ├─ Extract features
   └─ Create train/val/test splits

2. Dataset Creation
   ├─ MusicDataset with caching
   ├─ Data augmentation
   └─ Weighted sampling (if needed)

3. Model Initialization
   ├─ Create model architecture
   ├─ Initialize weights
   └─ Move to device

4. Training Setup
   ├─ Create optimizer
   ├─ Create scheduler
   ├─ Setup experiment tracking
   └─ Configure mixed precision

5. Training Loop
   ├─ Train epoch
   ├─ Validate
   ├─ Update scheduler
   ├─ Log metrics
   ├─ Save checkpoint
   └─ Early stopping check

6. Model Evaluation
   ├─ Test set evaluation
   ├─ Generate metrics
   └─ Save final model
```

## Optimizaciones Implementadas

### 1. Mixed Precision Training
- Entrenamiento con float16
- Reducción de memoria hasta 50%
- Aceleración en GPUs modernas

### 2. Gradient Accumulation
- Soporte para batch sizes grandes
- Acumulación de gradientes
- Entrenamiento estable

### 3. DataLoader Optimizado
- Multi-worker loading
- Pin memory para GPU
- Prefetching
- Persistent workers

### 4. Caching de Features
- Cache de features extraídas
- Reducción de tiempo de carga
- Reutilización entre epochs

## Nuevas Dependencias

```txt
wandb>=0.15.0          # Experiment tracking
tensorboard>=2.13.0    # TensorBoard integration
peft>=0.4.0            # LoRA fine-tuning
gradio>=4.0.0          # Interactive UI
matplotlib>=3.7.0      # Visualization
seaborn>=0.12.0        # Statistical plots
```

## Uso del Sistema de Entrenamiento

### Entrenamiento Básico

```python
from training.trainer import MusicModelTrainer, TrainingConfig
from training.data_loader import MusicDataset, MusicDataLoader
from training.experiment_tracker import ExperimentTracker
from core.deep_models import DeepGenreClassifier
import torch.nn as nn

# Create model
model = DeepGenreClassifier(input_size=169, num_genres=10)

# Create datasets
train_dataset = MusicDataset(train_samples, augment=True)
val_dataset = MusicDataset(val_samples, augment=False)

# Create loaders
loader = MusicDataLoader(batch_size=32, num_workers=4)
train_loader = loader.create_loader(train_dataset)
val_loader = loader.create_loader(val_dataset)

# Setup training
config = TrainingConfig(
    epochs=100,
    learning_rate=0.001,
    use_mixed_precision=True
)

tracker = ExperimentTracker("genre_classifier_v1", use_wandb=True)
trainer = MusicModelTrainer(model, config, device="cuda", experiment_tracker=tracker)

# Train
criterion = nn.CrossEntropyLoss()
history = trainer.train(train_loader, val_loader, criterion)
```

### Fine-tuning con LoRA

```python
from training.lora_finetuning import LoRATransformerFineTuner

# Load pre-trained transformer
transformer_model = load_pretrained_model()

# Apply LoRA
fine_tuner = LoRATransformerFineTuner(
    transformer_model,
    rank=8,
    alpha=16.0
)

lora_model = fine_tuner.get_model()

# Train with LoRA (only ~1% parameters trainable)
trainer = MusicModelTrainer(lora_model, config)
trainer.train(train_loader, val_loader, criterion)

# Save LoRA weights
fine_tuner.save_lora_weights("./lora_weights")
```

### Gradio Interface

```python
from gradio.music_analyzer_ui import MusicAnalyzerGradioUI

ui = MusicAnalyzerGradioUI()
ui.launch(server_port=7860, share=False)
```

Acceder en: `http://localhost:7860`

## Métricas y Evaluación

### Métricas Implementadas
- **Loss**: Cross-entropy, MSE
- **Accuracy**: Classification accuracy
- **F1 Score**: Para multi-class
- **Confusion Matrix**: Para análisis detallado

### Experiment Tracking
- **Weights & Biases**: Dashboard completo
- **TensorBoard**: Visualización de métricas
- **JSON Logs**: Logs estructurados

## Checkpointing

### Guardado Automático
- Checkpoints por epoch
- Mejor modelo (best model)
- Estado completo (model, optimizer, scheduler)
- Historial de entrenamiento

### Estructura de Checkpoints
```
checkpoints/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── ...
└── best_model.pt
```

## Próximos Pasos

1. ✅ Sistema de entrenamiento implementado
2. ✅ Experiment tracking configurado
3. ✅ LoRA fine-tuning disponible
4. ✅ Gradio interface creada
5. ⏳ Distributed training support
6. ⏳ Hyperparameter search
7. ⏳ Model ensembling
8. ⏳ AutoML capabilities

## Conclusión

El sistema de entrenamiento implementado en la versión 2.2.0 proporciona:

- ✅ **Training pipeline completo** con optimizaciones
- ✅ **Experiment tracking** con wandb y TensorBoard
- ✅ **LoRA fine-tuning** para eficiencia
- ✅ **Gradio interface** para interacción
- ✅ **DataLoader optimizado** con caching
- ✅ **Mixed precision** para aceleración
- ✅ **Checkpointing** automático
- ✅ **Early stopping** inteligente

El sistema está ahora preparado para entrenar y fine-tunear modelos de análisis musical de forma profesional y eficiente.

