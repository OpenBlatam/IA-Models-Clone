# Arquitectura Modular V2 - Dermatology AI

## 📁 Estructura de Directorios

La nueva arquitectura modular separa claramente las responsabilidades en módulos independientes:

```
dermatology_ai/
├── models/              # Modelos ML
│   ├── base.py         # Clases base e interfaces
│   ├── pytorch_models.py    # Modelos CNN personalizados
│   ├── vision_transformers.py  # Vision Transformers
│   └── ml_model_interface.py   # Interfaces legacy
│
├── data/                # Procesamiento de datos
│   ├── datasets.py      # Clases Dataset (PyTorch)
│   ├── transforms.py    # Transformaciones y aumentación
│   ├── preprocessing.py # Preprocesamiento de imágenes/videos
│   └── augmentation.py  # Pipeline de aumentación
│
├── training/            # Entrenamiento
│   ├── trainer.py       # Clase Trainer
│   ├── losses.py       # Funciones de pérdida
│   ├── optimizers.py    # Optimizadores y schedulers
│   └── metrics.py      # Métricas de evaluación
│
├── config/              # Configuraciones
│   ├── model_config.yaml  # Template de configuración
│   └── __init__.py     # Utilidades de configuración
│
└── core/                # Core business logic (existente)
```

## 🎯 Principios de Modularidad

### 1. Separación de Responsabilidades
- **Models**: Solo arquitecturas de modelos
- **Data**: Solo procesamiento de datos
- **Training**: Solo lógica de entrenamiento
- **Config**: Solo configuraciones

### 2. Interfaces Claras
Cada módulo expone interfaces bien definidas:

```python
# Models
from models import BaseModel, ModelFactory, ModelConfig

# Data
from data import SkinDataset, get_train_transforms, ImagePreprocessor

# Training
from training import Trainer, MultiTaskLoss, get_optimizer, get_scheduler

# Config
from config import load_config, save_config
```

### 3. Factory Pattern
Uso de factories para crear instancias:

```python
from models import ModelFactory

# Registrar modelo
ModelFactory.register("vit_skin", ViTSkinAnalyzer)

# Crear instancia
model = ModelFactory.create(
    "vit_skin",
    num_conditions=6,
    num_metrics=8
)
```

## 📦 Módulos Detallados

### Models (`models/`)

#### Base Classes (`models/base.py`)
- `BaseModel`: Clase abstracta base para todos los modelos
- `SkinAnalysisModel`: Base específica para análisis de piel
- `ModelFactory`: Factory para crear modelos
- `ModelConfig`: Configuración de modelos

**Características:**
- Inicialización automática de pesos
- Exportación a ONNX
- Cálculo de parámetros y tamaño del modelo

#### PyTorch Models (`models/pytorch_models.py`)
- `SkinAnalysisCNN`: CNN personalizada
- `SkinQualityRegressor`: Regresor multi-tarea
- `ConditionClassifier`: Clasificador multi-etiqueta
- `EnhancedSkinAnalyzer`: Analizador con atención
- `AttentionModule`: Módulo de auto-atención

#### Vision Transformers (`models/vision_transformers.py`)
- `VisionTransformer`: ViT desde cero
- `ViTSkinAnalyzer`: ViT multi-tarea
- `LoRAViT`: ViT con LoRA
- Componentes: `PatchEmbedding`, `MultiHeadSelfAttention`, `TransformerBlock`

### Data (`data/`)

#### Datasets (`data/datasets.py`)
- `SkinDataset`: Dataset para imágenes
- `SkinVideoDataset`: Dataset para videos
- `MultiTaskDataset`: Dataset para multi-tarea

**Características:**
- Soporte para múltiples formatos (numpy, PIL, paths)
- Cache opcional de imágenes
- Validación automática

#### Transforms (`data/transforms.py`)
- `get_train_transforms()`: Transformaciones de entrenamiento
- `get_val_transforms()`: Transformaciones de validación
- `get_test_transforms()`: Transformaciones de test
- `AugmentationPipeline`: Pipeline configurable

**Soporte:**
- Albumentations (avanzado)
- Torchvision (fallback)
- Diferentes niveles de aumentación (light, medium, strong)

#### Preprocessing (`data/preprocessing.py`)
- `ImagePreprocessor`: Preprocesamiento de imágenes
- `VideoPreprocessor`: Preprocesamiento de videos

**Funcionalidades:**
- Normalización
- Resizing
- Mejora de imagen (brightness, contrast, sharpness)
- Extracción de frames de video

### Training (`training/`)

#### Trainer (`training/trainer.py`)
- `Trainer`: Clase principal de entrenamiento

**Características:**
- Mixed precision training
- Gradient clipping
- Early stopping
- Learning rate scheduling
- Checkpointing automático
- Experiment tracking integrado

#### Losses (`training/losses.py`)
- `ConditionLoss`: Pérdida para clasificación
- `MetricLoss`: Pérdida para regresión
- `MultiTaskLoss`: Pérdida multi-tarea
- `FocalLoss`: Para desbalance de clases
- `DiceLoss`: Para segmentación

#### Optimizers (`training/optimizers.py`)
- `get_optimizer()`: Factory para optimizadores
- `get_scheduler()`: Factory para schedulers
- `get_optimizer_and_scheduler()`: Crear ambos

**Optimizadores soportados:**
- Adam, AdamW, SGD, RMSprop

**Schedulers soportados:**
- Cosine, Step, ReduceLROnPlateau, WarmupCosine, OneCycle

#### Metrics (`training/metrics.py`)
- `ClassificationMetrics`: Accuracy, Precision, Recall, F1, ROC-AUC
- `RegressionMetrics`: MSE, MAE, RMSE, R², Pearson
- `MetricCalculator`: Calculadora general de métricas

### Config (`config/`)

#### Configuration Management (`config/__init__.py`)
- `load_config()`: Cargar desde YAML
- `save_config()`: Guardar a YAML
- `get_config_value()`: Acceso con dot notation

#### Template (`config/model_config.yaml`)
Template completo con:
- Configuración del modelo
- Hiperparámetros de entrenamiento
- Configuración de datos
- Experiment tracking
- Checkpointing

## 🚀 Ejemplo de Uso Completo

### 1. Cargar Configuración

```python
from config import load_config

config = load_config("config/model_config.yaml")
```

### 2. Crear Modelo

```python
from models import ModelFactory, ModelConfig

# Opción 1: Usando Factory
model = ModelFactory.create(
    "vit_skin_analyzer",
    num_conditions=6,
    num_metrics=8
)

# Opción 2: Directamente
from models import ViTSkinAnalyzer
model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)
```

### 3. Preparar Datos

```python
from data import SkinDataset, get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

# Crear datasets
train_dataset = SkinDataset(
    images=train_images,
    labels={'conditions': train_conditions, 'metrics': train_metrics},
    transform=get_train_transforms(target_size=(224, 224)),
    cache_images=True
)

val_dataset = SkinDataset(
    images=val_images,
    labels={'conditions': val_conditions, 'metrics': val_metrics},
    transform=get_val_transforms(target_size=(224, 224))
)

# Crear data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)
```

### 4. Configurar Entrenamiento

```python
from training import Trainer, MultiTaskLoss, get_optimizer, get_scheduler

# Crear trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    use_mixed_precision=True,
    gradient_clip_val=1.0,
    early_stopping_patience=10
)

# Crear loss
loss_fn = MultiTaskLoss(
    condition_weight=1.0,
    metric_weight=1.0
)

# Crear optimizer y scheduler
optimizer = get_optimizer(
    model,
    optimizer_name="adamw",
    learning_rate=1e-4,
    weight_decay=1e-4
)

scheduler = get_scheduler(
    optimizer,
    scheduler_name="cosine",
    num_epochs=100
)
```

### 5. Entrenar

```python
trainer.fit(
    optimizer=optimizer,
    num_epochs=100,
    scheduler=scheduler,
    criterion=loss_fn,
    checkpoint_dir="./checkpoints"
)
```

## 🔧 Extensibilidad

### Agregar Nuevo Modelo

```python
from models.base import SkinAnalysisModel

class MyCustomModel(SkinAnalysisModel):
    def __init__(self, num_conditions=6, num_metrics=8):
        super().__init__(num_conditions, num_metrics, name="MyCustomModel")
        # Tu arquitectura aquí
    
    def forward(self, x):
        # Tu forward pass
        return {
            'conditions': conditions,
            'metrics': metrics
        }

# Registrar en factory
from models import ModelFactory
ModelFactory.register("my_custom", MyCustomModel)
```

### Agregar Nueva Loss

```python
from training.losses import BaseLoss

class MyCustomLoss(nn.Module):
    def forward(self, predictions, targets):
        # Tu lógica de pérdida
        return loss
```

### Agregar Nueva Métrica

```python
from training.metrics import MetricCalculator

class MyMetricCalculator(MetricCalculator):
    def calculate_custom_metric(self, predictions, targets):
        # Tu métrica personalizada
        return metric_value
```

## 📊 Ventajas de la Arquitectura Modular

1. **Mantenibilidad**: Código organizado y fácil de mantener
2. **Testabilidad**: Cada módulo se puede testear independientemente
3. **Reutilización**: Componentes reutilizables en diferentes contextos
4. **Escalabilidad**: Fácil agregar nuevos modelos, losses, métricas
5. **Claridad**: Responsabilidades bien definidas
6. **Configurabilidad**: Configuración centralizada en YAML

## 🔄 Migración desde Código Anterior

### Antes (Monolítico)
```python
from core.training import Trainer, SkinDataset
# Todo en un solo lugar
```

### Ahora (Modular)
```python
from training import Trainer
from data import SkinDataset
# Separado por responsabilidades
```

## 📝 Mejores Prácticas

1. **Usar Factory Pattern**: Para crear modelos, optimizadores, etc.
2. **Configuración en YAML**: Centralizar hiperparámetros
3. **Interfaces Claras**: Usar clases base para consistencia
4. **Documentación**: Documentar cada módulo
5. **Testing**: Testear cada módulo independientemente

## 🎓 Recursos

- Ver `ML_IMPROVEMENTS_V2.md` para detalles de ML
- Ver `config/model_config.yaml` para template de configuración
- Ver ejemplos en `examples/` (si existen)

---

**Arquitectura Modular V2 - Diseñada para Escalabilidad y Mantenibilidad**













