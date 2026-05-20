# Project Structure - Dermatology AI

## 📁 Estructura Reorganizada

```
dermatology_ai/
├── ml/                          # Machine Learning Core
│   ├── __init__.py             # Main ML exports
│   ├── models/                 # Model architectures
│   │   ├── base.py            # Base classes and interfaces
│   │   ├── pytorch_models.py  # PyTorch CNN models
│   │   ├── vision_transformers.py  # Vision Transformers
│   │   └── ml_model_interface.py   # Legacy interfaces
│   │
│   ├── training/               # Training components
│   │   ├── trainer.py         # Main trainer class
│   │   ├── losses.py          # Loss functions
│   │   ├── optimizers.py      # Optimizers and schedulers
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── distributed.py     # Distributed training
│   │
│   ├── data/                  # Data processing
│   │   ├── datasets.py        # Dataset classes
│   │   ├── transforms.py     # Transforms and augmentation
│   │   ├── preprocessing.py   # Preprocessing utilities
│   │   └── augmentation.py    # Augmentation pipeline
│   │
│   ├── experiments/           # Experiment management
│   │   ├── __init__.py        # Experiment exports
│   │   └── (links to core/experiment_tracker.py)
│   │
│   ├── inference/             # Inference engines
│   │   ├── __init__.py        # Inference exports
│   │   └── (links to utils/optimization, async_inference)
│   │
│   └── visualization/         # Visualization and demos
│       ├── __init__.py        # Visualization exports
│       └── (links to core/gradio_integration.py)
│
├── core/                       # Core Business Logic
│   ├── application/           # Use cases (hexagonal architecture)
│   ├── domain/                # Domain entities and services
│   ├── infrastructure/        # Infrastructure adapters
│   ├── composition_root.py    # Dependency injection
│   ├── experiment_tracker.py # Experiment tracking
│   ├── ml_model_manager.py    # Model management
│   └── ...                    # Other core components
│
├── utils/                      # Utilities
│   ├── optimization.py        # Basic optimizations
│   ├── advanced_optimization.py  # Advanced optimizations
│   ├── async_inference.py     # Async inference
│   ├── profiling.py           # Profiling utilities
│   ├── model_pruning.py       # Model pruning
│   └── ...                    # Other utilities
│
├── config/                     # Configuration
│   ├── settings.py            # Application settings
│   ├── model_config.yaml      # Model configuration template
│   └── __init__.py            # Config utilities
│
├── api/                        # API Layer
│   ├── controllers/           # Controllers
│   ├── routers/               # API routers
│   └── ...                    # API components
│
├── services/                   # Business Services
│   └── ...                    # Service implementations
│
├── tests/                      # Tests
│   └── ...                    # Test files
│
├── examples/                   # Examples
│   └── ...                    # Example scripts
│
├── scripts/                    # Utility scripts
│   └── ...                    # Deployment, health checks, etc.
│
└── docs/                       # Documentation
    ├── PROJECT_STRUCTURE.md   # This file
    ├── ML_IMPROVEMENTS_V2.md  # ML improvements
    ├── MODULAR_ARCHITECTURE_V2.md  # Architecture
    ├── ADVANCED_TRAINING_GUIDE.md  # Training guide
    ├── PERFORMANCE_OPTIMIZATIONS.md  # Performance
    └── ULTIMATE_OPTIMIZATION_GUIDE.md  # Ultimate optimizations
```

## 🎯 Organización por Responsabilidad

### 1. ML Module (`ml/`)
**Propósito:** Todo lo relacionado con Machine Learning

- **models/**: Arquitecturas de modelos
- **training/**: Componentes de entrenamiento
- **data/**: Procesamiento de datos
- **experiments/**: Gestión de experimentos
- **inference/**: Motores de inferencia
- **visualization/**: Demos y visualización

### 2. Core Module (`core/`)
**Propósito:** Lógica de negocio y arquitectura hexagonal

- **application/**: Casos de uso
- **domain/**: Entidades y servicios de dominio
- **infrastructure/**: Adaptadores de infraestructura
- Componentes de negocio

### 3. Utils Module (`utils/`)
**Propósito:** Utilidades y optimizaciones

- Optimizaciones de performance
- Profiling
- Model pruning
- Async inference

### 4. Config Module (`config/`)
**Propósito:** Configuraciones

- Settings de aplicación
- Templates de configuración
- Utilidades de configuración

## 📦 Imports Organizados

### Import desde ML Module

```python
# Import completo desde ML
from ml import (
    # Models
    ViTSkinAnalyzer,
    ModelFactory,
    # Training
    Trainer,
    MultiTaskLoss,
    get_optimizer,
    # Data
    SkinDataset,
    get_train_transforms,
    # Optimization
    FastInferenceEngine
)

# O imports específicos
from ml.models import ViTSkinAnalyzer
from ml.training import Trainer
from ml.data import SkinDataset
```

### Import desde Core

```python
from core.experiment_tracker import ExperimentTracker
from core.ml_model_manager import MLModelManager
```

### Import desde Utils

```python
from utils.optimization import compile_model
from utils.profiling import PerformanceMonitor
```

## 🔄 Migración de Imports

### Antes
```python
from models.pytorch_models import SkinAnalysisCNN
from training.trainer import Trainer
from data.datasets import SkinDataset
```

### Ahora (Recomendado)
```python
from ml import SkinAnalysisCNN, Trainer, SkinDataset
# O
from ml.models import SkinAnalysisCNN
from ml.training import Trainer
from ml.data import SkinDataset
```

## 📝 Convenciones de Organización

### 1. Separación por Capas
- **ML**: Machine Learning puro
- **Core**: Lógica de negocio
- **API**: Interfaz HTTP
- **Utils**: Utilidades generales

### 2. Imports Limpios
- Usar `ml/__init__.py` para exports principales
- Imports específicos cuando sea necesario
- Evitar imports circulares

### 3. Documentación
- Cada módulo tiene su `__init__.py` con exports
- Documentación en `docs/`
- Ejemplos en `examples/`

### 4. Tests
- Tests junto al código (`tests/`)
- Tests unitarios por módulo
- Tests de integración separados

## 🚀 Ejemplo de Uso con Nueva Estructura

### Entrenamiento Completo

```python
# Imports organizados
from ml import (
    ViTSkinAnalyzer,
    Trainer,
    SkinDataset,
    get_train_transforms,
    get_val_transforms,
    MultiTaskLoss,
    get_optimizer,
    get_scheduler,
    create_data_loaders
)
from ml.experiments import ExperimentTracker, ExperimentConfig
from ml.inference import FastInferenceEngine
from utils.advanced_optimization import enable_all_optimizations

# 1. Configuración
enable_all_optimizations()

# 2. Modelo
model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)

# 3. Datos
train_dataset = SkinDataset(
    images=train_images,
    labels={'conditions': train_conditions, 'metrics': train_metrics},
    transform=get_train_transforms()
)

val_dataset = SkinDataset(
    images=val_images,
    labels={'conditions': val_conditions, 'metrics': val_metrics},
    transform=get_val_transforms()
)

# 4. Data Loaders
loaders = create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size=32
)

# 5. Experiment Tracking
tracker = ExperimentTracker(use_wandb=True)
config = ExperimentConfig(
    experiment_id="exp_001",
    name="ViT Training",
    model_type="vision_transformer",
    hyperparameters={"lr": 1e-4, "batch_size": 32}
)
tracker.create_experiment(config)

# 6. Training
trainer = Trainer(
    model=model,
    train_loader=loaders['train'],
    val_loader=loaders['val'],
    experiment_tracker=tracker
)

loss_fn = MultiTaskLoss()
optimizer = get_optimizer(model, "adamw", lr=1e-4)
scheduler = get_scheduler(optimizer, "cosine", num_epochs=100)

trainer.fit(optimizer, num_epochs=100, scheduler=scheduler, criterion=loss_fn)

# 7. Inference
engine = FastInferenceEngine(model, use_compile=True)
output = engine.predict(input_tensor)
```

## 📚 Documentación por Módulo

### ML Module
- `ML_IMPROVEMENTS_V2.md`: Mejoras de ML
- `MODULAR_ARCHITECTURE_V2.md`: Arquitectura modular
- `ADVANCED_TRAINING_GUIDE.md`: Guía de entrenamiento

### Performance
- `PERFORMANCE_OPTIMIZATIONS.md`: Optimizaciones básicas
- `ULTIMATE_OPTIMIZATION_GUIDE.md`: Optimizaciones avanzadas

### Arquitectura
- `HEXAGONAL_ARCHITECTURE.md`: Arquitectura hexagonal
- `ARCHITECTURE_V7_FINAL.md`: Arquitectura final

## 🎯 Beneficios de la Nueva Estructura

1. **Claridad**: Separación clara de responsabilidades
2. **Mantenibilidad**: Fácil encontrar y modificar código
3. **Escalabilidad**: Fácil agregar nuevos componentes
4. **Testabilidad**: Tests organizados por módulo
5. **Reusabilidad**: Componentes reutilizables claramente definidos
6. **Documentación**: Documentación organizada por tema

## 🔧 Mejores Prácticas

1. **Usar imports desde `ml/`**: Para componentes ML principales
2. **Imports específicos**: Cuando solo necesites un componente
3. **No imports circulares**: Evitar dependencias circulares
4. **Documentar exports**: Cada `__init__.py` debe documentar exports
5. **Mantener estructura**: Seguir la estructura definida

---

**Project Structure - Organizado para Escalabilidad y Mantenibilidad**













