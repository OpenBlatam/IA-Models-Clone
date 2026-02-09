# Organization Guide - Dermatology AI

## 📁 Estructura Organizada por Responsabilidad

### 🎯 Principio: Separación Clara de Responsabilidades

```
dermatology_ai/
│
├── ml/                          # 🧠 Machine Learning Core
│   ├── models/                 # Arquitecturas de modelos
│   ├── training/               # Componentes de entrenamiento
│   ├── data/                   # Procesamiento de datos
│   ├── experiments/            # Gestión de experimentos
│   ├── inference/              # Motores de inferencia
│   └── visualization/          # Demos y visualización
│
├── core/                        # 💼 Business Logic
│   ├── application/            # Use cases (hexagonal)
│   ├── domain/                 # Domain entities
│   ├── infrastructure/         # Infrastructure adapters
│   └── ...                     # Core business components
│
├── api/                         # 🌐 API Layer
│   ├── controllers/            # Request handlers
│   ├── routers/                # API routes
│   └── middleware/             # API middleware
│
├── services/                    # 🔧 Business Services
│   └── ...                     # Service implementations
│
├── utils/                       # 🛠️ Utilities
│   ├── optimization.py         # Performance optimizations
│   ├── profiling.py            # Profiling tools
│   └── ...                     # Other utilities
│
├── config/                      # ⚙️ Configuration
│   ├── settings.py             # App settings
│   └── model_config.yaml       # Model config template
│
├── examples/                    # 📚 Examples
│   ├── training_example.py     # Training example
│   ├── inference_example.py    # Inference example
│   └── gradio_demo_example.py  # Gradio demo
│
├── tests/                       # 🧪 Tests
│   └── ...                     # Test files
│
├── docs/                        # 📖 Documentation
│   └── README.md               # Documentation index
│
└── scripts/                     # 🔨 Scripts
    └── ...                     # Utility scripts
```

## 📦 Imports Organizados

### Nivel 1: Imports Principales desde `ml/`

```python
# ✅ RECOMENDADO: Import desde ml/
from ml import (
    ViTSkinAnalyzer,      # Modelo
    Trainer,              # Entrenamiento
    SkinDataset,          # Datos
    get_train_transforms, # Transforms
    MultiTaskLoss,        # Loss
    get_optimizer,        # Optimizer
    FastInferenceEngine   # Inference
)
```

### Nivel 2: Imports Específicos

```python
# Para imports más específicos
from ml.models import ViTSkinAnalyzer, ModelFactory
from ml.training import Trainer, MultiTaskLoss
from ml.data import SkinDataset, get_train_transforms
from ml.inference import FastInferenceEngine
from ml.experiments import ExperimentTracker
from ml.visualization import GradioDemo
```

### Nivel 3: Imports Directos (Cuando sea necesario)

```python
# Solo cuando necesites algo muy específico
from models.pytorch_models import SkinAnalysisCNN
from training.losses import FocalLoss
from data.preprocessing import ImagePreprocessor
```

## 🎯 Casos de Uso Comunes

### 1. Entrenamiento Completo

```python
# ✅ Estilo recomendado
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
from utils.advanced_optimization import enable_all_optimizations

# Configuración
enable_all_optimizations()

# Modelo
model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)

# Datos
train_dataset = SkinDataset(
    images=train_images,
    labels={'conditions': train_conditions},
    transform=get_train_transforms()
)

# Entrenar
trainer = Trainer(model, train_loader, val_loader)
optimizer = get_optimizer(model, "adamw", lr=1e-4)
trainer.fit(optimizer, num_epochs=100)
```

### 2. Inferencia Optimizada

```python
# ✅ Estilo recomendado
from ml import ViTSkinAnalyzer
from ml.inference import FastInferenceEngine
from utils.optimization import compile_model, quantize_model
from utils.advanced_optimization import enable_all_optimizations

enable_all_optimizations()

model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)
model = compile_model(model)
model = quantize_model(model, "int8_dynamic")

engine = FastInferenceEngine(model, use_compile=False, use_quantization=False)
output = engine.predict(input_tensor)
```

### 3. Experiment Tracking

```python
# ✅ Estilo recomendado
from ml.experiments import ExperimentTracker, ExperimentConfig

tracker = ExperimentTracker(use_wandb=True, use_tensorboard=True)
config = ExperimentConfig(
    experiment_id="exp_001",
    name="ViT Training",
    model_type="vision_transformer",
    hyperparameters={"lr": 1e-4}
)
tracker.create_experiment(config)
```

### 4. Gradio Demo

```python
# ✅ Estilo recomendado
from ml.visualization import GradioDemo
from core.skin_analyzer import SkinAnalyzer

analyzer = SkinAnalyzer(use_advanced=True)
demo = GradioDemo(analyzer, title="Dermatology AI")
demo.launch(server_port=7860)
```

## 📋 Convenciones de Organización

### 1. Imports
- **Usar `ml/`**: Para componentes ML principales
- **Específicos**: Cuando solo necesites un componente
- **Evitar**: Imports directos de subdirectorios profundos

### 2. Naming
- **Clases**: PascalCase (`ViTSkinAnalyzer`)
- **Funciones**: snake_case (`get_train_transforms`)
- **Constantes**: UPPER_SNAKE_CASE (`MAX_BATCH_SIZE`)

### 3. Archivos
- **Un concepto por archivo**: Cada archivo tiene una responsabilidad clara
- **Nombres descriptivos**: `vision_transformers.py` no `vt.py`
- **Documentación**: Cada módulo tiene docstrings

### 4. Estructura
- **Separación por capa**: ML, Core, API, Utils
- **Separación por función**: Models, Training, Data
- **No dependencias circulares**: Evitar imports circulares

## 🔄 Migración de Código Existente

### Antes (Desorganizado)
```python
from models.pytorch_models import SkinAnalysisCNN
from training.trainer import Trainer
from data.datasets import SkinDataset
from core.experiment_tracker import ExperimentTracker
from utils.optimization import compile_model
```

### Después (Organizado)
```python
# Opción 1: Desde ml/
from ml import SkinAnalysisCNN, Trainer, SkinDataset
from ml.experiments import ExperimentTracker
from utils.optimization import compile_model

# Opción 2: Específico
from ml.models import SkinAnalysisCNN
from ml.training import Trainer
from ml.data import SkinDataset
from ml.experiments import ExperimentTracker
from utils.optimization import compile_model
```

## 📚 Documentación por Módulo

### ML Module
- **models/**: Arquitecturas y factory
- **training/**: Trainer, losses, optimizers, metrics
- **data/**: Datasets, transforms, preprocessing
- **experiments/**: Experiment tracking
- **inference/**: Inference engines
- **visualization/**: Gradio demos

### Core Module
- **application/**: Use cases
- **domain/**: Business logic
- **infrastructure/**: Adapters

### Utils Module
- **optimization.py**: Basic optimizations
- **advanced_optimization.py**: Advanced techniques
- **profiling.py**: Performance profiling
- **model_pruning.py**: Model compression

## 🎓 Mejores Prácticas

1. **Siempre usar imports desde `ml/` cuando sea posible**
2. **Agrupar imports por módulo**
3. **Evitar imports circulares**
4. **Documentar exports en `__init__.py`**
5. **Mantener estructura consistente**

## 📖 Referencias

- `PROJECT_STRUCTURE.md`: Estructura completa
- `ml/README.md`: Guía del módulo ML
- `docs/README.md`: Índice de documentación
- `examples/`: Ejemplos completos

---

**Organization Guide - Código Organizado, Fácil de Mantener**













