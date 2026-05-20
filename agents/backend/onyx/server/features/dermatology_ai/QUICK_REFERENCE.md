# Quick Reference Guide

## 🚀 Imports Rápidos

### ML Components (Recomendado)
```python
# Todo desde ml/
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

### Específicos
```python
from ml.models import ViTSkinAnalyzer
from ml.training import Trainer, MultiTaskLoss
from ml.data import SkinDataset, get_train_transforms
from ml.inference import FastInferenceEngine
from ml.experiments import ExperimentTracker
from ml.visualization import GradioDemo
```

### Utils
```python
from utils.optimization import compile_model, quantize_model
from utils.advanced_optimization import enable_all_optimizations
from utils.profiling import PerformanceMonitor
```

## 📝 Ejemplos Rápidos

### 1. Entrenamiento
```python
from ml import ViTSkinAnalyzer, Trainer, SkinDataset, get_train_transforms
from ml.training import MultiTaskLoss, get_optimizer, get_scheduler

model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)
dataset = SkinDataset(images, labels, transform=get_train_transforms())
trainer = Trainer(model, train_loader, val_loader)
optimizer = get_optimizer(model, "adamw", lr=1e-4)
trainer.fit(optimizer, num_epochs=100)
```

### 2. Inferencia
```python
from ml import ViTSkinAnalyzer
from ml.inference import FastInferenceEngine
from utils.optimization import compile_model

model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)
model = compile_model(model)
engine = FastInferenceEngine(model)
output = engine.predict(input_tensor)
```

### 3. Experiment Tracking
```python
from ml.experiments import ExperimentTracker, ExperimentConfig

tracker = ExperimentTracker(use_wandb=True)
config = ExperimentConfig(experiment_id="exp_001", name="Training", ...)
tracker.create_experiment(config)
```

### 4. Gradio Demo
```python
from ml.visualization import GradioDemo
from core.skin_analyzer import SkinAnalyzer

analyzer = SkinAnalyzer()
demo = GradioDemo(analyzer)
demo.launch(server_port=7860)
```

## 🎯 Configuración Rápida

```python
from config import load_config
from utils.advanced_optimization import enable_all_optimizations

# Cargar configuración
config = load_config("config/model_config.yaml")

# Habilitar optimizaciones
enable_all_optimizations()
```

## 📚 Documentación

- **Estructura**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Organización**: [ORGANIZATION_GUIDE.md](ORGANIZATION_GUIDE.md)
- **ML**: [ML_IMPROVEMENTS_V2.md](ML_IMPROVEMENTS_V2.md)
- **Training**: [ADVANCED_TRAINING_GUIDE.md](ADVANCED_TRAINING_GUIDE.md)
- **Performance**: [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md)
- **Optimization**: [ULTIMATE_OPTIMIZATION_GUIDE.md](ULTIMATE_OPTIMIZATION_GUIDE.md)

## 🔗 Enlaces Rápidos

- **Ejemplos**: [examples/](examples/)
- **Tests**: [tests/](tests/)
- **Config**: [config/](config/)
- **Docs**: [docs/](docs/)













