# ML Module - Machine Learning Components

## 📦 Módulo Principal de Machine Learning

Este módulo contiene todos los componentes relacionados con Machine Learning, organizados de manera clara y modular.

## 🏗️ Estructura

```
ml/
├── __init__.py              # Exports principales
├── models/                  # Arquitecturas de modelos
├── training/                # Componentes de entrenamiento
├── data/                    # Procesamiento de datos
├── experiments/             # Gestión de experimentos
├── inference/               # Motores de inferencia
└── visualization/           # Demos y visualización
```

## 🚀 Quick Start

### Import Básico

```python
from ml import (
    ViTSkinAnalyzer,
    Trainer,
    SkinDataset,
    get_train_transforms
)
```

### Entrenamiento Rápido

```python
from ml import (
    ViTSkinAnalyzer,
    Trainer,
    SkinDataset,
    MultiTaskLoss,
    get_optimizer,
    get_scheduler,
    create_data_loaders
)

# Modelo
model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)

# Datos
train_dataset = SkinDataset(images, labels, transform=get_train_transforms())
loaders = create_data_loaders(train_dataset, batch_size=32)

# Entrenar
trainer = Trainer(model, loaders['train'], loaders['val'])
optimizer = get_optimizer(model, "adamw", lr=1e-4)
scheduler = get_scheduler(optimizer, "cosine", num_epochs=100)
trainer.fit(optimizer, num_epochs=100, scheduler=scheduler)
```

### Inferencia Rápida

```python
from ml import ViTSkinAnalyzer
from ml.inference import FastInferenceEngine

model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)
engine = FastInferenceEngine(model, use_compile=True)
output = engine.predict(input_tensor)
```

## 📚 Documentación

- Ver `PROJECT_STRUCTURE.md` para estructura completa
- Ver `ML_IMPROVEMENTS_V2.md` para detalles de ML
- Ver `ADVANCED_TRAINING_GUIDE.md` para entrenamiento avanzado













