# Arquitectura Modular del Sistema de Enrutamiento

## 📁 Estructura de Directorios

```
core/
├── routing_models/          # Modelos de Deep Learning
│   ├── __init__.py
│   ├── base_model.py        # Clase base abstracta
│   ├── mlp_model.py         # Modelo MLP
│   ├── gnn_model.py         # Modelos GNN (GCN, GAT)
│   ├── transformer_model.py # Modelo Transformer
│   └── model_factory.py     # Factory para crear modelos
│
├── routing_data/            # Procesamiento de Datos
│   ├── __init__.py
│   ├── dataset.py           # Dataset y DataLoader
│   ├── preprocessing.py     # Preprocesamiento
│   └── augmentation.py      # Data Augmentation
│
├── routing_training/        # Entrenamiento
│   ├── __init__.py
│   ├── trainer.py          # Entrenador principal
│   ├── callbacks.py        # Callbacks (EarlyStopping, etc.)
│   └── metrics.py          # Cálculo de métricas
│
└── routing_config/          # Configuración
    ├── __init__.py
    ├── config_loader.py    # Cargador de YAML
    └── config_schema.py    # Esquemas de configuración
```

## 🏗️ Principios de Diseño

### 1. Separación de Responsabilidades
- **Modelos**: Solo arquitecturas de red
- **Datos**: Solo carga y preprocesamiento
- **Entrenamiento**: Solo lógica de entrenamiento
- **Configuración**: Solo gestión de configs

### 2. Interfaces y Abstracciones
- `BaseRouteModel`: Clase base para todos los modelos
- `Callback`: Sistema de callbacks extensible
- `ModelFactory`: Creación de modelos mediante factory pattern

### 3. Configuración Externa
- Todo configurable mediante YAML
- Esquemas de validación
- Fácil experimentación

## 📦 Módulos Principales

### routing_models
```python
from core.routing_models import ModelFactory, ModelConfig

# Crear modelo desde configuración
config = ModelConfig(
    input_dim=20,
    hidden_dims=[128, 256, 128],
    output_dim=4
)
model = ModelFactory.create_model("mlp", config)
```

### routing_data
```python
from core.routing_data import RouteDataset, RouteDataLoader, RoutePreprocessor

# Crear dataset
dataset = RouteDataset(features, targets, preprocessor=preprocessor)
train_dataset, val_dataset, test_dataset = dataset.split()

# Crear data loaders
train_loader, val_loader = RouteDataLoader.create_train_val_loaders(
    train_dataset, val_dataset, batch_size=32
)
```

### routing_training
```python
from core.routing_training import RouteTrainer, TrainingConfig
from core.routing_training.callbacks import EarlyStopping, ModelCheckpoint

# Crear entrenador
trainer = RouteTrainer(
    model=model,
    config=TrainingConfig(epochs=100),
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=[EarlyStopping(patience=20), ModelCheckpoint()]
)

# Entrenar
history = trainer.train()
```

### routing_config
```python
from core.routing_config import load_config, save_config

# Cargar configuración
config = load_config("config/default_config.yaml")

# Guardar configuración
save_config(config, "config/my_experiment.yaml")
```

## 🔧 Uso Completo

Ver ejemplo completo en: `examples/train_routing_model.py`

## ✅ Ventajas de la Arquitectura Modular

1. **Mantenibilidad**: Código organizado y fácil de mantener
2. **Extensibilidad**: Fácil agregar nuevos modelos o funcionalidades
3. **Testabilidad**: Cada módulo puede probarse independientemente
4. **Reutilización**: Componentes reutilizables en diferentes contextos
5. **Configurabilidad**: Todo configurable sin cambiar código
6. **Escalabilidad**: Fácil escalar a múltiples GPUs o distribuido

## 🚀 Próximos Pasos

- Agregar más modelos (Transformer, GNN avanzados)
- Implementar distributed training
- Agregar más callbacks (LearningRateFinder, etc.)
- Mejorar data augmentation
- Agregar validación de configuraciones


