# 📚 Guía Completa de Utilidades

Esta guía documenta todas las utilidades disponibles en el sistema de Deep Learning.

## 📦 Módulos de Utilidades

### 1. Model Utilities (`core/utils/model_utils.py`)

**Funciones de Inicialización:**
```python
from core.utils.model_utils import initialize_weights

initialize_weights(model, method="xavier_uniform", gain=1.0)
```

**Funciones de Información:**
```python
from core.utils.model_utils import count_parameters, get_model_size

total = count_parameters(model)
trainable = count_parameters(model, trainable_only=True)
size_mb = get_model_size(model, unit="MB")
```

**Funciones de Control:**
```python
from core.utils.model_utils import freeze_model, freeze_layers

freeze_model(model, freeze=True)
freeze_layers(model, ["encoder"], freeze=True)
```

**Funciones de Gradientes:**
```python
from core.utils.model_utils import get_gradient_norm, clip_gradients

grad_norm = get_gradient_norm(model)
clipped_norm = clip_gradients(model, max_norm=1.0)
```

**Funciones de Validación:**
```python
from core.utils.model_utils import check_for_nan_inf

result = check_for_nan_inf(model)
if result["has_nan"]:
    print("NaN detected!")
```

### 2. Training Utilities (`core/utils/training_utils.py`)

**Early Stopping:**
```python
from core.utils.training_utils import EarlyStopping

early_stopping = EarlyStopping(patience=5, min_delta=0.001)
if early_stopping(val_loss, model):
    break
```

**Optimizadores:**
```python
from core.utils.training_utils import create_optimizer

optimizer = create_optimizer(
    model,
    optimizer_type="adamw",
    learning_rate=1e-3,
    weight_decay=1e-5
)
```

**Schedulers:**
```python
from core.utils.training_utils import create_scheduler

scheduler = create_scheduler(
    optimizer,
    scheduler_type="cosine",
    T_max=num_epochs
)
```

**Entrenamiento:**
```python
from core.utils.training_utils import train_one_epoch, validate_one_epoch

train_loss, train_acc = train_one_epoch(
    model, train_loader, criterion, optimizer,
    device=device,
    use_mixed_precision=True
)

val_loss, val_acc = validate_one_epoch(
    model, val_loader, criterion, device=device
)
```

### 3. Data Utilities (`core/utils/data_utils.py`)

**División de Datos:**
```python
from core.utils.data_utils import create_data_splits

train_ds, val_ds, test_ds = create_data_splits(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

**DataLoaders:**
```python
from core.utils.data_utils import create_dataloader

train_loader = create_dataloader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

**Normalización:**
```python
from core.utils.data_utils import normalize_tensor

normalized, mean, std = normalize_tensor(tensor)
```

**Balanceo:**
```python
from core.utils.data_utils import balance_dataset, get_class_weights

balanced_ds = balance_dataset(dataset, labels, method="undersample")
class_weights = get_class_weights(labels)
```

### 4. Validation Utilities (`core/utils/validation_utils.py`)

**Validación de Configuraciones:**
```python
from core.utils.validation_utils import (
    validate_model_config,
    validate_training_config
)

is_valid, errors = validate_model_config(config)
is_valid, errors = validate_training_config(config)
```

**Validación de Modelos:**
```python
from core.utils.validation_utils import validate_model_output

is_valid, error, output = validate_model_output(
    model,
    input_shape=(784,),
    expected_output_shape=(10,)
)
```

**Validación de Gradientes:**
```python
from core.utils.validation_utils import validate_gradients

is_valid, grad_info = validate_gradients(model)
```

### 5. Performance Utilities (`core/utils/performance_utils.py`)

**Medición de Tiempo:**
```python
from core.utils.performance_utils import timer

with timer("Training epoch"):
    train_one_epoch(...)
```

**Profiling:**
```python
from core.utils.performance_utils import profile_model

stats = profile_model(model, input_shape=(784,), num_runs=100)
print(f"Throughput: {stats['throughput']} samples/sec")
```

**Memoria:**
```python
from core.utils.performance_utils import get_memory_usage, clear_cache

memory = get_memory_usage(device)
clear_cache(device)
```

### 6. Visualization Utilities (`core/utils/visualization_utils.py`)

**Historial de Entrenamiento:**
```python
from core.utils.visualization_utils import plot_training_history

image_bytes = plot_training_history(
    train_losses, val_losses, train_accs, val_accs
)
```

**Matriz de Confusión:**
```python
from core.utils.visualization_utils import plot_confusion_matrix

image_bytes = plot_confusion_matrix(y_true, y_pred, class_names)
```

**Feature Importance:**
```python
from core.utils.visualization_utils import plot_feature_importance

image_bytes = plot_feature_importance(feature_names, importances, top_k=10)
```

### 7. Checkpoint Utilities (`core/utils/checkpoint_utils.py`)

**Guardar/Cargar:**
```python
from core.utils.checkpoint_utils import save_checkpoint, load_checkpoint

save_checkpoint(
    model, optimizer, scheduler,
    epoch=epoch, loss=loss,
    filepath="checkpoint.pt",
    is_best=True
)

result = load_checkpoint(model, "checkpoint.pt", optimizer, scheduler)
```

**Gestión:**
```python
from core.utils.checkpoint_utils import list_checkpoints, cleanup_old_checkpoints

checkpoints = list_checkpoints("checkpoints/")
deleted = cleanup_old_checkpoints("checkpoints/", keep_last_n=5)
```

### 8. Debugging Utilities (`core/utils/debugging_utils.py`)

**Health Check:**
```python
from core.utils.debugging_utils import check_model_health

health = check_model_health(model)
if not health["is_healthy"]:
    print(f"Issues: {health['issues']}")
```

**Diagnóstico:**
```python
from core.utils.debugging_utils import diagnose_training_issue

diagnosis = diagnose_training_issue(model, loss)
print(f"Recommendations: {diagnosis['recommendations']}")
```

**Comparación:**
```python
from core.utils.debugging_utils import compare_models

comparison = compare_models(model1, model2, input_shape=(784,))
```

### 9. Export Utilities (`core/utils/export_utils.py`)

**ONNX:**
```python
from core.utils.export_utils import export_to_onnx

export_to_onnx(model, input_shape=(784,), output_path="model.onnx")
```

**TorchScript:**
```python
from core.utils.export_utils import export_to_torchscript

export_to_torchscript(
    model,
    input_shape=(784,),
    output_path="model.pt",
    method="trace"
)
```

**Resumen:**
```python
from core.utils.export_utils import export_model_summary

export_model_summary(model, input_shape=(784,), output_path="summary.txt")
```

## 🎯 Casos de Uso Comunes

### Pipeline Completo de Entrenamiento

```python
from core.base_model_service import BaseModelService
from core.utils import *

class MyService(BaseModelService):
    def train_complete(self):
        # 1. Crear modelo
        model = self.create_model(...)
        initialize_weights(model, "xavier_uniform")
        
        # 2. Validar
        is_valid, _, _ = validate_model_output(model, input_shape)
        
        # 3. Setup entrenamiento
        optimizer = create_optimizer(model, "adamw", lr=1e-3)
        scheduler = create_scheduler(optimizer, "cosine")
        early_stopping = EarlyStopping(patience=5)
        
        # 4. Entrenar
        for epoch in range(num_epochs):
            with timer(f"Epoch {epoch}"):
                train_loss, _ = train_one_epoch(...)
                val_loss, _ = validate_one_epoch(...)
                
                if early_stopping(val_loss, model):
                    break
        
        # 5. Visualizar
        plot_training_history(train_losses, val_losses)
        
        # 6. Exportar
        export_to_onnx(model, input_shape, "model.onnx")
```

## 📊 Estadísticas

- **50+ funciones** de utilidad
- **9 módulos** especializados
- **100% type-safe** con type hints
- **Cobertura completa** del pipeline de ML




