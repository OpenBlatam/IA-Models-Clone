# Quick Reference Guide

## 🚀 Imports Comunes

```python
# Models
from addiction_recovery_ai import (
    create_sentiment_analyzer,
    create_progress_predictor,
    create_relapse_predictor,
    create_llm_coach
)

# Optimization
from addiction_recovery_ai import (
    create_ultra_fast_inference,
    create_async_engine,
    create_embedding_cache
)

# Pipeline
from addiction_recovery_ai import (
    create_integrated_pipeline,
    create_inference_pipeline
)

# Validation
from addiction_recovery_ai import (
    validate_input,
    validate_features,
    validate_text
)

# Training
from addiction_recovery_ai import (
    TrainerFactory,
    create_tracker,
    create_checkpoint_manager
)

# Monitoring
from addiction_recovery_ai import (
    create_system_monitor,
    create_model_monitor
)

# Export
from addiction_recovery_ai import (
    export_to_onnx,
    export_to_torchscript
)
```

## 📝 Patrones Comunes

### 1. Inferencia Básica
```python
# Crear modelo
model = create_progress_predictor()

# Crear engine optimizado
engine = create_ultra_fast_inference(model)

# Predecir
input_tensor = torch.tensor([[0.3, 0.4, 0.5, 0.7]])
output = engine.predict(input_tensor)
```

### 2. Pipeline Integrado
```python
# Crear pipeline completo
pipeline = create_integrated_pipeline(
    model,
    enable_validation=True,
    enable_monitoring=True,
    enable_optimization=True
)

# Usar pipeline
output = pipeline.predict(input_tensor)
health = pipeline.get_health_status()
```

### 3. Training Completo
```python
# Setup
trainer = TrainerFactory.create("RecoveryModelTrainer", ...)
tracker = create_tracker("experiment_v1")
checkpoint_manager = create_checkpoint_manager("checkpoints")

# Training
trainer.train(optimizer, criterion, num_epochs=50)
```

### 4. Validación
```python
# Validar input
is_valid, error = validate_input(tensor, expected_shape=(1, 10))
if not is_valid:
    raise ValueError(error)

# Validar features
is_valid, error = validate_features([0.3, 0.4, 0.5], expected_length=3)
```

### 5. Monitoreo
```python
# System monitoring
system_monitor = create_system_monitor()
health = system_monitor.get_health_status()

# Model monitoring
model_monitor = create_model_monitor(model)
model_monitor.record_inference(10.5, success=True)
```

### 6. Exportación
```python
# Export to ONNX
export_to_onnx(model, input_shape=(1, 10), output_path="model.onnx")

# Export to TorchScript
export_to_torchscript(model, input_shape=(1, 10), output_path="model.pt")
```

## 🎯 Funciones Útiles

### Device Management
```python
from addiction_recovery_ai import get_device
device = get_device()  # Auto-detect best device
```

### Parameter Counting
```python
from addiction_recovery_ai import count_parameters
total = count_parameters(model)
trainable = count_parameters(model, trainable_only=True)
```

### Model Initialization
```python
from addiction_recovery_ai import initialize_model
initialize_model(model, method="xavier")
```

### Performance Optimization
```python
from addiction_recovery_ai import enable_optimizations
enable_optimizations()  # Enable all PyTorch optimizations
```

## 📊 Métricas

### Regression Metrics
```python
from addiction_recovery_ai import calculate_regression_metrics
metrics = calculate_regression_metrics(y_true, y_pred)
print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
```

### Classification Metrics
```python
from addiction_recovery_ai import calculate_classification_metrics
metrics = calculate_classification_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## 🔒 Seguridad

```python
from addiction_recovery_ai import (
    compute_model_hash,
    sanitize_input
)

# Compute hash
hash_value = compute_model_hash(model)

# Sanitize input
clean_input = sanitize_input(input_tensor, max_value=1.0)
```

## 💾 Caché

```python
from addiction_recovery_ai import create_smart_cache

cache = create_smart_cache(max_size=1000, ttl_seconds=3600)
cache.put("key", value)
value = cache.get("key")
```

## 📈 Visualización

```python
from addiction_recovery_ai import create_training_visualizer

viz = create_training_visualizer()
viz.plot_loss_curves(train_losses, val_losses)
```

---

**Version**: 3.4.0








