# Refactoring Final V2 - Testing, Ensembles & Error Handling

## Nuevas Mejoras Agregadas

### 1. Testing Module (`ml/testing/`) ✅

**Módulos de Testing Completos:**

#### `model_tester.py`
- `ModelTester`: Testing completo de modelos
  - `test_forward_pass()`: Probar forward pass
  - `test_backward_pass()`: Probar backward pass
  - `test_parameter_count()`: Contar parámetros
  - `test_memory_usage()`: Uso de memoria
  - `run_all_tests()`: Ejecutar todos los tests

#### `data_tester.py`
- `DataTester`: Testing de datos
  - `test_dataset()`: Probar dataset
  - `test_dataloader()`: Probar data loader
  - `validate_data_consistency()`: Validar consistencia

#### `integration_tester.py`
- `IntegrationTester`: Testing de integración
  - `test_training_step()`: Probar paso de entrenamiento
  - `test_validation_step()`: Probar paso de validación
  - `run_full_integration_test()`: Test completo end-to-end

**Uso:**
```python
from ml.testing import ModelTester, DataTester, IntegrationTester

# Test de modelo
tester = ModelTester(model, device)
results = tester.run_all_tests(input_shape=(1, 3, 224, 224))

# Test de datos
data_tester = DataTester()
dataset_results = data_tester.test_dataset(dataset)

# Test de integración
integration_tester = IntegrationTester(model, train_loader, val_loader)
full_results = integration_tester.run_full_integration_test()
```

### 2. Model Ensembles (`ml/models/ensemble.py`) ✅

**Ensemble Models:**
- `EnsembleModel`: Modelo ensemble
  - Soft voting (weighted average)
  - Hard voting (majority vote)
  - Configurable weights

- `EnsembleBuilder`: Builder para ensembles
  - `create_ensemble()`: Crear desde configuraciones
  - `create_diverse_ensemble()`: Crear ensemble diverso

**Uso:**
```python
from ml.models import EnsembleModel, EnsembleBuilder

# Crear ensemble manual
models = [model1, model2, model3]
ensemble = EnsembleModel(models, voting='soft', weights=[0.4, 0.3, 0.3])

# Crear ensemble desde config
ensemble = EnsembleBuilder.create_ensemble(
    model_configs=[config1, config2, config3],
    voting='soft'
)

# Ensemble diverso
ensemble = EnsembleBuilder.create_diverse_ensemble(
    base_config=config,
    num_models=5,
    diversity_params={'dropout': {'min': 0.1, 'max': 0.5}}
)
```

### 3. Error Handling (`ml/utils/error_handling.py`) ✅

**Manejo de Errores Completo:**

#### `ErrorHandler`
- `handle_nan_inf()`: Manejar NaN/Inf en tensores
- `safe_backward()`: Backward seguro con manejo de errores
- `retry_on_error()`: Decorator para reintentos
- `safe_model_forward()`: Forward seguro con fallback

#### `TrainingErrorHandler`
- `check_loss()`: Verificar pérdida por problemas
- Tracking de NaN/Inf losses consecutivos
- Auto-stop en caso de demasiados errores
- Gradient clipping integrado

**Uso:**
```python
from ml.utils import ErrorHandler, TrainingErrorHandler

# Manejo de NaN/Inf
tensor = ErrorHandler.handle_nan_inf(tensor)

# Backward seguro
result = ErrorHandler.safe_backward(loss, optimizer, max_grad_norm=1.0)

# Retry decorator
@ErrorHandler.retry_on_error(max_retries=3, delay=1.0)
def risky_operation():
    # código que puede fallar
    pass

# Error handler para entrenamiento
error_handler = TrainingErrorHandler(
    max_nan_losses=5,
    max_inf_losses=5,
    gradient_clip=1.0
)

# En el loop de entrenamiento
loss_check = error_handler.check_loss(loss)
if loss_check['should_stop']:
    break
```

## Arquitectura Completa Actualizada

```
ml/
├── models/              # 9 módulos (incluye ensemble)
│   └── ensemble.py     # ✅ NEW
├── training/           # 13 módulos
├── inference/          # 3 módulos
├── pipelines/          # 2 módulos
├── registry/           # 2 módulos
├── serving/            # 2 módulos
├── testing/            # ✅ NEW: 3 módulos
│   ├── model_tester.py
│   ├── data_tester.py
│   └── integration_tester.py
└── utils/              # 11 módulos (incluye error_handling)
    └── error_handling.py  # ✅ NEW
```

## Características Completas Finales

### Testing ✅
- Model testing (forward, backward, memory, parameters)
- Data testing (dataset, dataloader, consistency)
- Integration testing (end-to-end)
- Comprehensive test results

### Ensembles ✅
- Soft voting ensembles
- Hard voting ensembles
- Weighted ensembles
- Diverse ensemble creation

### Error Handling ✅
- NaN/Inf handling
- Safe backward pass
- Retry mechanisms
- Training error tracking
- Auto-stop on errors

### Entrenamiento ✅
- Trainer completo
- Distributed training
- Gradient accumulation
- Mixed precision
- Callbacks system
- Checkpoint management
- Experiment tracking
- Data augmentation
- Loss functions
- Optimizer factory
- Scheduler factory
- Training validation

### Inferencia ✅
- Model predictor
- Preprocessing
- Postprocessing
- Inference pipeline

### Deployment ✅
- ONNX export
- TorchScript export
- Quantization
- REST API
- Gradio integration

### Utilidades ✅
- Profiling
- Validation
- Metrics
- Visualization
- Configuration
- Debugging
- Monitoring
- Error handling

## Ejemplos de Uso Completos

### 1. Testing Completo Antes de Entrenar

```python
from ml.testing import IntegrationTester
from ml.utils import TrainingErrorHandler

# Test de integración
tester = IntegrationTester(model, train_loader, val_loader)
results = tester.run_full_integration_test()

if not results['all_passed']:
    print("Tests failed, fixing issues...")
    # Fix issues
else:
    print("All tests passed, starting training")

# Setup error handler
error_handler = TrainingErrorHandler(max_nan_losses=5)

# Entrenar con error handling
for epoch in range(num_epochs):
    for batch in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Check loss
        loss_check = error_handler.check_loss(loss)
        if loss_check['should_stop']:
            print("Stopping due to errors")
            break
        
        # Safe backward
        backward_result = ErrorHandler.safe_backward(loss, optimizer)
        if not backward_result['success']:
            print(f"Backward failed: {backward_result['error']}")
            continue
        
        optimizer.step()
```

### 2. Ensemble de Modelos

```python
from ml.models import EnsembleBuilder
from ml.pipelines import InferencePipeline

# Crear ensemble
ensemble = EnsembleBuilder.create_diverse_ensemble(
    base_config={
        'variant': 'mobilenet_v2',
        'num_classes': 10,
    },
    num_models=5,
    diversity_params={
        'dropout': {'min': 0.1, 'max': 0.3}
    }
)

# Usar ensemble para inferencia
pipeline = InferencePipeline(model=ensemble)
results = pipeline.predict('image.jpg')
```

### 3. Testing de Modelo

```python
from ml.testing import ModelTester

# Test completo del modelo
tester = ModelTester(model, device)
results = tester.run_all_tests(input_shape=(1, 3, 224, 224))

print(f"Forward pass: {results['forward_pass']['success']}")
print(f"Backward pass: {results['backward_pass']['success']}")
print(f"Parameters: {results['parameters']['total_parameters']:,}")
print(f"Memory: {results['memory']['allocated_mb']:.2f} MB")
```

## Estadísticas Finales

- **Total de Módulos**: 35+
- **Líneas de Código**: ~6000+
- **Características**: 60+
- **Testing Utilities**: 3 módulos completos
- **Error Handling**: Sistema completo
- **Ensembles**: Soporte completo

## Resumen

El framework ahora incluye:

1. ✅ **Testing Completo**: Model, data, e integration testing
2. ✅ **Ensembles**: Soft/hard voting, weighted ensembles
3. ✅ **Error Handling**: Manejo robusto de errores
4. ✅ **Auto-Recovery**: Retry mechanisms y auto-stop
5. ✅ **Production-Ready**: Todas las características necesarias

**El código está completamente refactorizado, testeado, y listo para producción con manejo robusto de errores y capacidades de ensemble.**



