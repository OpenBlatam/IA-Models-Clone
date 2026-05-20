# Ultra Modular Architecture V4 - Micro-Modules

## 🎯 Arquitectura Ultra-Granular con Micro-Módulos

Esta versión implementa una arquitectura extremadamente modular con micro-módulos que tienen una única responsabilidad específica.

## 📦 Micro-Módulos Implementados

### 1. Data Processors (`micro_modules/data_processors.py`)

**Normalizadores:**
- `StandardNormalizer` - Normalización estándar (mean=0, std=1)
- `MinMaxNormalizer` - Normalización min-max (0-1)

**Tokenizadores:**
- `SimpleTokenizer` - Tokenización simple basada en palabras
- `HuggingFaceTokenizer` - Wrapper para tokenizadores de HuggingFace

**Padders:**
- `ZeroPadder` - Padding con ceros
- `RepeatPadder` - Padding repitiendo último valor

**Augmenters:**
- `NoiseAugmenter` - Añade ruido
- `DropoutAugmenter` - Aplica dropout aleatorio

**Validators:**
- `TensorValidator` - Valida tensores (NaN/Inf)
- `ShapeValidator` - Valida formas
- `RangeValidator` - Valida rangos de valores

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers.micro_modules import (
    StandardNormalizer,
    ZeroPadder,
    TensorValidator
)

# Crear componentes
normalizer = StandardNormalizer()
padder = ZeroPadder()
validator = TensorValidator()

# Usar
data = torch.randn(10)
normalized = normalizer.normalize(data)
padded = padder.pad(normalized, target_length=20)
is_valid = validator.validate(padded)
```

### 2. Model Components (`micro_modules/model_components.py`)

**Inicialización:**
- `ModelInitializer` - Inicialización de pesos (Xavier, Kaiming, Orthogonal)

**Compilación:**
- `ModelCompiler` - Compilación con torch.compile

**Optimización:**
- `ModelOptimizer` - Mixed precision, TorchScript, pruning

**Cuantización:**
- `ModelQuantizer` - Cuantización dinámica y estática

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers.micro_modules import (
    ModelInitializer,
    ModelCompiler,
    ModelOptimizer
)

# Inicializar
ModelInitializer.initialize(model, method='xavier')

# Compilar
compiled_model = ModelCompiler.compile(model, mode="reduce-overhead")

# Optimizar
optimized_model = ModelOptimizer.enable_mixed_precision(compiled_model)
```

### 3. Training Components (`micro_modules/training_components.py`)

**Loss:**
- `LossCalculator` - Cálculo y gestión de funciones de pérdida

**Gradientes:**
- `GradientManager` - Clipping, validación, gestión de gradientes

**Learning Rate:**
- `LearningRateManager` - Gestión de schedulers y learning rates

**Checkpoints:**
- `CheckpointManager` - Guardado y carga de checkpoints

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers.micro_modules import (
    LossCalculator,
    GradientManager,
    LearningRateManager,
    CheckpointManager
)

# Loss
criterion = LossCalculator.create('mse')
loss = LossCalculator.calculate(predictions, targets, criterion)

# Gradientes
GradientManager.clip_gradients(model, max_norm=1.0)
grad_stats = GradientManager.check_gradients(model)

# Learning Rate
scheduler = LearningRateManager.create_scheduler(optimizer, 'reduce_on_plateau')

# Checkpoints
checkpoint_mgr = CheckpointManager("checkpoints")
checkpoint_mgr.save(model, optimizer, epoch=10, metrics={'loss': 0.5})
```

### 4. Inference Components (`micro_modules/inference_components.py`)

**Batch Processing:**
- `BatchProcessor` - Procesamiento eficiente por lotes

**Caching:**
- `CacheManager` - Gestión de caché para inferencia

**Formatting:**
- `OutputFormatter` - Formateo de salidas (numpy, list, dict)

**Post-Processing:**
- `PostProcessor` - Post-procesamiento (threshold, top-k, argmax)

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers.micro_modules import (
    BatchProcessor,
    CacheManager,
    OutputFormatter,
    PostProcessor
)

# Batch processing
processor = BatchProcessor(batch_size=32)
results = processor.process(inputs, model.forward)

# Caching
cache = CacheManager(max_size=1000)
cached_result = cache.get(cache_key) or model(input)
cache.set(cache_key, cached_result)

# Formatting
formatted = OutputFormatter.to_numpy(output)
formatted_list = OutputFormatter.to_list(output)

# Post-processing
post_processor = PostProcessor()
post_processor.add_processor(PostProcessor.threshold(0.5))
result = post_processor.process(output)
```

## 🏗️ Arquitectura Completa

```
Layers (6 capas principales)
  ↓
Micro-Modules (componentes ultra-granulares)
  ↓
Individual Components (responsabilidad única)
```

## 🎯 Principios de Diseño

1. **Single Responsibility**: Cada micro-módulo tiene una única responsabilidad
2. **Composability**: Los micro-módulos se pueden combinar fácilmente
3. **Reusability**: Componentes reutilizables en diferentes contextos
4. **Testability**: Fácil de testear individualmente
5. **Extensibility**: Fácil agregar nuevos micro-módulos

## 📊 Comparación de Modularidad

### Antes (V3)
- Capas principales
- Componentes medianos
- Algunas responsabilidades mezcladas

### Ahora (V4)
- Capas principales
- Micro-módulos granulares
- Responsabilidades únicas
- Componentes más pequeños y enfocados

## 🔧 Uso Completo

```python
from addiction_recovery_ai.core.layers.micro_modules import (
    # Data
    StandardNormalizer,
    ZeroPadder,
    TensorValidator,
    # Model
    ModelInitializer,
    ModelCompiler,
    # Training
    LossCalculator,
    GradientManager,
    # Inference
    BatchProcessor,
    CacheManager,
    OutputFormatter
)

# 1. Preparar datos
normalizer = StandardNormalizer()
padder = ZeroPadder()
validator = TensorValidator()

data = torch.randn(10)
data = normalizer.normalize(data)
data = padder.pad(data, 20)
assert validator.validate(data)

# 2. Preparar modelo
ModelInitializer.initialize(model, 'xavier')
compiled_model = ModelCompiler.compile(model)

# 3. Entrenar
criterion = LossCalculator.create('mse')
# ... training loop ...
GradientManager.clip_gradients(model)

# 4. Inferir
processor = BatchProcessor(batch_size=32)
cache = CacheManager()
formatter = OutputFormatter()

results = processor.process(inputs, compiled_model)
formatted = formatter.to_numpy(results[0])
```

## 🚀 Ventajas

1. **Granularidad Extrema**: Cada componente hace una cosa y la hace bien
2. **Fácil Testing**: Componentes pequeños son fáciles de testear
3. **Reutilización Máxima**: Componentes se pueden usar en cualquier contexto
4. **Mantenibilidad**: Código más fácil de entender y mantener
5. **Extensibilidad**: Fácil agregar nuevos micro-módulos

## 📚 Estructura de Archivos

```
core/layers/
├── __init__.py
├── interfaces.py
├── data_layer.py
├── model_layer.py
├── training_layer.py
├── inference_layer.py
├── service_layer.py
├── interface_layer.py
├── adapters.py
├── integration.py
├── utils.py
└── micro_modules/
    ├── __init__.py
    ├── data_processors.py
    ├── model_components.py
    ├── training_components.py
    └── inference_components.py
```

## 🎓 Mejores Prácticas

1. **Usar Micro-Módulos para Tareas Específicas**
   ```python
   normalizer = StandardNormalizer()
   data = normalizer.normalize(data)
   ```

2. **Componer Micro-Módulos**
   ```python
   pipeline = DataPipeline()
   pipeline.add_processor(StandardNormalizer())
   pipeline.add_processor(ZeroPadder(max_length=128))
   ```

3. **Reutilizar Componentes**
   ```python
   # Mismo normalizer en diferentes contextos
   normalizer = StandardNormalizer()
   train_data = normalizer.normalize(train_data)
   val_data = normalizer.normalize(val_data)
   ```

4. **Testear Individualmente**
   ```python
   def test_normalizer():
       normalizer = StandardNormalizer()
       data = torch.randn(10)
       normalized = normalizer.normalize(data)
       assert normalized.mean() < 1e-6
   ```

---

**Version**: 3.7.0  
**Status**: Ultra-Granular Modular ✅  
**Last Updated**: 2025



