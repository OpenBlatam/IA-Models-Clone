# Quick Start V4 - Ultra-Modular Architecture

## 🚀 Inicio Rápido con Micro-Módulos

### Instalación

```bash
pip install -r requirements.txt
```

### Uso Básico

#### 1. Data Processing

```python
from addiction_recovery_ai.core.layers.micro_modules import (
    StandardNormalizer,
    ZeroPadder,
    TensorValidator
)

# Crear procesadores
normalizer = StandardNormalizer()
padder = ZeroPadder()
validator = TensorValidator()

# Procesar datos
data = torch.randn(10)
data = normalizer.normalize(data)
data = padder.pad(data, 20)
assert validator.validate(data)
```

#### 2. Model Setup

```python
from addiction_recovery_ai.core.layers.micro_modules import (
    ModelInitializer,
    ModelCompiler
)

# Inicializar modelo
ModelInitializer.initialize(model, method='xavier')

# Compilar para inferencia rápida
compiled_model = ModelCompiler.compile(model)
```

#### 3. Training

```python
from addiction_recovery_ai.core.layers.micro_modules import (
    LossCalculator,
    GradientManager,
    LearningRateManager
)

# Crear loss
criterion = LossCalculator.create('mse')

# Gestionar gradientes
GradientManager.clip_gradients(model, max_norm=1.0)

# Gestionar learning rate
scheduler = LearningRateManager.create_scheduler(
    optimizer,
    'reduce_on_plateau',
    patience=5
)
```

#### 4. Inference

```python
from addiction_recovery_ai.core.layers.micro_modules import (
    BatchProcessor,
    CacheManager,
    OutputFormatter
)

# Procesar por lotes
processor = BatchProcessor(batch_size=32)
results = processor.process(inputs, model.forward)

# Caching
cache = CacheManager()
cached = cache.get(key) or model(input)

# Formatear salida
formatted = OutputFormatter.to_numpy(output)
```

### Workflow Completo

```python
from addiction_recovery_ai.core.layers import (
    WorkflowBuilder,
    StandardNormalizer,
    ModelInitializer,
    LossCalculator,
    BatchProcessor
)

# Crear workflow completo
workflow = (WorkflowBuilder("MyWorkflow")
    .with_model("RecoveryPredictor", config)
    .with_inference(use_mixed_precision=True)
    .build())

# Usar
prediction = workflow.predict(inputs)
```

### Utilidades Rápidas

```python
from addiction_recovery_ai.core.layers import (
    quick_model,
    quick_inference_engine,
    get_optimal_device
)

# Setup rápido
device = get_optimal_device()
model = quick_model("RecoveryPredictor", config, device)
engine = quick_inference_engine(model, device)
```

## 📚 Más Información

- `ULTRA_MODULAR_V4.md` - Documentación completa
- `examples/micro_modules_usage.py` - Ejemplos detallados
- `REFACTORING_COMPLETE_V4.md` - Resumen de refactorización

---

**Version**: 3.7.0  
**Quick Start**: Ready ✅



