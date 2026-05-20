# Quick Reference Guide - Modular ML Framework

## Import Guide

### Models
```python
from ml.models import MobileNetModel, MobileNetV2, MobileNetV3
from ml.models.mobilenet import MobileNetFactory, MobileNetConfig
```

### Training Components
```python
from ml.training import (
    MobileNetTrainer, TrainingConfig,
    OptimizerFactory, SchedulerFactory, LossFactory,
    EarlyStoppingCallback, ModelCheckpointCallback,
    DistributedTrainingManager, GradientAccumulator,
    AugmentationBuilder, CheckpointManager,
)
```

### Inference
```python
from ml.inference import (
    ModelPredictor, BatchPredictor,
    ImagePreprocessor, PredictionPostprocessor,
)
```

### Pipelines
```python
from ml.pipelines import TrainingPipeline, InferencePipeline
```

### Utilities
```python
from ml.utils import (
    ConfigLoader, Profiler, ModelExporter,
    QuantizationManager, MetricsCollector,
    TrainingVisualizer,
)
```

### Registry
```python
from ml.registry import model_registry, component_registry
```

## Quick Start Examples

### 1. Simple Training
```python
from ml.pipelines import TrainingPipeline

pipeline = TrainingPipeline(config_path='config.yaml')
pipeline.setup()
history = pipeline.train(train_loader, val_loader)
```

### 2. Custom Training
```python
from ml.models.mobilenet import MobileNetFactory, MobileNetConfig
from ml.training import (
    MobileNetTrainer, TrainingConfig,
    OptimizerFactory, SchedulerFactory, LossFactory,
)

# Create model
config = MobileNetConfig(variant='mobilenet_v2', num_classes=10)
model = MobileNetFactory.create_model(config)

# Create components
optimizer = OptimizerFactory.create_optimizer(model, 'adamw', lr=0.001)
scheduler = SchedulerFactory.create_scheduler(optimizer, 'cosine', T_max=50)
loss = LossFactory.create_loss('focal', alpha=1.0, gamma=2.0)

# Train
trainer = MobileNetTrainer(model, device, TrainingConfig())
history = trainer.train(train_loader, val_loader)
```

### 3. Inference
```python
from ml.pipelines import InferencePipeline

pipeline = InferencePipeline(model_path='model.pth')
result = pipeline.predict('image.jpg')
```

### 4. Export and Deploy
```python
from ml.utils import ModelExporter, QuantizationManager

# Export
ModelExporter.export_onnx(model, 'model.onnx')

# Quantize
quantized = QuantizationManager.quantize_dynamic(model)
```

## Module Map

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `models/` | Model definitions | MobileNetModel, MobileNetV2, MobileNetV3 |
| `training/` | Training infrastructure | MobileNetTrainer, Callbacks, Checkpoints |
| `inference/` | Inference utilities | ModelPredictor, Preprocessor, Postprocessor |
| `pipelines/` | High-level workflows | TrainingPipeline, InferencePipeline |
| `registry/` | Component registration | ModelRegistry, ComponentRegistry |
| `utils/` | Utilities | Profiler, Exporter, Quantization |

## Configuration

All components can be configured via YAML:

```yaml
model:
  variant: mobilenet_v2
  num_classes: 10

training:
  learning_rate: 0.001
  batch_size: 32
  optimizer: adamw
  scheduler: cosine
  loss: focal
```



