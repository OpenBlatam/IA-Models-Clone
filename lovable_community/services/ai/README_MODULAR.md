# Modular AI System - Quick Start Guide

## Structure Overview

The AI system is organized into **9 specialized modules**, each with a clear responsibility:

```
services/ai/
├── core/           # Base infrastructure
├── data/           # Data processing
├── training/       # Training operations
├── evaluation/     # Model evaluation
├── optimization/   # Model optimization
├── utils/          # Utilities (debugging, visualization, multi-GPU)
├── services/       # High-level AI services
├── tracking/       # Experiment tracking & versioning
└── interfaces/     # User interfaces
```

## Quick Import Examples

### By Module (Recommended)
```python
# Import from specific module
from services.ai.training import LoRAFineTuner, Trainer
from services.ai.evaluation import ModelEvaluator
from services.ai.utils import MemoryProfiler
from services.ai.services import EmbeddingServiceRefactored
```

### From Root (Also Works)
```python
# Import from root (backward compatible)
from services.ai import LoRAFineTuner, ModelEvaluator, EmbeddingService
```

## Module Quick Reference

### Core
```python
from services.ai.core import BaseAIService
```

### Data
```python
from services.ai.data import (
    TextDataset,           # PyTorch Dataset
    TextPreprocessor,     # Text preprocessing
    BatchProcessor,        # Batch processing
    DataAugmentation      # Data augmentation
)
```

### Training
```python
from services.ai.training import (
    LoRAFineTuner,        # LoRA fine-tuning
    Trainer,              # Complete trainer
    EarlyStopping,        # Early stopping
    ModelCheckpoint       # Checkpointing
)
```

### Evaluation
```python
from services.ai.evaluation import (
    ModelEvaluator,       # Model evaluator
    ClassificationMetrics # Metrics
)
```

### Optimization
```python
from services.ai.optimization import (
    ModelQuantizer,       # Quantization
    ONNXExporter          # ONNX export
)
```

### Utils
```python
from services.ai.utils import (
    MemoryProfiler,       # Memory profiling
    TrainingVisualizer,   # Visualization
    MultiGPUTrainer       # Multi-GPU
)
```

### Services
```python
from services.ai.services import (
    EmbeddingServiceRefactored,
    SentimentServiceRefactored,
    DiffusionService
)
```

### Tracking
```python
from services.ai.tracking import (
    ExperimentTracker,
    ModelRegistry
)
```

### Interfaces
```python
from services.ai.interfaces import GradioInterface
```

## Complete Example

```python
# 1. Data
from services.ai.data import TextPreprocessor, TextDataset
preprocessor = TextPreprocessor(lowercase=True)
dataset = TextDataset(texts, tokenizer)

# 2. Training
from services.ai.training import LoRAFineTuner, Trainer, EarlyStopping
fine_tuner = LoRAFineTuner("bert-base-uncased")
trainer = Trainer(model, optimizer, criterion, device)

# 3. Evaluation
from services.ai.evaluation import ModelEvaluator
evaluator = ModelEvaluator(model, device)
results = evaluator.evaluate(dataloader)

# 4. Visualization
from services.ai.utils import TrainingVisualizer
TrainingVisualizer.plot_training_curves(history)

# 5. Tracking
from services.ai.tracking import ExperimentTracker, ModelRegistry
with ExperimentTracker("project") as tracker:
    tracker.log_metric("accuracy", 0.95)
```

## Benefits

✅ **Clear organization** - Easy to find what you need
✅ **Modular imports** - Import only what you use
✅ **Backward compatible** - Old imports still work
✅ **Better maintainability** - Clear module boundaries
✅ **Scalable** - Easy to extend

See `MODULAR_STRUCTURE.md` for detailed documentation.















