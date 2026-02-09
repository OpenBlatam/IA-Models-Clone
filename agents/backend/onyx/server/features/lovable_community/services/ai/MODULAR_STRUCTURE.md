# Modular AI System Structure

## Overview

The AI system has been reorganized into a highly modular structure following best practices for deep learning projects. Each module has a clear, single responsibility.

## Directory Structure

```
services/ai/
├── __init__.py                    # Main exports (organized by module)
├── base_service.py                # Base class (moved to core/)
├── model_config.yaml              # Configuration
│
├── core/                          # Core infrastructure
│   └── __init__.py                # BaseAIService
│
├── data/                          # Data processing
│   └── __init__.py                # Data loading, preprocessing, augmentation
│
├── training/                      # Training operations
│   └── __init__.py                # Fine-tuning, trainers, callbacks
│
├── evaluation/                    # Model evaluation
│   └── __init__.py                # Metrics, evaluators, cross-validation
│
├── optimization/                  # Model optimization
│   └── __init__.py                # Quantization, pruning, ONNX
│
├── utils/                         # Utilities
│   └── __init__.py                # Debugging, profiling, visualization, multi-GPU
│
├── services/                      # High-level AI services
│   └── __init__.py                # Embeddings, sentiment, moderation, etc.
│
├── tracking/                      # Experiment tracking
│   └── __init__.py                # Experiment tracker, model versioning
│
└── interfaces/                    # User interfaces
    └── __init__.py                # Gradio interfaces
```

## Module Responsibilities

### 1. Core (`core/`)
**Purpose**: Base infrastructure for all AI services

**Contents**:
- `BaseAIService`: Base class with device management, mixed precision, context managers

**Usage**:
```python
from services.ai.core import BaseAIService

class MyService(BaseAIService):
    def __init__(self):
        super().__init__("my-model", "transformer")
```

### 2. Data (`data/`)
**Purpose**: All data-related operations

**Contents**:
- `TextDataset`: PyTorch Dataset
- `BatchProcessor`: Batch processing
- `TextPreprocessor`: Text preprocessing
- `TokenizationUtils`: Tokenization
- `DataAugmentation`: Data augmentation
- `FeatureExtractor`: Feature extraction

**Usage**:
```python
from services.ai.data import (
    TextDataset,
    TextPreprocessor,
    BatchProcessor
)

preprocessor = TextPreprocessor(lowercase=True)
dataset = TextDataset(texts, tokenizer)
processor = BatchProcessor(batch_size=32)
```

### 3. Training (`training/`)
**Purpose**: Training operations

**Contents**:
- `LoRAFineTuner`: LoRA fine-tuning
- `FullFineTuner`: Full fine-tuning
- `Trainer`: Complete training pipeline
- `EarlyStopping`: Early stopping callback
- `ModelCheckpoint`: Checkpoint callback
- `TrainingMetrics`: Training metrics

**Usage**:
```python
from services.ai.training import (
    LoRAFineTuner,
    Trainer,
    EarlyStopping,
    ModelCheckpoint
)

fine_tuner = LoRAFineTuner("bert-base-uncased")
trainer = Trainer(model, optimizer, criterion, device)
```

### 4. Evaluation (`evaluation/`)
**Purpose**: Model evaluation

**Contents**:
- `ClassificationMetrics`: Classification metrics
- `ModelEvaluator`: Model evaluator
- `cross_validate`: K-fold cross-validation

**Usage**:
```python
from services.ai.evaluation import (
    ModelEvaluator,
    ClassificationMetrics
)

evaluator = ModelEvaluator(model, device)
results = evaluator.evaluate(dataloader)
```

### 5. Optimization (`optimization/`)
**Purpose**: Model optimization

**Contents**:
- `ModelQuantizer`: Quantization
- `ModelPruner`: Pruning
- `ONNXExporter`: ONNX export
- `compare_models`: Model comparison

**Usage**:
```python
from services.ai.optimization import (
    ModelQuantizer,
    ONNXExporter
)

quantized = ModelQuantizer.quantize_dynamic(model)
ONNXExporter.export(model, "model.onnx", sample_input)
```

### 6. Utils (`utils/`)
**Purpose**: General utilities

**Contents**:
- `NaNInfDetector`: NaN/Inf detection
- `GradientChecker`: Gradient checking
- `MemoryProfiler`: Memory profiling
- `PerformanceProfiler`: Performance profiling
- `MultiGPUTrainer`: Multi-GPU support
- `TrainingVisualizer`: Visualization
- `ModelVisualizer`: Model visualization
- `MetricsVisualizer`: Metrics visualization

**Usage**:
```python
from services.ai.utils import (
    NaNInfDetector,
    MemoryProfiler,
    TrainingVisualizer
)

NaNInfDetector.check_model(model)
MemoryProfiler.log_memory_stats(device)
TrainingVisualizer.plot_training_curves(history)
```

### 7. Services (`services/`)
**Purpose**: High-level AI services

**Contents**:
- `EmbeddingService`: Semantic embeddings
- `SentimentService`: Sentiment analysis
- `ModerationService`: Content moderation
- `TextGenerationService`: Text generation
- `DiffusionService`: Image generation
- `RecommendationService`: Recommendations

**Usage**:
```python
from services.ai.services import (
    EmbeddingServiceRefactored,
    SentimentServiceRefactored,
    DiffusionService
)

embedding = EmbeddingServiceRefactored(db)
sentiment = SentimentServiceRefactored(db)
diffusion = DiffusionService()
```

### 8. Tracking (`tracking/`)
**Purpose**: Experiment tracking and model versioning

**Contents**:
- `ExperimentTracker`: Experiment tracking
- `ModelRegistry`: Model registry
- `ModelVersion`: Model version
- `load_model_config`: Load config

**Usage**:
```python
from services.ai.tracking import (
    ExperimentTracker,
    ModelRegistry
)

with ExperimentTracker("project", backend="wandb") as tracker:
    tracker.log_metric("accuracy", 0.95)

registry = ModelRegistry()
registry.register_model("model", "1.0.0", "./model.pth")
```

### 9. Interfaces (`interfaces/`)
**Purpose**: User interfaces

**Contents**:
- `GradioInterface`: Gradio web interfaces

**Usage**:
```python
from services.ai.interfaces import GradioInterface

interface = GradioInterface(
    text_generation_service=text_gen,
    diffusion_service=diffusion
)
interface.create_combined_interface()
```

## Import Patterns

### Import by Module (Recommended)
```python
# Import from specific module
from services.ai.training import LoRAFineTuner, Trainer
from services.ai.evaluation import ModelEvaluator
from services.ai.utils import MemoryProfiler
```

### Import from Root (Also Available)
```python
# Import from root (all modules)
from services.ai import (
    LoRAFineTuner,
    ModelEvaluator,
    MemoryProfiler
)
```

## Benefits of Modular Structure

### 1. Clear Separation of Concerns
- Each module has a single, well-defined responsibility
- Easy to understand what each module does
- Reduces coupling between components

### 2. Easy to Navigate
- Developers can quickly find what they need
- Clear organization by functionality
- Logical grouping of related code

### 3. Better Maintainability
- Changes in one module don't affect others
- Easy to test modules independently
- Clear dependencies

### 4. Scalability
- Easy to add new modules
- Easy to extend existing modules
- Clear extension points

### 5. Reusability
- Modules can be used independently
- Easy to import only what you need
- No unnecessary dependencies

## Migration Guide

### Old Import Style
```python
from services.ai import EmbeddingService
from services.ai import LoRAFineTuner
from services.ai import ModelEvaluator
```

### New Import Style (Recommended)
```python
from services.ai.services import EmbeddingService
from services.ai.training import LoRAFineTuner
from services.ai.evaluation import ModelEvaluator
```

### Both Styles Work
The root `__init__.py` still exports everything, so old imports continue to work for backward compatibility.

## Module Dependencies

```
core/
  └── (no dependencies on other modules)

data/
  └── depends on: core

training/
  └── depends on: core, data

evaluation/
  └── depends on: core, data

optimization/
  └── depends on: core

utils/
  └── depends on: core

services/
  └── depends on: core, data

tracking/
  └── (no dependencies on other modules)

interfaces/
  └── depends on: services
```

## Best Practices

### 1. Import from Specific Modules
```python
# Good
from services.ai.training import Trainer

# Also works, but less clear
from services.ai import Trainer
```

### 2. Use Module-Specific Imports in Large Projects
```python
# Better for large projects
from services.ai.training import Trainer, EarlyStopping
from services.ai.evaluation import ModelEvaluator
```

### 3. Keep Module Boundaries Clear
- Don't import from other modules unnecessarily
- Use dependency injection when needed
- Keep modules focused on their responsibility

## Conclusion

The modular structure provides:
- ✅ Clear organization
- ✅ Easy navigation
- ✅ Better maintainability
- ✅ Improved scalability
- ✅ Enhanced reusability
- ✅ Backward compatibility

All modules follow best practices and are production-ready.















