# AI System Architecture - Complete Guide

## Overview

The AI system is organized into **12 specialized modules**, each with clear responsibilities following deep learning best practices.

## Complete Module Structure

```
services/ai/
├── core/              # Base infrastructure (BaseAIService)
├── data/              # Data processing (loading, preprocessing, augmentation)
├── models/            # ✨ Custom model architectures
├── training/          # Training operations (fine-tuning, trainers, callbacks)
├── evaluation/        # Model evaluation (metrics, evaluators)
├── optimization/       # Model optimization (quantization, pruning, ONNX)
├── utils/             # Utilities (debugging, profiling, visualization, multi-GPU)
├── services/          # High-level AI services (embeddings, sentiment, etc.)
├── tracking/          # Experiment tracking & model versioning
├── interfaces/        # User interfaces (Gradio)
├── config/            # ✨ Configuration management
└── pipelines/         # ✨ End-to-end pipelines
```

## Module Details

### 1. Core (`core/`)
**Purpose**: Base infrastructure for all AI services

**Exports**:
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

**Exports**:
- `TextDataset`: PyTorch Dataset
- `BatchProcessor`: Batch processing
- `TextPreprocessor`: Advanced text preprocessing
- `TokenizationUtils`: Tokenization utilities
- `DataAugmentation`: Data augmentation
- `FeatureExtractor`: Feature extraction

**Usage**:
```python
from services.ai.data import TextPreprocessor, TextDataset

preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
dataset = TextDataset(texts, tokenizer, max_length=512)
```

### 3. Models (`models/`) ✨ NEW
**Purpose**: Custom model architectures

**Exports**:
- `ClassificationHead`: Custom classification head
- `RegressionHead`: Custom regression head
- `TransformerClassifier`: Complete transformer classifier
- `MultiTaskModel`: Multi-task learning model
- `WeightInitializer`: Weight initialization utilities
- `AttentionVisualizer`: Attention weight extraction

**Usage**:
```python
from services.ai.models import TransformerClassifier, WeightInitializer

# Create model
model = TransformerClassifier(
    model_name="bert-base-uncased",
    num_labels=3,
    dropout=0.1
)

# Initialize weights
WeightInitializer.apply_initialization(model, method="bert")
```

### 4. Training (`training/`)
**Purpose**: Training operations

**Exports**:
- `LoRAFineTuner`: LoRA fine-tuning
- `FullFineTuner`: Full fine-tuning
- `Trainer`: Complete training pipeline
- `EarlyStopping`: Early stopping callback
- `ModelCheckpoint`: Checkpoint callback
- `TrainingMetrics`: Training metrics

**Usage**:
```python
from services.ai.training import LoRAFineTuner, Trainer, EarlyStopping

fine_tuner = LoRAFineTuner("bert-base-uncased", r=16, alpha=32)
fine_tuner.load_model(num_labels=3)
history = fine_tuner.train(train_loader, val_loader, num_epochs=10)
```

### 5. Evaluation (`evaluation/`)
**Purpose**: Model evaluation

**Exports**:
- `ClassificationMetrics`: Classification metrics
- `ModelEvaluator`: Model evaluator
- `cross_validate`: K-fold cross-validation

**Usage**:
```python
from services.ai.evaluation import ModelEvaluator, ClassificationMetrics

evaluator = ModelEvaluator(model, device)
results = evaluator.evaluate(dataloader)
metrics = results["metrics"]  # accuracy, precision, recall, f1
```

### 6. Optimization (`optimization/`)
**Purpose**: Model optimization

**Exports**:
- `ModelQuantizer`: Quantization (INT8)
- `ModelPruner`: Pruning
- `ONNXExporter`: ONNX export
- `compare_models`: Model comparison

**Usage**:
```python
from services.ai.optimization import ModelQuantizer, ONNXExporter

quantized = ModelQuantizer.quantize_dynamic(model, dtype=torch.qint8)
ONNXExporter.export(model, "model.onnx", sample_input)
```

### 7. Utils (`utils/`)
**Purpose**: General utilities

**Exports**:
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

### 8. Services (`services/`)
**Purpose**: High-level AI services

**Exports**:
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

### 9. Tracking (`tracking/`)
**Purpose**: Experiment tracking and model versioning

**Exports**:
- `ExperimentTracker`: Experiment tracking (wandb/tensorboard/mlflow)
- `ModelRegistry`: Model registry
- `ModelVersion`: Model version
- `load_model_config`: Load config

**Usage**:
```python
from services.ai.tracking import ExperimentTracker, ModelRegistry

with ExperimentTracker("project", backend="wandb") as tracker:
    tracker.log_metric("accuracy", 0.95)

registry = ModelRegistry()
registry.register_model("model", "1.0.0", "./model.pth")
```

### 10. Interfaces (`interfaces/`)
**Purpose**: User interfaces

**Exports**:
- `GradioInterface`: Gradio web interfaces

**Usage**:
```python
from services.ai.interfaces import GradioInterface

interface = GradioInterface(
    text_generation_service=text_gen,
    diffusion_service=diffusion
)
interface.create_combined_interface(port=7860)
```

### 11. Config (`config/`) ✨ NEW
**Purpose**: Configuration management

**Exports**:
- `ConfigManager`: Configuration manager
- `get_config_manager`: Get global config manager
- `ModelConfig`: Model configuration
- `TrainingConfig`: Training configuration
- `LoRAConfig`: LoRA configuration

**Usage**:
```python
from services.ai.config import get_config_manager, ConfigManager

# Get config manager
config_manager = get_config_manager("model_config.yaml")

# Get model config
embedding_config = config_manager.get_model_config("embedding")
training_config = config_manager.get_training_config()
```

### 12. Pipelines (`pipelines/`) ✨ NEW
**Purpose**: End-to-end pipelines

**Exports**:
- `TrainingPipeline`: Complete training pipeline

**Usage**:
```python
from services.ai.pipelines import TrainingPipeline

pipeline = TrainingPipeline(
    model_name="bert-base-uncased",
    num_labels=3,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    use_lora=True
)

history = pipeline.train(
    num_epochs=10,
    learning_rate=2e-4,
    checkpoint_dir="./checkpoints"
)

results = pipeline.evaluate(test_loader)
pipeline.save_model("./models/final")
```

## Complete Example: End-to-End Training

```python
from services.ai.pipelines import TrainingPipeline
from services.ai.data import TextDataset, TextPreprocessor
from services.ai.tracking import ExperimentTracker, ModelRegistry
from services.ai.config import get_config_manager
from services.ai.utils import TrainingVisualizer
from transformers import AutoTokenizer

# 1. Load configuration
config_manager = get_config_manager("model_config.yaml")
training_config = config_manager.get_training_config()

# 2. Preprocess data
preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
train_texts = [preprocessor.preprocess(text) for text in raw_texts]

# 3. Create datasets
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = TextDataset(train_texts, tokenizer, metadata=[{"label": l} for l in labels])
val_dataset = TextDataset(val_texts, tokenizer, metadata=[{"label": l} for l in val_labels])

# 4. Setup pipeline
pipeline = TrainingPipeline(
    model_name="bert-base-uncased",
    num_labels=3,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    use_lora=True,
    config={
        "lora_r": 16,
        "lora_alpha": 32,
        "early_stopping_patience": 5
    }
)

# 5. Train with tracking
with ExperimentTracker("sentiment-classification", backend="wandb") as tracker:
    history = pipeline.train(
        num_epochs=10,
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        checkpoint_dir="./checkpoints",
        tracker=tracker
    )

# 6. Visualize
TrainingVisualizer.plot_training_curves(history, save_path="./plots/training.png")

# 7. Register model
registry = ModelRegistry()
registry.register_model(
    model_name="sentiment-classifier",
    version="1.0.0",
    model_path="./checkpoints/best_checkpoint.pt",
    metadata={
        "accuracy": history["val_accuracy"][-1],
        "f1": 0.93
    }
)
```

## Import Patterns

### Recommended: By Module
```python
from services.ai.training import LoRAFineTuner
from services.ai.evaluation import ModelEvaluator
from services.ai.models import TransformerClassifier
from services.ai.config import get_config_manager
from services.ai.pipelines import TrainingPipeline
```

### Also Works: From Root
```python
from services.ai import (
    LoRAFineTuner,
    ModelEvaluator,
    TransformerClassifier,
    TrainingPipeline
)
```

## Module Dependencies

```
core/
  └── (no dependencies)

data/
  └── depends on: core

models/
  └── depends on: core

training/
  └── depends on: core, data, models

evaluation/
  └── depends on: core, data, models

optimization/
  └── depends on: core, models

utils/
  └── depends on: core

services/
  └── depends on: core, data

tracking/
  └── (no dependencies)

interfaces/
  └── depends on: services

config/
  └── (no dependencies)

pipelines/
  └── depends on: training, evaluation, tracking, utils
```

## Best Practices Implemented

### ✅ Model Architecture
- Custom nn.Module classes
- Proper weight initialization (Xavier, Kaiming, BERT)
- Normalization layers
- Custom heads (classification, regression)
- Multi-task learning support

### ✅ Configuration
- YAML config files
- Environment variable support
- Pydantic validation
- Type checking
- Default values

### ✅ Pipelines
- End-to-end workflows
- Automatic setup
- Integrated callbacks
- Error handling
- Logging

### ✅ All Previous Best Practices
- Training, evaluation, optimization
- Debugging, profiling, visualization
- Multi-GPU, experiment tracking
- Interfaces, versioning

## Statistics

- **12 modules** specialized
- **100+ classes and functions**
- **Complete coverage** of deep learning workflows
- **Production-ready** code
- **Comprehensive documentation**

## Conclusion

The system is now:
- ✅ **Highly modular** - 12 specialized modules
- ✅ **Well organized** - Clear structure
- ✅ **Fully featured** - All capabilities
- ✅ **Production ready** - Best practices throughout
- ✅ **Well documented** - Complete documentation
- ✅ **Easy to use** - Clear APIs

All code follows PyTorch, Transformers, and Diffusers best practices.















