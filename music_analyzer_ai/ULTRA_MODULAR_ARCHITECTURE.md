# Ultra-Modular Architecture

## Overview
This document describes the ultra-modular architecture where every component is separated into its own micro-module, enabling maximum flexibility and composability.

## Architecture Principles

### 1. Micro-Modules
Every component is in its own file/module:
- Single Responsibility Principle
- Easy to test in isolation
- Easy to replace or extend

### 2. Composition Over Inheritance
Build complex systems by composing simple parts:
- ModelComposer for building models
- Compose for transformations
- Pipeline composition

### 3. Registry Pattern
Centralized registry for all components:
- Dynamic component discovery
- Easy registration of custom components
- Plugin-like extensibility

### 4. Factory Pattern
Factories for creating instances:
- Consistent creation interface
- Easy to extend with new types
- Configuration-driven creation

## Component Structure

### Core Components

#### Registry System (`core/registry.py`)
- **ComponentRegistry**: Central registry for all components
- **Registration methods**: Models, losses, optimizers, schedulers, metrics, callbacks
- **Discovery**: Get components by name
- **Global registry**: Singleton pattern for easy access

#### Composition System (`core/composition.py`)
- **ModelComposer**: Build complex models from components
- **SequentialComposer**: Build sequential models
- **ParallelComposer**: Build parallel processing branches
- **ComposedModel**: Runtime model composition

### Data Transformations (`data/transforms/`)

#### Audio Transforms (`audio_transforms.py`)
- **AudioNormalizer**: Normalize audio amplitude
- **AudioResampler**: Resample to target sample rate
- **AudioTrimmer**: Trim silence
- **AudioPadder**: Pad to target length
- **AudioAugmenter**: Apply augmentations

#### Feature Transforms (`feature_transforms.py`)
- **FeatureNormalizer**: Normalize features (standard, minmax, robust)
- **FeatureScaler**: Scale features by factor
- **FeatureSelector**: Select features by indices
- **FeatureCombiner**: Combine multiple feature arrays

#### Composition (`compose.py`)
- **Compose**: Chain transformations
- **ComposeTransforms**: Compose with different I/O types

### Training Loops (`training/loops/`)

#### Base Loop (`base_loop.py`)
- **BaseTrainingLoop**: Abstract base class
- **train_step()**: Abstract method for training step
- **train_epoch()**: Abstract method for training epoch
- **validate_step()**: Validation step implementation
- **validate_epoch()**: Validation epoch implementation

#### Standard Loop (`standard_loop.py`)
- **StandardTrainingLoop**: Single-GPU training
- **Gradient accumulation**: Support for large batches
- **Mixed precision**: FP16 training
- **Gradient clipping**: Prevent exploding gradients
- **NaN/Inf detection**: Automatic error handling

### Inference Pipelines (`inference/pipelines/`)

#### Base Pipeline (`base_pipeline.py`)
- **BaseInferencePipeline**: Abstract base class
- **preprocess()**: Preprocessing hook
- **postprocess()**: Postprocessing hook
- **predict()**: Abstract inference method

#### Standard Pipeline (`standard_pipeline.py`)
- **StandardInferencePipeline**: Single sample inference
- **Mixed precision**: FP16 inference
- **Automatic tensor conversion**: NumPy to PyTorch
- **Error handling**: Comprehensive error catching

## Usage Examples

### Using the Registry

```python
from music_analyzer_ai.core.registry import (
    get_registry,
    register_model,
    register_loss
)
from music_analyzer_ai.models.modular_transformer import ModularMusicClassifier

# Register components
registry = get_registry()
register_model("music_classifier", ModularMusicClassifier)

# Get component
ModelClass = registry.get_model("music_classifier")
model = ModelClass(input_dim=169, embed_dim=256)
```

### Composing Models

```python
from music_analyzer_ai.core.composition import ModelComposer
from music_analyzer_ai.models.architectures import (
    MusicFeatureEmbedding,
    MultiHeadAttention,
    ResidualFeedForward
)

# Build model using composer
composer = ModelComposer()

composer.add_component("embedding", MusicFeatureEmbedding(169, 256), is_input=True)
composer.add_component("attention", MultiHeadAttention(256, 8))
composer.add_component("ff", ResidualFeedForward(256, 1024))
composer.add_component("output", nn.Linear(256, 10), is_output=True)

composer.connect("embedding", "attention")
composer.connect("attention", "ff")
composer.connect("ff", "output")

model = composer.build()
```

### Composing Transformations

```python
from music_analyzer_ai.data.transforms import (
    AudioNormalizer,
    AudioResampler,
    AudioTrimmer,
    Compose
)

# Create transformation pipeline
transforms = Compose([
    AudioResampler(target_sr=22050),
    AudioTrimmer(top_db=20.0),
    AudioNormalizer(method="peak")
])

# Apply transformations
processed_audio, sr = transforms(audio, sr)
```

### Using Training Loops

```python
from music_analyzer_ai.training.loops import StandardTrainingLoop
from music_analyzer_ai.training.components import create_optimizer, MultiTaskLoss

# Create training loop
loop = StandardTrainingLoop(
    model=model,
    optimizer=create_optimizer("adamw", model.parameters()),
    loss_fn=MultiTaskLoss(task_losses={...}),
    gradient_accumulation_steps=4,
    max_grad_norm=1.0
)

# Train
for epoch in range(num_epochs):
    train_metrics = loop.train_epoch(train_loader, epoch)
    val_metrics = loop.validate_epoch(val_loader)
```

### Using Inference Pipelines

```python
from music_analyzer_ai.inference.pipelines import StandardInferencePipeline

# Create inference pipeline
pipeline = StandardInferencePipeline(
    model=model,
    preprocess_fn=preprocess_audio,
    postprocess_fn=postprocess_predictions
)

# Run inference
result = pipeline.predict(audio_data)
```

## Benefits of Ultra-Modular Architecture

### 1. Maximum Flexibility
- Mix and match any components
- Easy to experiment with different combinations
- No rigid structure constraints

### 2. Easy Testing
- Test each component in isolation
- Mock dependencies easily
- Fast unit tests

### 3. Easy Extension
- Add new components without modifying existing code
- Register custom components in registry
- Compose new models from existing components

### 4. Better Organization
- Clear separation of concerns
- Easy to find and understand code
- Self-documenting structure

### 5. Reusability
- Components can be used across projects
- Share components between models
- Build library of reusable components

## Component Registration

### Auto-Registration
Components can auto-register themselves:

```python
from music_analyzer_ai.core.registry import register_model

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ...
```

### Manual Registration
Register components manually:

```python
from music_analyzer_ai.core.registry import get_registry

registry = get_registry()
registry.register_model("my_model", MyModel)
registry.register_loss("my_loss", MyLoss)
```

## Composition Patterns

### Sequential Composition
```python
from music_analyzer_ai.core.composition import SequentialComposer

composer = SequentialComposer()
composer.add(nn.Linear(169, 256))
composer.add(nn.ReLU())
composer.add(nn.Linear(256, 10))

model = composer.build()  # nn.Sequential
```

### Parallel Composition
```python
from music_analyzer_ai.core.composition import ParallelComposer

composer = ParallelComposer()
composer.add_branch("genre", GenreHead())
composer.add_branch("mood", MoodHead())
composer.set_merge_strategy("concat")

model = composer.build()  # ParallelModel
```

### Complex Composition
```python
composer = ModelComposer()
composer.add_component("embed", Embedding(), is_input=True)
composer.add_component("branch1", Branch1())
composer.add_component("branch2", Branch2())
composer.add_component("merge", MergeLayer(), is_output=True)

composer.connect("embed", "branch1")
composer.connect("embed", "branch2")
composer.connect("branch1", "merge")
composer.connect("branch2", "merge")

model = composer.build()
```

## File Structure

```
music_analyzer_ai/
├── core/
│   ├── registry.py          # Component registry
│   └── composition.py       # Model composition
│
├── data/
│   └── transforms/          # Data transformations
│       ├── audio_transforms.py
│       ├── feature_transforms.py
│       └── compose.py
│
├── training/
│   └── loops/               # Training loops
│       ├── base_loop.py
│       └── standard_loop.py
│
├── inference/
│   └── pipelines/           # Inference pipelines
│       ├── base_pipeline.py
│       └── standard_pipeline.py
│
└── ... (other modules)
```

## Conclusion

The ultra-modular architecture provides:
- **Maximum flexibility**: Compose any combination of components
- **Easy testing**: Test components in isolation
- **Easy extension**: Add new components without modifying existing code
- **Better organization**: Clear separation of concerns
- **High reusability**: Components can be shared across projects

This architecture enables rapid experimentation, easy maintenance, and maximum code reuse.



