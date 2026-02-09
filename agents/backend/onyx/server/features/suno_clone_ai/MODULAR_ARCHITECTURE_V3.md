# Modular Architecture v3 - Deep Learning Best Practices

## Overview

This document describes the highly modular architecture implemented following deep learning best practices. The codebase is now organized into clear, focused modules with single responsibilities.

## 📁 Complete Module Structure

```
core/
├── data/                    # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   └── transforms.py        # Data transforms and augmentation
│
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── enhanced_music_model.py    # Custom transformer models
│   ├── lora_adapter.py            # LoRA fine-tuning
│   └── enhanced_diffusion.py      # Diffusion models
│
├── generators/              # Music generators
│   ├── __init__.py
│   ├── base_generator.py           # Abstract base class
│   ├── transformers_generator.py   # Transformers-based generator
│   └── refactored_generator.py    # Refactored version
│
├── training/                # Training components
│   ├── __init__.py
│   ├── enhanced_training.py       # Training pipeline
│   ├── optimizers.py              # Optimizer utilities
│   ├── losses.py                  # Loss functions
│   └── callbacks.py               # Training callbacks
│
├── utils/                   # Shared utilities
│   ├── __init__.py
│   ├── model_utils.py             # Model utilities
│   └── mixed_precision.py         # Mixed precision management
│
└── evaluation/              # Evaluation (future)
    └── metrics.py
```

## 🎯 Module Responsibilities

### 1. Data Module (`core/data/`)

**Purpose**: Handle all data-related operations

#### `dataset.py`
- `MusicDataset`: Main dataset for audio-text pairs
- `AudioTextDataset`: Simple dataset for preprocessed data
- Handles loading from JSON files or directories
- Audio resampling and trimming
- Metadata handling

#### `transforms.py`
- `AudioNormalize`: Normalize audio to [-1, 1]
- `AudioTrim`: Trim audio to specified length
- `AudioPad`: Pad audio to target length
- `AudioAugmentation`: Data augmentation (noise, time stretch, etc.)
- `ComposeTransforms`: Compose multiple transforms
- `create_audio_transform_pipeline()`: Factory function

**Usage**:
```python
from core.data import MusicDataset, create_audio_transform_pipeline

# Create transform pipeline
transform = create_audio_transform_pipeline(
    normalize=True,
    trim_to=32000 * 30,  # 30 seconds
    augment=True
)

# Create dataset
dataset = MusicDataset(
    data_path="./data",
    sample_rate=32000,
    max_duration=30,
    transform=transform
)
```

### 2. Models Module (`core/models/`)

**Purpose**: Define model architectures

#### Components:
- `EnhancedMusicModel`: Custom transformer architecture
- `LoRAAdapter`: LoRA fine-tuning support
- `EnhancedDiffusionGenerator`: Diffusion model implementation

**Usage**:
```python
from core.models import EnhancedMusicModel, add_lora_to_model

# Create model
model = EnhancedMusicModel(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

# Add LoRA for fine-tuning
adapter = add_lora_to_model(model, rank=8, alpha=16.0)
```

### 3. Generators Module (`core/generators/`)

**Purpose**: Music generation interfaces

#### Components:
- `BaseMusicGenerator`: Abstract base class
- `TransformersMusicGenerator`: Hugging Face Transformers implementation
- `RefactoredMusicGenerator`: Backward-compatible version

**Usage**:
```python
from core.generators import TransformersMusicGenerator

generator = TransformersMusicGenerator(
    model_name="facebook/musicgen-medium",
    use_mixed_precision=True
)

audio = generator.generate("Upbeat electronic music", duration=30)
```

### 4. Training Module (`core/training/`)

**Purpose**: Training pipeline and components

#### `enhanced_training.py`
- `EnhancedTrainingPipeline`: Complete training pipeline
- `EvaluationMetrics`: Evaluation metrics
- `create_train_val_test_split()`: Data splitting
- `cross_validate()`: K-fold cross-validation

#### `optimizers.py`
- `create_optimizer()`: Factory for optimizers (Adam, AdamW, SGD, etc.)
- `create_scheduler()`: Factory for LR schedulers
- `create_warmup_scheduler()`: Warmup scheduler
- `get_parameter_groups()`: Parameter grouping for different weight decay

#### `losses.py`
- `MSELoss`: Mean Squared Error
- `MAELoss`: Mean Absolute Error
- `SpectralLoss`: Spectral domain loss
- `CombinedLoss`: Combine multiple losses
- `create_loss_function()`: Factory function

#### `callbacks.py`
- `EarlyStopping`: Early stopping callback
- `ModelCheckpoint`: Model checkpointing
- `LearningRateMonitor`: LR monitoring

**Usage**:
```python
from core.training import (
    EnhancedTrainingPipeline,
    create_optimizer,
    create_scheduler,
    create_loss_function,
    EarlyStopping,
    ModelCheckpoint
)

# Create optimizer
optimizer = create_optimizer(
    model,
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=0.01
)

# Create scheduler
scheduler = create_scheduler(
    optimizer,
    scheduler_type="cosine",
    T_max=100
)

# Create loss
criterion = create_loss_function("mse")

# Create callbacks
early_stopping = EarlyStopping(patience=10)
checkpoint = ModelCheckpoint(save_dir="./checkpoints")

# Create training pipeline
pipeline = EnhancedTrainingPipeline(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch_size=4,
    use_mixed_precision=True
)

pipeline.setup_training(
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    early_stopping=early_stopping
)

# Train
history = pipeline.train(num_epochs=100)
```

### 5. Utils Module (`core/utils/`)

**Purpose**: Shared utilities

#### `model_utils.py`
- Weight initialization
- Gradient clipping
- NaN/Inf detection
- Device management
- GPU optimizations
- Model compilation
- Parameter counting

#### `mixed_precision.py`
- `MixedPrecisionManager`: Complete AMP management
- Automatic gradient scaling
- Loss scaling

**Usage**:
```python
from core.utils import (
    initialize_weights,
    clip_gradients,
    get_device,
    MixedPrecisionManager
)

# Initialize weights
model.apply(initialize_weights)

# Get device
device = get_device(use_gpu=True)

# Mixed precision
amp_manager = MixedPrecisionManager(enabled=True)
with amp_manager.autocast():
    output = model(input)
```

## 🔄 Complete Training Example

```python
from core.data import MusicDataset, create_audio_transform_pipeline
from core.models import EnhancedMusicModel
from core.training import (
    EnhancedTrainingPipeline,
    create_optimizer,
    create_scheduler,
    create_loss_function,
    EarlyStopping,
    ModelCheckpoint,
    create_train_val_test_split
)
from core.utils import set_seed, get_device

# Set seed for reproducibility
set_seed(42)

# Create transform pipeline
transform = create_audio_transform_pipeline(
    normalize=True,
    trim_to=32000 * 30,
    augment=True
)

# Load dataset
full_dataset = MusicDataset(
    data_path="./data",
    sample_rate=32000,
    max_duration=30,
    transform=transform
)

# Split dataset
train_ds, val_ds, test_ds = create_train_val_test_split(
    full_dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Create model
model = EnhancedMusicModel(
    vocab_size=32000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

# Get device
device = get_device(use_gpu=True)
model.to(device)

# Create optimizer
optimizer = create_optimizer(
    model,
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=0.01
)

# Create scheduler
scheduler = create_scheduler(
    optimizer,
    scheduler_type="cosine",
    T_max=100
)

# Create loss
criterion = create_loss_function("mse")

# Create callbacks
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
checkpoint = ModelCheckpoint(
    save_dir="./checkpoints",
    save_best=True,
    monitor="val_loss"
)

# Create training pipeline
pipeline = EnhancedTrainingPipeline(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch_size=4,
    num_workers=4,
    use_mixed_precision=True,
    gradient_clip_norm=1.0,
    device=device
)

# Setup training
pipeline.setup_training(
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    early_stopping=early_stopping,
    use_wandb=True,
    use_tensorboard=True
)

# Train
history = pipeline.train(
    num_epochs=100,
    save_dir="./checkpoints"
)
```

## ✨ Benefits of Modular Architecture

### 1. Single Responsibility
Each module has one clear purpose:
- `data/`: Data operations only
- `models/`: Model architectures only
- `generators/`: Generation interfaces only
- `training/`: Training components only
- `utils/`: Shared utilities only

### 2. Easy to Extend
Adding new features is straightforward:
- New dataset? Add to `data/dataset.py`
- New loss? Add to `training/losses.py`
- New optimizer? Add to `training/optimizers.py`
- New generator? Extend `BaseMusicGenerator`

### 3. Easy to Test
Each module can be tested independently:
```python
# Test dataset
def test_music_dataset():
    dataset = MusicDataset("./data")
    assert len(dataset) > 0
    item = dataset[0]
    assert 'audio' in item
    assert 'text' in item

# Test optimizer
def test_create_optimizer():
    model = create_test_model()
    optimizer = create_optimizer(model, optimizer_type="adamw")
    assert isinstance(optimizer, torch.optim.AdamW)

# Test loss
def test_spectral_loss():
    loss_fn = SpectralLoss()
    pred = torch.randn(1, 32000)
    target = torch.randn(1, 32000)
    loss = loss_fn(pred, target)
    assert loss.item() >= 0
```

### 4. Easy to Maintain
- Clear structure makes it easy to find code
- Changes are localized to specific modules
- No code duplication
- Consistent patterns throughout

### 5. Reusability
Components can be reused across projects:
```python
# Reuse dataset in different project
from core.data import MusicDataset

# Reuse optimizer factory
from core.training import create_optimizer

# Reuse utilities
from core.utils import initialize_weights
```

## 📊 Module Dependencies

```
data/          (no dependencies on other modules)
    ↓
models/        (uses utils/)
    ↓
generators/    (uses models/, utils/)
    ↓
training/      (uses data/, models/, utils/)
```

## 🎯 Design Principles

1. **Separation of Concerns**: Each module handles one aspect
2. **Dependency Injection**: Pass dependencies rather than creating them
3. **Factory Functions**: Use factories for complex object creation
4. **Interface Segregation**: Small, focused interfaces
5. **Open/Closed Principle**: Open for extension, closed for modification

## 🚀 Future Extensions

### Evaluation Module (`core/evaluation/`)
- Audio quality metrics
- Perceptual metrics
- Comparison utilities

### Inference Module (`core/inference/`)
- Batch inference
- Streaming inference
- Model serving

### Monitoring Module (`core/monitoring/`)
- Training monitoring
- Model performance tracking
- Resource usage tracking

## 📝 Best Practices Followed

✅ **Modular Design**: Clear separation of concerns
✅ **Single Responsibility**: Each module has one purpose
✅ **DRY Principle**: No code duplication
✅ **Type Hints**: Full type annotations
✅ **Documentation**: Comprehensive docstrings
✅ **Error Handling**: Consistent error handling
✅ **Logging**: Proper logging throughout
✅ **Testing**: Testable structure
✅ **PEP 8**: Code style compliance
✅ **Factory Pattern**: For complex object creation

## 🔧 Configuration

All modules can be configured via:
1. **YAML Config**: `config/hyperparameters.yaml`
2. **Environment Variables**: Via `config/settings.py`
3. **Code Parameters**: Direct initialization

This modular architecture makes the codebase:
- **Maintainable**: Easy to understand and modify
- **Extensible**: Easy to add new features
- **Testable**: Easy to test components
- **Reusable**: Components can be used independently
- **Scalable**: Easy to scale and optimize



