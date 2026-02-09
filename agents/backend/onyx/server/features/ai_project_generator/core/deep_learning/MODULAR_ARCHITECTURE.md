# Modular Deep Learning Architecture

## Overview

This module has been refactored to follow deep learning best practices with a highly modular architecture. The codebase is organized into specialized modules that can be used independently or together.

## Architecture Principles

1. **Object-Oriented for Models**: All model architectures inherit from `BaseModel` (nn.Module)
2. **Functional for Data**: Data loading uses functional programming patterns
3. **Best Practices**: Mixed precision, gradient accumulation, early stopping, etc.
4. **Modularity**: Each component can be used independently
5. **Type Safety**: Full type hints throughout
6. **Error Handling**: Comprehensive error handling and logging

## Module Structure

```
deep_learning/
├── models/              # Model architectures (nn.Module classes)
│   ├── base_model.py   # Abstract base class
│   ├── transformer_model.py
│   └── factory.py       # Model factory
│
├── data/               # Data loading (functional patterns)
│   ├── datasets.py     # Dataset classes
│   └── dataloader_utils.py
│
├── training/           # Training utilities
│   ├── trainer.py      # Main trainer with best practices
│   └── optimizers.py   # Optimizer/scheduler factories
│
├── evaluation/         # Evaluation and metrics
│   └── metrics.py
│
├── inference/          # Inference and Gradio
│   ├── inference_engine.py
│   └── gradio_apps.py
│
├── config/             # Configuration management
│   └── config_manager.py
│
└── utils/              # Utilities
    ├── device_utils.py
    └── experiment_tracking.py
```

## Usage Examples

### 1. Creating a Model

```python
from core.deep_learning.models import TransformerModel, create_model

# Direct instantiation
model = TransformerModel(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

# Or using factory
model = create_model(
    model_type='transformer',
    config={'vocab_size': 10000, 'd_model': 512}
)
```

### 2. Data Loading

```python
from core.deep_learning.data import TextDataset, create_dataloader, train_val_test_split

# Create dataset
dataset = TextDataset(
    texts=texts,
    labels=labels,
    tokenizer=tokenizer,
    max_length=512
)

# Split dataset
train_ds, val_ds, test_ds = train_val_test_split(dataset)

# Create data loaders
train_loader = create_dataloader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### 3. Training

```python
from core.deep_learning.training import (
    Trainer, TrainingConfig, EarlyStopping,
    create_optimizer, create_scheduler
)

# Create optimizer and scheduler
optimizer = create_optimizer(model, optimizer_type='adamw', learning_rate=1e-4)
scheduler = create_scheduler(optimizer, scheduler_type='cosine', num_epochs=10)

# Training config
config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    use_mixed_precision=True,
    gradient_accumulation_steps=1,
    early_stopping=EarlyStopping(patience=5)
)

# Create trainer
trainer = Trainer(model, config, optimizer, scheduler)

# Train
history = trainer.train(train_loader, val_loader)
```

### 4. Evaluation

```python
from core.deep_learning.evaluation import evaluate_model

metrics, info = evaluate_model(
    model,
    test_loader,
    device,
    task_type='classification',
    num_classes=10
)

print(metrics.to_dict())
```

### 5. Inference with Gradio

```python
from core.deep_learning.inference import create_text_classification_app

app = create_text_classification_app(
    model,
    tokenizer=tokenizer,
    class_names=['Class 0', 'Class 1']
)

app.launch(share=True)
```

### 6. Configuration Management

```python
from core.deep_learning.config import ConfigManager

# Load config
config_manager = ConfigManager()
config = config_manager.load(Path("config.yaml"))

# Get values
learning_rate = config_manager.get('training.learning_rate', default=1e-4)

# Set values
config_manager.set('training.learning_rate', 2e-4)

# Save config
config_manager.save(Path("config_updated.yaml"))
```

## Key Features

### Models
- ✅ Custom nn.Module classes
- ✅ Proper weight initialization
- ✅ Checkpoint saving/loading
- ✅ Device management
- ✅ Parameter counting

### Training
- ✅ Mixed precision (AMP)
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ Early stopping
- ✅ Learning rate scheduling
- ✅ Multi-GPU support
- ✅ NaN/Inf detection

### Data
- ✅ Efficient DataLoader configuration
- ✅ Train/val/test splitting
- ✅ Text and image datasets
- ✅ Custom collate functions

### Evaluation
- ✅ Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Regression metrics (MSE, MAE, RMSE, R²)
- ✅ Batch evaluation

### Inference
- ✅ Inference engine with error handling
- ✅ Batch inference
- ✅ Gradio integration
- ✅ Text and image classification apps

### Configuration
- ✅ YAML/JSON support
- ✅ Nested configuration access
- ✅ Configuration merging

## Best Practices Implemented

1. **GPU Utilization**: Automatic device detection, pin_memory, prefetch_factor
2. **Mixed Precision**: Automatic mixed precision with GradScaler
3. **Gradient Management**: Accumulation, clipping, proper zero_grad
4. **Error Handling**: Try-except blocks, NaN detection, logging
5. **Reproducibility**: Random seed setting, deterministic operations
6. **Experiment Tracking**: TensorBoard and W&B integration
7. **Modularity**: Independent, reusable components
8. **Type Safety**: Full type hints
9. **Documentation**: Comprehensive docstrings
10. **PEP 8**: Code style compliance

## Dependencies

All required dependencies are in `requirements.txt`:
- torch, torchvision, torchaudio
- transformers, diffusers
- gradio
- numpy, pandas
- tensorboard, wandb
- tqdm

## Migration from Legacy Code

The legacy generators are still available for backward compatibility. To migrate:

1. Replace direct model instantiation with `create_model()` or model classes
2. Use `create_dataloader()` instead of manual DataLoader creation
3. Use `Trainer` class instead of custom training loops
4. Use `evaluate_model()` for evaluation
5. Use `ConfigManager` for configuration

## Examples

See `examples/complete_example.py` for a full workflow example.



