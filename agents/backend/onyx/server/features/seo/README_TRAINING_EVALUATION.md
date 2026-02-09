# Model Training and Evaluation Framework

A comprehensive framework for training and evaluating deep learning models for SEO tasks, featuring efficient data loading, advanced training loops, and comprehensive evaluation metrics.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Data Loading](#data-loading)
7. [Training Configuration](#training-configuration)
8. [Training Process](#training-process)
9. [Evaluation](#evaluation)
10. [Advanced Features](#advanced-features)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)

## Overview

The Model Training and Evaluation Framework provides a robust, scalable solution for training deep learning models on SEO data. It includes:

- **Efficient Data Loading**: Optimized PyTorch DataLoader with multi-worker support
- **Advanced Training Loops**: Mixed precision training, gradient accumulation, early stopping
- **Comprehensive Evaluation**: Multiple metrics for classification and regression tasks
- **Flexible Configuration**: Easy-to-use configuration system for different training scenarios
- **Monitoring & Logging**: TensorBoard integration and detailed metrics tracking

## Key Features

### 🚀 Performance Optimizations
- **Mixed Precision Training**: Automatic mixed precision (AMP) for faster training
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Efficient Data Loading**: Optimized DataLoader with pin_memory and num_workers
- **Gradient Accumulation**: Support for large effective batch sizes

### 📊 Comprehensive Monitoring
- **Real-time Metrics**: Live tracking of loss, accuracy, and learning rate
- **TensorBoard Integration**: Visual training progress and model analysis
- **Early Stopping**: Automatic stopping based on validation metrics
- **Checkpoint Management**: Automatic saving of best and latest models

### 🔧 Flexible Configuration
- **Multiple Optimizers**: Adam, AdamW, SGD, RMSprop
- **Various Schedulers**: Cosine, Step, Plateau, Warmup Cosine
- **Custom Loss Functions**: Support for custom loss implementations
- **Task-Specific Metrics**: Classification and regression evaluation

## Architecture

```
model_training_evaluation.py
├── TrainingConfig          # Configuration dataclass
├── TrainingMetrics         # Metrics tracking
├── EfficientDataLoader     # Optimized data loading
├── ModelTrainer           # Main training loop
└── ModelEvaluator         # Evaluation utilities
```

### Core Components

1. **TrainingConfig**: Centralized configuration management
2. **EfficientDataLoader**: Optimized PyTorch DataLoader wrapper
3. **ModelTrainer**: Comprehensive training loop with monitoring
4. **ModelEvaluator**: Multi-metric evaluation system

## Installation

```bash
# Install required dependencies
pip install torch torchvision torchaudio
pip install tensorboard scikit-learn matplotlib seaborn
pip install transformers tqdm pandas numpy

# For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Training Example

```python
from model_training_evaluation import TrainingConfig, ModelTrainer, ModelEvaluator
from torch.utils.data import Dataset, DataLoader

# 1. Create your dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return (features, labels) or dict with 'features' and 'labels' keys
        return self.data[idx]

# 2. Create your model
model = YourModel()

# 3. Setup training configuration
config = TrainingConfig(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    optimizer="adamw",
    scheduler="cosine"
)

# 4. Train the model
trainer = ModelTrainer(config)
metrics = trainer.train()

# 5. Evaluate the model
evaluator = ModelEvaluator(model)
test_loader = DataLoader(test_dataset, batch_size=32)
test_metrics = evaluator.evaluate(test_loader, task_type="classification")
```

## Data Loading

### EfficientDataLoader Features

The `EfficientDataLoader` class provides optimized data loading with:

- **Multi-worker Support**: Parallel data loading with `num_workers`
- **Memory Pinning**: `pin_memory=True` for faster GPU transfer
- **Distributed Training**: Automatic DistributedSampler for multi-GPU
- **Flexible Batching**: Configurable batch size and drop_last

### Custom Dataset Example

```python
class SEODataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text if tokenizer provided
        if self.tokenizer:
            encoding = self.tokenizer(
                item['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(item['labels'])
            }
        else:
            # Feature-based approach
            return {
                'features': torch.tensor(item['features']),
                'labels': torch.tensor(item['labels'])
            }
```

## Training Configuration

### TrainingConfig Parameters

```python
config = TrainingConfig(
    # Model configuration
    model=your_model,
    model_name="seo_model",
    
    # Data configuration
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    
    # Training configuration
    epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-5,
    optimizer="adamw",  # "adam", "adamw", "sgd", "rmsprop"
    scheduler="cosine",  # "cosine", "step", "plateau", "warmup_cosine"
    gradient_clip_val=1.0,
    gradient_accumulation_steps=1,
    
    # Loss configuration
    loss_function="cross_entropy",  # "cross_entropy", "focal", "mse", "mae"
    class_weights=class_weights_tensor,
    
    # Device configuration
    device="cuda",
    use_mixed_precision=True,
    use_distributed=False,
    
    # Logging configuration
    log_dir="./logs",
    tensorboard=True,
    
    # Checkpoint configuration
    checkpoint_dir="./checkpoints",
    save_best_only=True
)
```

### Optimizer Options

```python
# Adam
optimizer="adam"

# AdamW (recommended for transformers)
optimizer="adamw"

# SGD with momentum
optimizer="sgd"

# RMSprop
optimizer="rmsprop"
```

### Scheduler Options

```python
# Cosine annealing
scheduler="cosine"

# Step decay
scheduler="step"

# Reduce on plateau
scheduler="plateau"

# Warmup cosine (for transformers)
scheduler="warmup_cosine"
```

## Training Process

### Training Loop Features

1. **Mixed Precision Training**: Automatic FP16 training for speed
2. **Gradient Clipping**: Prevents gradient explosion
3. **Gradient Accumulation**: Effective large batch sizes
4. **Early Stopping**: Stops when validation doesn't improve
5. **Checkpointing**: Saves best and latest models

### Training Monitoring

```python
# Access training metrics
metrics = trainer.train()

print(f"Best validation loss: {metrics.best_val_loss:.4f}")
print(f"Best validation accuracy: {metrics.best_val_accuracy:.4f}")
print(f"Training time: {sum(metrics.epoch_times):.2f} seconds")

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(metrics.train_loss, label='Train Loss')
plt.plot(metrics.val_loss, label='Val Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(metrics.train_accuracy, label='Train Accuracy')
plt.plot(metrics.val_accuracy, label='Val Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
```

### TensorBoard Integration

```python
# View training progress
tensorboard --logdir=./logs

# Or programmatically
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs')

# Add custom metrics
writer.add_scalar('Custom/Metric', value, step)
writer.add_histogram('Weights/Linear', model.fc.weight, step)
```

## Evaluation

### ModelEvaluator Features

- **Multi-task Support**: Handles classification and regression
- **Comprehensive Metrics**: Accuracy, F1, ROC-AUC, MSE, MAE, R²
- **Confusion Matrix**: Detailed classification analysis
- **Report Generation**: Detailed evaluation reports

### Evaluation Example

```python
# Create evaluator
evaluator = ModelEvaluator(model)

# Evaluate on test set
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_metrics = evaluator.evaluate(test_loader, task_type="classification")

# Print results
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")

# Generate detailed report
report = evaluator.generate_report(test_loader, task_type="classification")
print(report)
```

### Multi-task Evaluation

```python
# For multi-task models
test_metrics = evaluator.evaluate(test_loader, task_type="regression")

print(f"Test MSE: {test_metrics['mse']:.4f}")
print(f"Test MAE: {test_metrics['mae']:.4f}")
print(f"Test R²: {test_metrics['r2_score']:.4f}")
print(f"Test RMSE: {test_metrics['rmse']:.4f}")
```

## Advanced Features

### Custom Loss Functions

```python
class CustomSEOLoss(nn.Module):
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or torch.ones(4)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        total_loss = 0.0
        for i in range(predictions.size(1)):
            task_loss = self.mse_loss(predictions[:, i], targets[:, i])
            total_loss += self.task_weights[i] * task_loss
        return total_loss

# Use custom loss
trainer = ModelTrainer(config)
trainer.criterion = CustomSEOLoss()
metrics = trainer.train()
```

### Distributed Training

```python
# Multi-GPU training
config = TrainingConfig(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    use_distributed=True,
    local_rank=0  # Set by torchrun
)

# Run with torchrun
# torchrun --nproc_per_node=4 train_script.py
```

### Checkpoint Management

```python
# Save checkpoint
trainer.save_checkpoint("model_checkpoint.pth")

# Load checkpoint
trainer.load_checkpoint("model_checkpoint.pth")

# Resume training
metrics = trainer.train()
```

## Best Practices

### 1. Data Loading Optimization

```python
# Optimal DataLoader settings
config = TrainingConfig(
    batch_size=32,
    num_workers=4,  # Adjust based on CPU cores
    pin_memory=True,  # Faster GPU transfer
    drop_last=True,  # Consistent batch sizes
    shuffle=True
)
```

### 2. Learning Rate Scheduling

```python
# For transformers
config = TrainingConfig(
    scheduler="warmup_cosine",
    warmup_steps=1000,
    learning_rate=5e-5
)

# For CNN/MLP
config = TrainingConfig(
    scheduler="cosine",
    learning_rate=1e-3
)
```

### 3. Mixed Precision Training

```python
# Enable for faster training
config = TrainingConfig(
    use_mixed_precision=True,
    gradient_clip_val=1.0  # Important for mixed precision
)
```

### 4. Early Stopping

```python
# Prevent overfitting
config = TrainingConfig(
    early_stopping_patience=10,
    early_stopping_min_delta=1e-4
)
```

### 5. Model Checkpointing

```python
# Save best model only
config = TrainingConfig(
    save_best_only=True,
    save_interval=5,
    max_checkpoints=5
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   config.batch_size = 16
   
   # Enable gradient accumulation
   config.gradient_accumulation_steps = 2
   
   # Use mixed precision
   config.use_mixed_precision = True
   ```

2. **Slow Training**
   ```python
   # Increase num_workers
   config.num_workers = 8
   
   # Enable pin_memory
   config.pin_memory = True
   
   # Use mixed precision
   config.use_mixed_precision = True
   ```

3. **Poor Convergence**
   ```python
   # Adjust learning rate
   config.learning_rate = 1e-4
   
   # Use better optimizer
   config.optimizer = "adamw"
   
   # Add weight decay
   config.weight_decay = 1e-5
   ```

4. **Overfitting**
   ```python
   # Reduce model capacity
   # Add dropout
   # Use early stopping
   config.early_stopping_patience = 10
   ```

### Performance Monitoring

```python
# Monitor GPU usage
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Monitor training speed
import time
start_time = time.time()
# ... training code ...
print(f"Training time: {time.time() - start_time:.2f} seconds")
```

## API Reference

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Model configuration
    model: nn.Module
    model_name: str = "seo_model"
    
    # Data configuration
    train_dataset: Dataset
    val_dataset: Dataset = None
    test_dataset: Dataset = None
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    
    # Training configuration
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Loss configuration
    loss_function: str = "cross_entropy"
    class_weights: Optional[torch.Tensor] = None
    
    # Device configuration
    device: str = "cuda"
    use_mixed_precision: bool = True
    use_distributed: bool = False
    local_rank: int = -1
    
    # Logging configuration
    log_dir: str = "./logs"
    tensorboard: bool = True
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    max_checkpoints: int = 5
```

### ModelTrainer

```python
class ModelTrainer:
    def __init__(self, config: TrainingConfig)
    def train(self) -> TrainingMetrics
    def train_epoch(self, epoch: int) -> Dict[str, float]
    def validate_epoch(self) -> Dict[str, float]
    def save_checkpoint(self, filename: str, is_best: bool = False)
    def load_checkpoint(self, filepath: str) -> int
```

### ModelEvaluator

```python
class ModelEvaluator:
    def __init__(self, model: nn.Module, device: str = "cuda")
    def evaluate(self, data_loader: DataLoader, task_type: str = "classification") -> Dict[str, float]
    def generate_report(self, data_loader: DataLoader, task_type: str = "classification") -> str
```

### EfficientDataLoader

```python
class EfficientDataLoader:
    def __init__(self, config: TrainingConfig)
    def get_train_loader(self) -> DataLoader
    def get_val_loader(self) -> DataLoader
    def get_test_loader(self) -> DataLoader
    def set_epoch(self, epoch: int)
```

## Examples

See `example_training_evaluation.py` for comprehensive examples including:

- Basic training with feature-based datasets
- Custom loss function training
- Transformer model training
- Checkpoint loading and resuming
- Model comparison and hyperparameter tuning
- Multi-task learning scenarios

## Contributing

When contributing to the training framework:

1. Follow the existing code style
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation for API changes
5. Test with different model architectures

## License

This framework is part of the SEO Deep Learning System and follows the same licensing terms. 