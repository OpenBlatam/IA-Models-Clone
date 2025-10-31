# Comprehensive Training and Evaluation System

## Overview

This document provides a comprehensive overview of the advanced training and evaluation system implemented for deep learning models. The system integrates with all components of our deep learning framework and provides robust training loops, evaluation metrics, and monitoring capabilities.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Training Manager](#training-manager)
3. [Evaluator](#evaluator)
4. [Training Modes](#training-modes)
5. [Evaluation Modes](#evaluation-modes)
6. [Metrics and Monitoring](#metrics-and-monitoring)
7. [Advanced Features](#advanced-features)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)

## System Architecture

### Core Components

The training and evaluation system consists of several key components:

```python
class TrainingManager:
    """Comprehensive training manager."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.performance_monitor = PerformanceMonitor()
```

### Configuration System

```python
@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    lr_scheduler_patience: int = 5
    
    # Advanced features
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    checkpoint_save_frequency: int = 5
    
    # Monitoring
    log_frequency: int = 100
    eval_frequency: int = 500
    tensorboard_logging: bool = True
    wandb_logging: bool = False
```

## Training Manager

### Key Features

1. **Automatic Device Setup**: Supports CPU, CUDA, and MPS
2. **Mixed Precision Training**: Automatic mixed precision for faster training
3. **Gradient Accumulation**: Support for large effective batch sizes
4. **Early Stopping**: Prevents overfitting with configurable patience
5. **Checkpointing**: Automatic model saving and loading
6. **Logging Integration**: TensorBoard and Weights & Biases support

### Training Loop

```python
def train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               loss_function: nn.Module, train_dataloader: DataLoader,
               epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Prepare batch
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch
        else:
            inputs, targets = batch, None
        
        inputs = inputs.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        
        # Forward pass with mixed precision
        if self.config.mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if targets is not None:
                    loss = loss_function(outputs, targets)
                else:
                    loss = loss_function(outputs)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard training
            outputs = model(inputs)
            if targets is not None:
                loss = loss_function(outputs, targets)
            else:
                loss = loss_function(outputs)
            
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        # Update metrics
        epoch_loss += loss.item()
        num_batches += 1
        self.global_step += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
        # Logging
        if self.global_step % self.config.log_frequency == 0:
            self._log_training_step(loss.item(), optimizer.param_groups[0]['lr'])
    
    return {
        'epoch_loss': epoch_loss / num_batches,
        'num_batches': num_batches
    }
```

### Optimizer Setup

```python
def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
    """Setup optimizer."""
    if self.config.optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    elif self.config.optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    elif self.config.optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    return optimizer
```

### Scheduler Setup

```python
def setup_scheduler(self, optimizer: torch.optim.Optimizer, 
                   total_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
    """Setup learning rate scheduler."""
    if self.config.scheduler_type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
    elif self.config.scheduler_type.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif self.config.scheduler_type.lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.config.lr_scheduler_patience, factor=0.5
        )
    elif self.config.scheduler_type.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.config.learning_rate, total_steps=total_steps
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    
    return scheduler
```

## Evaluator

### Comprehensive Evaluation

The evaluator provides comprehensive evaluation capabilities:

```python
class Evaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.metrics_tracker = MetricsTracker()
    
    def compute_metrics(self, predictions: torch.Tensor, 
                       targets: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Determine task type based on predictions shape
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            # Classification task
            return self._compute_classification_metrics(predictions_np, targets_np)
        elif len(predictions.shape) == 1 or predictions.shape[1] == 1:
            # Regression task
            return self._compute_regression_metrics(predictions_np, targets_np)
        else:
            # Generation task
            return self._compute_generation_metrics(predictions_np, targets_np)
```

### Classification Metrics

```python
def _compute_classification_metrics(self, predictions: np.ndarray, 
                                  targets: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    # Convert to class predictions
    if len(predictions.shape) == 2:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    metrics = {}
    
    # Accuracy
    if 'accuracy' in self.config.classification_metrics:
        metrics['accuracy'] = accuracy_score(targets, pred_classes)
    
    # Precision, Recall, F1
    if any(metric in self.config.classification_metrics for metric in ['precision', 'recall', 'f1']):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, pred_classes, average='weighted'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
    
    # Confusion Matrix
    if 'confusion_matrix' in self.config.classification_metrics:
        cm = confusion_matrix(targets, pred_classes)
        metrics['confusion_matrix'] = cm
        
        # Plot confusion matrix
        if self.config.plot_metrics:
            self._plot_confusion_matrix(cm)
    
    return metrics
```

### Regression Metrics

```python
def _compute_regression_metrics(self, predictions: np.ndarray, 
                               targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    metrics = {}
    
    # MSE
    if 'mse' in self.config.regression_metrics:
        metrics['mse'] = np.mean((predictions - targets) ** 2)
    
    # MAE
    if 'mae' in self.config.regression_metrics:
        metrics['mae'] = np.mean(np.abs(predictions - targets))
    
    # RMSE
    if 'rmse' in self.config.regression_metrics:
        metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # R² Score
    if 'r2_score' in self.config.regression_metrics:
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        metrics['r2_score'] = 1 - (ss_res / ss_tot)
    
    return metrics
```

## Training Modes

### Supported Modes

```python
class TrainingMode(Enum):
    """Training modes."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    FEDERATED = "federated"
```

### Mode-Specific Configurations

#### Supervised Learning
```python
# Standard supervised training
training_config = TrainingConfig(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    mixed_precision=True,
    early_stopping_patience=10
)
```

#### Unsupervised Learning
```python
# Unsupervised training (e.g., autoencoders)
training_config = TrainingConfig(
    num_epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    mixed_precision=False,  # May not help with reconstruction loss
    early_stopping_patience=5
)
```

#### Semi-Supervised Learning
```python
# Semi-supervised training
training_config = TrainingConfig(
    num_epochs=200,
    batch_size=32,
    learning_rate=1e-4,
    gradient_accumulation_steps=2,  # Larger effective batch size
    early_stopping_patience=15
)
```

## Evaluation Modes

### Supported Modes

```python
class EvaluationMode(Enum):
    """Evaluation modes."""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    CROSS_VALIDATION = "cross_validation"
    ENSEMBLE = "ensemble"
```

### Cross-Validation

```python
def cross_validate(self, model_class: type, train_dataset: Dataset,
                   config: TrainingConfig, n_folds: int = 5) -> Dict[str, List[float]]:
    """Perform cross-validation."""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        self.logger.info(f"Training fold {fold + 1}/{n_folds}")
        
        # Create fold-specific datasets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=train_subsampler
        )
        val_loader = DataLoader(
            train_dataset, batch_size=config.eval_batch_size, sampler=val_subsampler
        )
        
        # Train model
        model = model_class()
        trainer = TrainingManager(config)
        evaluator = Evaluator(EvaluationConfig())
        
        training_results = trainer.train(
            model, train_loader, val_loader, evaluator=evaluator
        )
        
        # Get best validation metrics
        best_metrics = training_results['training_history']['val_metrics'][-1]
        fold_metrics.append(best_metrics)
    
    # Aggregate results
    aggregated_metrics = {}
    for metric_name in fold_metrics[0].keys():
        values = [fold[metric_name] for fold in fold_metrics]
        aggregated_metrics[f"{metric_name}_mean"] = np.mean(values)
        aggregated_metrics[f"{metric_name}_std"] = np.std(values)
    
    return aggregated_metrics
```

## Metrics and Monitoring

### Real-Time Monitoring

```python
# TensorBoard logging
if self.writer:
    self.writer.add_scalar('Loss/Train', train_results['epoch_loss'], epoch)
    if val_results:
        self.writer.add_scalar('Loss/Val', val_results.get('eval_loss', 0), epoch)
    self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

# WandB logging
if self.config.wandb_logging:
    log_dict = {
        'train_loss': train_results['epoch_loss'],
        'learning_rate': optimizer.param_groups[0]['lr'],
        'epoch': epoch
    }
    if val_results:
        log_dict.update(val_results)
    wandb.log(log_dict)
```

### Metrics Tracking

```python
class MetricsTracker:
    """Track and manage training metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))
            self.metrics[key] = value
    
    def get_best_metric(self, metric_name: str, minimize: bool = True) -> Tuple[float, int]:
        """Get best value for a metric."""
        if metric_name not in self.history:
            return None, None
        
        values = [value for _, value in self.history[metric_name]]
        if minimize:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return values[best_idx], self.history[metric_name][best_idx][0]
```

## Advanced Features

### Early Stopping

```python
# Early stopping implementation
if val_results and self.config.early_stopping_patience > 0:
    current_metric = val_results.get('eval_loss', float('inf'))
    if current_metric < self.best_metric:
        self.best_metric = current_metric
        self.best_epoch = epoch
        self.early_stopping_counter = 0
        
        # Save best model
        if self.config.save_best_model:
            self.save_checkpoint(model, optimizer, epoch, "best")
    else:
        self.early_stopping_counter += 1
    
    if self.early_stopping_counter >= self.config.early_stopping_patience:
        self.logger.info(f"Early stopping at epoch {epoch + 1}")
        break
```

### Checkpointing

```python
def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, name: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': self.config,
        'best_metric': self.best_metric,
        'best_epoch': self.best_epoch
    }
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{name}.pth"
    torch.save(checkpoint, checkpoint_path)
    self.logger.info(f"Checkpoint saved: {checkpoint_path}")
```

### Gradient Monitoring

```python
# Gradient monitoring integration
if hasattr(self, 'gradient_monitor'):
    self.gradient_monitor.update()

# Gradient clipping
if self.config.gradient_clipping:
    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
```

## Usage Examples

### Basic Training

```python
# Setup
training_config = TrainingConfig(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    mixed_precision=True,
    tensorboard_logging=True
)

eval_config = EvaluationConfig(
    compute_metrics=True,
    save_predictions=True,
    plot_metrics=True
)

# Create components
trainer = TrainingManager(training_config)
evaluator = Evaluator(eval_config)

# Train model
training_results = trainer.train(
    model, train_loader, val_loader, 
    task_type=TaskType.CLASSIFICATION,
    evaluator=evaluator
)
```

### Advanced Training

```python
# Advanced configuration
training_config = TrainingConfig(
    num_epochs=200,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-5,
    gradient_clipping=True,
    max_grad_norm=1.0,
    optimizer_type="adamw",
    scheduler_type="cosine",
    mixed_precision=True,
    gradient_accumulation_steps=4,  # Effective batch size = 16 * 4 = 64
    early_stopping_patience=15,
    checkpoint_save_frequency=5,
    tensorboard_logging=True,
    wandb_logging=True
)

# Training with advanced features
trainer = TrainingManager(training_config)
training_results = trainer.train(
    model, train_loader, val_loader,
    task_type=TaskType.CLASSIFICATION,
    evaluator=evaluator
)
```

### Cross-Validation

```python
# Cross-validation
cv_results = evaluator.cross_validate(
    SimpleModel, dataset, training_config, n_folds=5
)

print(f"Cross-validation results: {cv_results}")
```

### Model Evaluation

```python
# Evaluate final model
final_metrics = evaluator.evaluate_model(
    model, test_loader, trainer.device, nn.CrossEntropyLoss()
)

print(f"Final evaluation metrics: {final_metrics}")
```

## Best Practices

### 1. Configuration Management

- Use appropriate learning rates for different model sizes
- Enable mixed precision for faster training
- Use gradient accumulation for large effective batch sizes
- Implement early stopping to prevent overfitting

### 2. Monitoring

- Use TensorBoard for real-time monitoring
- Enable Weights & Biases for experiment tracking
- Monitor gradient norms and learning rates
- Track validation metrics regularly

### 3. Checkpointing

- Save best model based on validation metrics
- Save checkpoints at regular intervals
- Implement model loading for resuming training
- Use appropriate checkpoint naming

### 4. Evaluation

- Use appropriate metrics for each task type
- Implement cross-validation for robust evaluation
- Save predictions for further analysis
- Plot confusion matrices and learning curves

### 5. Performance Optimization

- Use mixed precision training when possible
- Implement gradient accumulation for large models
- Use appropriate batch sizes for your hardware
- Monitor memory usage and optimize accordingly

## Conclusion

The comprehensive training and evaluation system provides:

1. **Robust Training Loops**: Support for various training modes and optimizations
2. **Comprehensive Evaluation**: Multiple metrics and evaluation modes
3. **Advanced Monitoring**: Real-time logging and visualization
4. **Checkpointing**: Automatic model saving and loading
5. **Cross-Validation**: Robust model evaluation
6. **Performance Optimization**: Mixed precision, gradient accumulation, etc.

This system serves as a complete solution for training and evaluating deep learning models, with the flexibility to adapt to different requirements while maintaining high performance and reliability standards. 