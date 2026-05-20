# Early Stopping and Learning Rate Scheduling System

## Overview

This document provides a comprehensive overview of the early stopping and learning rate scheduling system that implements advanced training optimization techniques for deep learning applications.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Early Stopping Implementation](#early-stopping-implementation)
3. [Learning Rate Scheduling](#learning-rate-scheduling)
4. [Training Manager](#training-manager)
5. [Configuration Options](#configuration-options)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Monitoring and Visualization](#monitoring-and-visualization)

## System Architecture

### Core Components

The early stopping and learning rate scheduling system consists of several key components:

```python
class EarlyStopping:
    """Comprehensive early stopping implementation."""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Early stopping state
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        
        # History tracking
        self.history = {
            'epochs': [],
            'metrics': [],
            'best_scores': [],
            'patience_counter': []
        }
```

### Advanced Scheduler

```python
class AdvancedScheduler:
    """Advanced learning rate scheduler with multiple strategies."""
    
    def __init__(self, optimizer: Optimizer, config: SchedulerConfig):
        self.optimizer = optimizer
        self.config = config
        self.logger = self._setup_logging()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # History tracking
        self.history = {
            'epochs': [],
            'learning_rates': [],
            'metrics': []
        }
        
        # Current epoch
        self.current_epoch = 0
```

### Training Manager

```python
class TrainingManager:
    """Training manager with early stopping and learning rate scheduling."""
    
    def __init__(self, model: nn.Module, optimizer: Optimizer,
                 early_stopping_config: EarlyStoppingConfig,
                 scheduler_config: SchedulerConfig):
        self.model = model
        self.optimizer = optimizer
        self.logger = self._setup_logging()
        
        # Initialize early stopping and scheduler
        self.early_stopping = EarlyStopping(early_stopping_config)
        self.scheduler = AdvancedScheduler(optimizer, scheduler_config)
        
        # Training state
        self.current_epoch = 0
        self.training_history = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'learning_rates': []
        }
```

## Early Stopping Implementation

### Configuration

```python
@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    # Basic settings
    patience: int = 10
    min_delta: float = 0.0
    mode: EarlyStoppingMode = EarlyStoppingMode.MIN
    monitor: MonitorMetric = MonitorMetric.VAL_LOSS
    
    # Advanced settings
    restore_best_weights: bool = True
    verbose: bool = True
    save_checkpoint: bool = True
    checkpoint_path: str = "best_model.pth"
    
    # Custom settings
    custom_monitor: Optional[Callable] = None
    custom_metric_name: str = "custom_metric"
    
    # Monitoring
    log_early_stopping: bool = True
    save_history: bool = True
    history_file: str = "early_stopping_history.json"
```

### Early Stopping Modes

```python
class EarlyStoppingMode(Enum):
    """Early stopping modes."""
    MIN = "min"  # Stop when metric decreases
    MAX = "max"  # Stop when metric increases
```

### Monitor Metrics

```python
class MonitorMetric(Enum):
    """Metrics to monitor for early stopping and scheduling."""
    LOSS = "loss"
    ACCURACY = "accuracy"
    VAL_LOSS = "val_loss"
    VAL_ACCURACY = "val_accuracy"
    TRAIN_LOSS = "train_loss"
    TRAIN_ACCURACY = "train_accuracy"
    CUSTOM = "custom"
```

### Core Early Stopping Logic

```python
def __call__(self, epoch: int, metric: float, model: nn.Module) -> bool:
    """Check if training should stop early."""
    # Get metric name
    metric_name = self.config.monitor.value if self.config.monitor != MonitorMetric.CUSTOM else self.config.custom_metric_name
    
    # Determine if metric improved
    if self.config.mode == EarlyStoppingMode.MIN:
        improved = metric < self.best_score - self.config.min_delta
    else:
        improved = metric > self.best_score + self.config.min_delta
    
    if improved:
        # Update best score
        self.best_score = metric
        self.best_epoch = epoch
        self.counter = 0
        
        # Save best weights
        if self.config.restore_best_weights:
            self.best_weights = model.state_dict().copy()
        
        # Save checkpoint
        if self.config.save_checkpoint:
            self._save_checkpoint(model, epoch, metric)
        
        if self.config.verbose:
            self.logger.info(f"Epoch {epoch}: {metric_name} improved to {metric:.6f}")
    else:
        self.counter += 1
        if self.config.verbose:
            self.logger.info(f"Epoch {epoch}: {metric_name} did not improve. Patience: {self.counter}/{self.config.patience}")
    
    # Update history
    self.history['epochs'].append(epoch)
    self.history['metrics'].append(metric)
    self.history['best_scores'].append(self.best_score)
    self.history['patience_counter'].append(self.counter)
    
    # Check if should stop
    if self.counter >= self.config.patience:
        self.early_stop = True
        if self.config.verbose:
            self.logger.info(f"Early stopping triggered at epoch {epoch}")
        
        # Restore best weights
        if self.config.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.config.verbose:
                self.logger.info(f"Restored best weights from epoch {self.best_epoch}")
    
    # Save history
    if self.config.save_history:
        self._save_history()
    
    return self.early_stop
```

### Checkpoint Management

```python
def _save_checkpoint(self, model: nn.Module, epoch: int, metric: float):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metric': metric,
        'best_score': self.best_score,
        'best_epoch': self.best_epoch,
        'config': self.config
    }
    
    torch.save(checkpoint, self.config.checkpoint_path)
    
    if self.config.log_early_stopping:
        self.logger.info(f"Checkpoint saved: {self.config.checkpoint_path}")

def load_checkpoint(self, model: nn.Module) -> Dict[str, Any]:
    """Load model checkpoint."""
    if os.path.exists(self.config.checkpoint_path):
        checkpoint = torch.load(self.config.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore early stopping state
        self.best_score = checkpoint['best_score']
        self.best_epoch = checkpoint['best_epoch']
        
        if self.config.log_early_stopping:
            self.logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        
        return checkpoint
    else:
        self.logger.warning(f"No checkpoint found at {self.config.checkpoint_path}")
        return {}
```

## Learning Rate Scheduling

### Scheduler Types

```python
class SchedulerType(Enum):
    """Learning rate scheduler types."""
    STEP = "step"
    MULTI_STEP = "multi_step"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"
    ONE_CYCLE = "one_cycle"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    CYCLIC = "cyclic"
    LAMBDA = "lambda"
    CUSTOM = "custom"
```

### Configuration

```python
@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduling."""
    # Scheduler type
    scheduler_type: SchedulerType = SchedulerType.COSINE_ANNEALING
    
    # Basic settings
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-1
    
    # Step-based settings
    step_size: int = 30
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [30, 60, 90])
    
    # Cosine annealing settings
    T_max: int = 100
    eta_min: float = 1e-6
    
    # One cycle settings
    epochs: int = 100
    steps_per_epoch: int = 100
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    
    # Reduce on plateau settings
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr_factor: float = 0.01
    
    # Cyclic settings
    base_lr: float = 1e-6
    max_lr: float = 1e-3
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode: str = "triangular"
    gamma_scale: float = 1.0
    scale_fn: Optional[Callable] = None
    scale_mode: str = "cycle"
    
    # Custom settings
    custom_scheduler: Optional[Callable] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring
    log_scheduling: bool = True
    save_scheduler_state: bool = True
    scheduler_state_file: str = "scheduler_state.pth"
```

### Scheduler Creation

```python
def _create_scheduler(self) -> _LRScheduler:
    """Create learning rate scheduler."""
    if self.config.scheduler_type == SchedulerType.STEP:
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma
        )
    
    elif self.config.scheduler_type == SchedulerType.MULTI_STEP:
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config.milestones,
            gamma=self.config.gamma
        )
    
    elif self.config.scheduler_type == SchedulerType.EXPONENTIAL:
        return torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.config.gamma
        )
    
    elif self.config.scheduler_type == SchedulerType.COSINE_ANNEALING:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.T_max,
            eta_min=self.config.eta_min
        )
    
    elif self.config.scheduler_type == SchedulerType.COSINE_ANNEALING_WARM_RESTARTS:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.T_max,
            T_mult=2,
            eta_min=self.config.eta_min
        )
    
    elif self.config.scheduler_type == SchedulerType.ONE_CYCLE:
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.max_lr,
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            pct_start=self.config.pct_start,
            anneal_strategy=self.config.anneal_strategy
        )
    
    elif self.config.scheduler_type == SchedulerType.REDUCE_LR_ON_PLATEAU:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.factor,
            patience=self.config.patience,
            threshold=self.config.threshold,
            threshold_mode=self.config.threshold_mode,
            cooldown=self.config.cooldown,
            min_lr=self.config.min_lr * self.config.min_lr_factor
        )
    
    elif self.config.scheduler_type == SchedulerType.CYCLIC:
        return torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.config.base_lr,
            max_lr=self.config.max_lr,
            step_size_up=self.config.step_size_up,
            step_size_down=self.config.step_size_down,
            mode=self.config.mode,
            gamma=self.config.gamma_scale,
            scale_fn=self.config.scale_fn,
            scale_mode=self.config.scale_mode
        )
    
    elif self.config.scheduler_type == SchedulerType.LAMBDA:
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: self.config.gamma ** epoch
        )
    
    elif self.config.scheduler_type == SchedulerType.CUSTOM:
        if self.config.custom_scheduler is None:
            raise ValueError("Custom scheduler function required for CUSTOM scheduler type")
        return self.config.custom_scheduler(self.optimizer, **self.config.custom_params)
    
    else:
        raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
```

### Scheduler Step

```python
def step(self, metric: Optional[float] = None):
    """Step the scheduler."""
    if self.config.scheduler_type == SchedulerType.REDUCE_LR_ON_PLATEAU:
        if metric is None:
            raise ValueError("Metric required for ReduceLROnPlateau scheduler")
        self.scheduler.step(metric)
    else:
        self.scheduler.step()
    
    # Update history
    current_lr = self.optimizer.param_groups[0]['lr']
    self.history['epochs'].append(self.current_epoch)
    self.history['learning_rates'].append(current_lr)
    if metric is not None:
        self.history['metrics'].append(metric)
    
    self.current_epoch += 1
    
    if self.config.log_scheduling:
        self.logger.info(f"Epoch {self.current_epoch}: LR = {current_lr:.6f}")
```

## Training Manager

### Training Epoch

```python
def train_epoch(self, train_dataloader: torch.utils.data.DataLoader,
               val_dataloader: torch.utils.data.DataLoader,
               criterion: nn.Module) -> Dict[str, float]:
    """Train for one epoch."""
    self.model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_dataloader):
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update scheduler for OneCycleLR
        if self.scheduler.config.scheduler_type == SchedulerType.ONE_CYCLE:
            self.scheduler.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
    
    # Calculate training metrics
    train_loss /= len(train_dataloader)
    train_accuracy = train_correct / train_total
    
    # Validation
    self.model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in val_dataloader:
            output = self.model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    
    # Calculate validation metrics
    val_loss /= len(val_dataloader)
    val_accuracy = val_correct / val_total
    
    # Update scheduler (except OneCycleLR which is updated per batch)
    if self.scheduler.config.scheduler_type != SchedulerType.ONE_CYCLE:
        if self.scheduler.config.scheduler_type == SchedulerType.REDUCE_LR_ON_PLATEAU:
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
    
    # Update training history
    self.training_history['epochs'].append(self.current_epoch)
    self.training_history['train_losses'].append(train_loss)
    self.training_history['val_losses'].append(val_loss)
    self.training_history['train_accuracies'].append(train_accuracy)
    self.training_history['val_accuracies'].append(val_accuracy)
    self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
    
    # Check early stopping
    monitor_metric = self._get_monitor_metric(val_loss, val_accuracy)
    should_stop = self.early_stopping(self.current_epoch, monitor_metric, self.model)
    
    self.current_epoch += 1
    
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'learning_rate': self.optimizer.param_groups[0]['lr'],
        'should_stop': should_stop
    }
```

### Monitor Metric Selection

```python
def _get_monitor_metric(self, val_loss: float, val_accuracy: float) -> float:
    """Get the metric to monitor based on early stopping configuration."""
    monitor = self.early_stopping.config.monitor
    
    if monitor == MonitorMetric.VAL_LOSS:
        return val_loss
    elif monitor == MonitorMetric.VAL_ACCURACY:
        return val_accuracy
    elif monitor == MonitorMetric.LOSS:
        return val_loss  # Default to validation loss
    elif monitor == MonitorMetric.ACCURACY:
        return val_accuracy  # Default to validation accuracy
    elif monitor == MonitorMetric.CUSTOM:
        if self.early_stopping.config.custom_monitor is not None:
            return self.early_stopping.config.custom_monitor(val_loss, val_accuracy)
        else:
            return val_loss  # Default to validation loss
    else:
        return val_loss  # Default to validation loss
```

### Complete Training Loop

```python
def train(self, train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          criterion: nn.Module, max_epochs: int = 100) -> Dict[str, Any]:
    """Train the model with early stopping and learning rate scheduling."""
    self.logger.info("Starting training with early stopping and learning rate scheduling")
    
    training_results = []
    
    for epoch in range(max_epochs):
        # Train one epoch
        epoch_results = self.train_epoch(train_dataloader, val_dataloader, criterion)
        training_results.append(epoch_results)
        
        # Log progress
        self.logger.info(
            f"Epoch {epoch + 1}/{max_epochs}: "
            f"Train Loss: {epoch_results['train_loss']:.4f}, "
            f"Val Loss: {epoch_results['val_loss']:.4f}, "
            f"Train Acc: {epoch_results['train_accuracy']:.4f}, "
            f"Val Acc: {epoch_results['val_accuracy']:.4f}, "
            f"LR: {epoch_results['learning_rate']:.6f}"
        )
        
        # Check early stopping
        if epoch_results['should_stop']:
            self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Save final states
    self.early_stopping.save_history()
    self.scheduler.save_state()
    
    return {
        'training_results': training_results,
        'best_epoch': self.early_stopping.get_best_epoch(),
        'best_score': self.early_stopping.get_best_score(),
        'total_epochs': len(training_results)
    }
```

## Configuration Options

### Early Stopping Configuration

#### **Basic Settings**
- `patience`: Number of epochs to wait before stopping (default: 10)
- `min_delta`: Minimum change in monitored metric (default: 0.0)
- `mode`: Early stopping mode - MIN or MAX (default: MIN)
- `monitor`: Metric to monitor (default: VAL_LOSS)

#### **Advanced Settings**
- `restore_best_weights`: Restore best model weights (default: True)
- `verbose`: Enable verbose logging (default: True)
- `save_checkpoint`: Save model checkpoints (default: True)
- `checkpoint_path`: Path for checkpoint file (default: "best_model.pth")

#### **Custom Settings**
- `custom_monitor`: Custom monitoring function
- `custom_metric_name`: Name for custom metric

#### **Monitoring**
- `log_early_stopping`: Enable logging (default: True)
- `save_history`: Save training history (default: True)
- `history_file`: Path for history file (default: "early_stopping_history.json")

### Scheduler Configuration

#### **Basic Settings**
- `scheduler_type`: Type of scheduler to use
- `initial_lr`: Initial learning rate (default: 1e-3)
- `min_lr`: Minimum learning rate (default: 1e-6)
- `max_lr`: Maximum learning rate (default: 1e-1)

#### **Step-based Settings**
- `step_size`: Step size for StepLR (default: 30)
- `gamma`: Learning rate decay factor (default: 0.1)
- `milestones`: Milestones for MultiStepLR (default: [30, 60, 90])

#### **Cosine Annealing Settings**
- `T_max`: Maximum number of iterations (default: 100)
- `eta_min`: Minimum learning rate (default: 1e-6)

#### **One Cycle Settings**
- `epochs`: Number of epochs (default: 100)
- `steps_per_epoch`: Steps per epoch (default: 100)
- `pct_start`: Percentage of training for warmup (default: 0.3)
- `anneal_strategy`: Annealing strategy (default: "cos")

#### **Reduce on Plateau Settings**
- `factor`: Factor by which to reduce learning rate (default: 0.1)
- `patience`: Number of epochs to wait (default: 10)
- `threshold`: Threshold for measuring improvement (default: 1e-4)
- `threshold_mode`: Threshold mode (default: "rel")
- `cooldown`: Number of epochs to wait after reducing LR (default: 0)
- `min_lr_factor`: Factor for minimum learning rate (default: 0.01)

#### **Cyclic Settings**
- `base_lr`: Base learning rate (default: 1e-6)
- `max_lr`: Maximum learning rate (default: 1e-3)
- `step_size_up`: Steps to increase LR (default: 2000)
- `step_size_down`: Steps to decrease LR (default: 2000)
- `mode`: Cyclic mode (default: "triangular")
- `gamma_scale`: Scale factor (default: 1.0)
- `scale_fn`: Custom scaling function
- `scale_mode`: Scaling mode (default: "cycle")

## Usage Examples

### Basic Early Stopping

```python
# Create early stopping configuration
early_stopping_config = EarlyStoppingConfig(
    patience=10,
    monitor=MonitorMetric.VAL_LOSS,
    mode=EarlyStoppingMode.MIN,
    verbose=True,
    save_checkpoint=True
)

# Create early stopping
early_stopping = EarlyStopping(early_stopping_config)

# Use in training loop
for epoch in range(max_epochs):
    # Train and validate
    val_loss = train_and_validate()
    
    # Check early stopping
    if early_stopping(epoch, val_loss, model):
        print(f"Early stopping triggered at epoch {epoch}")
        break
```

### Learning Rate Scheduling

```python
# Create scheduler configuration
scheduler_config = SchedulerConfig(
    scheduler_type=SchedulerType.COSINE_ANNEALING,
    T_max=100,
    eta_min=1e-6
)

# Create scheduler
scheduler = AdvancedScheduler(optimizer, scheduler_config)

# Use in training loop
for epoch in range(max_epochs):
    # Training
    train_epoch()
    
    # Step scheduler
    scheduler.step()
    
    # Get current learning rate
    current_lr = scheduler.get_lr()[0]
    print(f"Epoch {epoch}: LR = {current_lr:.6f}")
```

### Complete Training with Both

```python
# Create configurations
early_stopping_config = EarlyStoppingConfig(
    patience=15,
    monitor=MonitorMetric.VAL_LOSS,
    mode=EarlyStoppingMode.MIN,
    verbose=True
)

scheduler_config = SchedulerConfig(
    scheduler_type=SchedulerType.REDUCE_LR_ON_PLATEAU,
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Create training manager
training_manager = TrainingManager(
    model, optimizer,
    early_stopping_config,
    scheduler_config
)

# Train model
results = training_manager.train(
    train_dataloader, val_dataloader, criterion, max_epochs=100
)

print(f"Training completed in {results['total_epochs']} epochs")
print(f"Best epoch: {results['best_epoch']}")
print(f"Best score: {results['best_score']:.6f}")
```

### Different Scheduler Types

#### **Cosine Annealing**
```python
scheduler_config = SchedulerConfig(
    scheduler_type=SchedulerType.COSINE_ANNEALING,
    T_max=100,
    eta_min=1e-6
)
```

#### **One Cycle**
```python
scheduler_config = SchedulerConfig(
    scheduler_type=SchedulerType.ONE_CYCLE,
    max_lr=1e-2,
    epochs=100,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.3
)
```

#### **Reduce on Plateau**
```python
scheduler_config = SchedulerConfig(
    scheduler_type=SchedulerType.REDUCE_LR_ON_PLATEAU,
    factor=0.1,
    patience=10,
    min_lr=1e-6
)
```

#### **Cyclic**
```python
scheduler_config = SchedulerConfig(
    scheduler_type=SchedulerType.CYCLIC,
    base_lr=1e-6,
    max_lr=1e-3,
    step_size_up=2000,
    step_size_down=2000
)
```

## Best Practices

### 1. Early Stopping Configuration

- **Patience**: Set based on dataset size and complexity
  - Small datasets: 5-10 epochs
  - Large datasets: 10-20 epochs
- **Monitor**: Use validation loss for most cases
- **Mode**: MIN for loss, MAX for accuracy
- **Restore weights**: Always enable for best results

### 2. Learning Rate Scheduling

- **Cosine Annealing**: Good for most cases
- **One Cycle**: Excellent for fast convergence
- **Reduce on Plateau**: Good for stable training
- **Cyclic**: Good for finding optimal learning rates

### 3. Integration

- **Monitor validation metrics**: Use validation loss/accuracy
- **Save checkpoints**: Enable for model recovery
- **Log progress**: Enable verbose logging
- **Plot results**: Visualize training progress

### 4. Hyperparameter Tuning

- **Learning rate range**: 1e-6 to 1e-1
- **Patience**: 5-20 epochs
- **Scheduler type**: Match to problem type
- **Monitor metric**: Choose based on task

## Monitoring and Visualization

### Early Stopping History

```python
def plot_history(self, save_path: Optional[str] = None):
    """Plot early stopping history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot metric over epochs
    axes[0, 0].plot(self.history['epochs'], self.history['metrics'], label='Current Metric')
    axes[0, 0].plot(self.history['epochs'], self.history['best_scores'], label='Best Score', linestyle='--')
    axes[0, 0].axvline(x=self.best_epoch, color='r', linestyle=':', label=f'Best Epoch ({self.best_epoch})')
    axes[0, 0].set_title('Metric Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Metric')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot patience counter
    axes[0, 1].plot(self.history['epochs'], self.history['patience_counter'], label='Patience Counter')
    axes[0, 1].axhline(y=self.config.patience, color='r', linestyle='--', label=f'Patience Limit ({self.config.patience})')
    axes[0, 1].set_title('Patience Counter')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Counter')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot metric improvement
    metric_improvements = [self.history['best_scores'][i] - self.history['metrics'][i] for i in range(len(self.history['metrics']))]
    axes[1, 0].plot(self.history['epochs'], metric_improvements, label='Improvement')
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Metric Improvement')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Improvement')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot early stopping decision
    should_stop = [1 if counter >= self.config.patience else 0 for counter in self.history['patience_counter']]
    axes[1, 1].plot(self.history['epochs'], should_stop, label='Should Stop', marker='o')
    axes[1, 1].set_title('Early Stopping Decision')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Stop (1) / Continue (0)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Early stopping history plot saved to {save_path}")
    
    plt.show()
```

### Learning Rate Schedule Visualization

```python
def plot_schedule(self, save_path: Optional[str] = None):
    """Plot learning rate schedule."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot learning rate over epochs
    axes[0, 0].plot(self.history['epochs'], self.history['learning_rates'], label='Learning Rate')
    axes[0, 0].set_title('Learning Rate Schedule')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Learning Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale('log')
    
    # Plot learning rate changes
    if len(self.history['learning_rates']) > 1:
        lr_changes = [self.history['learning_rates'][i] - self.history['learning_rates'][i-1] 
                     for i in range(1, len(self.history['learning_rates']))]
        axes[0, 1].plot(self.history['epochs'][1:], lr_changes, label='LR Change')
        axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, 1].set_title('Learning Rate Changes')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Change')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot metrics vs learning rate
    if len(self.history['metrics']) > 0:
        axes[1, 0].scatter(self.history['learning_rates'], self.history['metrics'], alpha=0.6)
        axes[1, 0].set_title('Metrics vs Learning Rate')
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('Metric')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True)
    
    # Plot learning rate distribution
    axes[1, 1].hist(self.history['learning_rates'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Learning Rate Distribution')
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Learning rate schedule plot saved to {save_path}")
    
    plt.show()
```

### Training History Visualization

```python
def plot_training_history(self, save_path: Optional[str] = None):
    """Plot training history."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot losses
    axes[0, 0].plot(self.training_history['epochs'], self.training_history['train_losses'], label='Train Loss')
    axes[0, 0].plot(self.training_history['epochs'], self.training_history['val_losses'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracies
    axes[0, 1].plot(self.training_history['epochs'], self.training_history['train_accuracies'], label='Train Accuracy')
    axes[0, 1].plot(self.training_history['epochs'], self.training_history['val_accuracies'], label='Val Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate
    axes[0, 2].plot(self.training_history['epochs'], self.training_history['learning_rates'], label='Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].set_yscale('log')
    
    # Plot loss vs learning rate
    axes[1, 0].scatter(self.training_history['learning_rates'], self.training_history['val_losses'], alpha=0.6)
    axes[1, 0].set_title('Validation Loss vs Learning Rate')
    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True)
    
    # Plot accuracy vs learning rate
    axes[1, 1].scatter(self.training_history['learning_rates'], self.training_history['val_accuracies'], alpha=0.6)
    axes[1, 1].set_title('Validation Accuracy vs Learning Rate')
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True)
    
    # Plot early stopping history
    self.early_stopping.plot_history()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()
```

## Conclusion

The early stopping and learning rate scheduling system provides:

1. **Comprehensive Early Stopping**: Multiple modes, metrics, and monitoring options
2. **Advanced Learning Rate Scheduling**: Multiple scheduler types with customization
3. **Training Manager**: Integrated training with both early stopping and scheduling
4. **Checkpoint Management**: Automatic saving and loading of best models
5. **History Tracking**: Complete training history and visualization
6. **Flexible Configuration**: Extensive configuration options for different use cases
7. **Monitoring and Logging**: Comprehensive logging and progress tracking
8. **Visualization Tools**: Complete plotting and analysis capabilities

This system ensures optimal training with automatic stopping and learning rate optimization for robust model development in deep learning applications. 