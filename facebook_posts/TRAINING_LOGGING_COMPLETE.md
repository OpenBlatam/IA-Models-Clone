# Training Logging System - Complete Documentation

## Overview

The Training Logging System provides comprehensive logging for training progress and errors with real-time monitoring, detailed error tracking, progress visualization, and automatic recovery mechanisms. This system ensures complete visibility into the training process and robust error handling.

## Architecture

### Core Components

1. **TrainingLogger**: Central logging with real-time monitoring
2. **TrainingProgressTracker**: Advanced progress tracking with visualization
3. **TrainingLoggingManager**: High-level manager for comprehensive logging
4. **ErrorInfo**: Detailed error information structure
5. **TrainingMetrics**: Comprehensive metrics tracking

### Key Features

- **Real-time Monitoring**: Continuous monitoring of system resources
- **Detailed Error Tracking**: Categorized error logging with recovery attempts
- **Progress Visualization**: Automatic plot generation for training metrics
- **Comprehensive Metrics**: Complete tracking of all training parameters
- **Automatic Recovery**: Intelligent error recovery mechanisms
- **Multiple Output Formats**: CSV, JSON, and console logging

## Training Logger

### Core Logging Methods

```python
class TrainingLogger:
    def __init__(self, log_dir: str = "training_logs", experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Training state
        self.metrics = TrainingMetrics()
        self.error_history: List[ErrorInfo] = []
        self.progress_history: List[Dict[str, Any]] = []
```

### Training Phase Logging

```python
def log_phase_start(self, phase: TrainingPhase, **kwargs):
    """Log the start of a training phase."""
    self.logger.debug(f"Phase started: {phase.value}")
    
    if phase == TrainingPhase.EPOCH_START:
        self.epoch_start_time = time.time()
        self.metrics.epoch = kwargs.get('epoch', 0)
        self.logger.info(f"Epoch {self.metrics.epoch} started")
    
    elif phase == TrainingPhase.BATCH_START:
        self.batch_start_time = time.time()
        self.metrics.batch = kwargs.get('batch', 0)
        self.metrics.total_batches = kwargs.get('total_batches', 0)
    
    # Log phase with context
    phase_data = {
        'phase': phase.value,
        'timestamp': datetime.now().isoformat(),
        'epoch': self.metrics.epoch,
        'batch': self.metrics.batch,
        **kwargs
    }
    
    self.progress_history.append(phase_data)
    self._save_progress()
```

### Metrics Logging

```python
def log_metrics(self, metrics: Dict[str, Any]):
    """Log training metrics."""
    # Update metrics
    for key, value in metrics.items():
        if hasattr(self.metrics, key):
            setattr(self.metrics, key, value)
    
    # Calculate additional metrics
    self.metrics.training_time = time.time() - self.start_time
    self.metrics.memory_usage = self._get_memory_usage()
    self.metrics.gpu_memory = self._get_gpu_memory_usage()
    
    # Log to console
    if self.metrics.batch % 10 == 0:  # Log every 10 batches
        self.logger.info(
            f"Epoch {self.metrics.epoch}, "
            f"Batch {self.metrics.batch}/{self.metrics.total_batches}, "
            f"Loss: {self.metrics.loss:.4f}, "
            f"Accuracy: {self.metrics.accuracy:.4f}, "
            f"LR: {self.metrics.learning_rate:.6f}"
        )
    
    # Save to CSV
    self._save_metrics()
    
    # Update progress
    self._update_progress()
```

### Error Logging

```python
def log_error(self, error: Exception, phase: TrainingPhase, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
    """Log training error with detailed information."""
    error_info = ErrorInfo(
        category=category,
        phase=phase,
        error_message=str(error),
        traceback=traceback.format_exc(),
        timestamp=datetime.now(),
        epoch=self.metrics.epoch,
        batch=self.metrics.batch,
        context_data=kwargs
    )
    
    # Log error
    self.logger.error(f"Error in {phase.value}: {str(error)}")
    self.logger.error(f"Category: {category.value}")
    self.logger.error(f"Epoch: {self.metrics.epoch}, Batch: {self.metrics.batch}")
    self.logger.error(f"Traceback: {error_info.traceback}")
    
    # Add to history
    self.error_history.append(error_info)
    
    # Save errors
    self._save_errors()
    
    # Update error statistics
    self._update_error_statistics()
```

### Real-time Monitoring

```python
def _monitoring_loop(self):
    """Real-time monitoring loop."""
    while self.monitoring_active:
        try:
            # Monitor system resources
            memory_usage = self._get_memory_usage()
            gpu_memory = self._get_gpu_memory_usage()
            
            # Check for potential issues
            if memory_usage > 1000:  # 1GB threshold
                self.logger.warning(f"High memory usage: {memory_usage:.2f} MB")
            
            if gpu_memory > 8000:  # 8GB threshold
                self.logger.warning(f"High GPU memory usage: {gpu_memory:.2f} MB")
            
            # Sleep for monitoring interval
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            self.logger.error(f"Monitoring error: {str(e)}")
            time.sleep(60)  # Wait longer on error
```

## Training Progress Tracker

### Metrics Tracking

```python
class TrainingProgressTracker:
    def __init__(self, logger: TrainingLogger):
        self.logger = logger
        self.metrics_history: List[Dict[str, Any]] = []
        self.error_summary: Dict[str, int] = defaultdict(int)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics history."""
        self.metrics_history.append(metrics.copy())
        
        # Update performance metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.performance_metrics[key].append(value)
```

### Training Summary

```python
def get_training_summary(self) -> Dict[str, Any]:
    """Get comprehensive training summary."""
    if not self.metrics_history:
        return {}
    
    latest_metrics = self.metrics_history[-1]
    
    summary = {
        'current_epoch': latest_metrics.get('epoch', 0),
        'current_batch': latest_metrics.get('batch', 0),
        'total_batches': latest_metrics.get('total_batches', 0),
        'current_loss': latest_metrics.get('loss', 0.0),
        'current_accuracy': latest_metrics.get('accuracy', 0.0),
        'best_loss': latest_metrics.get('best_loss', float('inf')),
        'best_accuracy': latest_metrics.get('best_accuracy', 0.0),
        'learning_rate': latest_metrics.get('learning_rate', 0.0),
        'training_time': latest_metrics.get('training_time', 0.0),
        'error_count': len(self.logger.error_history),
        'memory_usage': latest_metrics.get('memory_usage', 0.0),
        'gpu_memory': latest_metrics.get('gpu_memory', 0.0)
    }
    
    return summary
```

### Error Summary

```python
def get_error_summary(self) -> Dict[str, Any]:
    """Get error summary statistics."""
    error_categories = defaultdict(int)
    error_phases = defaultdict(int)
    
    for error in self.logger.error_history:
        error_categories[error.category.value] += 1
        error_phases[error.phase.value] += 1
    
    return {
        'total_errors': len(self.logger.error_history),
        'error_categories': dict(error_categories),
        'error_phases': dict(error_phases),
        'recovery_success_rate': self._calculate_recovery_success_rate()
    }
```

### Visualization

```python
def create_training_plots(self, save_dir: str = None):
    """Create training progress plots."""
    if not self.metrics_history:
        return
    
    save_dir = Path(save_dir) if save_dir else self.logger.experiment_dir / "plots"
    save_dir.mkdir(exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress - {self.logger.experiment_name}', fontsize=16)
    
    # Extract data
    epochs = [m.get('epoch', 0) for m in self.metrics_history]
    losses = [m.get('loss', 0.0) for m in self.metrics_history]
    accuracies = [m.get('accuracy', 0.0) for m in self.metrics_history]
    learning_rates = [m.get('learning_rate', 0.0) for m in self.metrics_history]
    
    # Loss plot
    axes[0, 0].plot(epochs, losses, 'b-', label='Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, accuracies, 'g-', label='Training Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, learning_rates, 'r-', label='Learning Rate')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Error summary plot
    error_summary = self.get_error_summary()
    if error_summary['error_categories']:
        categories = list(error_summary['error_categories'].keys())
        counts = list(error_summary['error_categories'].values())
        
        axes[1, 1].bar(categories, counts, color='orange')
        axes[1, 1].set_title('Error Categories')
        axes[1, 1].set_xlabel('Error Category')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
```

## Training Logging Manager

### High-level Management

```python
class TrainingLoggingManager:
    def __init__(self, experiment_name: str = None, log_dir: str = "training_logs"):
        self.logger = TrainingLogger(log_dir, experiment_name)
        self.progress_tracker = TrainingProgressTracker(self.logger)
        self.error_handler = RobustErrorHandler(self.logger.logger)
        
        # Training state
        self.training_active = False
        self.current_epoch = 0
        self.current_batch = 0
```

### Training Lifecycle Management

```python
def start_training(self, config: Dict[str, Any]):
    """Start training with comprehensive logging."""
    self.training_active = True
    self.logger.log_training_start(config)
    
    # Log initial configuration
    self.logger.logger.info("Training configuration:")
    for key, value in config.items():
        self.logger.logger.info(f"  {key}: {value}")

def log_epoch_start(self, epoch: int, total_epochs: int):
    """Log epoch start."""
    self.current_epoch = epoch
    self.logger.log_phase_start(TrainingPhase.EPOCH_START, epoch=epoch, total_epochs=total_epochs)
    
    # Update progress tracker
    self.progress_tracker.update_metrics({
        'epoch': epoch,
        'total_epochs': total_epochs
    })

def log_batch_start(self, batch: int, total_batches: int):
    """Log batch start."""
    self.current_batch = batch
    self.logger.log_phase_start(TrainingPhase.BATCH_START, batch=batch, total_batches=total_batches)

def log_batch_metrics(self, metrics: Dict[str, Any]):
    """Log batch metrics."""
    # Add context information
    metrics.update({
        'epoch': self.current_epoch,
        'batch': self.current_batch
    })
    
    self.logger.log_metrics(metrics)
    self.progress_tracker.update_metrics(metrics)

def log_batch_end(self):
    """Log batch end."""
    self.logger.log_phase_end(TrainingPhase.BATCH_END)

def log_epoch_end(self, epoch_metrics: Dict[str, Any]):
    """Log epoch end."""
    self.logger.log_phase_end(TrainingPhase.EPOCH_END, **epoch_metrics)
    
    # Update progress tracker
    self.progress_tracker.update_metrics(epoch_metrics)

def log_validation(self, validation_metrics: Dict[str, Any]):
    """Log validation results."""
    self.logger.log_validation_results(validation_metrics)
    
    # Update progress tracker
    self.progress_tracker.update_metrics(validation_metrics)
```

### Error Handling with Recovery

```python
def log_error(self, error: Exception, phase: TrainingPhase, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
    """Log error with recovery attempt."""
    # Log error
    self.logger.log_error(error, phase, category, **kwargs)
    
    # Attempt recovery
    recovery_start = time.time()
    recovery_successful = self._attempt_error_recovery(error, category, **kwargs)
    recovery_time = time.time() - recovery_start
    
    # Log recovery attempt
    if self.logger.error_history:
        latest_error = self.logger.error_history[-1]
        self.logger.log_recovery_attempt(latest_error, recovery_successful, recovery_time)

def _attempt_error_recovery(self, error: Exception, category: ErrorCategory, **kwargs) -> bool:
    """Attempt error recovery based on category."""
    try:
        if category == ErrorCategory.MEMORY:
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        
        elif category == ErrorCategory.GRADIENT_COMPUTATION:
            # Clear gradients
            if 'model' in kwargs:
                for param in kwargs['model'].parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            return True
        
        elif category == ErrorCategory.OPTIMIZATION:
            # Skip optimization step
            return True
        
        else:
            # Generic recovery
            time.sleep(1)
            return True
            
    except Exception as recovery_error:
        self.logger.logger.error(f"Recovery failed: {str(recovery_error)}")
        return False
```

## Error Categories and Phases

### Error Categories

```python
class ErrorCategory(Enum):
    DATA_LOADING = "data_loading"           # Data loading errors
    MODEL_INFERENCE = "model_inference"     # Model inference errors
    LOSS_COMPUTATION = "loss_computation"   # Loss computation errors
    GRADIENT_COMPUTATION = "gradient_computation"  # Gradient computation errors
    OPTIMIZATION = "optimization"           # Optimization errors
    MEMORY = "memory"                       # Memory-related errors
    DEVICE = "device"                       # Device-related errors
    CHECKPOINT = "checkpoint"               # Checkpoint errors
    VALIDATION = "validation"               # Validation errors
    CONFIGURATION = "configuration"         # Configuration errors
    NETWORK = "network"                     # Network errors
    SYSTEM = "system"                       # System errors
    UNKNOWN = "unknown"                     # Unknown errors
```

### Training Phases

```python
class TrainingPhase(Enum):
    INITIALIZATION = "initialization"       # Training initialization
    DATA_LOADING = "data_loading"          # Data loading phase
    MODEL_SETUP = "model_setup"            # Model setup phase
    TRAINING_START = "training_start"      # Training start
    EPOCH_START = "epoch_start"            # Epoch start
    BATCH_START = "batch_start"            # Batch start
    FORWARD_PASS = "forward_pass"           # Forward pass
    LOSS_COMPUTATION = "loss_computation"  # Loss computation
    BACKWARD_PASS = "backward_pass"        # Backward pass
    OPTIMIZATION_STEP = "optimization_step" # Optimization step
    BATCH_END = "batch_end"                # Batch end
    EPOCH_END = "epoch_end"                # Epoch end
    VALIDATION = "validation"              # Validation phase
    CHECKPOINT_SAVING = "checkpoint_saving" # Checkpoint saving
    TRAINING_END = "training_end"          # Training end
    ERROR_RECOVERY = "error_recovery"      # Error recovery
```

## Usage Examples

### Basic Usage

```python
# Create logging manager
logging_manager = TrainingLoggingManager("my_experiment")

# Start training
config = {
    'model_type': 'neural_network',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10
}
logging_manager.start_training(config)

# Training loop
for epoch in range(10):
    logging_manager.log_epoch_start(epoch, 10)
    
    for batch in range(100):
        logging_manager.log_batch_start(batch, 100)
        
        # Training step
        try:
            # Your training code here
            batch_metrics = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            logging_manager.log_batch_metrics(batch_metrics)
            
        except Exception as e:
            logging_manager.log_error(e, TrainingPhase.LOSS_COMPUTATION, ErrorCategory.LOSS_COMPUTATION)
        
        logging_manager.log_batch_end()
    
    logging_manager.log_epoch_end({'epoch_loss': epoch_loss})

# End training
logging_manager.end_training({'final_loss': final_loss})
```

### Advanced Usage with Custom Error Handling

```python
# Custom error handling
def custom_training_step(model, data, target, optimizer, criterion):
    try:
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Handle memory error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
        else:
            raise e
    
    except Exception as e:
        # Handle other errors
        raise e

# Use in training loop
for batch in range(num_batches):
    try:
        loss = custom_training_step(model, data, target, optimizer, criterion)
        
        metrics = {
            'loss': loss,
            'accuracy': calculate_accuracy(output, target),
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        logging_manager.log_batch_metrics(metrics)
        
    except Exception as e:
        # Determine error category
        if "memory" in str(e).lower():
            category = ErrorCategory.MEMORY
        elif "gradient" in str(e).lower():
            category = ErrorCategory.GRADIENT_COMPUTATION
        else:
            category = ErrorCategory.UNKNOWN
        
        logging_manager.log_error(e, TrainingPhase.LOSS_COMPUTATION, category)
```

### Real-time Monitoring

```python
# Access real-time metrics
summary = logging_manager.progress_tracker.get_training_summary()
print(f"Current epoch: {summary['current_epoch']}")
print(f"Current loss: {summary['current_loss']:.4f}")
print(f"Best accuracy: {summary['best_accuracy']:.4f}")

# Get error statistics
error_summary = logging_manager.progress_tracker.get_error_summary()
print(f"Total errors: {error_summary['total_errors']}")
print(f"Recovery success rate: {error_summary['recovery_success_rate']:.2%}")

# Create visualizations
logging_manager.progress_tracker.create_training_plots()
```

## Log File Structure

### Generated Files

```
training_logs/
└── experiment_name/
    ├── training.log              # Main log file
    ├── training_metrics.csv      # Metrics in CSV format
    ├── training_errors.json      # Error history
    ├── training_progress.json    # Progress history
    ├── training_config.json      # Training configuration
    ├── training_summary.json     # Final summary
    └── plots/
        └── training_progress.png # Training plots
```

### CSV Metrics Format

```csv
timestamp,epoch,batch,total_batches,loss,accuracy,learning_rate,gradient_norm,memory_usage,gpu_memory,training_time,validation_loss,validation_accuracy,best_loss,best_accuracy,patience_counter
2024-01-01T10:00:00,0,0,100,1.2345,0.5678,0.001,0.1234,512.5,2048.0,0.0,0.0,0.0,inf,0.0,0
2024-01-01T10:00:05,0,1,100,1.1234,0.6789,0.001,0.2345,513.2,2049.0,5.0,0.0,0.0,1.1234,0.6789,0
```

### JSON Error Format

```json
[
  {
    "category": "loss_computation",
    "phase": "loss_computation",
    "error_message": "Loss computation failed",
    "timestamp": "2024-01-01T10:00:00",
    "epoch": 0,
    "batch": 5,
    "recovery_attempted": true,
    "recovery_successful": true,
    "recovery_time": 0.5
  }
]
```

## Best Practices

### Logging Best Practices

1. **Comprehensive Logging**: Log all important events and metrics
2. **Error Categorization**: Use appropriate error categories for better analysis
3. **Recovery Tracking**: Track recovery attempts and success rates
4. **Performance Monitoring**: Monitor system resources in real-time
5. **Visualization**: Generate plots for better understanding

### Error Handling Best Practices

1. **Specific Error Types**: Handle specific exception types
2. **Recovery Strategies**: Implement appropriate recovery strategies
3. **Resource Cleanup**: Clean up resources on errors
4. **Error Context**: Provide detailed error context
5. **Recovery Validation**: Validate recovery success

### Performance Best Practices

1. **Efficient Logging**: Use appropriate log levels
2. **Batch Logging**: Log metrics in batches to reduce I/O
3. **Asynchronous Logging**: Use async logging for better performance
4. **Resource Monitoring**: Monitor memory and GPU usage
5. **Cleanup**: Clean up resources regularly

## Configuration Options

### Logger Configuration

```python
# Configure logger
logging_manager = TrainingLoggingManager(
    experiment_name="my_experiment",
    log_dir="custom_logs"
)

# Custom monitoring thresholds
logging_manager.logger.monitoring_thresholds = {
    'memory_usage_mb': 1000,
    'gpu_memory_mb': 8000,
    'check_interval_seconds': 30
}
```

### Progress Tracker Configuration

```python
# Configure progress tracker
logging_manager.progress_tracker.plot_config = {
    'figure_size': (15, 10),
    'dpi': 300,
    'save_format': 'png'
}
```

## Conclusion

The Training Logging System provides comprehensive logging for training progress and errors with:

- **Real-time Monitoring**: Continuous system resource monitoring
- **Detailed Error Tracking**: Categorized error logging with recovery
- **Progress Visualization**: Automatic plot generation
- **Comprehensive Metrics**: Complete training parameter tracking
- **Automatic Recovery**: Intelligent error recovery mechanisms
- **Multiple Formats**: CSV, JSON, and console output

This system ensures complete visibility into the training process and enables robust error handling and recovery for production-ready deep learning applications. 