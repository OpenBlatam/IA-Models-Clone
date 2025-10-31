# Advanced Logging System Guide

## Overview

This guide documents the comprehensive advanced logging system implemented for AI training operations, specifically focusing on **proper logging for training progress and errors**.

## Key Features

### 🎯 **Comprehensive Training Progress Tracking**
- **Real-time batch progress** logging with detailed metrics
- **Epoch-level tracking** with start/end logging
- **Performance metrics** including timing and memory usage
- **Progress visualization** with milestone achievements

### 🛡️ **Advanced Error Logging**
- **Structured error logging** with context and traceback
- **Error categorization** by severity and type
- **Error recovery tracking** and statistics
- **Automatic error context** capture

### 📊 **Metrics and Analytics**
- **Training metrics** (loss, accuracy, learning rate, gradient norm)
- **System metrics** (memory usage, performance timing)
- **Model information** logging
- **Hyperparameter tracking**

### 🔄 **Log Management**
- **Automatic log rotation** with configurable size limits
- **Multiple log files** for different purposes
- **JSON-structured logging** for machine-readable data
- **Training summaries** with statistics

## System Architecture

### Core Components

#### 1. AdvancedLogger Class
The main logging orchestrator that provides:
- **Multiple specialized loggers** (main, training, errors, metrics)
- **Structured data logging** with dataclasses
- **Context managers** for safe operations
- **Performance monitoring** and statistics

#### 2. TrainingProgressTracker Class
Progress tracking system that provides:
- **Real-time progress updates** with percentages
- **Milestone achievements** (25%, 50%, 75%, 90%)
- **Epoch and overall progress** tracking
- **Visual progress indicators**

#### 3. TrainingMetrics Dataclass
Structured metrics container:
```python
@dataclass
class TrainingMetrics:
    epoch: int
    batch: int
    total_batches: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    memory_usage: Optional[Dict[str, float]] = None
    batch_time: Optional[float] = None
    data_time: Optional[float] = None
    timestamp: Optional[datetime] = None
```

#### 4. ErrorLog Dataclass
Structured error logging:
```python
@dataclass
class ErrorLog:
    error_type: str
    error_message: str
    operation: str
    timestamp: datetime
    traceback: str
    context: Dict[str, Any]
    severity: str = "ERROR"
```

## Usage Examples

### Basic Setup

```python
from advanced_logging_system import AdvancedLogger, TrainingProgressTracker

# Initialize advanced logger
advanced_logger = AdvancedLogger(
    log_dir="logs",
    experiment_name="my_experiment",
    log_level=logging.INFO
)

# Start training
config = {
    "model": "ResNet18",
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 1e-3
}
advanced_logger.start_training(config)
```

### Training Loop Integration

```python
# Log hyperparameters
hyperparams = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingWarmRestarts"
}
advanced_logger.log_hyperparameters(hyperparams)

# Log model information
advanced_logger.log_model_info(model)

# Training loop
for epoch in range(num_epochs):
    # Start epoch
    advanced_logger.start_epoch(epoch + 1, num_epochs)
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        # Training step
        loss = train_step(data, targets)
        accuracy = calculate_accuracy(outputs, targets)
        
        # Log batch progress
        advanced_logger.log_batch_progress(
            epoch=epoch + 1,
            batch=batch_idx + 1,
            total_batches=len(dataloader),
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer.param_groups[0]['lr'],
            gradient_norm=calculate_gradient_norm(model)
        )
    
    # Validation
    val_metrics = validate(model, val_dataloader)
    is_best = val_metrics['accuracy'] > best_accuracy
    
    # Log validation results
    advanced_logger.log_validation(epoch + 1, val_metrics, is_best)
    
    # End epoch
    advanced_logger.end_epoch(epoch + 1, val_metrics)

# End training
final_metrics = {
    "final_loss": train_losses[-1],
    "final_accuracy": val_accuracies[-1],
    "best_accuracy": best_accuracy
}
advanced_logger.end_training(final_metrics)
```

### Error Handling Integration

```python
try:
    # Training operation
    with advanced_logger.training_context("model_training"):
        train_epoch(model, dataloader)
        
except Exception as e:
    # Log error with context
    advanced_logger.log_error(
        error=e,
        operation="model_training",
        context={
            "epoch": current_epoch,
            "batch": current_batch,
            "model_state": "training"
        },
        severity="ERROR"
    )
```

### Progress Tracking

```python
# Initialize progress tracker
progress_tracker = TrainingProgressTracker(advanced_logger)

# Update progress during training
for epoch in range(num_epochs):
    for batch in range(num_batches):
        # Update progress
        progress_tracker.update_progress(epoch + 1, batch + 1, num_batches, num_epochs)
        
        # This will automatically log:
        # - Epoch progress percentage
        # - Overall progress percentage
        # - Milestone achievements (25%, 50%, 75%, 90%)
```

## Log Files Generated

### 1. Main Log (`experiment_main.log`)
- **General application logs**
- **System information**
- **Configuration details**
- **High-level training events**

### 2. Training Log (`experiment_training.log`)
- **Training progress details**
- **Batch-level metrics**
- **Epoch summaries**
- **Validation results**

### 3. Error Log (`experiment_errors.log`)
- **Detailed error information**
- **Stack traces**
- **Error context**
- **Recovery attempts**

### 4. Metrics Log (`experiment_metrics.jsonl`)
- **Structured metrics data**
- **JSON lines format**
- **Machine-readable data**
- **Time-series metrics**

### 5. Training Summary (`experiment_summary.json`)
- **Final training statistics**
- **Performance summaries**
- **Error counts**
- **Completion information**

## Advanced Features

### Memory Usage Monitoring

```python
# Log current memory usage
advanced_logger.log_memory_usage()

# This logs:
# - CPU memory usage (GB)
# - GPU memory usage (allocated, reserved, max allocated)
# - Memory trends over time
```

### Performance Metrics

```python
# Log performance metrics
advanced_logger.log_performance_metrics(batch_time, data_time)

# This tracks:
# - Batch processing time
# - Data loading time
# - Average performance over time
# - Performance bottlenecks
```

### Context Managers

```python
# Safe operation context
with advanced_logger.training_context("data_preprocessing"):
    # All operations are automatically logged
    # Errors are captured with context
    # Timing is automatically measured
    preprocess_data()
```

### Training Summaries

```python
# Get comprehensive training summary
summary = advanced_logger.get_training_summary()

# Returns:
# {
#     "total_metrics": 1000,
#     "total_errors": 5,
#     "training_duration": "0:15:30",
#     "loss_stats": {
#         "min": 0.1,
#         "max": 2.5,
#         "mean": 0.8,
#         "std": 0.3
#     },
#     "accuracy_stats": {
#         "min": 0.3,
#         "max": 0.95,
#         "mean": 0.85,
#         "std": 0.1
#     }
# }
```

## Integration with Optimization Demo

### Updated OptimizedTrainer

```python
class OptimizedTrainer:
    def __init__(self, model, config, advanced_logger=None):
        self.advanced_logger = advanced_logger
        self.progress_tracker = None
        if advanced_logger:
            self.progress_tracker = TrainingProgressTracker(advanced_logger)
    
    def train_epoch(self, dataloader, epoch=1, total_epochs=1):
        # Start epoch logging
        if self.advanced_logger:
            self.advanced_logger.start_epoch(epoch, total_epochs)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Training step with comprehensive logging
            loss, accuracy = self._train_step(data, targets)
            
            # Log batch progress
            if self.advanced_logger:
                self.advanced_logger.log_batch_progress(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    total_batches=len(dataloader),
                    loss=loss,
                    accuracy=accuracy,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    gradient_norm=self._calculate_gradient_norm()
                )
                
                # Update progress tracker
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        epoch, batch_idx + 1, len(dataloader), total_epochs
                    )
        
        # End epoch logging
        if self.advanced_logger:
            self.advanced_logger.end_epoch(epoch, metrics)
```

## Configuration Options

### Logger Configuration

```python
advanced_logger = AdvancedLogger(
    log_dir="logs",                    # Log directory
    experiment_name="my_experiment",   # Experiment name
    log_level=logging.INFO,           # Logging level
    max_log_files=10,                 # Max log files to keep
    max_log_size=10 * 1024 * 1024    # Max log file size (10MB)
)
```

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information and progress
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

## Best Practices

### 1. Comprehensive Error Logging

```python
# Always log errors with context
try:
    operation()
except Exception as e:
    advanced_logger.log_error(
        error=e,
        operation="operation_name",
        context={
            "epoch": current_epoch,
            "batch": current_batch,
            "model_state": model.training,
            "device": next(model.parameters()).device
        }
    )
```

### 2. Regular Progress Updates

```python
# Log progress at regular intervals
if batch_idx % 10 == 0:  # Every 10 batches
    advanced_logger.log_batch_progress(...)
    advanced_logger.log_memory_usage()
```

### 3. Performance Monitoring

```python
# Monitor performance bottlenecks
start_time = time.time()
data_time = time.time() - data_start_time
batch_time = time.time() - batch_start_time

advanced_logger.log_performance_metrics(batch_time, data_time)
```

### 4. Structured Data Logging

```python
# Use structured metrics for analysis
metrics = TrainingMetrics(
    epoch=epoch,
    batch=batch,
    total_batches=total_batches,
    loss=loss,
    accuracy=accuracy,
    learning_rate=lr,
    gradient_norm=grad_norm,
    memory_usage=memory_info,
    batch_time=batch_time,
    data_time=data_time,
    timestamp=datetime.now()
)
```

## Testing and Validation

### Running Tests

```bash
# Run comprehensive logging tests
python test_advanced_logging.py
```

### Test Coverage

The test suite covers:
- **Logging system initialization**
- **Training progress logging**
- **Error logging and recovery**
- **Metrics and performance logging**
- **Progress tracking**
- **Context managers**
- **Log file creation and rotation**
- **Training summaries**
- **Integration with optimization demo**

## Monitoring and Analysis

### Real-time Monitoring

```python
# Monitor training in real-time
summary = advanced_logger.get_training_summary()
print(f"Training Progress: {summary['total_metrics']} metrics logged")
print(f"Errors Encountered: {summary['total_errors']}")
print(f"Current Loss: {summary['loss_stats']['mean']:.4f}")
```

### Log Analysis

```python
# Analyze JSON metrics
import json
import pandas as pd

with open("logs/experiment_metrics.jsonl", "r") as f:
    metrics = [json.loads(line) for line in f]

df = pd.DataFrame(metrics)
print(df.describe())
```

## Production Deployment

### Log Rotation

```python
# Automatic log rotation
advanced_logger = AdvancedLogger(
    max_log_files=10,        # Keep 10 log files
    max_log_size=50*1024*1024  # 50MB per file
)
```

### Error Alerting

```python
# Monitor error rates
summary = advanced_logger.get_training_summary()
if summary['total_errors'] > threshold:
    send_alert(f"High error rate: {summary['total_errors']} errors")
```

### Performance Monitoring

```python
# Monitor training performance
if summary['loss_stats']['mean'] > threshold:
    logger.warning("Training loss is high, consider adjusting hyperparameters")
```

## Conclusion

The advanced logging system provides:

1. **📊 Comprehensive Training Progress Tracking** - Real-time monitoring of all training metrics
2. **🛡️ Advanced Error Logging** - Structured error capture with context and recovery
3. **📈 Performance Monitoring** - Memory usage, timing, and bottleneck detection
4. **🔄 Automatic Log Management** - Rotation, organization, and cleanup
5. **📋 Training Summaries** - Statistical analysis and performance insights
6. **🔧 Easy Integration** - Seamless integration with existing training loops
7. **📱 Progress Visualization** - Milestone achievements and progress indicators
8. **🎯 Production Ready** - Enterprise-grade logging for production environments

This system ensures that AI training operations are **fully monitored, debuggable, and analyzable**, providing comprehensive insights into training progress and error patterns. 