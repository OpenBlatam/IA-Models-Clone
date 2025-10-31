# Comprehensive Logging System for SEO Evaluation

## Overview

The Comprehensive Logging System provides advanced logging capabilities for training progress and errors in the Ultra-Optimized SEO Evaluation System. It features structured logging, real-time monitoring, error tracking, and comprehensive metrics collection.

## Key Features

### 🚀 **Core Logging Capabilities**
- **Structured Logging**: JSON-formatted logs with context and metadata
- **Multi-Output Support**: Console, file, JSON, and remote logging
- **Async Logging**: Non-blocking log handling for high-performance applications
- **Thread-Safe**: Multi-threaded environment support

### 📊 **Training Metrics Logging**
- **Step-by-Step Logging**: Detailed training step metrics (loss, accuracy, learning rate)
- **Epoch Summaries**: Comprehensive epoch completion summaries
- **Progress Tracking**: CSV and JSONL output for analysis
- **Performance Metrics**: Training time, memory usage, gradient norms

### 🛡️ **Error Tracking & Analysis**
- **Error Classification**: Categorization by type and severity
- **Recovery Tracking**: Monitor error recovery attempts and success rates
- **Trend Analysis**: Identify patterns and peak error times
- **Context Preservation**: Maintain error context for debugging

### 💻 **System Monitoring**
- **Resource Metrics**: CPU, memory, disk, and network usage
- **GPU Monitoring**: CUDA memory allocation and utilization
- **Real-Time Monitoring**: Continuous system health tracking
- **Performance Profiling**: Operation timing and resource consumption

### 🔧 **Advanced Features**
- **Configurable Logging**: Flexible configuration for different environments
- **Log Rotation**: Automatic file rotation and backup management
- **Context Managers**: Performance tracking and resource management
- **Integration Ready**: Easy integration with existing systems

## Architecture

### Core Components

```
ComprehensiveLogger
├── TrainingMetricsLogger    # Training progress and metrics
├── ErrorTracker            # Error tracking and analysis
├── SystemMonitor          # System resource monitoring
└── StructuredFormatter    # Log formatting and output
```

### Logging Flow

```
Application → ComprehensiveLogger → Multiple Handlers
    ↓
├── Console Output (Structured)
├── File Output (Rotating)
├── JSON Output (Machine-readable)
├── Training Metrics (CSV/JSONL)
├── Error Tracking (JSONL)
└── System Metrics (JSONL)
```

## Installation

### Basic Installation

```bash
pip install -r requirements_comprehensive_logging.txt
```

### Core Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **psutil**: System monitoring

### Optional Dependencies

- **loguru**: Enhanced logging features
- **rich**: Beautiful log formatting
- **nvidia-ml-py3**: GPU monitoring
- **matplotlib/seaborn**: Log visualization

## Usage

### Basic Setup

```python
from comprehensive_logging import setup_logging

# Initialize with default configuration
logger = setup_logging("seo_evaluation")

# Or with custom configuration
logger = setup_logging(
    name="seo_evaluation",
    log_level="DEBUG",
    log_dir="./logs",
    enable_console=True,
    enable_file=True,
    log_training_metrics=True,
    log_system_metrics=True
)
```

### Training Logging

```python
# Log training steps
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        # Training logic
        loss = model.train_step(batch)
        
        # Log training step
        logger.log_training_step(
            epoch=epoch,
            step=step,
            loss=loss,
            accuracy=accuracy,
            learning_rate=optimizer.param_groups[0]['lr'],
            gradient_norm=gradient_norm,
            memory_usage=torch.cuda.memory_allocated() / 1e9
        )
    
    # Log epoch summary
    logger.log_epoch_summary(
        epoch=epoch,
        train_loss=avg_train_loss,
        val_loss=avg_val_loss,
        train_accuracy=avg_train_acc,
        val_accuracy=avg_val_acc
    )
```

### Error Logging

```python
try:
    # Risky operation
    result = model.inference(input_data)
except Exception as e:
    # Log error with context
    logger.log_error(
        error=e,
        context={
            "operation": "model_inference",
            "input_shape": input_data.shape,
            "model_state": "training"
        },
        severity="ERROR",
        recovery_attempted=False
    )
    
    # Attempt recovery
    try:
        result = model.inference(input_data.to('cpu'))
        logger.error_tracker.track_recovery_success(
            error_type=type(e).__name__,
            recovery_method="cpu_fallback"
        )
    except Exception as recovery_error:
        logger.log_error(
            error=recovery_error,
            context={"recovery_attempt": "cpu_fallback"},
            severity="CRITICAL"
        )
```

### Performance Tracking

```python
# Using context manager
with logger.performance_tracking("data_loading"):
    data = load_large_dataset()

# Manual performance logging
start_time = time.time()
result = expensive_operation()
duration = time.time() - start_time
logger.log_performance("expensive_operation", duration, result_size=len(result))
```

### System Monitoring

```python
# System monitoring starts automatically
# Access system metrics
system_metrics = logger.system_monitor._collect_system_metrics()
print(f"CPU Usage: {system_metrics['cpu_percent']}%")
print(f"Memory Usage: {system_metrics['memory_percent']}%")

# Stop monitoring when done
logger.system_monitor.stop_monitoring()
```

## Configuration

### LoggingConfig Options

```python
@dataclass
class LoggingConfig:
    log_level: str = "INFO"                    # Logging level
    log_dir: str = "./logs"                    # Log directory
    max_file_size: int = 10 * 1024 * 1024     # Max log file size (10MB)
    backup_count: int = 5                      # Number of backup files
    enable_console: bool = True                # Console output
    enable_file: bool = True                   # File output
    enable_json: bool = True                   # JSON output
    log_training_metrics: bool = True          # Training metrics logging
    log_system_metrics: bool = True            # System monitoring
    log_gpu_metrics: bool = True               # GPU monitoring
    enable_async_logging: bool = True          # Async log handling
    enable_thread_safety: bool = True          # Thread safety
    structured_format: bool = True             # Structured logging
    include_context: bool = True               # Include context in logs
    max_queue_size: int = 1000                # Async queue size
```

### Environment-Specific Configurations

```python
# Development environment
dev_logger = setup_logging(
    log_level="DEBUG",
    enable_console=True,
    enable_file=True,
    log_debug=True
)

# Production environment
prod_logger = setup_logging(
    log_level="INFO",
    enable_console=False,
    enable_file=True,
    enable_json=True,
    log_system_metrics=True,
    max_file_size=100 * 1024 * 1024  # 100MB
)

# Testing environment
test_logger = setup_logging(
    log_level="WARNING",
    enable_console=True,
    enable_file=False,
    log_training_metrics=False
)
```

## Output Files

### Generated Log Files

```
logs/
├── application.log          # Main application logs
├── application.jsonl        # JSON-formatted logs
├── training_metrics.jsonl   # Training step metrics
├── training_progress.csv    # Training progress (CSV)
├── errors.jsonl            # Error tracking
├── error_summary.json      # Error analysis summary
└── system_metrics.jsonl    # System resource metrics
```

### Log File Formats

#### Training Metrics (JSONL)
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "epoch": 1,
  "step": 100,
  "loss": 0.234,
  "accuracy": 0.89,
  "learning_rate": 0.001,
  "gradient_norm": 1.2,
  "memory_usage": 2.5,
  "time_elapsed": 3600.5
}
```

#### Error Tracking (JSONL)
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "error_type": "RuntimeError",
  "error_message": "CUDA out of memory",
  "severity": "ERROR",
  "context": {"operation": "training", "batch_size": 32},
  "stack_trace": "...",
  "recovery_attempted": true
}
```

#### System Metrics (JSONL)
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "cpu_percent": 45.2,
  "cpu_count": 8,
  "memory_total": 17179869184,
  "memory_available": 8589934592,
  "memory_percent": 50.0,
  "gpu_0_memory_allocated": 2147483648,
  "gpu_0_memory_reserved": 3221225472
}
```

## Integration with SEO Evaluation System

### Ultra-Optimized Integration

```python
from evaluation_metrics_ultra_optimized import UltraOptimizedSEOTrainer
from comprehensive_logging import setup_logging

class LoggedSEOTrainer(UltraOptimizedSEOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logging("seo_trainer")
    
    def train_step(self, batch_data):
        try:
            with self.logger.performance_tracking("train_step"):
                result = super().train_step(batch_data)
                
                # Log training metrics
                self.logger.log_training_step(
                    epoch=self.current_epoch,
                    step=self.current_step,
                    loss=result['loss'],
                    accuracy=result.get('accuracy'),
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    gradient_norm=self._get_gradient_norm()
                )
                
                return result
                
        except Exception as e:
            self.logger.log_error(
                error=e,
                context={"batch_size": len(batch_data)},
                severity="ERROR"
            )
            raise
```

### Gradio Interface Integration

```python
from gradio_user_friendly_interface import SEOGradioUserFriendlyInterface
from comprehensive_logging import setup_logging

class LoggedSEOGradioInterface(SEOGradioUserFriendlyInterface):
    def __init__(self):
        super().__init__()
        self.logger = setup_logging("gradio_interface")
    
    async def start_training(self, dummy_data_size: int = 100):
        try:
            self.logger.log_info("Starting training simulation", {
                "data_size": dummy_data_size,
                "user": "gradio_user"
            })
            
            result = await super().start_training(dummy_data_size)
            
            self.logger.log_info("Training simulation completed", {
                "data_size": dummy_data_size,
                "status": "success"
            })
            
            return result
            
        except Exception as e:
            self.logger.log_error(
                error=e,
                context={"operation": "start_training", "data_size": dummy_data_size},
                severity="ERROR"
            )
            raise
```

## Performance Considerations

### Async Logging Benefits
- **Non-blocking**: Logging doesn't interfere with training
- **Queue-based**: Efficient memory management
- **Configurable**: Adjustable queue sizes and flush intervals

### File Rotation
- **Automatic**: Prevents log files from growing too large
- **Backup Management**: Configurable number of backup files
- **Size Control**: Configurable maximum file sizes

### Memory Management
- **Efficient Storage**: JSONL format for easy parsing
- **Context Preservation**: Structured data without duplication
- **Cleanup**: Automatic resource cleanup on shutdown

## Monitoring and Analysis

### Real-Time Monitoring

```python
# Get current logging status
summary = logger.get_logging_summary()
print(f"Total errors: {summary['error_analysis']['total_errors']}")
print(f"Training steps: {summary['training_metrics']['total_steps']}")

# Monitor error trends
error_analysis = logger.error_tracker.get_error_analysis()
print(f"Recovery rate: {error_analysis['recovery_success_rate']:.2%}")
print(f"Most common error: {error_analysis['most_common_errors'][0]}")
```

### Log Analysis Tools

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training progress
df = pd.read_csv("logs/training_progress.csv")

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(df['step'], df['loss'])
plt.title('Training Loss Over Time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()

# Load error summary
with open("logs/error_summary.json") as f:
    error_summary = json.load(f)

# Analyze error patterns
error_df = pd.DataFrame(error_summary['error_timeline'])
error_df['timestamp'] = pd.to_datetime(error_df['timestamp'])
error_df.set_index('timestamp').resample('H').count().plot()
```

## Testing

### Unit Tests

```python
import pytest
from comprehensive_logging import setup_logging, LoggingConfig

def test_logging_setup():
    logger = setup_logging("test_logger")
    assert logger is not None
    assert logger.config.log_level == "INFO"

def test_training_logging():
    logger = setup_logging("test_training")
    
    # Test training step logging
    logger.log_training_step(epoch=1, step=1, loss=0.5)
    
    # Verify metrics were logged
    summary = logger.get_logging_summary()
    assert summary['training_metrics']['total_steps'] == 1

def test_error_tracking():
    logger = setup_logging("test_errors")
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error(e, context={"test": True})
    
    # Verify error was tracked
    analysis = logger.get_logging_summary()
    assert analysis['error_analysis']['total_errors'] == 1
```

### Integration Tests

```python
def test_seo_trainer_integration():
    from evaluation_metrics_ultra_optimized import UltraOptimizedConfig
    
    config = UltraOptimizedConfig()
    trainer = LoggedSEOTrainer(config)
    
    # Test that logging is properly integrated
    assert hasattr(trainer, 'logger')
    assert trainer.logger is not None
```

## Troubleshooting

### Common Issues

#### Log Files Not Created
- Check directory permissions
- Verify `log_dir` path exists
- Ensure sufficient disk space

#### Performance Issues
- Reduce `max_queue_size` for memory-constrained environments
- Disable async logging if not needed
- Adjust monitoring intervals

#### Missing Dependencies
- Install required packages: `pip install -r requirements_comprehensive_logging.txt`
- Check PyTorch CUDA availability for GPU monitoring

### Debug Mode

```python
# Enable debug logging
logger = setup_logging(
    log_level="DEBUG",
    log_debug=True,
    enable_console=True
)

# Check configuration
print(logger.config)
print(logger.get_logging_summary())
```

## Future Enhancements

### Planned Features
- **Remote Logging**: Cloud-based log aggregation
- **Real-time Dashboards**: Web-based monitoring interfaces
- **Machine Learning**: Anomaly detection in logs
- **Advanced Analytics**: Predictive error analysis
- **Integration APIs**: Third-party service connectors

### Extensibility
- **Custom Handlers**: User-defined logging outputs
- **Plugin System**: Modular logging components
- **Custom Metrics**: Application-specific monitoring
- **Export Formats**: Additional output formats

## Contributing

### Development Setup
1. Clone the repository
2. Install development dependencies
3. Run tests: `pytest tests/`
4. Follow PEP 8 style guidelines

### Code Standards
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for new features
- Error handling for edge cases

## License

This comprehensive logging system is part of the Ultra-Optimized SEO Evaluation System and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Enable debug logging for detailed information
4. Check system requirements and dependencies
