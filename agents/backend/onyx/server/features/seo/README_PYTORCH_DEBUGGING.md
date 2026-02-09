# PyTorch Debugging Tools Integration

## Overview

The Comprehensive Logging System now includes **integrated PyTorch debugging tools** that provide advanced debugging capabilities for deep learning models, specifically designed for the SEO evaluation system. This integration leverages PyTorch's built-in debugging features like `autograd.detect_anomaly()`, gradient debugging, memory monitoring, and tensor analysis.

## Key Features

### 🚀 **Core PyTorch Debugging Capabilities**
- **Autograd Anomaly Detection**: `torch.autograd.set_detect_anomaly()` integration
- **Gradient Debugging**: Comprehensive gradient analysis and monitoring
- **Memory Debugging**: CUDA and CPU memory usage tracking
- **Tensor Debugging**: Detailed tensor information and validation
- **Model State Debugging**: Model parameter analysis and health monitoring

### 🔧 **Advanced Debugging Tools**
- **PyTorch Profiler**: Performance profiling and bottleneck identification
- **Context Managers**: Easy-to-use debugging contexts for specific operations
- **Real-time Monitoring**: Continuous debugging during training
- **Comprehensive Logging**: All debugging information integrated with logging system

### 📊 **Debugging Analytics**
- **Gradient History**: Track gradient norms and anomalies over time
- **Memory Tracking**: Monitor memory usage patterns and fragmentation
- **Error Context**: Rich debugging context for error analysis
- **Performance Metrics**: Detailed performance analysis and optimization insights

## Architecture

### Core Components

```
ComprehensiveLogger
├── PyTorchDebugTools          # PyTorch debugging engine
│   ├── Anomaly Detection      # autograd.detect_anomaly()
│   ├── Gradient Debugging     # Gradient analysis and monitoring
│   ├── Memory Debugging       # Memory usage tracking
│   ├── Tensor Debugging       # Tensor validation and analysis
│   ├── Model State Debugging  # Model health monitoring
│   └── Profiler Integration   # PyTorch profiler
├── TrainingMetricsLogger      # Training progress and metrics
├── ErrorTracker              # Error tracking and analysis
└── SystemMonitor            # System resource monitoring
```

### Debugging Flow

```
Training Operation → PyTorchDebugTools → Comprehensive Logging
        ↓
├── Gradient Analysis (NaN/Inf detection)
├── Memory Usage Monitoring
├── Tensor Validation
├── Model State Analysis
├── Performance Profiling
└── Structured Logging Output
```

## Installation

### Dependencies

The PyTorch debugging tools require the following packages:

```bash
# Core PyTorch
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=0.15.0

# Comprehensive Logging System
pip install -r requirements_comprehensive_logging.txt
```

### Configuration

Enable PyTorch debugging in your logging configuration:

```python
from comprehensive_logging import LoggingConfig

config = LoggingConfig(
    enable_pytorch_debugging=True,
    enable_autograd_anomaly_detection=False,  # Can be expensive
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_tensor_debugging=True,
    enable_profiler=False,  # Can be expensive
    max_grad_norm=1.0,
    memory_fraction=0.8
)
```

## Usage

### Basic Setup

```python
from comprehensive_logging import setup_logging

# Initialize with PyTorch debugging enabled
logger = setup_logging(
    "seo_evaluation",
    enable_pytorch_debugging=True,
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_tensor_debugging=True
)
```

### Gradient Debugging

```python
# Debug model gradients during training
output = model(input_data)
loss = criterion(output, target)
loss.backward()

# Comprehensive gradient debugging
gradient_debug = logger.debug_model_gradients(model, loss)

# Access debugging information
print(f"Total gradient norm: {gradient_debug['total_grad_norm']}")
print(f"Parameters with NaN: {gradient_debug['nan_gradients']}")
print(f"Parameters with Inf: {gradient_debug['inf_gradients']}")
```

### Memory Debugging

```python
# Debug PyTorch memory usage
memory_debug = logger.debug_model_memory()

# Access memory information
if torch.cuda.is_available():
    print(f"CUDA Memory Allocated: {memory_debug['cuda_memory_allocated']:.2f} GB")
    print(f"CUDA Memory Reserved: {memory_debug['cuda_memory_reserved']:.2f} GB")
    print(f"Memory Fragmentation: {memory_debug['cuda_memory_fragmentation']:.2%}")
```

### Tensor Debugging

```python
# Debug tensor information
tensor = torch.randn(10, 20, requires_grad=True)
tensor_debug = logger.debug_tensor(tensor, "input_tensor")

# Access tensor debugging information
print(f"Shape: {tensor_debug['input_tensor_shape']}")
print(f"Device: {tensor_debug['input_tensor_device']}")
print(f"Requires Grad: {tensor_debug['input_tensor_requires_grad']}")
print(f"Has NaN: {tensor_debug['input_tensor_has_nan']}")
print(f"Has Inf: {tensor_debug['input_tensor_has_inf']}")
```

### Model State Debugging

```python
# Debug model state
model_debug = logger.debug_model_state(model)

# Access model debugging information
print(f"Training Mode: {model_debug['model_training_mode']}")
print(f"Total Parameters: {model_debug['total_parameters']}")
print(f"Trainable Parameters: {model_debug['trainable_parameters']}")
print(f"Parameters with NaN: {model_debug['parameters_with_nan']}")
print(f"Parameters with Inf: {model_debug['parameters_with_inf']}")
```

### Training Step with Comprehensive Debugging

```python
# Log training step with full debugging
debug_info = logger.log_training_step_with_debug(
    epoch=epoch,
    step=step,
    loss=loss.item(),
    model=model,
    accuracy=accuracy,
    learning_rate=optimizer.param_groups[0]['lr']
)

# Access comprehensive debugging information
gradient_debug = debug_info['gradient_debug']
memory_debug = debug_info['memory_debug']
model_debug = debug_info['model_debug']
```

## Context Managers

### PyTorch Debugging Context

```python
# Use context manager for specific operations
with logger.pytorch_debugging("model_inference", enable_anomaly_detection=True):
    # This operation will have anomaly detection enabled
    output = model(input_data)
    # Anomaly detection automatically disabled after context
```

### Gradient Debugging Context

```python
# Debug gradients for specific operations
with logger.gradient_debugging(model, "forward_pass"):
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    # Gradients automatically debugged after context
```

### Performance Tracking with Debugging

```python
# Combine performance tracking with PyTorch debugging
with logger.performance_tracking("training_step"):
    with logger.pytorch_debugging("backward_pass", enable_anomaly_detection=True):
        loss.backward()
        # Both performance and debugging information logged
```

## Advanced Features

### Autograd Anomaly Detection

```python
# Enable anomaly detection for debugging
logger.enable_autograd_anomaly_detection(True)

# This will now detect and report any anomalies in the computation graph
output = model(input_data)
loss = criterion(output, target)
loss.backward()  # Will show detailed error if anomaly detected

# Disable when not needed (can be expensive)
logger.enable_autograd_anomaly_detection(False)
```

### PyTorch Profiler

```python
# Start profiling
logger.start_profiler()

# Run operations to profile
for i in range(100):
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Stop profiling
logger.stop_profiler()

# Profiler traces saved to ./logs/profiler/
```

### Custom Debugging Context

```python
# Add custom debugging context
logger.pytorch_debug.debug_context['custom_info'] = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'Adam'
}

# This context will be included in all debugging operations
```

## Integration with SEO Evaluation System

### SEO Trainer Integration

```python
from evaluation_metrics_ultra_optimized import UltraOptimizedSEOTrainer
from comprehensive_logging import setup_logging

class DebuggedSEOTrainer(UltraOptimizedSEOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = setup_logging(
            "seo_trainer",
            enable_pytorch_debugging=True,
            enable_gradient_debugging=True,
            enable_memory_debugging=True
        )
    
    def train_step(self, batch_data):
        try:
            # Enable anomaly detection for this step
            with self.logger.pytorch_debugging("train_step", enable_anomaly_detection=True):
                result = super().train_step(batch_data)
                
                # Debug gradients after backward pass
                gradient_debug = self.logger.debug_model_gradients(
                    self.model, 
                    result['loss']
                )
                
                # Log with debugging information
                self.logger.log_training_step_with_debug(
                    epoch=self.current_epoch,
                    step=self.current_step,
                    loss=result['loss'].item(),
                    model=self.model,
                    accuracy=result.get('accuracy'),
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )
                
                return result
                
        except Exception as e:
            # Log error with debugging context
            self.logger.log_error(
                error=e,
                context={
                    "operation": "train_step",
                    "batch_size": len(batch_data),
                    "model_state": self.logger.debug_model_state(self.model),
                    "memory_state": self.logger.debug_model_memory()
                },
                severity="ERROR"
            )
            raise
```

### Gradio Interface Integration

```python
from gradio_user_friendly_interface import SEOGradioUserFriendlyInterface
from comprehensive_logging import setup_logging

class DebuggedSEOGradioInterface(SEOGradioUserFriendlyInterface):
    def __init__(self):
        super().__init__()
        self.logger = setup_logging(
            "gradio_interface",
            enable_pytorch_debugging=True,
            enable_gradient_debugging=True
        )
    
    async def start_training(self, dummy_data_size: int = 100):
        try:
            with self.logger.pytorch_debugging("gradio_training", enable_anomaly_detection=True):
                result = await super().start_training(dummy_data_size)
                
                # Log training completion with debugging info
                self.logger.log_info("Training completed", {
                    "data_size": dummy_data_size,
                    "status": "success",
                    "model_state": self.logger.debug_model_state(self.model) if hasattr(self, 'model') else None
                })
                
                return result
                
        except Exception as e:
            self.logger.log_error(
                error=e,
                context={
                    "operation": "start_training",
                    "data_size": dummy_data_size
                },
                severity="ERROR"
            )
            raise
```

## Configuration Options

### LoggingConfig PyTorch Debugging Options

```python
@dataclass
class LoggingConfig:
    # PyTorch Debugging Tools Configuration
    enable_pytorch_debugging: bool = True
    enable_autograd_anomaly_detection: bool = False  # Can be expensive
    enable_gradient_debugging: bool = True
    enable_memory_debugging: bool = True
    enable_profiler: bool = False  # Can be expensive
    enable_tensor_debugging: bool = True
    max_grad_norm: float = 1.0
    memory_fraction: float = 0.8
    enable_cuda_memory_stats: bool = True
```

### Environment-Specific Configurations

```python
# Development environment
dev_config = LoggingConfig(
    enable_pytorch_debugging=True,
    enable_autograd_anomaly_detection=True,  # Enable for development
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_tensor_debugging=True,
    enable_profiler=True  # Enable for development
)

# Production environment
prod_config = LoggingConfig(
    enable_pytorch_debugging=True,
    enable_autograd_anomaly_detection=False,  # Disable for performance
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_tensor_debugging=False,  # Disable for performance
    enable_profiler=False  # Disable for performance
)

# Testing environment
test_config = LoggingConfig(
    enable_pytorch_debugging=True,
    enable_autograd_anomaly_detection=True,  # Enable for testing
    enable_gradient_debugging=True,
    enable_memory_debugging=False,  # Disable for testing
    enable_tensor_debugging=True,
    enable_profiler=False  # Disable for testing
)
```

## Output and Analysis

### Debug Information Structure

```json
{
  "gradient_debug": {
    "total_grad_norm": 1.234,
    "param_count": 150,
    "nan_gradients": 0,
    "inf_gradients": 0,
    "loss_value": 0.567,
    "loss_requires_grad": true,
    "loss_grad_fn": "MseLossBackward0"
  },
  "memory_debug": {
    "cuda_memory_allocated": 2.5,
    "cuda_memory_reserved": 3.2,
    "cuda_memory_fragmentation": 0.22,
    "gpu_0_allocated": 2.5,
    "gpu_0_reserved": 3.2
  },
  "model_debug": {
    "model_training_mode": true,
    "total_parameters": 15000,
    "trainable_parameters": 15000,
    "parameters_with_nan": 0,
    "parameters_with_inf": 0
  }
}
```

### Logging Summary with PyTorch Debug

```python
# Get comprehensive summary including PyTorch debugging
summary = logger.get_logging_summary()

# Access PyTorch debugging summary
pytorch_debug = summary['pytorch_debug']
print(f"Anomaly Detection: {pytorch_debug['anomaly_detection_enabled']}")
print(f"Profiler Active: {pytorch_debug['profiler_active']}")
print(f"Gradient History Entries: {len(pytorch_debug['gradient_history'])}")

# Access gradient history
for entry in pytorch_debug['gradient_history'][-5:]:  # Last 5 entries
    print(f"Time: {entry['timestamp']}, Norm: {entry['total_norm']:.4f}")
```

## Performance Considerations

### Memory Overhead

- **Gradient Debugging**: Minimal overhead, only during backward pass
- **Memory Debugging**: Very low overhead, uses existing PyTorch APIs
- **Tensor Debugging**: Low overhead, only when explicitly called
- **Anomaly Detection**: Can be expensive, use selectively
- **Profiler**: Significant overhead, use only when needed

### Optimization Tips

```python
# Use selective debugging for production
config = LoggingConfig(
    enable_pytorch_debugging=True,
    enable_autograd_anomaly_detection=False,  # Disable for performance
    enable_gradient_debugging=True,           # Keep for monitoring
    enable_memory_debugging=True,             # Keep for monitoring
    enable_tensor_debugging=False,            # Disable for performance
    enable_profiler=False                     # Disable for performance
)

# Use context managers for selective debugging
with logger.pytorch_debugging("critical_operation", enable_anomaly_detection=True):
    # Only enable expensive debugging for critical operations
    critical_result = critical_operation()
```

## Troubleshooting

### Common Issues

#### Anomaly Detection Not Working
```python
# Check if anomaly detection is properly enabled
if logger.pytorch_debug.anomaly_detection_enabled:
    print("Anomaly detection is active")
else:
    print("Anomaly detection is not active")

# Manually enable if needed
logger.enable_autograd_anomaly_detection(True)
```

#### Memory Debugging Issues
```python
# Check CUDA availability
if torch.cuda.is_available():
    print("CUDA is available for memory debugging")
else:
    print("CUDA not available, memory debugging limited to CPU")

# Check memory debugging configuration
if logger.config.enable_memory_debugging:
    print("Memory debugging is enabled")
else:
    print("Memory debugging is disabled")
```

#### Profiler Issues
```python
# Check profiler configuration
if logger.config.enable_profiler:
    print("Profiler is configured")
    if logger.pytorch_debug.profiler_active:
        print("Profiler is currently active")
    else:
        print("Profiler is not active")
else:
    print("Profiler is not configured")
```

### Debug Mode

```python
# Enable comprehensive debugging for troubleshooting
debug_config = LoggingConfig(
    log_level="DEBUG",
    enable_pytorch_debugging=True,
    enable_autograd_anomaly_detection=True,
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_tensor_debugging=True,
    enable_profiler=True
)

debug_logger = setup_logging("debug_logger", **debug_config.__dict__)

# Check configuration
print(debug_logger.config)
print(debug_logger.get_logging_summary())
```

## Testing

### Run PyTorch Debugging Tests

```bash
# Run comprehensive tests
python test_pytorch_debugging.py

# Run specific test classes
python -m unittest test_pytorch_debugging.TestPyTorchDebugTools
python -m unittest test_pytorch_debugging.TestComprehensiveLoggerPyTorchDebugging
python -m unittest test_pytorch_debugging.TestPyTorchDebuggingIntegration
```

### Test Coverage

The test suite covers:
- **PyTorchDebugTools**: All debugging tool functionality
- **ComprehensiveLogger Integration**: PyTorch debugging integration
- **Context Managers**: All context manager functionality
- **SEO Integration**: Integration with SEO evaluation system
- **Performance Tests**: High-volume debugging performance
- **Error Handling**: Error scenarios with debugging context

## Future Enhancements

### Planned Features

- **Real-time Debugging Dashboard**: Web-based debugging interface
- **Advanced Profiling**: More sophisticated performance analysis
- **Memory Leak Detection**: Automatic memory leak detection
- **Gradient Clipping Integration**: Automatic gradient clipping based on debugging
- **Distributed Debugging**: Multi-GPU debugging support

### Extensibility

- **Custom Debug Handlers**: User-defined debugging operations
- **Plugin System**: Modular debugging components
- **Custom Metrics**: Application-specific debugging metrics
- **Export Formats**: Additional debugging output formats

## Contributing

### Development Setup

1. Clone the repository
2. Install development dependencies
3. Run tests: `python test_pytorch_debugging.py`
4. Follow PEP 8 style guidelines

### Code Standards

- Type hints for all functions
- Comprehensive docstrings
- Unit tests for new features
- Error handling for edge cases
- Performance considerations

## License

This PyTorch debugging integration is part of the Comprehensive Logging System and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for debugging details
3. Enable debug logging for detailed information
4. Check PyTorch version compatibility
5. Verify CUDA availability for GPU debugging

## Examples

### Complete Training Loop with Debugging

```python
from comprehensive_logging import setup_logging
import torch
import torch.nn as nn

# Setup logging with PyTorch debugging
logger = setup_logging(
    "seo_training",
    enable_pytorch_debugging=True,
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_tensor_debugging=True
)

try:
    # Create model and data
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    
    x = torch.randn(64, 100)
    y = torch.randint(0, 10, (64,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with comprehensive debugging
    for epoch in range(5):
        for step in range(10):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x)
            loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            
            # Debug before optimization
            gradient_debug = logger.debug_model_gradients(model, loss, epoch=epoch, step=step)
            memory_debug = logger.debug_model_memory(epoch=epoch, step=step)
            model_debug = logger.debug_model_state(model, epoch=epoch, step=step)
            
            # Optimizer step
            optimizer.step()
            
            # Log with debugging
            logger.log_training_step_with_debug(
                epoch=epoch,
                step=step,
                loss=loss.item(),
                model=model,
                accuracy=0.8,
                learning_rate=optimizer.param_groups[0]['lr']
            )
        
        # Log epoch summary
        logger.log_epoch_summary(
            epoch=epoch,
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.8,
            val_accuracy=0.75
        )
    
    # Get comprehensive summary
    summary = logger.get_logging_summary()
    print("Training completed with comprehensive debugging!")

finally:
    # Cleanup
    logger.cleanup()
```

This comprehensive PyTorch debugging integration provides powerful debugging capabilities for the SEO evaluation system while maintaining high performance and ease of use.
