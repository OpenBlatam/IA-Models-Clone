# PyTorch Debugging Tools Integration

Comprehensive integration of PyTorch's built-in debugging tools including `autograd.detect_anomaly()`, profiler, and other debugging utilities for enhanced error detection and performance analysis.

## Overview

The PyTorch debugging system provides a comprehensive suite of debugging tools that integrate seamlessly with the email sequence training pipeline. It includes:

- **Autograd Anomaly Detection**: Detects and reports gradient computation issues
- **PyTorch Profiler**: Performance profiling and bottleneck identification
- **Memory Tracking**: Monitor memory usage and detect memory leaks
- **Gradient Checking**: Validate gradients for NaN/Inf values and explosion/vanishing
- **Forward/Backward Debugging**: Detailed analysis of model computations
- **Training Session Debugging**: Complete training loop debugging

## Features

### 1. Autograd Anomaly Detection

Detects issues in gradient computation using PyTorch's `autograd.detect_anomaly()`:

```python
from core.pytorch_debugging import create_pytorch_debugger

debugger = create_pytorch_debugger(enable_anomaly_detection=True)

with debugger.anomaly_detection():
    # Your training code here
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
```

**Benefits:**
- Automatically detects gradient computation errors
- Provides detailed stack traces for debugging
- Helps identify numerical instability issues

### 2. PyTorch Profiler

Performance profiling to identify bottlenecks:

```python
with debugger.profiling(output_file="profile.json"):
    # Your training code here
    for epoch in range(num_epochs):
        train_epoch()
```

**Features:**
- CPU and CUDA activity profiling
- Memory usage tracking
- Shape recording for tensors
- Stack trace collection
- Chrome trace export for visualization

### 3. Memory Tracking

Monitor memory usage and detect memory leaks:

```python
with debugger.memory_tracking():
    # Your training code here
    train_model()
```

**Capabilities:**
- CPU memory usage monitoring
- CUDA memory allocation tracking
- Memory leak detection
- Memory snapshot comparison

### 4. Gradient Checking

Validate gradients for common issues:

```python
# Check gradients after backward pass
gradient_norm = debugger.check_gradients(model, gradient_threshold=1.0)

# Check model parameters
param_stats = debugger.check_model_parameters(model)
```

**Checks:**
- NaN gradient detection
- Inf gradient detection
- Gradient explosion detection
- Gradient vanishing detection
- Parameter value validation

### 5. Forward/Backward Debugging

Detailed analysis of model computations:

```python
# Debug forward pass with layer hooks
outputs = debugger.debug_forward_pass(model, inputs, layer_hooks=True)

# Debug backward pass
debugger.debug_backward_pass(loss)
```

**Features:**
- Layer-by-layer activation analysis
- Input/output shape tracking
- Statistical analysis of activations
- Timing information

## Integration with Training Pipeline

### Enhanced Training Optimizer Integration

The PyTorch debugging tools are fully integrated with the enhanced training optimizer:

```python
from core.enhanced_training_optimizer import create_enhanced_training_optimizer

optimizer = create_enhanced_training_optimizer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    debug_mode=True,
    enable_pytorch_debugging=True,
    enable_profiling=False,
    max_epochs=100,
    learning_rate=0.001
)

results = await optimizer.train()
```

### Training Session Debugging

Complete training session debugging with context managers:

```python
from core.pytorch_debugging import debug_training_session

with debug_training_session(
    model=model,
    logger=logger,
    debug_mode=True,
    enable_anomaly_detection=True,
    enable_profiling=True,
    enable_memory_tracking=True,
    enable_gradient_checking=True
) as debugger:
    # Your training code here
    train_model()
```

## Configuration Options

### Debugger Configuration

```python
debugger = create_pytorch_debugger(
    logger=logger,                    # Training logger instance
    debug_mode=True,                  # Enable debug mode
    enable_anomaly_detection=True,    # Enable autograd anomaly detection
    enable_profiling=False,           # Enable PyTorch profiling
    enable_memory_tracking=True,      # Enable memory tracking
    enable_gradient_checking=True,    # Enable gradient checking
    log_dir="debug_logs"             # Debug log directory
)
```

### Training Optimizer Configuration

```python
optimizer = create_enhanced_training_optimizer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    debug_mode=True,                  # Enable debug mode
    enable_pytorch_debugging=True,    # Enable PyTorch debugging
    enable_profiling=False,           # Enable profiling
    max_epochs=100,
    learning_rate=0.001,
    gradient_clip=1.0
)
```

## Debug Reports and Logging

### Debug Summary

Get a comprehensive summary of debugging activities:

```python
summary = debugger.get_debug_summary()
print(json.dumps(summary, indent=2))
```

**Summary includes:**
- Anomaly detection status
- Profiling activity status
- Memory tracking statistics
- Gradient violation counts
- Debug event counts

### Debug Report Saving

Save detailed debug reports to files:

```python
report_path = debugger.save_debug_report("debug_report.json")
```

**Report includes:**
- Debug summary
- Debug events timeline
- Memory snapshots
- Gradient violations
- Performance metrics

### Integration with Training Logger

Debug events are automatically logged through the training logger:

```python
# Debug events are logged with appropriate log levels
logger.log_info("Debug event: Anomaly detection enabled")
logger.log_warning("Debug event: Gradient explosion detected")
logger.log_error("Debug event: NaN parameter detected")
```

## Performance Considerations

### Debugging Overhead

- **Anomaly Detection**: Minimal overhead, recommended for development
- **Profiling**: Moderate overhead, use selectively
- **Memory Tracking**: Low overhead, safe for production
- **Gradient Checking**: Low overhead, recommended for training

### Best Practices

1. **Development vs Production**:
   - Use anomaly detection and gradient checking in development
   - Disable profiling in production unless needed
   - Keep memory tracking enabled for monitoring

2. **Selective Debugging**:
   - Enable specific debugging features as needed
   - Use profiling only when investigating performance issues
   - Combine debugging tools strategically

3. **Resource Management**:
   - Clear debug data periodically to prevent memory accumulation
   - Save debug reports before clearing data
   - Monitor debug log file sizes

## Error Handling and Recovery

### Automatic Error Detection

The debugging system automatically detects and reports:

- **Gradient Computation Errors**: NaN/Inf gradients, gradient explosion
- **Parameter Issues**: NaN/Inf parameters, parameter value anomalies
- **Memory Issues**: Memory leaks, excessive memory usage
- **Performance Issues**: Slow operations, bottlenecks

### Error Recovery

```python
# Automatic NaN/Inf handling in forward pass
if torch.isnan(x).any() or torch.isinf(x).any():
    x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

# Gradient clipping to prevent explosion
gradient_norm = debugger.check_gradients(model, gradient_threshold=1.0)
```

### Custom Error Handling

```python
try:
    with debugger.anomaly_detection():
        # Training code
        pass
except Exception as e:
    # Handle specific debugging errors
    debugger._log_debug_event("custom_error", str(e))
    # Implement recovery strategy
```

## Visualization and Analysis

### Debug Event Timeline

Debug events are logged with timestamps for timeline analysis:

```python
# Events are automatically timestamped
debug_event = {
    "timestamp": "2024-01-01T12:00:00",
    "event_type": "gradient_explosion",
    "message": "Gradient explosion detected",
    "data": {"gradient_norm": 15.5, "threshold": 1.0}
}
```

### Performance Analysis

Profiling results can be visualized using Chrome DevTools:

```python
# Profiling results are saved as Chrome trace files
with debugger.profiling(output_file="profile.json"):
    train_model()

# Open profile.json in Chrome DevTools for visualization
```

### Memory Analysis

Memory snapshots provide detailed memory usage analysis:

```python
# Memory snapshots include CPU and CUDA memory
snapshot = {
    "cpu_memory_percent": 45.2,
    "cuda_allocated": 1024 * 1024 * 100,  # 100MB
    "cuda_reserved": 1024 * 1024 * 150,   # 150MB
    "timestamp": "2024-01-01T12:00:00"
}
```

## Example Usage Scenarios

### 1. Development and Debugging

```python
# Full debugging for development
debugger = create_pytorch_debugger(
    debug_mode=True,
    enable_anomaly_detection=True,
    enable_profiling=True,
    enable_memory_tracking=True,
    enable_gradient_checking=True
)

with debugger.anomaly_detection():
    with debugger.profiling():
        with debugger.memory_tracking():
            train_model()
```

### 2. Production Monitoring

```python
# Lightweight monitoring for production
debugger = create_pytorch_debugger(
    debug_mode=False,
    enable_anomaly_detection=False,
    enable_profiling=False,
    enable_memory_tracking=True,
    enable_gradient_checking=True
)

with debugger.memory_tracking():
    train_model()
```

### 3. Performance Investigation

```python
# Profiling for performance investigation
debugger = create_pytorch_debugger(
    debug_mode=True,
    enable_profiling=True,
    enable_memory_tracking=True
)

with debugger.profiling(output_file="performance_profile.json"):
    train_model()
```

### 4. Training Optimizer Integration

```python
# Integrated debugging with training optimizer
optimizer = create_enhanced_training_optimizer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    debug_mode=True,
    enable_pytorch_debugging=True,
    enable_profiling=False,
    max_epochs=100
)

results = await optimizer.train()
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install torch torchvision torchaudio
   pip install psutil numpy matplotlib plotly
   ```

2. **Memory Issues**:
   - Clear debug data periodically
   - Monitor debug log file sizes
   - Use selective debugging features

3. **Performance Issues**:
   - Disable profiling in production
   - Use gradient checking selectively
   - Monitor debugging overhead

### Debug Data Management

```python
# Clear debug data to free memory
debugger.clear_debug_data()

# Save debug report before clearing
debugger.save_debug_report("final_debug_report.json")
debugger.clear_debug_data()
```

## API Reference

### PyTorchDebugger Class

```python
class PyTorchDebugger:
    def __init__(self, logger=None, debug_mode=False, ...)
    def anomaly_detection(self, enabled=None)
    def profiling(self, enabled=None, output_file=None)
    def memory_tracking(self, enabled=None)
    def check_gradients(self, model, gradient_threshold=1.0)
    def check_model_parameters(self, model)
    def debug_forward_pass(self, model, inputs, layer_hooks=True)
    def debug_backward_pass(self, loss, retain_graph=False)
    def debug_training_step(self, model, inputs, targets, loss_fn, optimizer, gradient_threshold=1.0)
    def get_debug_summary(self)
    def save_debug_report(self, filename=None)
    def clear_debug_data(self)
```

### Utility Functions

```python
def create_pytorch_debugger(logger=None, debug_mode=True, ...)
def debug_training_session(model, logger=None, debug_mode=True, ...)
```

## Conclusion

The PyTorch debugging tools integration provides comprehensive debugging capabilities for the email sequence training pipeline. It enables:

- **Early Detection**: Catch issues before they cause training failures
- **Performance Analysis**: Identify bottlenecks and optimize training
- **Memory Management**: Monitor and prevent memory leaks
- **Gradient Validation**: Ensure stable training dynamics
- **Comprehensive Logging**: Detailed debugging information for analysis

By integrating these tools into the training pipeline, developers can build more robust and efficient email sequence models with better error detection and debugging capabilities. 