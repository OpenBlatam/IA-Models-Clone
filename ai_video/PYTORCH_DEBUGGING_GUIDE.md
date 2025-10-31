# PyTorch Debugging Tools Guide

## Overview

This guide documents the comprehensive implementation of **PyTorch's built-in debugging tools** including `autograd.detect_anomaly()` and other debugging utilities for AI training operations.

## Key Features

### 🔍 **Autograd Anomaly Detection**
- **`autograd.detect_anomaly()`** integration for automatic gradient anomaly detection
- **Context managers** for safe anomaly detection
- **Automatic error reporting** with detailed tracebacks
- **Configurable detection modes** (detect_anomaly vs set_detect_anomaly)

### 📊 **Gradient Checking**
- **Gradient validation** for NaN, Inf, and extreme values
- **Gradient norm monitoring** for exploding/vanishing gradients
- **Parameter gradient analysis** with detailed reporting
- **Automatic gradient clipping** recommendations

### 💾 **Memory Tracking**
- **Real-time memory usage** monitoring (CPU and GPU)
- **Memory leak detection** with snapshots
- **Memory trend analysis** over training iterations
- **Automatic memory cleanup** recommendations

### ⚡ **Performance Profiling**
- **PyTorch profiler integration** for performance analysis
- **CPU and CUDA activity** tracking
- **Memory profiling** with detailed statistics
- **Performance bottleneck** identification

### 🔧 **Tensor Debugging**
- **Tensor property analysis** (shape, dtype, device, requires_grad)
- **Value range monitoring** (min, max, mean, std)
- **NaN/Inf detection** in tensors
- **Tensor validation** with detailed reporting

### 🛠️ **Model Debugging**
- **Model architecture analysis** with parameter counts
- **Gradient debugging** for all model parameters
- **Common training issues** detection
- **Model state validation**

## System Architecture

### Core Components

#### 1. PyTorchDebugger Class
The main debugging orchestrator that provides:
- **Context managers** for all debugging features
- **Configuration management** for debugging options
- **Statistics tracking** and reporting
- **Integration** with training loops

#### 2. DebugConfig Dataclass
Configuration for debugging features:
```python
@dataclass
class DebugConfig:
    enable_anomaly_detection: bool = True
    enable_grad_check: bool = True
    enable_memory_tracking: bool = True
    enable_profiling: bool = True
    enable_tensor_debugging: bool = True
    anomaly_detection_mode: str = "detect_anomaly"
    grad_check_numerical: bool = True
    grad_check_analytical: bool = True
    memory_tracking_interval: int = 100
    profiling_interval: int = 50
```

#### 3. DebugTrainer Class
Training wrapper with integrated debugging:
- **Comprehensive training step** with all debugging features
- **Automatic issue detection** and reporting
- **Integration** with existing training loops
- **Debug state management**

## Usage Examples

### Basic Setup

```python
from pytorch_debugging_tools import PyTorchDebugger, DebugConfig, DebugTrainer

# Initialize debugger with configuration
debug_config = DebugConfig(
    enable_anomaly_detection=True,
    enable_grad_check=True,
    enable_memory_tracking=True,
    enable_profiling=True,
    enable_tensor_debugging=True
)
debugger = PyTorchDebugger(debug_config)
```

### Anomaly Detection

```python
# Enable anomaly detection for training
with debugger.anomaly_detection():
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, targets)
    
    # Backward pass (anomalies will be detected here)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Gradient Checking

```python
# Enable gradient checking
with debugger.grad_check(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    outputs = model(data)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Gradients will be automatically checked for:
    # - NaN values
    # - Infinite values
    # - Extremely large values
    # - Very small values (potential dead neurons)
```

### Memory Tracking

```python
# Enable memory tracking
with debugger.memory_tracking():
    # Training operations
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    outputs = model(data)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Memory usage will be tracked and reported
```

### Performance Profiling

```python
# Enable profiling
with debugger.profiling():
    # Training operations
    for epoch in range(num_epochs):
        for batch in dataloader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Profiling results will be automatically generated
```

### Tensor Debugging

```python
# Debug individual tensors
data = torch.randn(32, 784)
debugger.debug_tensor(data, "input_data")

# Debug model outputs
outputs = model(data)
debugger.debug_tensor(outputs, "model_outputs")

# Debug loss
loss = criterion(outputs, targets)
debugger.debug_tensor(loss, "loss")
```

### Model Debugging

```python
# Debug entire model
debugger.debug_model(model)

# Debug gradients after backward pass
loss.backward()
debugger.debug_gradients(model)
```

### Common Issues Detection

```python
# Check for common training issues
issues = debugger.check_for_common_issues(model, loss)

if issues:
    logger.warning("Training issues detected:")
    for issue in issues:
        logger.warning(f"  - {issue}")
```

## Integration with Training Loops

### Using DebugTrainer

```python
# Create debug trainer
debug_trainer = DebugTrainer(model, debugger)

# Training step with comprehensive debugging
result = debug_trainer.training_step(data, targets, criterion, optimizer)

# Check for issues
if result.get('issues'):
    logger.warning(f"Debug issues: {result['issues']}")
```

### Manual Integration

```python
# Manual integration with existing training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        # Enable debugging for this batch
        with debugger.anomaly_detection():
            with debugger.grad_check(model):
                with debugger.memory_tracking():
                    # Training step
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Debug tensors periodically
        if batch_idx % 100 == 0:
            debugger.debug_tensor(data, f"batch_{batch_idx}_data")
            debugger.debug_tensor(outputs, f"batch_{batch_idx}_outputs")
            debugger.debug_tensor(loss, f"batch_{batch_idx}_loss")
```

## Advanced Features

### Configuration Management

```python
# Custom debug configuration
debug_config = DebugConfig(
    enable_anomaly_detection=True,
    enable_grad_check=True,
    enable_memory_tracking=True,
    enable_profiling=False,  # Disable profiling for faster training
    enable_tensor_debugging=True,
    anomaly_detection_mode="set_detect_anomaly",  # Alternative mode
    memory_tracking_interval=50,  # Track memory every 50 iterations
    profiling_interval=100  # Profile every 100 iterations
)

debugger = PyTorchDebugger(debug_config)
```

### Debug State Management

```python
# Enable all debugging features
debugger.enable_all_debugging()

# Disable all debugging features
debugger.disable_all_debugging()

# Get debug summary
summary = debugger.get_debug_summary()
print(f"Anomaly count: {summary['anomaly_count']}")
print(f"Gradient check count: {summary['grad_check_count']}")
print(f"Memory snapshots: {summary['memory_snapshots_count']}")
```

### Error Recovery

```python
# Automatic error recovery with debugging
try:
    with debugger.anomaly_detection():
        with debugger.grad_check(model):
            # Training operations
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
except Exception as e:
    # Debugger will provide detailed error information
    logger.error(f"Training error: {e}")
    
    # Get debug summary for analysis
    summary = debugger.get_debug_summary()
    logger.info(f"Debug summary: {summary}")
```

## Integration with Optimization Demo

### Updated OptimizedTrainer

```python
class OptimizedTrainer:
    def __init__(self, model, config, advanced_logger=None, debugger=None):
        self.debugger = debugger
        self.debug_trainer = None
        if debugger:
            self.debug_trainer = DebugTrainer(model, debugger)
    
    def train_epoch(self, dataloader, epoch=1, total_epochs=1):
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Use debug trainer if available
            if self.debug_trainer:
                result = self.debug_trainer.training_step(
                    data, targets, self.criterion, self.optimizer
                )
                
                # Log any issues detected
                if result.get('issues'):
                    logger.warning(f"Debug issues: {result['issues']}")
            else:
                # Standard training with basic debugging
                with self.debugger.anomaly_detection():
                    with self.debugger.grad_check(self.model):
                        # Training operations
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        self.optimizer.step()
```

## Debugging Best Practices

### 1. Selective Debugging

```python
# Enable debugging only when needed
if debug_mode:
    with debugger.anomaly_detection():
        with debugger.grad_check(model):
            # Training operations
            pass
else:
    # Standard training without debugging overhead
    pass
```

### 2. Periodic Debugging

```python
# Debug periodically to avoid performance impact
if batch_idx % 100 == 0:
    with debugger.memory_tracking():
        with debugger.profiling():
            # Training operations
            pass
```

### 3. Issue-Specific Debugging

```python
# Enable specific debugging based on issues
if loss.item() > threshold:
    # Enable comprehensive debugging for problematic batches
    with debugger.anomaly_detection():
        with debugger.grad_check(model):
            with debugger.tensor_debugging():
                # Training operations
                pass
```

### 4. Memory Management

```python
# Clear debug data periodically
if epoch % 10 == 0:
    debugger.clear_debug_data()
```

## Performance Considerations

### Debugging Overhead

- **Anomaly detection**: ~5-10% overhead
- **Gradient checking**: ~2-5% overhead
- **Memory tracking**: ~1-3% overhead
- **Profiling**: ~10-20% overhead
- **Tensor debugging**: ~1-2% overhead

### Optimization Strategies

```python
# Minimize debugging overhead
debug_config = DebugConfig(
    enable_anomaly_detection=True,  # Keep for critical issues
    enable_grad_check=False,        # Disable if not needed
    enable_memory_tracking=False,   # Disable for performance
    enable_profiling=False,         # Disable for production
    enable_tensor_debugging=False   # Disable for performance
)
```

## Testing and Validation

### Running Tests

```bash
# Run comprehensive debugging tests
python test_pytorch_debugging.py
```

### Test Coverage

The test suite covers:
- **Anomaly detection** with various scenarios
- **Gradient checking** with different models
- **Memory tracking** with large tensors
- **Performance profiling** with realistic workloads
- **Tensor debugging** with edge cases
- **Model debugging** with complex architectures
- **Common issues detection** with problematic data
- **DebugTrainer integration** with training loops
- **Configuration management** with different settings

## Production Deployment

### Debugging in Production

```python
# Production configuration with minimal debugging
debug_config = DebugConfig(
    enable_anomaly_detection=True,  # Keep for critical issues
    enable_grad_check=False,        # Disable for performance
    enable_memory_tracking=False,   # Disable for performance
    enable_profiling=False,         # Disable for performance
    enable_tensor_debugging=False   # Disable for performance
)
```

### Monitoring and Alerting

```python
# Monitor debugging statistics
summary = debugger.get_debug_summary()

if summary['anomaly_count'] > threshold:
    send_alert(f"High anomaly count: {summary['anomaly_count']}")

if summary['grad_check_count'] > threshold:
    send_alert(f"High gradient check count: {summary['grad_check_count']}")
```

## Troubleshooting

### Common Issues

1. **Anomaly detection not working**
   - Check if `autograd.detect_anomaly()` is properly imported
   - Verify PyTorch version compatibility
   - Ensure context manager is properly used

2. **Memory tracking issues**
   - Check if `psutil` is installed
   - Verify CUDA availability for GPU memory tracking
   - Ensure proper cleanup of memory snapshots

3. **Profiling performance impact**
   - Reduce profiling frequency
   - Use selective profiling for specific operations
   - Consider disabling profiling in production

4. **Gradient checking false positives**
   - Adjust gradient norm thresholds
   - Check for legitimate large gradients in your model
   - Verify gradient clipping implementation

### Debug Information

```python
# Get comprehensive debug information
summary = debugger.get_debug_summary()
print(f"Debug Summary: {json.dumps(summary, indent=2)}")

# Check debug state
print(f"Debug State: {debugger.debug_state}")

# Get memory snapshots
print(f"Memory Snapshots: {len(debugger.memory_snapshots)}")

# Get profiling data
print(f"Profiling Data: {len(debugger.profiling_data)}")
```

## Conclusion

The PyTorch debugging tools provide:

1. **🔍 Comprehensive Anomaly Detection** - Automatic detection of gradient anomalies
2. **📊 Advanced Gradient Checking** - Validation of gradient quality and stability
3. **💾 Memory Management** - Real-time memory usage monitoring and leak detection
4. **⚡ Performance Profiling** - Detailed performance analysis and bottleneck identification
5. **🔧 Tensor Validation** - Comprehensive tensor property and value checking
6. **🛠️ Model Debugging** - Complete model state and gradient analysis
7. **🔄 Easy Integration** - Seamless integration with existing training loops
8. **⚙️ Configurable** - Flexible configuration for different use cases
9. **📈 Production Ready** - Optimized for production deployment
10. **🧪 Well Tested** - Comprehensive test suite for validation

This system ensures that **PyTorch training operations are fully debuggable, monitorable, and analyzable**, providing comprehensive insights into training behavior and automatic detection of common issues. 