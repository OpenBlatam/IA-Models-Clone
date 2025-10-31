# PyTorch Debugging Guide for Video-OpusClip

This guide covers the comprehensive PyTorch debugging tools integrated into the Video-OpusClip system, with a focus on `autograd.detect_anomaly()` and other built-in PyTorch debugging features.

## Table of Contents

1. [Overview](#overview)
2. [Autograd Anomaly Detection](#autograd-anomaly-detection)
3. [Gradient Debugging](#gradient-debugging)
4. [Memory Debugging](#memory-debugging)
5. [Model Debugging](#model-debugging)
6. [Training Debugging](#training-debugging)
7. [CUDA Debugging](#cuda-debugging)
8. [Profiling](#profiling)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Examples](#examples)

## Overview

The PyTorch debugging system provides comprehensive tools for debugging deep learning models, with special focus on:

- **Autograd Anomaly Detection**: Detecting gradient computation issues
- **Gradient Analysis**: Monitoring gradient flow and detecting anomalies
- **Memory Management**: Tracking tensor memory usage
- **Model Inspection**: Analyzing model parameters and structure
- **Training Monitoring**: Debugging training loops and optimization
- **CUDA Optimization**: GPU memory and operation debugging
- **Performance Profiling**: Detailed operation profiling

## Autograd Anomaly Detection

### What is autograd.detect_anomaly()?

`torch.autograd.detect_anomaly()` is PyTorch's built-in tool for detecting anomalies in gradient computation. It helps identify:

- NaN gradients
- Infinite gradients
- Gradient computation errors
- Backward pass issues

### Usage Examples

#### Basic Anomaly Detection

```python
from pytorch_debug_tools import AutogradAnomalyDetector

# Initialize detector
config = PyTorchDebugConfig(enable_autograd_anomaly=True)
anomaly_detector = AutogradAnomalyDetector(config)

# Use in training loop
with anomaly_detector.detect_anomaly():
    loss = model(inputs, targets)
    loss.backward()
```

#### Different Detection Modes

```python
# Detect and warn
with anomaly_detector.detect_anomaly(mode="warn"):
    loss.backward()

# Detect and raise exception
with anomaly_detector.detect_anomaly(mode="raise"):
    loss.backward()

# Detect only (default)
with anomaly_detector.detect_anomaly(mode="detect"):
    loss.backward()
```

#### Integration with Training

```python
class OptimizedTrainer:
    def __init__(self, debug_config: PyTorchDebugConfig):
        self.anomaly_detector = AutogradAnomalyDetector(debug_config)
    
    def train_step(self, model, inputs, targets):
        with self.anomaly_detector.detect_anomaly():
            outputs = model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
```

### Anomaly Detection Report

```python
# Get detailed anomaly report
report = anomaly_detector.get_anomaly_report()
print(f"Total anomalies: {report['total_anomalies']}")
print(f"Anomaly history: {report['anomalies']}")
```

## Gradient Debugging

### Gradient Analysis

Monitor gradient flow and detect issues:

```python
from pytorch_debug_tools import GradientDebugger

gradient_debugger = GradientDebugger(config)

# Check gradients after backward pass
gradient_info = gradient_debugger.check_gradients(model, step=0)
print(f"Gradient norm: {gradient_info['statistics']['total_norm']}")
print(f"Anomalies: {gradient_info['anomalies']}")
```

### Gradient Flow Analysis

```python
# Analyze gradient flow through model
flow_analysis = gradient_debugger.analyze_gradient_flow(model)
print(f"Vanishing gradients: {flow_analysis['vanishing_gradients']}")
print(f"Exploding gradients: {flow_analysis['exploding_gradients']}")
```

### Gradient Clipping

```python
# Clip gradients and monitor
total_norm = gradient_debugger.clip_gradients(model, max_norm=1.0)
print(f"Gradients clipped. Total norm: {total_norm}")
```

## Memory Debugging

### Tensor Memory Tracking

```python
from pytorch_debug_tools import PyTorchMemoryDebugger

memory_debugger = PyTorchMemoryDebugger(config)

# Track specific tensors
tensor = torch.randn(1000, 1000)
memory_debugger.track_tensor(tensor, "my_tensor")

# Take memory snapshots
snapshot = memory_debugger.take_memory_snapshot("before_training")
print(f"GPU memory: {snapshot['gpu_memory']}")
```

### Memory Analysis

```python
# Get comprehensive memory report
report = memory_debugger.get_memory_report()
print(f"Memory snapshots: {len(report['snapshots'])}")
print(f"Tracked tensors: {len(report['tensor_tracker'])}")
```

### Memory Cleanup

```python
# Clear memory and garbage collect
memory_debugger.clear_memory()
```

## Model Debugging

### Model Inspection

```python
from pytorch_debug_tools import ModelDebugger

model_debugger = ModelDebugger(config)

# Comprehensive model inspection
inspection = model_debugger.inspect_model(model, input_shape=(32, 10))
print(f"Total parameters: {inspection['model_info']['total_parameters']}")
print(f"Parameter anomalies: {inspection['parameter_info']['statistics']}")
```

### Input Validation

```python
# Validate model inputs
validation = model_debugger._validate_model_inputs(model, (32, 10))
if validation['forward_pass_successful']:
    print("Model inputs are valid")
else:
    print(f"Input validation failed: {validation['error']}")
```

## Training Debugging

### Training Step Debugging

```python
from pytorch_debug_tools import TrainingDebugger

training_debugger = TrainingDebugger(config)

# Debug training step
debug_info = training_debugger.debug_training_step(
    model, optimizer, loss_fn, inputs, targets, step=0
)
print(f"Loss computation: {debug_info['loss_computation']}")
```

### Training Loop Integration

```python
def debug_training_loop(model, dataloader, optimizer, loss_fn):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Debug each step
        debug_info = training_debugger.debug_training_step(
            model, optimizer, loss_fn, inputs, targets, batch_idx
        )
        
        # Check for issues
        if debug_info['loss_computation']['computation_successful']:
            # Continue training
            pass
        else:
            print(f"Training step failed: {debug_info['loss_computation']['error']}")
            break
```

## CUDA Debugging

### CUDA Operations Debugging

```python
from pytorch_debug_tools import CUDADebugger

cuda_debugger = CUDADebugger(config)

# Debug CUDA operations
cuda_info = cuda_debugger.debug_cuda_operations("model_forward")
print(f"GPU memory: {cuda_info['memory_info']}")
print(f"Device info: {cuda_info['device_info']}")
```

### CUDA Synchronization

```python
# Force CUDA synchronization for debugging
config.cuda_synchronize = True
cuda_info = cuda_debugger.debug_cuda_operations("synchronized_operation")
```

## Profiling

### Operation Profiling

```python
from pytorch_debug_tools import PyTorchProfiler

profiler = PyTorchProfiler(config)

# Profile operations
with profiler.profile_operation("model_training") as prof:
    for batch in dataloader:
        outputs = model(batch)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

# Get profiling results
results = profiler.get_profiler_report()
print(f"Profiling results: {results['profiler_results']}")
```

### Memory Profiling

```python
# Enable memory profiling
config.profile_memory = True
config.profile_cuda = True

with profiler.profile_operation("memory_intensive_operation") as prof:
    large_tensor = torch.randn(10000, 10000, device='cuda')
    result = model(large_tensor)
```

## Best Practices

### 1. Selective Debugging

Enable debugging only when needed to avoid performance impact:

```python
# Enable only necessary debugging features
config = PyTorchDebugConfig(
    enable_autograd_anomaly=True,  # Always useful
    enable_gradient_debugging=False,  # Disable if not needed
    enable_memory_debugging=True,  # Enable for memory issues
    enable_profiling=False  # Disable in production
)
```

### 2. Gradual Debugging

Start with basic debugging and add more as needed:

```python
# Step 1: Basic anomaly detection
with anomaly_detector.detect_anomaly():
    loss.backward()

# Step 2: Add gradient debugging if issues found
if issues_found:
    gradient_info = gradient_debugger.check_gradients(model)

# Step 3: Add memory debugging if needed
if memory_issues:
    memory_snapshot = memory_debugger.take_memory_snapshot()
```

### 3. Performance Monitoring

Monitor debugging overhead:

```python
import time

start_time = time.time()
with anomaly_detector.detect_anomaly():
    # Your operations here
    pass
debug_time = time.time() - start_time
print(f"Debugging overhead: {debug_time:.4f}s")
```

### 4. Error Recovery

Implement graceful error recovery:

```python
try:
    with anomaly_detector.detect_anomaly():
        loss.backward()
except Exception as e:
    logger.error(f"Gradient computation error: {e}")
    # Implement recovery strategy
    optimizer.zero_grad()
    # Try with reduced learning rate or different batch
```

## Troubleshooting

### Common Issues and Solutions

#### 1. NaN Gradients

**Problem**: Gradients become NaN during training

**Solution**:
```python
# Enable anomaly detection
with anomaly_detector.detect_anomaly():
    loss.backward()

# Check gradient norms
gradient_info = gradient_debugger.check_gradients(model)
if gradient_info['anomalies']:
    print("NaN gradients detected")
    # Implement gradient clipping or learning rate reduction
```

#### 2. Memory Leaks

**Problem**: GPU memory keeps increasing

**Solution**:
```python
# Track tensor memory
memory_debugger.track_tensor(tensor, "problematic_tensor")

# Take periodic snapshots
snapshot = memory_debugger.take_memory_snapshot("checkpoint")

# Clear memory
memory_debugger.clear_memory()
```

#### 3. Slow Training

**Problem**: Training is unexpectedly slow

**Solution**:
```python
# Profile operations
with profiler.profile_operation("slow_operation") as prof:
    # Your slow operation
    pass

# Check CUDA operations
cuda_info = cuda_debugger.debug_cuda_operations("slow_operation")
```

#### 4. Model Convergence Issues

**Problem**: Model doesn't converge or loss doesn't decrease

**Solution**:
```python
# Analyze gradient flow
flow_analysis = gradient_debugger.analyze_gradient_flow(model)
if flow_analysis['vanishing_gradients']:
    print("Vanishing gradients detected")
    # Consider different activation functions or initialization

if flow_analysis['exploding_gradients']:
    print("Exploding gradients detected")
    # Implement gradient clipping
```

### Debug Configuration Examples

#### Development Configuration

```python
config = PyTorchDebugConfig(
    enable_autograd_anomaly=True,
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_model_debugging=True,
    enable_training_debugging=True,
    enable_cuda_debugging=True,
    enable_profiling=True,
    save_debug_reports=True
)
```

#### Production Configuration

```python
config = PyTorchDebugConfig(
    enable_autograd_anomaly=False,
    enable_gradient_debugging=False,
    enable_memory_debugging=False,
    enable_model_debugging=False,
    enable_training_debugging=False,
    enable_cuda_debugging=False,
    enable_profiling=False,
    save_debug_reports=False
)
```

#### Debugging Specific Issues

```python
# For gradient issues
config = PyTorchDebugConfig(
    enable_autograd_anomaly=True,
    enable_gradient_debugging=True,
    gradient_norm_threshold=1e3,
    gradient_clip_threshold=1.0
)

# For memory issues
config = PyTorchDebugConfig(
    enable_memory_debugging=True,
    track_tensor_memory=True,
    memory_snapshot_frequency=10
)

# For performance issues
config = PyTorchDebugConfig(
    enable_profiling=True,
    profile_memory=True,
    profile_cuda=True
)
```

## Examples

### Complete Training Debugging Example

```python
from pytorch_debug_tools import PyTorchDebugManager, PyTorchDebugConfig

# Initialize debug manager
config = PyTorchDebugConfig(
    enable_autograd_anomaly=True,
    enable_gradient_debugging=True,
    enable_memory_debugging=True,
    enable_training_debugging=True
)

debug_manager = PyTorchDebugManager(config)

# Create model and data
model = YourModel()
train_loader = YourDataLoader()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Debug training loop
debug_manager.debug_training_loop(
    model, train_loader, optimizer, loss_fn, num_epochs=1
)

# Generate comprehensive report
report = debug_manager.generate_comprehensive_report()
print("Debug report generated successfully")
```

### Quick Debugging Setup

```python
# Quick setup for immediate debugging
debug_manager = enable_pytorch_debugging()

# Use in your existing code
with debug_manager.anomaly_detector.detect_anomaly():
    # Your existing training code
    loss.backward()

# Check status
status = debug_manager.get_debug_status()
print(f"Debug status: {status}")
```

### Integration with Existing Code

```python
# Add to existing training loop
def train_with_debugging(model, dataloader, optimizer, loss_fn):
    debug_manager = PyTorchDebugManager()
    
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Debug training step
            with debug_manager.anomaly_detector.detect_anomaly():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                
                # Check gradients
                gradient_info = debug_manager.gradient_debugger.check_gradients(model)
                
                optimizer.step()
            
            # Memory snapshot every 100 batches
            if batch_idx % 100 == 0:
                debug_manager.memory_debugger.take_memory_snapshot(f"epoch_{epoch}_batch_{batch_idx}")
    
    # Generate final report
    report = debug_manager.generate_comprehensive_report()
    return report
```

This comprehensive PyTorch debugging system provides powerful tools for identifying and resolving issues in your Video-OpusClip training pipeline. Use these tools strategically to maintain optimal performance while ensuring robust debugging capabilities. 