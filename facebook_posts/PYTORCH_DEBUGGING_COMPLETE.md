# PyTorch Debugging Tools - Complete Documentation

## Overview

The PyTorch Debugging Tools system provides comprehensive debugging capabilities using PyTorch's built-in debugging tools like `autograd.detect_anomaly()`, gradient checking, memory profiling, and performance monitoring. This system ensures robust error detection and analysis for deep learning applications.

## Architecture

### Core Components

1. **PyTorchDebugger**: Central debugging system with built-in tools
2. **PyTorchDebugManager**: High-level manager for debugging operations
3. **DebugConfig**: Comprehensive configuration for debugging modes
4. **DebugInfo**: Detailed debug information structure
5. **DebugMode**: Different debugging modes and levels

### Key Features

- **Anomaly Detection**: Using `torch.autograd.detect_anomaly()`
- **Gradient Checking**: Comprehensive gradient analysis
- **Memory Profiling**: CPU and GPU memory monitoring
- **Performance Profiling**: Execution time and bottleneck detection
- **Activation Logging**: Detailed activation tracking
- **Weight Logging**: Parameter monitoring and analysis
- **Comprehensive Reporting**: Detailed debug reports and visualizations

## PyTorch Debugger

### Core Debugging Methods

```python
class PyTorchDebugger:
    def __init__(self, config: DebugConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.debug_history: List[DebugInfo] = []
        self.anomaly_detection_active = False
        self.gradient_checking_active = False
        self.memory_profiling_active = False
        self.performance_profiling_active = False
```

### Anomaly Detection Setup

```python
def _setup_anomaly_detection(self):
    """Setup PyTorch anomaly detection."""
    try:
        if self.config.anomaly_detection_mode == "warn":
            torch.autograd.set_detect_anomaly(True)
            self.logger.info("PyTorch anomaly detection enabled (warn mode)")
        elif self.config.anomaly_detection_mode == "raise":
            torch.autograd.set_detect_anomaly(True)
            self.logger.info("PyTorch anomaly detection enabled (raise mode)")
        
        self.anomaly_detection_active = True
        
    except Exception as e:
        self.logger.error(f"Failed to setup anomaly detection: {str(e)}")
```

### Debug Context Manager

```python
@contextmanager
def debug_context(self, operation: str, model_name: str = "unknown", epoch: int = 0, batch: int = 0):
    """Context manager for debugging operations."""
    debug_info = DebugInfo(
        timestamp=datetime.now(),
        mode=self.config.mode,
        level=self.config.level,
        operation=operation,
        model_name=model_name,
        epoch=epoch,
        batch=batch
    )
    
    start_time = time.time()
    start_memory = self._get_memory_usage()
    start_gpu_memory = self._get_gpu_memory_usage()
    
    try:
        # Enable debugging tools if needed
        if self.anomaly_detection_active:
            torch.autograd.set_detect_anomaly(True)
        
        yield debug_info
        
    except Exception as e:
        debug_info.anomaly_detected = True
        self.logger.error(f"Debug anomaly detected in {operation}: {str(e)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    finally:
        # Calculate metrics
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_gpu_memory = self._get_gpu_memory_usage()
        
        debug_info.execution_time = end_time - start_time
        debug_info.memory_usage = end_memory - start_memory
        debug_info.gpu_memory = end_gpu_memory - start_gpu_memory
        
        # Disable debugging tools
        if self.anomaly_detection_active:
            torch.autograd.set_detect_anomaly(False)
        
        # Save debug info
        self.debug_history.append(debug_info)
        self._save_debug_info(debug_info)
```

### Forward Pass Debugging

```python
def debug_forward_pass(self, model: nn.Module, input_data: torch.Tensor, **kwargs):
    """Debug forward pass with comprehensive monitoring."""
    with self.debug_context("forward_pass", model.__class__.__name__, **kwargs) as debug_info:
        # Log input information
        self.logger.debug(f"Forward pass input shape: {input_data.shape}")
        self.logger.debug(f"Input dtype: {input_data.dtype}")
        self.logger.debug(f"Input device: {input_data.device}")
        
        # Check for NaN/Inf in input
        if torch.isnan(input_data).any():
            self.logger.warning("NaN detected in input data")
            debug_info.anomaly_detected = True
        
        if torch.isinf(input_data).any():
            self.logger.warning("Inf detected in input data")
            debug_info.anomaly_detected = True
        
        # Perform forward pass
        output = model(input_data)
        
        # Log output information
        self.logger.debug(f"Forward pass output shape: {output.shape}")
        self.logger.debug(f"Output dtype: {output.dtype}")
        
        # Check for NaN/Inf in output
        if torch.isnan(output).any():
            self.logger.warning("NaN detected in output")
            debug_info.anomaly_detected = True
        
        if torch.isinf(output).any():
            self.logger.warning("Inf detected in output")
            debug_info.anomaly_detected = True
        
        # Log activations if enabled
        if self.config.log_activations:
            self._log_activations(model, debug_info)
        
        return output
```

### Backward Pass Debugging

```python
def debug_backward_pass(self, loss: torch.Tensor, model: nn.Module, **kwargs):
    """Debug backward pass with gradient monitoring."""
    with self.debug_context("backward_pass", model.__class__.__name__, **kwargs) as debug_info:
        # Log loss information
        self.logger.debug(f"Loss value: {loss.item()}")
        self.logger.debug(f"Loss shape: {loss.shape}")
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss).any():
            self.logger.warning("NaN detected in loss")
            debug_info.anomaly_detected = True
        
        if torch.isinf(loss).any():
            self.logger.warning("Inf detected in loss")
            debug_info.anomaly_detected = True
        
        # Perform backward pass
        loss.backward()
        
        # Check gradients
        gradient_norm = self._check_gradients(model)
        debug_info.gradient_norm = gradient_norm
        
        # Log gradient information
        self.logger.debug(f"Gradient norm: {gradient_norm}")
        
        # Check for gradient anomalies
        if gradient_norm > 10.0:
            self.logger.warning(f"High gradient norm: {gradient_norm}")
            debug_info.anomaly_detected = True
        
        if gradient_norm == 0.0:
            self.logger.warning("Zero gradient norm detected")
            debug_info.anomaly_detected = True
        
        # Log gradients if enabled
        if self.config.log_gradients:
            self._log_gradients(model, debug_info)
        
        return gradient_norm
```

### Gradient Checking

```python
def _check_gradients(self, model: nn.Module) -> float:
    """Check gradients for anomalies."""
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Check for NaN/Inf in gradients
            if torch.isnan(param.grad).any():
                self.logger.warning(f"NaN detected in gradients of {param}")
            
            if torch.isinf(param.grad).any():
                self.logger.warning(f"Inf detected in gradients of {param}")
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
    
    return total_norm
```

### Memory Profiling

```python
def debug_memory_usage(self, **kwargs):
    """Debug memory usage with detailed profiling."""
    with self.debug_context("memory_profiling", **kwargs) as debug_info:
        # Get memory information
        cpu_memory = self._get_memory_usage()
        gpu_memory = self._get_gpu_memory_usage()
        
        debug_info.memory_usage = cpu_memory
        debug_info.gpu_memory = gpu_memory
        
        # Log memory information
        self.logger.debug(f"CPU memory usage: {cpu_memory:.2f} MB")
        self.logger.debug(f"GPU memory usage: {gpu_memory:.2f} MB")
        
        # Check for memory issues
        if cpu_memory > 1000:  # 1GB threshold
            self.logger.warning(f"High CPU memory usage: {cpu_memory:.2f} MB")
            debug_info.memory_leak_detected = True
        
        if gpu_memory > 8000:  # 8GB threshold
            self.logger.warning(f"High GPU memory usage: {gpu_memory:.2f} MB")
            debug_info.memory_leak_detected = True
        
        # Save memory history
        memory_data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'epoch': debug_info.epoch,
            'batch': debug_info.batch
        }
        self.memory_history.append(memory_data)
        
        return cpu_memory, gpu_memory
```

### Performance Profiling

```python
def debug_performance(self, operation: str, **kwargs):
    """Debug performance with timing and profiling."""
    with self.debug_context("performance_profiling", operation, **kwargs) as debug_info:
        # Get performance information
        execution_time = debug_info.execution_time
        
        # Log performance information
        self.logger.debug(f"Operation: {operation}")
        self.logger.debug(f"Execution time: {execution_time:.4f} seconds")
        
        # Check for performance issues
        if execution_time > 10.0:  # 10 second threshold
            self.logger.warning(f"Slow operation detected: {execution_time:.4f} seconds")
            debug_info.performance_issue_detected = True
        
        # Save performance history
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'execution_time': execution_time,
            'epoch': debug_info.epoch,
            'batch': debug_info.batch
        }
        self.performance_history.append(performance_data)
        
        return execution_time
```

## PyTorch Debug Manager

### High-level Management

```python
class PyTorchDebugManager:
    def __init__(self, config: DebugConfig):
        self.debugger = PyTorchDebugger(config)
        self.logger = self.debugger.logger
        self.training_logger = None
    
    def setup_training_logging(self, experiment_name: str = None):
        """Setup training logging integration."""
        self.training_logger = TrainingLoggingManager(experiment_name)
```

### Complete Training Step Debugging

```python
def debug_training_step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       loss_fn: Callable, data_batch: torch.Tensor,
                       target_batch: torch.Tensor, epoch: int = 0, batch: int = 0):
    """Debug complete training step."""
    # Debug forward pass
    output = self.debugger.debug_forward_pass(
        model, data_batch,
        epoch=epoch, batch=batch
    )
    
    # Debug loss computation
    loss = loss_fn(output, target_batch)
    
    # Debug backward pass
    gradient_norm = self.debugger.debug_backward_pass(
        loss, model,
        epoch=epoch, batch=batch
    )
    
    # Debug optimization step
    param_norm = self.debugger.debug_optimization_step(
        optimizer, model,
        epoch=epoch, batch=batch
    )
    
    # Debug memory usage
    cpu_memory, gpu_memory = self.debugger.debug_memory_usage(
        epoch=epoch, batch=batch
    )
    
    # Debug performance
    execution_time = self.debugger.debug_performance(
        "training_step",
        epoch=epoch, batch=batch
    )
    
    # Log to training logger if available
    if self.training_logger:
        metrics = {
            'loss': loss.item(),
            'gradient_norm': gradient_norm,
            'param_norm': param_norm,
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'execution_time': execution_time
        }
        self.training_logger.log_batch_metrics(metrics)
    
    return {
        'loss': loss.item(),
        'gradient_norm': gradient_norm,
        'param_norm': param_norm,
        'cpu_memory': cpu_memory,
        'gpu_memory': gpu_memory,
        'execution_time': execution_time
    }
```

## Debug Configuration

### Debug Modes

```python
class DebugMode(Enum):
    NONE = "none"                           # No debugging
    ANOMALY_DETECTION = "anomaly_detection" # Only anomaly detection
    GRADIENT_CHECKING = "gradient_checking" # Only gradient checking
    MEMORY_PROFILING = "memory_profiling"   # Only memory profiling
    PERFORMANCE_PROFILING = "performance_profiling" # Only performance profiling
    FULL_DEBUG = "full_debug"               # All debugging features
```

### Debug Levels

```python
class DebugLevel(Enum):
    BASIC = "basic"           # Basic debugging
    INTERMEDIATE = "intermediate" # Intermediate debugging
    ADVANCED = "advanced"     # Advanced debugging
    EXPERT = "expert"         # Expert-level debugging
```

### Configuration Options

```python
@dataclass
class DebugConfig:
    mode: DebugMode = DebugMode.NONE
    level: DebugLevel = DebugLevel.BASIC
    enable_anomaly_detection: bool = False
    enable_gradient_checking: bool = False
    enable_memory_profiling: bool = False
    enable_performance_profiling: bool = False
    anomaly_detection_mode: str = "warn"  # "warn" or "raise"
    gradient_checking_frequency: int = 100
    memory_profiling_frequency: int = 50
    performance_profiling_frequency: int = 10
    save_debug_info: bool = True
    debug_output_dir: str = "debug_outputs"
    log_gradients: bool = False
    log_activations: bool = False
    log_weights: bool = False
    log_memory: bool = True
    log_performance: bool = True
```

## Usage Examples

### Basic Usage

```python
# Create debug configuration
config = DebugConfig(
    mode=DebugMode.FULL_DEBUG,
    level=DebugLevel.ADVANCED,
    enable_anomaly_detection=True,
    enable_gradient_checking=True,
    enable_memory_profiling=True,
    enable_performance_profiling=True
)

# Create debug manager
debug_manager = PyTorchDebugManager(config)

# Debug training step
model = nn.Sequential(nn.Linear(784, 10))
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

data_batch = torch.randn(32, 784)
target_batch = torch.randint(0, 10, (32,))

metrics = debug_manager.debug_training_step(
    model, optimizer, loss_fn, data_batch, target_batch
)

print(f"Loss: {metrics['loss']:.4f}")
print(f"Gradient norm: {metrics['gradient_norm']:.4f}")
print(f"Execution time: {metrics['execution_time']:.4f}s")
```

### Advanced Usage with Custom Configuration

```python
# Custom debug configuration
config = DebugConfig(
    mode=DebugMode.ANOMALY_DETECTION,
    level=DebugLevel.EXPERT,
    enable_anomaly_detection=True,
    anomaly_detection_mode="raise",  # Raise exceptions on anomalies
    log_gradients=True,
    log_activations=True,
    log_weights=True,
    save_debug_info=True,
    debug_output_dir="custom_debug_outputs"
)

# Create debug manager
debug_manager = PyTorchDebugManager(config)

# Debug model inference
input_data = torch.randn(1, 784)
inference_result = debug_manager.debug_model_inference(model, input_data)

print(f"Inference time: {inference_result['execution_time']:.4f}s")
print(f"Memory usage: {inference_result['cpu_memory']:.2f} MB")
```

### Integration with Training Loop

```python
# Setup debug manager with training logging
debug_manager = PyTorchDebugManager(config)
debug_manager.setup_training_logging("debug_experiment")

# Training loop with debugging
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            # Debug training step
            metrics = debug_manager.debug_training_step(
                model, optimizer, loss_fn, data, target,
                epoch=epoch, batch=batch_idx
            )
            
            # Log metrics
            print(f"Epoch {epoch}, Batch {batch_idx}:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Gradient norm: {metrics['gradient_norm']:.4f}")
            
        except Exception as e:
            print(f"Error in training step: {e}")
            # Debug information is automatically logged
            continue

# Create comprehensive report
report = debug_manager.create_comprehensive_report()
```

### Custom Debug Context

```python
# Custom debugging with context manager
with debug_manager.debugger.debug_context("custom_operation", "my_model") as debug_info:
    # Your custom operation here
    result = model(input_data)
    
    # Debug info is automatically collected
    print(f"Operation completed in {debug_info.execution_time:.4f}s")
    print(f"Memory usage: {debug_info.memory_usage:.2f} MB")
```

## Debug Reports and Visualizations

### Comprehensive Report

```python
def create_debug_report(self) -> Dict[str, Any]:
    """Create comprehensive debug report."""
    report = {
        'summary': {
            'total_debug_operations': len(self.debug_history),
            'anomalies_detected': sum(1 for info in self.debug_history if info.anomaly_detected),
            'memory_leaks_detected': sum(1 for info in self.debug_history if info.memory_leak_detected),
            'performance_issues_detected': sum(1 for info in self.debug_history if info.performance_issue_detected),
            'debug_mode': self.config.mode.value,
            'debug_level': self.config.level.value
        },
        'performance_analysis': {
            'average_execution_time': np.mean([info.execution_time for info in self.debug_history]),
            'max_execution_time': max([info.execution_time for info in self.debug_history], default=0),
            'min_execution_time': min([info.execution_time for info in self.debug_history], default=0)
        },
        'memory_analysis': {
            'average_memory_usage': np.mean([info.memory_usage for info in self.debug_history]),
            'max_memory_usage': max([info.memory_usage for info in self.debug_history], default=0),
            'average_gpu_memory': np.mean([info.gpu_memory for info in self.debug_history]),
            'max_gpu_memory': max([info.gpu_memory for info in self.debug_history], default=0)
        },
        'gradient_analysis': {
            'average_gradient_norm': np.mean([info.gradient_norm for info in self.debug_history]),
            'max_gradient_norm': max([info.gradient_norm for info in self.debug_history], default=0),
            'gradient_anomalies': sum(1 for info in self.debug_history if info.gradient_norm > 10.0)
        },
        'operation_breakdown': defaultdict(int)
    }
    
    # Count operations
    for info in self.debug_history:
        report['operation_breakdown'][info.operation] += 1
    
    report['operation_breakdown'] = dict(report['operation_breakdown'])
    
    return report
```

### Debug Visualizations

```python
def create_debug_plots(self):
    """Create debug visualization plots."""
    if not self.debug_history:
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PyTorch Debug Analysis', fontsize=16)
    
    # Extract data
    timestamps = [info.timestamp for info in self.debug_history]
    execution_times = [info.execution_time for info in self.debug_history]
    memory_usage = [info.memory_usage for info in self.debug_history]
    gpu_memory = [info.gpu_memory for info in self.debug_history]
    gradient_norms = [info.gradient_norm for info in self.debug_history]
    
    # Execution time plot
    axes[0, 0].plot(timestamps, execution_times, 'b-')
    axes[0, 0].set_title('Execution Time')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True)
    
    # Memory usage plot
    axes[0, 1].plot(timestamps, memory_usage, 'g-', label='CPU Memory')
    axes[0, 1].plot(timestamps, gpu_memory, 'r-', label='GPU Memory')
    axes[0, 1].set_title('Memory Usage')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Memory (MB)')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True)
    
    # Gradient norm plot
    axes[1, 0].plot(timestamps, gradient_norms, 'orange')
    axes[1, 0].set_title('Gradient Norm')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True)
    
    # Anomaly detection plot
    anomaly_timestamps = [info.timestamp for info in self.debug_history if info.anomaly_detected]
    anomaly_counts = [1] * len(anomaly_timestamps)
    
    if anomaly_timestamps:
        axes[1, 1].scatter(anomaly_timestamps, anomaly_counts, color='red', alpha=0.7)
    axes[1, 1].set_title('Anomaly Detection')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Anomalies')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(self.debug_dir / 'debug_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
```

## Best Practices

### Debugging Best Practices

1. **Selective Debugging**: Enable only necessary debugging features
2. **Performance Impact**: Be aware of debugging overhead
3. **Memory Monitoring**: Monitor memory usage during debugging
4. **Error Handling**: Handle debugging exceptions gracefully
5. **Log Management**: Manage debug log files efficiently

### Anomaly Detection Best Practices

1. **Use Sparingly**: Enable only when needed due to performance impact
2. **Proper Context**: Use in specific debugging contexts
3. **Error Handling**: Handle detected anomalies appropriately
4. **Logging**: Log anomaly details for analysis
5. **Cleanup**: Disable after debugging

### Memory Profiling Best Practices

1. **Regular Monitoring**: Monitor memory usage regularly
2. **Threshold Setting**: Set appropriate memory thresholds
3. **Leak Detection**: Monitor for memory leaks
4. **Cleanup**: Ensure proper memory cleanup
5. **Documentation**: Document memory usage patterns

### Performance Profiling Best Practices

1. **Baseline Measurement**: Establish performance baselines
2. **Bottleneck Identification**: Identify performance bottlenecks
3. **Optimization**: Optimize slow operations
4. **Monitoring**: Monitor performance trends
5. **Documentation**: Document performance characteristics

## Configuration Examples

### Basic Debugging

```python
config = DebugConfig(
    mode=DebugMode.ANOMALY_DETECTION,
    level=DebugLevel.BASIC,
    enable_anomaly_detection=True,
    anomaly_detection_mode="warn"
)
```

### Advanced Debugging

```python
config = DebugConfig(
    mode=DebugMode.FULL_DEBUG,
    level=DebugLevel.EXPERT,
    enable_anomaly_detection=True,
    enable_gradient_checking=True,
    enable_memory_profiling=True,
    enable_performance_profiling=True,
    log_gradients=True,
    log_activations=True,
    log_weights=True,
    save_debug_info=True
)
```

### Production Debugging

```python
config = DebugConfig(
    mode=DebugMode.MEMORY_PROFILING,
    level=DebugLevel.INTERMEDIATE,
    enable_memory_profiling=True,
    enable_performance_profiling=True,
    log_memory=True,
    log_performance=True,
    save_debug_info=False  # Disable in production
)
```

## Conclusion

The PyTorch Debugging Tools system provides comprehensive debugging capabilities using PyTorch's built-in tools:

- **Anomaly Detection**: Using `torch.autograd.detect_anomaly()`
- **Gradient Analysis**: Comprehensive gradient checking and monitoring
- **Memory Profiling**: CPU and GPU memory monitoring
- **Performance Profiling**: Execution time and bottleneck detection
- **Comprehensive Reporting**: Detailed debug reports and visualizations
- **Integration**: Seamless integration with training systems

This system ensures robust error detection and analysis for production-ready deep learning applications while maintaining performance and providing detailed insights into model behavior. 