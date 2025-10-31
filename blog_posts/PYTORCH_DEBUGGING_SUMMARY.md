# 🔍 PyTorch Debugging Tools Implementation Summary

## Overview

This document summarizes the comprehensive PyTorch debugging tools integration implemented in the Gradio app. The system provides advanced debugging capabilities including autograd anomaly detection, profiling, memory tracking, gradient validation, and tensor/model debugging.

## 🎯 Key Features

### 1. **PyTorchDebugger Class**

#### **Core Debugging Tools**
```python
class PyTorchDebugger:
    """Comprehensive PyTorch debugging utilities."""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.anomaly_detection_enabled = False
        self.profiler_active = False
        self.memory_tracking_enabled = False
        self.gradient_tracking_enabled = False
        self.debug_info = {}
```

#### **Autograd Anomaly Detection**
```python
def enable_anomaly_detection(self, enabled: bool = True):
    """Enable/disable autograd anomaly detection."""
    try:
        if enabled and not self.anomaly_detection_enabled:
            torch.autograd.set_detect_anomaly(True)
            self.anomaly_detection_enabled = True
            logger.info("✅ PyTorch autograd anomaly detection enabled")
        elif not enabled and self.anomaly_detection_enabled:
            torch.autograd.set_detect_anomaly(False)
            self.anomaly_detection_enabled = False
            logger.info("❌ PyTorch autograd anomaly detection disabled")
    except Exception as e:
        logger.error(f"Failed to toggle anomaly detection: {e}")
```

**Features:**
- Automatic detection of backward pass errors
- Detailed error reporting with stack traces
- Performance impact monitoring
- Safe enable/disable with error handling

**Usage:**
```python
# Enable anomaly detection
pytorch_debugger.enable_anomaly_detection(True)

# Perform operations that might cause issues
output = model(input_data)
loss = criterion(output, target)
loss.backward()  # Will catch any autograd errors

# Disable when done
pytorch_debugger.enable_anomaly_detection(False)
```

### 2. **PyTorch Profiler Integration**

#### **Profiler Configuration**
```python
def start_profiler(self, record_shapes: bool = True, profile_memory: bool = True, 
                  with_stack: bool = True, use_cuda: bool = True):
    """Start PyTorch profiler for performance analysis."""
    try:
        if not self.profiler_active:
            profiler_config = {
                'record_shapes': record_shapes,
                'profile_memory': profile_memory,
                'with_stack': with_stack,
                'use_cuda': use_cuda and torch.cuda.is_available()
            }
            
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA if use_cuda and torch.cuda.is_available() else None
                ],
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=2
                )
            )
            
            self.profiler.start()
            self.profiler_active = True
            logger.info("✅ PyTorch profiler started")
            
    except Exception as e:
        logger.error(f"Failed to start profiler: {e}")
```

**Features:**
- CPU and CUDA activity profiling
- Memory usage tracking
- Operation shape recording
- Stack trace collection
- Chrome trace export

**Profiler Summary Logging:**
```python
def _log_profiler_summary(self):
    """Log profiler summary information."""
    try:
        if hasattr(self, 'profiler'):
            # Get key metrics from profiler
            key_averages = self.profiler.key_averages()
            
            # Log CPU and CUDA time
            cpu_time = sum(event.cpu_time_total for event in key_averages) / 1000
            cuda_time = sum(event.cuda_time_total for event in key_averages) / 1000 if torch.cuda.is_available() else 0
            
            logger.info(f"Profiler Summary - CPU Time: {cpu_time:.3f}s, CUDA Time: {cuda_time:.3f}s")
            
            # Log top operations by time
            top_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:5]
            for i, op in enumerate(top_ops):
                logger.info(f"  Top {i+1}: {op.name} - CPU: {op.cpu_time_total/1000:.3f}s, CUDA: {op.cuda_time_total/1000:.3f}s")
                
    except Exception as e:
        logger.error(f"Failed to log profiler summary: {e}")
```

### 3. **Memory Tracking System**

#### **Memory Statistics**
```python
def get_memory_stats(self):
    """Get current memory statistics."""
    try:
        stats = {
            'cpu_memory_percent': psutil.virtual_memory().percent,
            'cpu_memory_available_gb': psutil.virtual_memory().available / 1024**3
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                'gpu_memory_max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3
            })
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        return {}
```

**Memory Tracking Features:**
- CPU memory usage monitoring
- GPU memory allocation tracking
- Peak memory usage recording
- Memory leak detection
- Automatic cache clearing

### 4. **Gradient Validation**

#### **Gradient Checking**
```python
def check_gradients(self, model: torch.nn.Module, log_gradients: bool = False):
    """Check gradients for NaN/Inf values."""
    try:
        if not self.gradient_tracking_enabled:
            return True, "Gradient tracking not enabled"
        
        has_nan = False
        has_inf = False
        gradient_info = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_nan = torch.isnan(param.grad).any().item()
                grad_inf = torch.isinf(param.grad).any().item()
                
                if grad_nan or grad_inf:
                    has_nan = has_nan or grad_nan
                    has_inf = has_inf or grad_inf
                    
                    if log_gradients:
                        gradient_info[name] = {
                            'has_nan': grad_nan,
                            'has_inf': grad_inf,
                            'grad_norm': param.grad.norm().item(),
                            'param_norm': param.norm().item()
                        }
        
        if has_nan or has_inf:
            error_msg = f"Gradient issues detected: NaN={has_nan}, Inf={has_inf}"
            if log_gradients:
                logger.error(f"{error_msg} - Details: {gradient_info}")
            return False, error_msg
        
        return True, "Gradients are valid"
        
    except Exception as e:
        logger.error(f"Failed to check gradients: {e}")
        return False, f"Gradient check failed: {e}"
```

**Gradient Validation Features:**
- NaN/Inf detection in gradients
- Gradient norm monitoring
- Parameter norm tracking
- Detailed gradient information logging
- Automatic gradient validation during training

### 5. **Tensor Debugging**

#### **Tensor Analysis**
```python
def debug_tensor(self, tensor: torch.Tensor, name: str = "tensor", log_details: bool = True):
    """Debug tensor properties and values."""
    try:
        debug_info = {
            'name': name,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad,
            'is_leaf': tensor.is_leaf,
            'numel': tensor.numel(),
            'memory_size_mb': tensor.element_size() * tensor.numel() / 1024**2
        }
        
        # Check for NaN/Inf values
        if tensor.numel() > 0:
            debug_info.update({
                'has_nan': torch.isnan(tensor).any().item(),
                'has_inf': torch.isinf(tensor).any().item(),
                'min_value': tensor.min().item() if tensor.numel() > 0 else None,
                'max_value': tensor.max().item() if tensor.numel() > 0 else None,
                'mean_value': tensor.mean().item() if tensor.numel() > 0 else None,
                'std_value': tensor.std().item() if tensor.numel() > 0 else None
            })
        
        if log_details:
            logger.info(f"Tensor Debug - {name}: {debug_info}")
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Failed to debug tensor {name}: {e}")
        return {'name': name, 'error': str(e)}
```

**Tensor Debugging Features:**
- Shape and dtype analysis
- Device placement tracking
- Memory usage calculation
- NaN/Inf value detection
- Statistical analysis (min, max, mean, std)
- Gradient tracking status

### 6. **Model Debugging**

#### **Model Analysis**
```python
def debug_model(self, model: torch.nn.Module, log_details: bool = True):
    """Debug model parameters and structure."""
    try:
        debug_info = {
            'model_name': model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'modules': len(list(model.modules())),
            'device': next(model.parameters()).device if list(model.parameters()) else None
        }
        
        # Check parameter statistics
        param_stats = {}
        for name, param in model.named_parameters():
            param_stats[name] = {
                'shape': list(param.shape),
                'requires_grad': param.requires_grad,
                'has_nan': torch.isnan(param).any().item(),
                'has_inf': torch.isinf(param).any().item(),
                'norm': param.norm().item(),
                'grad_norm': param.grad.norm().item() if param.grad is not None else None
            }
        
        debug_info['parameter_stats'] = param_stats
        
        if log_details:
            logger.info(f"Model Debug - {debug_info['model_name']}: {debug_info}")
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Failed to debug model: {e}")
        return {'error': str(e)}
```

**Model Debugging Features:**
- Parameter count analysis
- Trainable parameter identification
- Module structure analysis
- Parameter statistics (norm, gradients)
- NaN/Inf detection in parameters
- Device placement tracking

### 7. **Context Manager**

#### **Debug Context**
```python
class PyTorchDebugContext:
    """Context manager for PyTorch debugging operations."""
    
    def __init__(self, debugger: PyTorchDebugger, operation_name: str):
        self.debugger = debugger
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        """Enter debugging context."""
        self.start_time = time.time()
        self.start_memory = self.debugger.get_memory_stats()
        
        logger.info(f"🔍 Starting debug context: {self.operation_name}")
        return self.debugger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit debugging context."""
        end_time = time.time()
        end_memory = self.debugger.get_memory_stats()
        
        duration = end_time - self.start_time
        
        # Log operation summary
        logger.info(f"🔍 Debug context completed: {self.operation_name} - Duration: {duration:.3f}s")
        
        # Log memory changes
        if self.start_memory and end_memory:
            cpu_diff = end_memory.get('cpu_memory_percent', 0) - self.start_memory.get('cpu_memory_percent', 0)
            gpu_diff = end_memory.get('gpu_memory_allocated_gb', 0) - self.start_memory.get('gpu_memory_allocated_gb', 0)
            
            logger.info(f"  Memory changes - CPU: {cpu_diff:+.1f}%, GPU: {gpu_diff:+.2f}GB")
        
        # Log any exceptions
        if exc_type is not None:
            logger.error(f"  Exception in {self.operation_name}: {exc_type.__name__}: {exc_val}")
            return False  # Re-raise the exception
        
        return True
```

**Context Manager Features:**
- Automatic timing measurement
- Memory usage tracking
- Exception handling
- Operation summary logging
- Resource cleanup

## 🔧 Enhanced Functions with PyTorch Debugging

### 1. **Enhanced `safe_inference()`**
```python
def safe_inference(pipeline: Any, prompt: str, num_images: int, generator: Any, 
                  use_mixed_precision: bool, debug_mode: bool = False) -> Tuple[Any, str]:
    """Safely perform inference with comprehensive error handling and PyTorch debugging."""
    start_time = time.time()
    
    # Use PyTorch debugging context if debug mode is enabled
    debug_context = pytorch_debugger.context_manager("inference") if debug_mode else None
    
    try:
        if debug_context:
            debug_context.__enter__()
        
        # Enable PyTorch debugging tools if in debug mode
        if debug_mode:
            pytorch_debugger.enable_anomaly_detection(True)
            pytorch_debugger.enable_memory_tracking(True)
            
            # Start profiler for detailed performance analysis
            pytorch_debugger.start_profiler(
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                use_cuda=torch.cuda.is_available()
            )
        
        # Debug pipeline if in debug mode
        if debug_mode and hasattr(pipeline, 'unet'):
            pytorch_debugger.debug_model(pipeline.unet, log_details=True)
        
        # Perform inference with debugging
        inference_start = time.time()
        try:
            if use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = pipeline(prompt, num_images_per_prompt=num_images, generator=generator)
            else:
                output = pipeline(prompt, num_images_per_prompt=num_images, generator=generator)
        except Exception as inference_error:
            # Log detailed inference error with PyTorch debugging info
            if debug_mode:
                logger.error(f"Inference failed with error: {inference_error}")
                logger.error(f"PyTorch version: {torch.__version__}")
                logger.error(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.error(f"CUDA version: {torch.version.cuda}")
                    logger.error(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
            raise inference_error
        
        # Debug output tensors if in debug mode
        if debug_mode and hasattr(output, 'images'):
            for i, img in enumerate(output.images[:2]):  # Debug first 2 images
                if hasattr(img, 'shape'):
                    pytorch_debugger.debug_tensor(torch.tensor(img), f"output_image_{i}", log_details=True)
        
        # Stop profiler and export results if in debug mode
        if debug_mode:
            pytorch_debugger.stop_profiler(export_path=f"logs/inference_profile_{int(time.time())}.json")
        
        return output, ""
        
    except Exception as e:
        # Log error with context
        log_error_with_context(e, "inference_general_error", {
            "num_images": num_images,
            "use_mixed_precision": use_mixed_precision,
            "memory_before": memory_before,
            "inference_time": time.time() - start_time
        })
        
        return None, error_msg
        
    finally:
        # Clean up debugging tools
        if debug_mode:
            pytorch_debugger.enable_anomaly_detection(False)
            pytorch_debugger.enable_memory_tracking(False)
        
        if debug_context:
            debug_context.__exit__(None, None, None)
```

### 2. **Enhanced `generate()` Function**
```python
def generate(prompt, model_name, seed, num_images, debug_mode, use_mixed_precision, use_multi_gpu, use_ddp, gradient_accumulation_steps):
    """Enhanced generate function with comprehensive error handling and debugging."""
    
    # Enable PyTorch debugging tools if debug mode is requested
    if debug_mode:
        # Enable comprehensive PyTorch debugging
        pytorch_debugger.enable_anomaly_detection(True)
        pytorch_debugger.enable_memory_tracking(True)
        pytorch_debugger.enable_gradient_tracking(True)
        
        # Start profiler for detailed performance analysis
        pytorch_debugger.start_profiler(
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            use_cuda=torch.cuda.is_available()
        )
        
        logger.info("✅ PyTorch debugging tools enabled: anomaly detection, memory tracking, gradient tracking, profiler")
        
        # Debug pipeline model if available
        if hasattr(pipeline, 'unet'):
            pytorch_debugger.debug_model(pipeline.unet, log_details=True)
    
    try:
        # Perform inference...
        
        # Stop profiler and export results if in debug mode
        if debug_mode:
            pytorch_debugger.stop_profiler(export_path=f"logs/generation_profile_{int(time.time())}.json")
            
            # Disable debugging tools
            pytorch_debugger.enable_anomaly_detection(False)
            pytorch_debugger.enable_memory_tracking(False)
            pytorch_debugger.enable_gradient_tracking(False)
            
            # Get final memory stats
            final_memory_stats = pytorch_debugger.get_memory_stats()
            
            logger.info("✅ PyTorch debugging tools disabled and profiler results exported")
            
    except Exception as e:
        # Clean up debugging tools on error
        if debug_mode:
            pytorch_debugger.stop_profiler()
            pytorch_debugger.enable_anomaly_detection(False)
            pytorch_debugger.enable_memory_tracking(False)
            pytorch_debugger.enable_gradient_tracking(False)
```

## 📊 Debugging Output Examples

### 1. **Anomaly Detection Output**
```
[2024-01-15 10:30:00] INFO - ✅ PyTorch autograd anomaly detection enabled
[2024-01-15 10:30:15] ERROR - Function 'AddBackward0' returned nan for argument 1
[2024-01-15 10:30:15] ERROR - ❌ PyTorch autograd anomaly detection disabled
```

### 2. **Profiler Output**
```
[2024-01-15 10:30:20] INFO - ✅ PyTorch profiler started
[2024-01-15 10:30:25] INFO - Profiler Summary - CPU Time: 2.345s, CUDA Time: 1.234s
[2024-01-15 10:30:25] INFO -   Top 1: aten::conv2d - CPU: 0.456s, CUDA: 0.234s
[2024-01-15 10:30:25] INFO -   Top 2: aten::add - CPU: 0.234s, CUDA: 0.123s
[2024-01-15 10:30:25] INFO - ❌ PyTorch profiler stopped
```

### 3. **Memory Tracking Output**
```
[2024-01-15 10:30:30] INFO - ✅ Memory tracking enabled
[2024-01-15 10:30:35] INFO - Memory stats: {'cpu_memory_percent': 75.2, 'gpu_memory_allocated_gb': 2.5}
[2024-01-15 10:30:40] INFO - ❌ Memory tracking disabled
```

### 4. **Gradient Validation Output**
```
[2024-01-15 10:30:45] INFO - ✅ Gradient tracking enabled
[2024-01-15 10:30:50] INFO - Step 1: Gradients are valid
[2024-01-15 10:30:55] ERROR - Gradient issues detected: NaN=True, Inf=False - Details: {'fc1.weight': {'has_nan': True, 'grad_norm': 0.0}}
[2024-01-15 10:31:00] INFO - ❌ Gradient tracking disabled
```

### 5. **Tensor Debugging Output**
```
[2024-01-15 10:31:05] INFO - Tensor Debug - input_tensor: {'name': 'input_tensor', 'shape': [32, 784], 'dtype': 'torch.float32', 'device': 'cuda:0', 'has_nan': False, 'has_inf': False, 'min_value': -2.345, 'max_value': 2.456, 'mean_value': 0.123, 'std_value': 1.234}
```

### 6. **Model Debugging Output**
```
[2024-01-15 10:31:10] INFO - Model Debug - SimpleNeuralNetwork: {'model_name': 'SimpleNeuralNetwork', 'total_parameters': 101770, 'trainable_parameters': 101770, 'modules': 8, 'device': 'cuda:0', 'parameter_stats': {...}}
```

## 🎯 Benefits of PyTorch Debugging Integration

### 1. **Error Detection and Diagnosis**
- **Autograd anomaly detection**: Automatic detection of backward pass errors
- **Gradient validation**: NaN/Inf detection in gradients
- **Tensor analysis**: Comprehensive tensor property checking
- **Model debugging**: Parameter and structure analysis

### 2. **Performance Analysis**
- **Profiler integration**: Detailed CPU/CUDA performance analysis
- **Memory tracking**: Real-time memory usage monitoring
- **Operation timing**: Precise timing of operations
- **Bottleneck identification**: Performance bottleneck detection

### 3. **Debugging Support**
- **Context management**: Automatic resource management
- **Error logging**: Comprehensive error context preservation
- **Export capabilities**: Profiler results export
- **Integration**: Seamless integration with existing logging

### 4. **Production Readiness**
- **Safe enable/disable**: Safe toggling of debugging features
- **Performance impact**: Minimal overhead when disabled
- **Error handling**: Robust error handling and cleanup
- **Resource management**: Automatic resource cleanup

## 📈 Usage Patterns

### 1. **Development Mode**
```python
# Enable all debugging tools for development
pytorch_debugger.enable_anomaly_detection(True)
pytorch_debugger.enable_memory_tracking(True)
pytorch_debugger.enable_gradient_tracking(True)
pytorch_debugger.start_profiler()

# Perform operations
model = SimpleNeuralNetwork()
output = model(input_data)
loss = criterion(output, target)
loss.backward()

# Clean up
pytorch_debugger.stop_profiler(export_path="logs/dev_profile.json")
pytorch_debugger.enable_anomaly_detection(False)
pytorch_debugger.enable_memory_tracking(False)
pytorch_debugger.enable_gradient_tracking(False)
```

### 2. **Context Manager Usage**
```python
# Use context manager for automatic resource management
with pytorch_debugger.context_manager("training_step"):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### 3. **Error Investigation**
```python
# Enable debugging for error investigation
pytorch_debugger.enable_anomaly_detection(True)
pytorch_debugger.enable_memory_tracking(True)

try:
    # Suspect operation
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
except Exception as e:
    # Get debugging information
    memory_stats = pytorch_debugger.get_memory_stats()
    log_error_with_context(e, "training_error", {
        'memory_stats': memory_stats,
        'model_info': pytorch_debugger.debug_model(model)
    })
finally:
    # Clean up
    pytorch_debugger.enable_anomaly_detection(False)
    pytorch_debugger.enable_memory_tracking(False)
```

## 🚀 Best Practices

### 1. **Debugging Best Practices**
- **Enable selectively**: Only enable debugging tools when needed
- **Use context managers**: Use context managers for automatic cleanup
- **Monitor performance**: Be aware of performance impact
- **Export results**: Export profiler results for analysis

### 2. **Error Handling Best Practices**
- **Comprehensive cleanup**: Always clean up debugging tools
- **Error context**: Include debugging information in error logs
- **Safe toggling**: Use safe enable/disable methods
- **Resource management**: Monitor resource usage

### 3. **Performance Best Practices**
- **Minimal overhead**: Disable debugging in production
- **Selective profiling**: Profile only specific operations
- **Memory monitoring**: Monitor memory usage during debugging
- **Export analysis**: Export results for offline analysis

## 📝 Conclusion

The PyTorch debugging tools integration provides:

1. **Comprehensive Debugging**: Full suite of PyTorch debugging capabilities
2. **Performance Analysis**: Detailed profiling and performance monitoring
3. **Error Detection**: Automatic detection of common PyTorch issues
4. **Resource Management**: Automatic resource cleanup and management
5. **Production Ready**: Safe for production use with minimal overhead
6. **Integration**: Seamless integration with existing logging system

This implementation ensures that PyTorch operations are fully debuggable, performance issues are easily identifiable, and errors are quickly diagnosable while maintaining production readiness and minimal performance impact. 