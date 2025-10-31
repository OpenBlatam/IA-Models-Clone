# PyTorch Debugging Tools Guide

## Overview

This guide covers the comprehensive PyTorch debugging tools system that provides built-in debugging utilities like `autograd.detect_anomaly()` and other advanced debugging features. The system includes memory debugging, CUDA debugging, model validation, and performance profiling tools.

## 🔧 Available PyTorch Debugging Systems

### 1. PyTorch Debugging Tools (`pytorch_debugging_tools.py`)
**Port**: 7871
**Description**: Comprehensive debugging tools with autograd.detect_anomaly() and more

**Features**:
- **Autograd Anomaly Detection**: `autograd.detect_anomaly()` for gradient anomaly detection
- **Memory Debugging**: Memory leak detection and GPU memory monitoring
- **CUDA Debugging**: CUDA error detection and device synchronization
- **Model Debugging**: Forward/backward pass validation and debugging
- **Performance Profiling**: Model performance profiling with torch.profiler
- **Tensor Validation**: NaN/Inf detection and tensor property validation
- **Gradient Validation**: Gradient explosion/vanishing detection

## 🚀 Quick Start

### Installation

1. **Install Dependencies**:
```bash
pip install -r requirements_gradio_demos.txt
pip install torch torchvision torchaudio
```

2. **Launch PyTorch Debugging Tools**:
```bash
# Launch PyTorch debugging tools
python demo_launcher.py --demo pytorch-debugging

# Launch all debugging systems
python demo_launcher.py --all
```

### Direct Launch

```bash
# PyTorch debugging tools
python pytorch_debugging_tools.py
```

## 🔧 PyTorch Debugging Features

### Autograd Anomaly Detection

**Core Autograd Debugging**:
```python
def _setup_autograd_anomaly_detection(self):
    """Setup autograd anomaly detection"""
    try:
        if torch.autograd.anomaly_mode._enabled:
            logger.info("Autograd anomaly detection already enabled")
        else:
            autograd.detect_anomaly()
            self.autograd_anomaly_enabled = True
            logger.info("Autograd anomaly detection enabled")
    except Exception as e:
        logger.error(f"Failed to setup autograd anomaly detection: {e}")
```

**Enable/Disable Debug Mode**:
```python
def enable_debug_mode(self, training_logger: TrainingLogger = None):
    """Enable comprehensive debug mode"""
    self.training_logger = training_logger
    
    # Enable autograd anomaly detection
    if self.config.enable_autograd_anomaly and not self.autograd_anomaly_enabled:
        self._setup_autograd_anomaly_detection()
    
    # Enable memory debugging
    if self.config.enable_memory_debugging and not self.memory_debugging_enabled:
        self._setup_memory_debugging()
    
    # Enable CUDA debugging
    if self.config.enable_cuda_debugging and not self.cuda_debugging_enabled:
        self._setup_cuda_debugging()
    
    logger.info("Comprehensive debug mode enabled")

def disable_debug_mode(self):
    """Disable debug mode"""
    if self.autograd_anomaly_enabled:
        autograd.detect_anomaly(False)
        self.autograd_anomaly_enabled = False
    
    if self.memory_debugging_enabled:
        self.memory_debugging_enabled = False
    
    if self.cuda_debugging_enabled:
        os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
        self.cuda_debugging_enabled = False
    
    if self.profiling_enabled:
        self.profiling_enabled = False
    
    logger.info("Debug mode disabled")
```

### Tensor Validation

**Comprehensive Tensor Checking**:
```python
def check_tensor_validity(self, tensor: torch.Tensor, tensor_name: str = "tensor") -> Dict[str, Any]:
    """Check tensor validity and properties"""
    validity_info = {
        "name": tensor_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
        "is_leaf": tensor.is_leaf,
        "is_valid": True,
        "issues": []
    }
    
    # Check for NaN values
    if self.config.enable_nan_detection:
        if torch.isnan(tensor).any():
            validity_info["is_valid"] = False
            validity_info["issues"].append("Contains NaN values")
            self._log_tensor_issue(tensor_name, "NaN detected", tensor)
    
    # Check for Inf values
    if self.config.enable_inf_detection:
        if torch.isinf(tensor).any():
            validity_info["is_valid"] = False
            validity_info["issues"].append("Contains Inf values")
            self._log_tensor_issue(tensor_name, "Inf detected", tensor)
    
    # Check for extreme values
    if tensor.numel() > 0:
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        
        validity_info.update({
            "max_value": max_val,
            "min_value": min_val,
            "mean_value": mean_val,
            "std_value": std_val
        })
        
        # Check for extreme values
        if abs(max_val) > 1e6 or abs(min_val) > 1e6:
            validity_info["issues"].append("Contains extreme values")
            self._log_tensor_issue(tensor_name, "Extreme values detected", tensor)
    
    return validity_info
```

**Tensor Issue Logging**:
```python
def _log_tensor_issue(self, tensor_name: str, issue: str, tensor: torch.Tensor):
    """Log tensor issues"""
    tensor_info = {
        "name": tensor_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
        "numel": tensor.numel()
    }
    
    if tensor.numel() > 0:
        tensor_info.update({
            "min_value": tensor.min().item(),
            "max_value": tensor.max().item(),
            "mean_value": tensor.mean().item(),
            "std_value": tensor.std().item()
        })
    
    debug_event = DebugEvent(
        timestamp=datetime.now(),
        event_type="TENSOR_ISSUE",
        severity="WARNING",
        message=f"Tensor issue detected: {issue}",
        context={"tensor_name": tensor_name, "issue": issue},
        tensor_info=tensor_info
    )
    
    self._log_debug_event(debug_event)
```

### Gradient Validation

**Gradient Validity Checking**:
```python
def check_gradient_validity(self, model: nn.Module) -> Dict[str, Any]:
    """Check gradient validity for model parameters"""
    gradient_info = {
        "total_parameters": 0,
        "parameters_with_gradients": 0,
        "gradient_norms": {},
        "gradient_issues": [],
        "is_valid": True
    }
    
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient_info["total_parameters"] += param.numel()
            param_count += 1
            
            if param.grad is not None:
                gradient_info["parameters_with_gradients"] += param.numel()
                
                # Check gradient validity
                grad_validity = self.check_tensor_validity(param.grad, f"grad_{name}")
                
                if not grad_validity["is_valid"]:
                    gradient_info["is_valid"] = False
                    gradient_info["gradient_issues"].append({
                        "parameter": name,
                        "issues": grad_validity["issues"]
                    })
                
                # Calculate gradient norm
                param_norm = param.grad.norm().item()
                gradient_info["gradient_norms"][name] = param_norm
                total_norm += param_norm ** 2
    
    gradient_info["total_gradient_norm"] = total_norm ** 0.5
    
    # Check for gradient explosion
    if gradient_info["total_gradient_norm"] > 10.0:
        gradient_info["gradient_issues"].append("Gradient explosion detected")
        gradient_info["is_valid"] = False
    
    # Check for gradient vanishing
    if gradient_info["total_gradient_norm"] < 1e-6:
        gradient_info["gradient_issues"].append("Gradient vanishing detected")
        gradient_info["is_valid"] = False
    
    return gradient_info
```

### Model Input Validation

**Input Validation**:
```python
def validate_model_inputs(self, model: nn.Module, *args, **kwargs) -> Dict[str, Any]:
    """Validate model inputs"""
    validation_info = {
        "inputs_valid": True,
        "input_shapes": [],
        "input_types": [],
        "input_devices": [],
        "validation_issues": []
    }
    
    # Validate positional arguments
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            tensor_info = self.check_tensor_validity(arg, f"input_{i}")
            validation_info["input_shapes"].append(tensor_info["shape"])
            validation_info["input_types"].append(tensor_info["dtype"])
            validation_info["input_devices"].append(tensor_info["device"])
            
            if not tensor_info["is_valid"]:
                validation_info["inputs_valid"] = False
                validation_info["validation_issues"].append({
                    "input_index": i,
                    "issues": tensor_info["issues"]
                })
    
    # Validate keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            tensor_info = self.check_tensor_validity(value, f"input_{key}")
            validation_info["input_shapes"].append(tensor_info["shape"])
            validation_info["input_types"].append(tensor_info["dtype"])
            validation_info["input_devices"].append(tensor_info["device"])
            
            if not tensor_info["is_valid"]:
                validation_info["inputs_valid"] = False
                validation_info["validation_issues"].append({
                    "input_key": key,
                    "issues": tensor_info["issues"]
                })
    
    return validation_info
```

### Forward Pass Debugging

**Comprehensive Forward Pass Debugging**:
```python
def debug_forward_pass(self, model: nn.Module, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """Debug model forward pass with comprehensive validation"""
    debug_info = {
        "input_validation": {},
        "forward_pass_successful": False,
        "output_validation": {},
        "memory_usage": {},
        "performance_metrics": {},
        "errors": []
    }
    
    try:
        # Validate inputs
        debug_info["input_validation"] = self.validate_model_inputs(model, *args, **kwargs)
        
        if not debug_info["input_validation"]["inputs_valid"]:
            raise ValueError("Model inputs validation failed")
        
        # Record memory before forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            debug_info["memory_usage"]["before"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved()
            }
        
        # Perform forward pass with timing
        start_time = time.time()
        with torch.no_grad():
            output = model(*args, **kwargs)
        forward_time = time.time() - start_time
        
        debug_info["performance_metrics"]["forward_time"] = forward_time
        debug_info["forward_pass_successful"] = True
        
        # Record memory after forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            debug_info["memory_usage"]["after"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved()
            }
        
        # Validate output
        if isinstance(output, torch.Tensor):
            debug_info["output_validation"] = self.check_tensor_validity(output, "model_output")
        elif isinstance(output, (list, tuple)):
            debug_info["output_validation"] = {
                "type": "sequence",
                "length": len(output),
                "elements": [self.check_tensor_validity(item, f"output_{i}") 
                           for i, item in enumerate(output) if isinstance(item, torch.Tensor)]
            }
        elif isinstance(output, dict):
            debug_info["output_validation"] = {
                "type": "dict",
                "keys": list(output.keys()),
                "elements": {key: self.check_tensor_validity(value, f"output_{key}")
                           for key, value in output.items() if isinstance(value, torch.Tensor)}
            }
        
    except Exception as e:
        debug_info["errors"].append({
            "type": type(e).__name__,
            "message": str(e),
            "stack_trace": traceback.format_exc()
        })
        
        # Log error
        error_event = DebugEvent(
            timestamp=datetime.now(),
            event_type="FORWARD_PASS_ERROR",
            severity="ERROR",
            message=f"Forward pass failed: {e}",
            context={"model_type": type(model).__name__},
            stack_trace=traceback.format_exc()
        )
        self._log_debug_event(error_event)
    
    return output, debug_info
```

### Backward Pass Debugging

**Backward Pass Validation**:
```python
def debug_backward_pass(self, loss: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
    """Debug model backward pass with gradient validation"""
    debug_info = {
        "loss_validation": {},
        "backward_pass_successful": False,
        "gradient_validation": {},
        "memory_usage": {},
        "performance_metrics": {},
        "errors": []
    }
    
    try:
        # Validate loss
        debug_info["loss_validation"] = self.check_tensor_validity(loss, "loss")
        
        if not debug_info["loss_validation"]["is_valid"]:
            raise ValueError("Loss validation failed")
        
        # Record memory before backward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            debug_info["memory_usage"]["before"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved()
            }
        
        # Perform backward pass with timing
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        debug_info["performance_metrics"]["backward_time"] = backward_time
        debug_info["backward_pass_successful"] = True
        
        # Record memory after backward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            debug_info["memory_usage"]["after"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved()
            }
        
        # Validate gradients
        debug_info["gradient_validation"] = self.check_gradient_validity(model)
        
    except Exception as e:
        debug_info["errors"].append({
            "type": type(e).__name__,
            "message": str(e),
            "stack_trace": traceback.format_exc()
        })
        
        # Log error
        error_event = DebugEvent(
            timestamp=datetime.now(),
            event_type="BACKWARD_PASS_ERROR",
            severity="ERROR",
            message=f"Backward pass failed: {e}",
            context={"model_type": type(model).__name__},
            stack_trace=traceback.format_exc()
        )
        self._log_debug_event(error_event)
    
    return debug_info
```

## 📊 Performance Profiling

### Model Profiling

**Comprehensive Model Profiling**:
```python
def profile_model(self, model: nn.Module, input_data: torch.Tensor, 
                 num_iterations: int = 10) -> Dict[str, Any]:
    """Profile model performance"""
    if not self.profiling_enabled:
        return {"error": "Profiling not enabled"}
    
    profile_info = {
        "model_info": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        },
        "performance_metrics": {},
        "memory_metrics": {},
        "profiler_output": None
    }
    
    try:
        # Warm up
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile forward pass
        forward_times = []
        memory_usage = []
        
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            for _ in range(num_iterations):
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                with torch.no_grad():
                    output = model(input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                forward_times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
        
        # Calculate statistics
        profile_info["performance_metrics"] = {
            "mean_forward_time": np.mean(forward_times),
            "std_forward_time": np.std(forward_times),
            "min_forward_time": np.min(forward_times),
            "max_forward_time": np.max(forward_times),
            "throughput": num_iterations / sum(forward_times)
        }
        
        if torch.cuda.is_available():
            profile_info["memory_metrics"] = {
                "mean_memory_usage": np.mean(memory_usage),
                "max_memory_usage": np.max(memory_usage),
                "current_memory_allocated": torch.cuda.memory_allocated(),
                "current_memory_reserved": torch.cuda.memory_reserved()
            }
        
        # Save profiler output
        profiler_output_file = self.debug_output_dir / "profiler_output.txt"
        with open(profiler_output_file, 'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        profile_info["profiler_output"] = str(profiler_output_file)
        
    except Exception as e:
        profile_info["error"] = str(e)
        
        error_event = DebugEvent(
            timestamp=datetime.now(),
            event_type="PROFILING_ERROR",
            severity="ERROR",
            message=f"Model profiling failed: {e}",
            context={"model_type": type(model).__name__},
            stack_trace=traceback.format_exc()
        )
        self._log_debug_event(error_event)
    
    return profile_info
```

## 💾 Memory Debugging

### Memory Leak Detection

**Memory Leak Checking**:
```python
def check_memory_leaks(self, model: nn.Module, num_iterations: int = 100) -> Dict[str, Any]:
    """Check for memory leaks in model"""
    memory_info = {
        "initial_memory": {},
        "final_memory": {},
        "memory_growth": {},
        "potential_leak": False,
        "memory_snapshots": []
    }
    
    try:
        # Record initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_info["initial_memory"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        
        # Create dummy input
        input_data = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        
        # Run multiple iterations
        for i in range(num_iterations):
            with torch.no_grad():
                output = model(input_data)
            
            if i % 10 == 0:  # Record memory every 10 iterations
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    snapshot = {
                        "iteration": i,
                        "allocated": torch.cuda.memory_allocated(),
                        "reserved": torch.cuda.memory_reserved()
                    }
                    memory_info["memory_snapshots"].append(snapshot)
            
            # Clear some references
            del output
            if i % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Record final memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_info["final_memory"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
            
            # Check for memory growth
            allocated_growth = (memory_info["final_memory"]["allocated"] - 
                              memory_info["initial_memory"]["allocated"])
            reserved_growth = (memory_info["final_memory"]["reserved"] - 
                             memory_info["initial_memory"]["reserved"])
            
            memory_info["memory_growth"] = {
                "allocated_growth_mb": allocated_growth / (1024 * 1024),
                "reserved_growth_mb": reserved_growth / (1024 * 1024)
            }
            
            # Check for potential leak (more than 10MB growth)
            if allocated_growth > 10 * 1024 * 1024:  # 10MB
                memory_info["potential_leak"] = True
                
                leak_event = DebugEvent(
                    timestamp=datetime.now(),
                    event_type="MEMORY_LEAK_DETECTED",
                    severity="WARNING",
                    message="Potential memory leak detected",
                    context=memory_info["memory_growth"]
                )
                self._log_debug_event(leak_event)
        
    except Exception as e:
        memory_info["error"] = str(e)
        
        error_event = DebugEvent(
            timestamp=datetime.now(),
            event_type="MEMORY_CHECK_ERROR",
            severity="ERROR",
            message=f"Memory leak check failed: {e}",
            stack_trace=traceback.format_exc()
        )
        self._log_debug_event(error_event)
    
    return memory_info
```

## 🔧 Debug Configuration

### Configuration Options

**Debug Configuration Structure**:
```python
@dataclass
class DebugConfiguration:
    """Configuration for PyTorch debugging tools"""
    enable_autograd_anomaly: bool = True
    enable_memory_debugging: bool = True
    enable_cuda_debugging: bool = True
    enable_model_debugging: bool = True
    enable_performance_profiling: bool = True
    enable_gradient_checking: bool = True
    enable_nan_detection: bool = True
    enable_inf_detection: bool = True
    enable_shape_validation: bool = True
    enable_dtype_validation: bool = True
    enable_device_validation: bool = True
    debug_log_level: str = "INFO"
    save_debug_info: bool = True
    debug_output_dir: str = "debug_output"
```

### Debug Context Manager

**Context Manager for Debug Operations**:
```python
@contextlib.contextmanager
def debug_context(self, context_name: str = "debug_operation"):
    """Context manager for debug operations"""
    debug_event = DebugEvent(
        timestamp=datetime.now(),
        event_type="DEBUG_CONTEXT_START",
        severity="INFO",
        message=f"Debug context started: {context_name}",
        context={"context_name": context_name}
    )
    
    self._log_debug_event(debug_event)
    
    try:
        yield self
    except Exception as e:
        error_event = DebugEvent(
            timestamp=datetime.now(),
            event_type="DEBUG_CONTEXT_ERROR",
            severity="ERROR",
            message=f"Error in debug context {context_name}: {e}",
            context={"context_name": context_name, "error": str(e)},
            stack_trace=traceback.format_exc()
        )
        self._log_debug_event(error_event)
        raise
    finally:
        end_event = DebugEvent(
            timestamp=datetime.now(),
            event_type="DEBUG_CONTEXT_END",
            severity="INFO",
            message=f"Debug context ended: {context_name}",
            context={"context_name": context_name}
        )
        self._log_debug_event(end_event)
```

## 📊 Debug Event Logging

### Event Structure

**Debug Event Information**:
```python
@dataclass
class DebugEvent:
    """Debug event information"""
    timestamp: datetime
    event_type: str
    severity: str
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    tensor_info: Optional[Dict[str, Any]] = None
    gradient_info: Optional[Dict[str, Any]] = None
    memory_info: Optional[Dict[str, Any]] = None
    performance_info: Optional[Dict[str, Any]] = None
```

### Event Logging

**Debug Event Logging**:
```python
def _log_debug_event(self, event: DebugEvent):
    """Log debug event"""
    self.debug_events.append(event)
    
    # Log to training logger if available
    if self.training_logger:
        if event.severity == "ERROR":
            self.training_logger.log_error(
                Exception(event.message),
                event.context,
                event.severity
            )
        elif event.severity == "WARNING":
            self.training_logger.log_warning(event.message, event.context)
        else:
            self.training_logger.log_info(event.message, event.context)
    
    # Save debug event to file
    if self.config.save_debug_info:
        debug_file = self.debug_output_dir / "debug_events.json"
        event_dict = asdict(event)
        event_dict["timestamp"] = event_dict["timestamp"].isoformat()
        
        with open(debug_file, 'a') as f:
            f.write(json.dumps(event_dict) + '\n')
```

## 🎯 Usage Examples

### Basic Debug Mode

```python
from pytorch_debugging_tools import PyTorchDebugger, DebugConfiguration

# Create debug configuration
config = DebugConfiguration(
    enable_autograd_anomaly=True,
    enable_memory_debugging=True,
    enable_cuda_debugging=True,
    enable_performance_profiling=True
)

# Create debugger
debugger = PyTorchDebugger(config)

# Enable debug mode
debugger.enable_debug_mode()

# Use debug context
with debugger.debug_context("training_step"):
    # Your training code here
    pass

# Disable debug mode
debugger.disable_debug_mode()
```

### Tensor Validation

```python
# Check tensor validity
tensor = torch.randn(3, 3)
validity_info = debugger.check_tensor_validity(tensor, "my_tensor")

if not validity_info["is_valid"]:
    print(f"Tensor issues: {validity_info['issues']}")
```

### Model Debugging

```python
# Debug forward pass
model = nn.Linear(10, 1)
input_data = torch.randn(1, 10)

output, debug_info = debugger.debug_forward_pass(model, input_data)

if debug_info["forward_pass_successful"]:
    print(f"Forward pass successful: {debug_info['performance_metrics']}")
else:
    print(f"Forward pass failed: {debug_info['errors']}")
```

### Gradient Validation

```python
# Debug backward pass
loss = torch.nn.functional.mse_loss(output, torch.randn(1, 1))
debug_info = debugger.debug_backward_pass(loss, model)

if debug_info["backward_pass_successful"]:
    gradient_info = debug_info["gradient_validation"]
    if not gradient_info["is_valid"]:
        print(f"Gradient issues: {gradient_info['gradient_issues']}")
```

### Model Profiling

```python
# Profile model performance
profile_info = debugger.profile_model(model, input_data, num_iterations=20)

print(f"Model parameters: {profile_info['model_info']['total_parameters']}")
print(f"Mean forward time: {profile_info['performance_metrics']['mean_forward_time']:.4f}s")
print(f"Throughput: {profile_info['performance_metrics']['throughput']:.2f} samples/s")
```

### Memory Leak Detection

```python
# Check for memory leaks
memory_info = debugger.check_memory_leaks(model, num_iterations=100)

if memory_info["potential_leak"]:
    print(f"Potential memory leak detected: {memory_info['memory_growth']}")
else:
    print("No memory leaks detected")
```

## 📋 Debug Summary and Reports

### Debug Summary

**Get Debug Summary**:
```python
def get_debug_summary(self) -> Dict[str, Any]:
    """Get debug summary"""
    if not self.debug_events:
        return {"message": "No debug events recorded"}
    
    event_types = {}
    severities = {}
    
    for event in self.debug_events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        severities[event.severity] = severities.get(event.severity, 0) + 1
    
    return {
        "total_events": len(self.debug_events),
        "event_types": event_types,
        "severities": severities,
        "debug_mode_enabled": {
            "autograd_anomaly": self.autograd_anomaly_enabled,
            "memory_debugging": self.memory_debugging_enabled,
            "cuda_debugging": self.cuda_debugging_enabled,
            "profiling": self.profiling_enabled
        },
        "recent_events": [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "severity": event.severity,
                "message": event.message
            }
            for event in self.debug_events[-10:]  # Last 10 events
        ]
    }
```

### Debug Report Generation

**Save Debug Report**:
```python
def save_debug_report(self, filename: str = None) -> str:
    """Save comprehensive debug report"""
    if filename is None:
        filename = f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report_file = self.debug_output_dir / filename
    
    report = {
        "debug_configuration": asdict(self.config),
        "debug_summary": self.get_debug_summary(),
        "debug_events": [asdict(event) for event in self.debug_events],
        "performance_metrics": self.performance_metrics,
        "memory_snapshots": self.memory_snapshots
    }
    
    # Convert datetime objects to strings
    for event_dict in report["debug_events"]:
        event_dict["timestamp"] = event_dict["timestamp"].isoformat()
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Debug report saved to: {report_file}")
    return str(report_file)
```

## 🔧 Integration with Training

### Training Integration

**Integration with Training Logger**:
```python
from training_logging_system import TrainingLogger
from pytorch_debugging_tools import PyTorchDebugger

# Create training logger
training_logger = TrainingLogger("logs", "debug_experiment")

# Create debugger with training logger integration
debugger = PyTorchDebugger()
debugger.enable_debug_mode(training_logger)

# Start training session
training_logger.log_training_start({"debug_mode": True})

# Training loop with debugging
for epoch in range(10):
    for step in range(100):
        with debugger.debug_context(f"epoch_{epoch}_step_{step}"):
            # Forward pass debugging
            output, forward_debug = debugger.debug_forward_pass(model, input_data)
            
            # Loss calculation
            loss = criterion(output, target)
            
            # Backward pass debugging
            backward_debug = debugger.debug_backward_pass(loss, model)
            
            # Log training step
            training_logger.log_training_step(step, loss.item())
    
    # Check gradients periodically
    gradient_info = debugger.check_gradient_validity(model)
    if not gradient_info["is_valid"]:
        training_logger.log_warning("Gradient issues detected", gradient_info)

# End training session
training_logger.log_training_end({"final_debug_summary": debugger.get_debug_summary()})
```

## 🎯 Best Practices

### Debug Mode Best Practices

1. **Selective Debugging**: Enable only necessary debugging features
2. **Performance Impact**: Be aware that debugging adds overhead
3. **Memory Monitoring**: Monitor memory usage during debugging
4. **Error Handling**: Handle debug errors gracefully
5. **Log Management**: Manage debug log files and cleanup

### Autograd Anomaly Detection Best Practices

1. **Use Sparingly**: Enable only when debugging specific issues
2. **Performance Impact**: Significant performance overhead
3. **Error Context**: Provides detailed error information
4. **Stack Traces**: Use stack traces for debugging
5. **Disable in Production**: Always disable in production code

### Memory Debugging Best Practices

1. **Regular Monitoring**: Monitor memory usage regularly
2. **Leak Detection**: Run memory leak detection periodically
3. **Cleanup**: Ensure proper cleanup after debugging
4. **Memory Snapshots**: Take memory snapshots at key points
5. **GPU Memory**: Monitor GPU memory separately

### Performance Profiling Best Practices

1. **Warm-up**: Always warm up models before profiling
2. **Multiple Iterations**: Profile over multiple iterations
3. **Statistical Analysis**: Use statistical analysis of results
4. **Bottleneck Identification**: Focus on bottlenecks
5. **Profile Output**: Save and analyze profiler output

## 📚 API Reference

### PyTorchDebugger Methods

**Core Methods**:
- `enable_debug_mode(training_logger)` → Enable comprehensive debug mode
- `disable_debug_mode()` → Disable debug mode
- `debug_context(context_name)` → Context manager for debug operations
- `check_tensor_validity(tensor, tensor_name)` → Check tensor validity
- `check_gradient_validity(model)` → Check gradient validity
- `validate_model_inputs(model, *args, **kwargs)` → Validate model inputs
- `debug_forward_pass(model, *args, **kwargs)` → Debug forward pass
- `debug_backward_pass(loss, model)` → Debug backward pass
- `profile_model(model, input_data, num_iterations)` → Profile model performance
- `check_memory_leaks(model, num_iterations)` → Check for memory leaks

**Utility Methods**:
- `get_debug_summary()` → Get debug summary
- `save_debug_report(filename)` → Save debug report
- `_log_debug_event(event)` → Log debug event
- `_log_tensor_issue(tensor_name, issue, tensor)` → Log tensor issues

### Data Structures

**DebugConfiguration**:
```python
@dataclass
class DebugConfiguration:
    enable_autograd_anomaly: bool = True
    enable_memory_debugging: bool = True
    enable_cuda_debugging: bool = True
    enable_model_debugging: bool = True
    enable_performance_profiling: bool = True
    enable_gradient_checking: bool = True
    enable_nan_detection: bool = True
    enable_inf_detection: bool = True
    enable_shape_validation: bool = True
    enable_dtype_validation: bool = True
    enable_device_validation: bool = True
    debug_log_level: str = "INFO"
    save_debug_info: bool = True
    debug_output_dir: str = "debug_output"
```

**DebugEvent**:
```python
@dataclass
class DebugEvent:
    timestamp: datetime
    event_type: str
    severity: str
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    tensor_info: Optional[Dict[str, Any]] = None
    gradient_info: Optional[Dict[str, Any]] = None
    memory_info: Optional[Dict[str, Any]] = None
    performance_info: Optional[Dict[str, Any]] = None
```

## 🔮 Future Enhancements

### Planned Features

1. **Advanced Profiling**: More detailed performance profiling
2. **Memory Optimization**: Automatic memory optimization suggestions
3. **Error Prediction**: ML-based error prediction and prevention
4. **Distributed Debugging**: Multi-GPU debugging support
5. **Real-time Monitoring**: Real-time debugging dashboard

### Technology Integration

1. **TensorBoard Integration**: Enhanced TensorBoard integration
2. **Weights & Biases**: W&B debugging integration
3. **MLflow**: MLflow experiment tracking
4. **Prometheus**: Prometheus metrics integration
5. **Grafana**: Grafana dashboard integration

---

**Comprehensive PyTorch Debugging Tools for Reliable AI Development! 🔧**

For more information, see the main documentation or run:
```bash
python demo_launcher.py --help
``` 