# Robust Error Handling with Try-Except Blocks Guide

## Overview

This guide covers the comprehensive robust error handling system implemented with try-except blocks for error-prone operations, especially in data loading and model inference. The system provides retry mechanisms, fallback strategies, and graceful error recovery for critical AI operations.

## 🛡️ Available Robust Error Handling Systems

### 1. Robust Error Handling System (`robust_error_handling.py`)
**Port**: 7869
**Description**: Comprehensive try-except blocks for data loading and model inference

**Features**:
- **Data Loading Error Handling**: Retry mechanisms and fallback strategies
- **Model Inference Error Handling**: Device fallback and memory management
- **File Operation Error Handling**: Recovery mechanisms and validation
- **Network Operation Error Handling**: Timeout management and retry logic
- **Memory Operation Error Handling**: Cleanup strategies and monitoring
- **GPU Operation Error Handling**: Device management and memory optimization

## 🚀 Quick Start

### Installation

1. **Install Dependencies**:
```bash
pip install -r requirements_gradio_demos.txt
pip install psutil memory-profiler h5py pandas requests
```

2. **Launch Robust Error Handling System**:
```bash
# Launch robust error handling system
python demo_launcher.py --demo robust-error-handling

# Launch all error handling systems
python demo_launcher.py --all
```

### Direct Launch

```bash
# Robust error handling system
python robust_error_handling.py
```

## 🛡️ Robust Error Handling Features

### Try-Except Block Implementation

**Core Error Handling Pattern**:
```python
def safe_execute_with_retry(self, operation_type: str, func: Callable, 
                           *args, **kwargs) -> Tuple[Any, ErrorContext]:
    """Execute function with comprehensive error handling and retry logic"""
    
    context = ErrorContext(
        operation=operation_type,
        timestamp=datetime.now(),
        max_retries=self.retry_strategies.get(operation_type, {}).get('max_retries', 3)
    )
    
    for attempt in range(max_retries + 1):
        try:
            # Execute the function
            result = func(*args, **kwargs)
            context.success = True
            context.recovery_strategy = "success"
            return result, context
            
        except Exception as e:
            # Handle error with retry logic
            context.retry_count = attempt
            context.error_type = type(e).__name__
            context.error_message = str(e)
            context.stack_trace = traceback.format_exc()
            
            # Check if error is retryable
            if type(e).__name__ not in retryable_errors or attempt == max_retries:
                break
            
            # Exponential backoff
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            time.sleep(delay)
    
    # Try fallback strategy
    fallback_func = self.fallback_strategies.get(operation_type)
    if fallback_func:
        try:
            result = fallback_func(*args, **kwargs)
            context.success = True
            context.recovery_strategy = "fallback_success"
            return result, context
        except Exception as fallback_error:
            context.error_message = f"Original: {last_error}, Fallback: {fallback_error}"
            context.recovery_strategy = "fallback_failed"
    
    return None, context
```

### Data Loading Error Handling

**Supported Data Types**:
- **Text Files**: UTF-8 and Latin-1 encoding support
- **Images**: PIL Image loading with format validation
- **JSON Files**: JSON parsing with error recovery
- **Pickle Files**: Python object serialization
- **HDF5 Files**: Hierarchical data format
- **CSV Files**: Pandas DataFrame loading
- **NumPy Files**: Array data loading

**Data Loading with Error Handling**:
```python
def safe_data_loading(self, file_path: str, data_type: str = "auto") -> Tuple[Any, ErrorContext]:
    """Safely load data with comprehensive error handling"""
    
    def load_data():
        if data_type == "auto":
            data_type = self._detect_data_type(file_path)
        
        if data_type == "image":
            return self._load_image(file_path)
        elif data_type == "text":
            return self._load_text(file_path)
        elif data_type == "json":
            return self._load_json(file_path)
        elif data_type == "pickle":
            return self._load_pickle(file_path)
        elif data_type == "hdf5":
            return self._load_hdf5(file_path)
        elif data_type == "csv":
            return self._load_csv(file_path)
        elif data_type == "numpy":
            return self._load_numpy(file_path)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    return self.safe_execute_with_retry("data_loading", load_data)
```

**Text Loading with Encoding Fallback**:
```python
def _load_text(self, file_path: str) -> str:
    """Load text file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load text file {file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load text file {file_path}: {e}")
```

### Model Inference Error Handling

**Inference with Device Fallback**:
```python
def safe_model_inference(self, model: nn.Module, input_data: Any, 
                       device: str = "auto") -> Tuple[Any, ErrorContext]:
    """Safely perform model inference with comprehensive error handling"""
    
    def perform_inference():
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Move model to device
        model.to(device)
        model.eval()
        
        # Prepare input data
        if isinstance(input_data, (list, tuple)):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        elif isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
        
        # Move input to device
        input_tensor = input_tensor.to(device)
        
        # Perform inference
        with torch.no_grad():
            try:
                output = model(input_tensor)
                return output.cpu().numpy() if device == "cuda" else output.numpy()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try to free memory and retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    raise e
                else:
                    raise e
    
    return self.safe_execute_with_retry("model_inference", perform_inference)
```

**Batch Inference with Error Handling**:
```python
def batch_inference(self, model: nn.Module, input_batch: List[Any], 
                   device: str = "auto", batch_size: int = 32) -> List[Any]:
    """Perform batch inference with error handling"""
    results = []
    
    for i in range(0, len(input_batch), batch_size):
        batch = input_batch[i:i + batch_size]
        
        try:
            result, context = self.error_handler.safe_model_inference(model, batch, device)
            if context.success:
                results.extend(result)
            else:
                # Handle failed batch
                results.extend([None] * len(batch))
        except Exception as e:
            # Add dummy results for failed batch
            results.extend([None] * len(batch))
    
    return results
```

### File Operation Error Handling

**Comprehensive File Operations**:
```python
def safe_file_operations(self, operation: str, file_path: str, 
                       data: Any = None, mode: str = "r") -> Tuple[Any, ErrorContext]:
    """Safely perform file operations with comprehensive error handling"""
    
    def perform_file_operation():
        if operation == "read":
            return self._read_file(file_path, mode)
        elif operation == "write":
            return self._write_file(file_path, data, mode)
        elif operation == "delete":
            return self._delete_file(file_path)
        elif operation == "exists":
            return os.path.exists(file_path)
        elif operation == "size":
            return os.path.getsize(file_path)
        else:
            raise ValueError(f"Unsupported file operation: {operation}")
    
    return self.safe_execute_with_retry("file_operations", perform_file_operation)
```

**File Writing with Directory Creation**:
```python
def _write_file(self, file_path: str, data: Any, mode: str = "w") -> bool:
    """Write file with error handling"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, mode, encoding='utf-8' if 'b' not in mode else None) as f:
            f.write(data)
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to write file {file_path}: {e}")
```

### Network Operation Error Handling

**HTTP Requests with Retry Logic**:
```python
def safe_network_operations(self, url: str, method: str = "GET", 
                          data: Any = None, timeout: float = 30.0) -> Tuple[Any, ErrorContext]:
    """Safely perform network operations with comprehensive error handling"""
    
    def perform_network_operation():
        if method.upper() == "GET":
            response = requests.get(url, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, timeout=timeout)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
    
    return self.safe_execute_with_retry("network_operations", perform_network_operation)
```

### Memory Operation Error Handling

**Memory Management with Cleanup**:
```python
def safe_memory_operations(self, operation: str, size: int = None) -> Tuple[Any, ErrorContext]:
    """Safely perform memory operations with comprehensive error handling"""
    
    def perform_memory_operation():
        if operation == "allocate":
            return self._allocate_memory(size)
        elif operation == "free":
            return self._free_memory()
        elif operation == "check":
            return self._check_memory_usage()
        else:
            raise ValueError(f"Unsupported memory operation: {operation}")
    
    return self.safe_execute_with_retry("memory_operations", perform_memory_operation)
```

**Memory Allocation with Cleanup**:
```python
def _allocate_memory(self, size: int) -> bool:
    """Allocate memory with error handling"""
    try:
        # Simulate memory allocation
        test_array = np.zeros(size, dtype=np.float32)
        return True
    except MemoryError:
        # Try to free some memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise MemoryError(f"Failed to allocate {size} bytes")
```

### GPU Operation Error Handling

**GPU Memory Management**:
```python
def safe_gpu_operations(self, operation: str, device_id: int = 0) -> Tuple[Any, ErrorContext]:
    """Safely perform GPU operations with comprehensive error handling"""
    
    def perform_gpu_operation():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        if operation == "memory_info":
            return self._get_gpu_memory_info(device_id)
        elif operation == "clear_cache":
            return self._clear_gpu_cache(device_id)
        elif operation == "device_info":
            return self._get_gpu_device_info(device_id)
        else:
            raise ValueError(f"Unsupported GPU operation: {operation}")
    
    return self.safe_execute_with_retry("gpu_operations", perform_gpu_operation)
```

**GPU Memory Information**:
```python
def _get_gpu_memory_info(self, device_id: int) -> Dict[str, float]:
    """Get GPU memory information with error handling"""
    try:
        torch.cuda.set_device(device_id)
        memory_allocated = torch.cuda.memory_allocated(device_id)
        memory_reserved = torch.cuda.memory_reserved(device_id)
        memory_total = torch.cuda.get_device_properties(device_id).total_memory
        
        return {
            'allocated_gb': memory_allocated / (1024**3),
            'reserved_gb': memory_reserved / (1024**3),
            'total_gb': memory_total / (1024**3),
            'free_gb': (memory_total - memory_reserved) / (1024**3)
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get GPU memory info: {e}")
```

## 🔧 Retry Strategies

### Configuration

**Retry Strategy Configuration**:
```python
retry_strategies = {
    'data_loading': {
        'max_retries': 5,
        'base_delay': 1.0,
        'max_delay': 30.0,
        'backoff_factor': 2.0,
        'retryable_errors': [
            'FileNotFoundError', 'PermissionError', 'ConnectionError',
            'TimeoutError', 'OSError', 'IOError'
        ]
    },
    'model_inference': {
        'max_retries': 3,
        'base_delay': 0.5,
        'max_delay': 10.0,
        'backoff_factor': 1.5,
        'retryable_errors': [
            'RuntimeError', 'CUDAError', 'OutOfMemoryError',
            'TimeoutError', 'ConnectionError'
        ]
    },
    'file_operations': {
        'max_retries': 3,
        'base_delay': 0.5,
        'max_delay': 5.0,
        'backoff_factor': 1.2,
        'retryable_errors': [
            'FileNotFoundError', 'PermissionError', 'OSError',
            'IOError', 'TimeoutError'
        ]
    },
    'network_operations': {
        'max_retries': 5,
        'base_delay': 2.0,
        'max_delay': 60.0,
        'backoff_factor': 2.0,
        'retryable_errors': [
            'ConnectionError', 'TimeoutError', 'requests.RequestException',
            'socket.error', 'urllib.error.URLError'
        ]
    },
    'memory_operations': {
        'max_retries': 2,
        'base_delay': 1.0,
        'max_delay': 10.0,
        'backoff_factor': 1.5,
        'retryable_errors': [
            'MemoryError', 'OutOfMemoryError', 'RuntimeError'
        ]
    },
    'gpu_operations': {
        'max_retries': 3,
        'base_delay': 1.0,
        'max_delay': 15.0,
        'backoff_factor': 2.0,
        'retryable_errors': [
            'CUDAError', 'OutOfMemoryError', 'RuntimeError'
        ]
    }
}
```

### Exponential Backoff

**Backoff Calculation**:
```python
# Calculate delay with exponential backoff
delay = min(base_delay * (backoff_factor ** attempt), max_delay)

# Example progression:
# Attempt 0: 1.0s
# Attempt 1: 2.0s
# Attempt 2: 4.0s
# Attempt 3: 8.0s
# Attempt 4: 16.0s (capped at max_delay)
```

## 🛡️ Fallback Strategies

### Data Loading Fallback

**Fallback Data Loading**:
```python
def _fallback_data_loading(self, file_path: str, data_type: str = "auto") -> Any:
    """Fallback strategy for data loading"""
    try:
        # Try to load as text as last resort
        if data_type != "text":
            return self._load_text(file_path)
        else:
            # Return empty/default data
            if data_type == "image":
                return Image.new('RGB', (100, 100), color='gray')
            elif data_type == "json":
                return {}
            elif data_type == "csv":
                return pd.DataFrame()
            elif data_type == "numpy":
                return np.array([])
            else:
                return ""
    except Exception as e:
        self.debugger.log_debug_event("FALLBACK_DATA_LOADING_FAILED", 
                                    f"Fallback data loading failed: {e}", "ERROR", error=e)
        raise
```

### Model Inference Fallback

**Device Fallback Strategy**:
```python
def _fallback_model_inference(self, model: nn.Module, input_data: Any, device: str = "auto") -> Any:
    """Fallback strategy for model inference"""
    try:
        # Try CPU inference as fallback
        if device != "cpu":
            model.cpu()
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.cpu()
            elif isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data).float()
            
            with torch.no_grad():
                output = model(input_data)
                return output.numpy()
        else:
            # Return dummy output
            return np.zeros((1, 10))  # Dummy classification output
    except Exception as e:
        self.debugger.log_debug_event("FALLBACK_MODEL_INFERENCE_FAILED", 
                                    f"Fallback model inference failed: {e}", "ERROR", error=e)
        raise
```

## 📊 Error Context Tracking

### Error Context Structure

**Error Context Information**:
```python
@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    recovery_strategy: str = ""
    success: bool = False
```

### Error Summary

**Error Summary Generation**:
```python
def get_error_summary(self) -> Dict[str, Any]:
    """Get summary of error handling activities"""
    if not self.error_contexts:
        return {"message": "No errors recorded"}
    
    error_types = defaultdict(int)
    operation_types = defaultdict(int)
    recovery_strategies = defaultdict(int)
    
    for context in self.error_contexts:
        error_types[context.error_type] += 1
        operation_types[context.operation] += 1
        recovery_strategies[context.recovery_strategy] += 1
    
    return {
        "total_errors": len(self.error_contexts),
        "error_types": dict(error_types),
        "operation_types": dict(operation_types),
        "recovery_strategies": dict(recovery_strategies),
        "success_rate": len([c for c in self.error_contexts if c.success]) / len(self.error_contexts),
        "recent_errors": [
            {
                "operation": c.operation,
                "error_type": c.error_type,
                "error_message": c.error_message,
                "retry_count": c.retry_count,
                "recovery_strategy": c.recovery_strategy,
                "success": c.success,
                "timestamp": c.timestamp.isoformat()
            }
            for c in list(self.error_contexts)[-10:]  # Last 10 errors
        ]
    }
```

## 🎯 Best Practices

### Try-Except Block Best Practices

1. **Specific Exception Handling**: Catch specific exceptions rather than generic ones
2. **Proper Error Logging**: Log errors with context and stack traces
3. **Graceful Degradation**: Provide fallback mechanisms for failed operations
4. **Resource Cleanup**: Ensure proper cleanup in finally blocks
5. **Retry Logic**: Implement exponential backoff for transient errors

### Data Loading Best Practices

1. **File Validation**: Validate file existence and permissions before loading
2. **Encoding Handling**: Handle different text encodings gracefully
3. **Memory Management**: Monitor memory usage during large file loading
4. **Format Validation**: Validate data format and structure
5. **Caching**: Implement caching for frequently accessed data

### Model Inference Best Practices

1. **Device Management**: Handle GPU/CPU device switching gracefully
2. **Memory Monitoring**: Monitor GPU memory usage and cleanup
3. **Input Validation**: Validate input data types and shapes
4. **Batch Processing**: Handle batch inference with error recovery
5. **Performance Optimization**: Optimize inference for different devices

### File Operation Best Practices

1. **Path Validation**: Validate file paths and create directories if needed
2. **Permission Handling**: Handle file permission errors gracefully
3. **Atomic Operations**: Use atomic operations for critical file writes
4. **Backup Strategies**: Implement backup strategies for important files
5. **Cleanup**: Clean up temporary files and resources

## 🔍 Troubleshooting

### Common Issues

**Memory Issues**:
```python
# Handle out of memory errors
try:
    result = model(input_data)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        torch.cuda.empty_cache()
        gc.collect()
        # Retry with smaller batch or different device
```

**File Permission Issues**:
```python
# Handle file permission errors
try:
    with open(file_path, 'r') as f:
        data = f.read()
except PermissionError:
    # Try alternative location or different permissions
    pass
```

**Network Timeout Issues**:
```python
# Handle network timeouts
try:
    response = requests.get(url, timeout=30)
except requests.Timeout:
    # Implement retry logic with exponential backoff
    pass
```

### Debug Mode

**Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode in error handler
error_handler.debugger.config.debug_log_level = "DEBUG"
```

## 📚 API Reference

### RobustErrorHandler Methods

**Core Methods**:
- `safe_execute_with_retry(operation_type, func, *args, **kwargs)` → `(Any, ErrorContext)`
- `safe_data_loading(file_path, data_type)` → `(Any, ErrorContext)`
- `safe_model_inference(model, input_data, device)` → `(Any, ErrorContext)`
- `safe_file_operations(operation, file_path, data, mode)` → `(Any, ErrorContext)`
- `safe_network_operations(url, method, data, timeout)` → `(Any, ErrorContext)`
- `safe_memory_operations(operation, size)` → `(Any, ErrorContext)`
- `safe_gpu_operations(operation, device_id)` → `(Any, ErrorContext)`

**Utility Methods**:
- `get_error_summary()` → `Dict[str, Any]`
- `_detect_data_type(file_path)` → `str`
- `_load_image(file_path)` → `Image.Image`
- `_load_text(file_path)` → `str`
- `_load_json(file_path)` → `Dict[str, Any]`

### RobustDataLoader Methods

**Data Loading Methods**:
- `load_dataset(file_paths, data_type)` → `List[Any]`
- `load_with_cache(file_path, data_type)` → `Any`

### RobustModelInference Methods

**Inference Methods**:
- `inference_with_fallback(model, input_data, device, fallback_device)` → `Any`
- `batch_inference(model, input_batch, device, batch_size)` → `List[Any]`

## 🎯 Usage Examples

### Basic Error Handling

```python
from robust_error_handling import RobustErrorHandler

# Create error handler
error_handler = RobustErrorHandler()

# Safe data loading
data, context = error_handler.safe_data_loading("data.json", "json")
if context.success:
    print(f"Data loaded successfully: {data}")
else:
    print(f"Failed to load data: {context.error_message}")
```

### Model Inference with Fallback

```python
# Safe model inference
result, context = error_handler.safe_model_inference(model, input_data, "cuda")
if context.success:
    print(f"Inference successful: {result.shape}")
else:
    print(f"Inference failed: {context.error_message}")
    print(f"Recovery strategy: {context.recovery_strategy}")
```

### File Operations

```python
# Safe file operations
result, context = error_handler.safe_file_operations("write", "output.txt", "Hello World")
if context.success:
    print("File written successfully")
else:
    print(f"File operation failed: {context.error_message}")
```

### Network Operations

```python
# Safe network operations
result, context = error_handler.safe_network_operations("https://api.example.com/data")
if context.success:
    print(f"Network request successful: {result}")
else:
    print(f"Network request failed: {context.error_message}")
```

### Memory Operations

```python
# Safe memory operations
result, context = error_handler.safe_memory_operations("check")
if context.success:
    print(f"Memory usage: {result}")
else:
    print(f"Memory check failed: {context.error_message}")
```

### GPU Operations

```python
# Safe GPU operations
result, context = error_handler.safe_gpu_operations("memory_info", device_id=0)
if context.success:
    print(f"GPU memory: {result}")
else:
    print(f"GPU operation failed: {context.error_message}")
```

## 🔮 Future Enhancements

### Planned Features

1. **Advanced Retry Strategies**: ML-based retry strategy optimization
2. **Predictive Error Handling**: Predict and prevent errors before they occur
3. **Distributed Error Handling**: Multi-node error handling coordination
4. **Real-time Error Monitoring**: Live error monitoring and alerting
5. **Automated Error Recovery**: Fully automated error recovery systems

### Technology Integration

1. **Cloud Integration**: Cloud-based error handling and monitoring
2. **Container Support**: Container-aware error handling
3. **Microservices**: Microservice error handling patterns
4. **Event Streaming**: Real-time error event streaming
5. **Machine Learning**: ML-powered error prediction and prevention

---

**Robust Error Handling with Try-Except Blocks for Reliable AI Operations! 🛡️**

For more information, see the main documentation or run:
```bash
python demo_launcher.py --help
``` 