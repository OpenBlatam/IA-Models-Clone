# 🔄 Async/Sync Function Patterns

## Overview

This guide covers the proper use of `def` vs `async def` in Python functions, with clear patterns for mixing synchronous and asynchronous code.

## Core Principles

### Use `def` for:
- **CPU-bound operations** (math, validation, formatting)
- **Simple data transformations**
- **Business logic calculations**
- **Pure functions** with no side effects
- **Fast operations** that don't block

### Use `async def` for:
- **I/O operations** (database, file system, network)
- **Long-running operations**
- **Operations that can benefit from concurrency**
- **Operations that wait for external resources**

## Patterns

### 1. Pure Synchronous Functions

```python
def validate_input_data(data: Dict[str, Any]) -> bool:
    """Synchronous validation - CPU-bound operation."""
    if not isinstance(data, dict):
        return False
    
    required_fields = ['prompt', 'width', 'height']
    return all(field in data for field in required_fields)

def calculate_processing_time(start_time: float) -> float:
    """Synchronous calculation - simple math operation."""
    return time.time() - start_time

def format_file_size(size_bytes: int) -> str:
    """Synchronous formatting - string manipulation."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
```

### 2. Pure Asynchronous Functions

```python
async def fetch_video_data(video_id: str) -> Optional[Dict[str, Any]]:
    """Async database query - I/O-bound operation."""
    await asyncio.sleep(0.1)  # Simulate async database call
    return {
        "id": video_id,
        "status": "processing",
        "created_at": time.time()
    }

async def save_video_file_async(video_data: bytes, path: str) -> bool:
    """Async file I/O - I/O-bound operation."""
    try:
        await asyncio.to_thread(_write_file_sync, path, video_data)
        return True
    except Exception as e:
        logger.error(f"Failed to save video: {e}")
        return False
```

### 3. Mixed Sync/Async Functions

```python
async def process_video_with_validation(
    video_data: Dict[str, Any],
    config_path: str
) -> Dict[str, Any]:
    """Mixed sync/async function - validation + async processing."""
    
    # 1. Synchronous validation first
    if not validate_input_data(video_data):
        raise ValueError("Invalid video data")
    
    # 2. Synchronous config loading
    config = load_config_sync(config_path)
    if not config:
        raise ValueError("Failed to load config")
    
    # 3. Async video processing
    result = await generate_video_async(
        video_data["prompt"], 
        config
    )
    
    # 4. Synchronous result formatting
    result["file_size"] = format_file_size(len(str(result)))
    
    return result
```

## Best Practices

### 1. Validation First (Sync)

Always do validation synchronously at the start of async functions:

```python
async def process_video_request(request: Dict[str, Any]) -> Dict[str, Any]:
    # SYNC: Validate input first (fast operation)
    if not validate_video_request(request):
        raise ValueError("Invalid video request")
    
    # SYNC: Calculate estimates (CPU-bound)
    estimated_time = calculate_estimated_time(
        request['num_steps'], 
        request['width'], 
        request['height']
    )
    
    # ASYNC: Process video (I/O-bound)
    result = await generate_video_async(request)
    
    return result
```

### 2. Use `asyncio.to_thread()` for Sync Operations

When you need to run sync functions in async context:

```python
async def save_video_file(video_bytes: bytes, file_path: str) -> bool:
    """Async file I/O operation."""
    try:
        # Use asyncio.to_thread for file I/O
        await asyncio.to_thread(_write_video_file_sync, file_path, video_bytes)
        return True
    except Exception as e:
        logger.error(f"Failed to save video file: {e}")
        return False

def _write_video_file_sync(file_path: str, video_bytes: bytes) -> None:
    """Synchronous file write helper."""
    with open(file_path, 'wb') as f:
        f.write(video_bytes)
```

### 3. Concurrent Async Operations

Use `asyncio.gather()` for concurrent async operations:

```python
async def process_batch_requests(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # SYNC: Validate all requests
    valid_requests = validate_batch_requests(requests)
    
    # ASYNC: Process all requests concurrently
    tasks = [process_video_request(req) for req in valid_requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # SYNC: Process results
    successful_results = [
        result for result in results 
        if not isinstance(result, Exception)
    ]
    
    return successful_results
```

### 4. Error Handling Patterns

```python
# SYNC: Error classification
def classify_error(error: Exception) -> str:
    """Synchronous error classification - CPU-bound."""
    if isinstance(error, ValueError):
        return "validation_error"
    elif isinstance(error, FileNotFoundError):
        return "file_error"
    else:
        return "unknown_error"

# ASYNC: Error handling wrapper
async def handle_async_operation(operation: callable, context: str, *args, **kwargs):
    """Async error handling wrapper."""
    try:
        return await operation(*args, **kwargs)
    except Exception as e:
        # SYNC: Format error message
        error_message = format_error_message(e, context)
        logger.error(error_message)
        
        # ASYNC: Log error to database
        await log_error_async(error_message, context)
        
        raise
```

## Common Patterns

### 1. Configuration Management

```python
# SYNC: Config validation
def validate_config(config: Dict[str, Any]) -> bool:
    """Synchronous config validation - CPU-bound."""
    required_sections = ['model', 'processing', 'output']
    return all(section in config for section in required_sections)

# ASYNC: Config loading
async def load_config_async(config_path: str) -> Optional[Dict[str, Any]]:
    """Async config loading - I/O-bound."""
    try:
        # Use asyncio.to_thread for file I/O
        config = await asyncio.to_thread(_load_config_sync, config_path)
        
        # SYNC: Validate config
        if not validate_config(config):
            logger.error("Invalid configuration")
            return None
        
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None
```

### 2. Resource Planning

```python
# SYNC: Resource calculation
def calculate_batch_resources(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Synchronous resource calculation - CPU-bound."""
    total_steps = sum(req['num_steps'] for req in requests)
    total_pixels = sum(req['width'] * req['height'] for req in requests)
    estimated_memory = total_pixels * 4 * 3  # 4 bytes per pixel, 3 channels
    
    return {
        "total_steps": total_steps,
        "total_pixels": total_pixels,
        "estimated_memory_mb": estimated_memory / 1024 / 1024,
        "batch_size": len(requests)
    }
```

### 3. Data Formatting

```python
# SYNC: Data formatting
def format_video_metadata(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous data formatting - string manipulation."""
    return {
        "id": video_data.get("id", "unknown"),
        "status": video_data.get("status", "unknown"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", 
                                  time.localtime(video_data.get("created_at", time.time()))),
        "file_size": f"{video_data.get('file_size', 0) / 1024 / 1024:.1f} MB",
        "duration": f"{video_data.get('duration', 0):.2f}s"
    }
```

## Performance Considerations

### 1. Don't Block the Event Loop

❌ **Bad**: Blocking operations in async functions
```python
async def bad_example():
    # This blocks the event loop!
    time.sleep(5)  # Use await asyncio.sleep(5) instead
```

✅ **Good**: Non-blocking operations
```python
async def good_example():
    # This doesn't block the event loop
    await asyncio.sleep(5)
    
    # Or use asyncio.to_thread for CPU-bound operations
    result = await asyncio.to_thread(cpu_intensive_function)
```

### 2. Use Executors for CPU-Bound Operations

```python
async def process_large_dataset(data: List[Any]) -> List[Any]:
    """Process large dataset with CPU-bound operations."""
    
    # Use executor for CPU-bound operations
    processed_data = await asyncio.to_thread(
        process_data_sync,  # CPU-intensive function
        data
    )
    
    return processed_data

def process_data_sync(data: List[Any]) -> List[Any]:
    """CPU-intensive synchronous processing."""
    # Heavy computation here
    return [item * 2 for item in data]
```

### 3. Concurrent I/O Operations

```python
async def fetch_multiple_resources(urls: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple resources concurrently."""
    
    # Create tasks for concurrent execution
    tasks = [fetch_single_resource(url) for url in urls]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = [
        result for result in results 
        if not isinstance(result, Exception)
    ]
    
    return valid_results
```

## Summary

- **Use `def`** for fast, CPU-bound operations
- **Use `async def`** for I/O-bound operations
- **Validate synchronously** at the start of async functions
- **Use `asyncio.to_thread()`** for sync operations in async context
- **Use `asyncio.gather()`** for concurrent async operations
- **Don't block the event loop** with sync operations in async functions
- **Format results synchronously** at the end of async functions

This approach ensures optimal performance and maintainable code structure. 