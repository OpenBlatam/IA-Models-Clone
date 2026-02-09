# Async/Sync Patterns - CPU vs I/O Bound Operations
================================================

## Overview

This document outlines the proper use of `def` for pure, CPU-bound routines and `async def` for network- or I/O-bound operations, following the RORO pattern and named exports conventions.

## Table of Contents

1. [CPU-Bound Operations (`def`)](#cpu-bound-operations-def)
2. [I/O-Bound Operations (`async def`)](#io-bound-operations-async-def)
3. [Mixed Operations](#mixed-operations)
4. [Integration with Existing Codebase](#integration-with-existing-codebase)
5. [Best Practices](#best-practices)
6. [Performance Considerations](#performance-considerations)
7. [Error Handling](#error-handling)
8. [Examples from Current Codebase](#examples-from-current-codebase)

## CPU-Bound Operations (`def`)

### When to Use `def`

Use `def` for functions that are:
- **Pure functions** (no side effects)
- **CPU-intensive computations**
- **Mathematical operations**
- **Data transformations**
- **Validation logic**
- **Algorithm implementations**

### Examples

**Mathematical Operations:**
```python
def calculate_loss_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate loss function - CPU-bound pure function."""
    try:
        predictions = params["predictions"]
        targets = params["targets"]
        loss_type = params.get("loss_type", "mse")
        
        if loss_type == "mse":
            loss = torch.nn.MSELoss()(predictions, targets)
        elif loss_type == "cross_entropy":
            loss = torch.nn.CrossEntropyLoss()(predictions, targets)
        
        return {
            "is_successful": True,
            "result": loss.item(),
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

**Data Processing:**
```python
def normalize_tensor_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tensor data - CPU-bound pure function."""
    try:
        tensor_data = params["tensor_data"]
        normalization_type = params.get("normalization_type", "standard")
        
        if normalization_type == "standard":
            mean = torch.mean(tensor_data)
            std = torch.std(tensor_data)
            normalized = (tensor_data - mean) / (std + 1e-8)
        elif normalization_type == "minmax":
            min_val = torch.min(tensor_data)
            max_val = torch.max(tensor_data)
            normalized = (tensor_data - min_val) / (max_val - min_val + 1e-8)
        
        return {
            "is_successful": True,
            "result": normalized,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

**Validation Logic:**
```python
def validate_model_configuration(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate model configuration - CPU-bound pure function."""
    try:
        config = params["config"]
        
        # CPU-intensive validation
        is_input_valid = config.get("input_dimension", 0) > 0
        is_output_valid = config.get("output_dimension", 0) > 0
        are_hidden_layers_valid = all(
            size > 0 for size in config.get("hidden_layer_sizes", [])
        )
        is_learning_rate_valid = 0 < config.get("learning_rate", 0) < 1
        
        validation_errors = []
        if not is_input_valid:
            validation_errors.append("Invalid input dimension")
        if not is_output_valid:
            validation_errors.append("Invalid output dimension")
        if not are_hidden_layers_valid:
            validation_errors.append("Invalid hidden layer sizes")
        if not is_learning_rate_valid:
            validation_errors.append("Invalid learning rate")
        
        is_config_valid = len(validation_errors) == 0
        
        return {
            "is_successful": True,
            "result": {
                "is_valid": is_config_valid,
                "errors": validation_errors
            },
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

## I/O-Bound Operations (`async def`)

### When to Use `async def`

Use `async def` for functions that involve:
- **File I/O operations**
- **Network requests**
- **Database queries**
- **API calls**
- **WebSocket connections**
- **Streaming operations**
- **External service calls**

### Examples

**File Operations:**
```python
async def load_training_data_async(params: Dict[str, Any]) -> Dict[str, Any]:
    """Load training data from file - I/O-bound async function."""
    try:
        file_path = params["file_path"]
        data_format = params.get("data_format", "json")
        
        # I/O-bound operation
        async with aiofiles.open(file_path, 'r') as file:
            content = await file.read()
        
        if data_format == "json":
            import json
            data = json.loads(content)
        elif data_format == "csv":
            import csv
            from io import StringIO
            data = list(csv.DictReader(StringIO(content)))
        
        return {
            "is_successful": True,
            "result": data,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

**Network Requests:**
```python
async def fetch_external_api_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch data from external API - I/O-bound async function."""
    try:
        api_url = params["api_url"]
        headers = params.get("headers", {})
        timeout = params.get("timeout", 30)
        
        # I/O-bound network operation
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                else:
                    raise Exception(f"API request failed with status {response.status}")
        
        return {
            "is_successful": True,
            "result": data,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

**Model Checkpoint Operations:**
```python
async def save_model_checkpoint_async(params: Dict[str, Any]) -> Dict[str, Any]:
    """Save model checkpoint to disk - I/O-bound async function."""
    try:
        model_state = params["model_state"]
        checkpoint_path = params["checkpoint_path"]
        
        # I/O-bound operation
        async with aiofiles.open(checkpoint_path, 'w') as file:
            import json
            await file.write(json.dumps(model_state))
        
        return {
            "is_successful": True,
            "result": checkpoint_path,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

## Mixed Operations

### Combining CPU and I/O Operations

When you need to combine CPU-bound and I/O-bound operations:

```python
async def train_model_with_async_data_loading(params: Dict[str, Any]) -> Dict[str, Any]:
    """Train model with async data loading - mixed CPU and I/O operations."""
    try:
        model = params["model"]
        data_path = params["data_path"]
        training_config = params["training_config"]
        
        # I/O-bound: Load training data
        data_result = await load_training_data_async({"file_path": data_path})
        if not data_result["is_successful"]:
            return data_result
        
        training_data = data_result["result"]
        
        # CPU-bound: Process data
        processed_result = process_batch_data({
            "batch_data": training_data,
            "processing_type": "standard"
        })
        if not processed_result["is_successful"]:
            return processed_result
        
        processed_data = processed_result["result"]
        
        # CPU-bound: Training loop
        for epoch in range(training_config.get("epochs", 10)):
            # CPU-intensive training
            loss_result = calculate_loss_function({
                "predictions": model(processed_data),
                "targets": training_data["targets"],
                "loss_type": "mse"
            })
            
            if not loss_result["is_successful"]:
                return loss_result
        
        # I/O-bound: Save model
        save_result = await save_model_checkpoint_async({
            "model_state": model.state_dict(),
            "checkpoint_path": training_config.get("checkpoint_path", "model.pth")
        })
        
        return save_result
        
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

## Integration with Existing Codebase

### Current Codebase Analysis

The existing codebase already demonstrates good patterns:

**✅ Good CPU-bound Examples:**
- `calculate_loss_function` - Mathematical operations
- `validate_model_configuration` - Validation logic
- `normalize_tensor_data` - Data transformations
- `compute_gradient_descent` - Algorithm implementations

**✅ Good I/O-bound Examples:**
- `load_training_data_async` - File operations
- `fetch_external_api_data` - Network requests
- `save_model_checkpoint_async` - File I/O
- `upload_results_to_storage` - Network operations

### Recommended Refactoring

**Before (Mixed Responsibilities):**
```python
def process_training_data(data_path, model):
    # ❌ Mixing I/O and CPU operations
    with open(data_path, 'r') as f:
        data = json.load(f)  # I/O operation
    
    # CPU operations
    processed_data = normalize_data(data)
    loss = calculate_loss(model(processed_data))
    
    # I/O operation
    with open('results.json', 'w') as f:
        json.dump({'loss': loss}, f)
```

**After (Separated Responsibilities):**
```python
async def process_training_data_async(data_path, model):
    # I/O-bound: Load data
    data_result = await load_training_data_async({"file_path": data_path})
    if not data_result["is_successful"]:
        return data_result
    
    # CPU-bound: Process data
    processed_result = normalize_tensor_data({"tensor_data": data_result["result"]})
    if not processed_result["is_successful"]:
        return processed_result
    
    # CPU-bound: Calculate loss
    loss_result = calculate_loss_function({
        "predictions": model(processed_result["result"]),
        "targets": data_result["result"]["targets"]
    })
    
    # I/O-bound: Save results
    save_result = await save_results_async({
        "results": {"loss": loss_result["result"]},
        "file_path": "results.json"
    })
    
    return save_result
```

## Best Practices

### 1. Function Naming

**CPU-bound functions:**
```python
def calculate_loss_function()
def normalize_tensor_data()
def validate_model_configuration()
def compute_gradient_descent()
def process_batch_data()
```

**I/O-bound functions:**
```python
async def load_training_data_async()
async def save_model_checkpoint_async()
async def fetch_external_api_data()
async def download_model_weights_async()
async def upload_results_to_storage()
```

### 2. Error Handling

**Consistent RORO Pattern:**
```python
def cpu_bound_function(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # CPU operations
        result = perform_calculation(params)
        return {
            "is_successful": True,
            "result": result,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }

async def io_bound_function(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # I/O operations
        result = await perform_io_operation(params)
        return {
            "is_successful": True,
            "result": result,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

### 3. Named Exports

```python
__all__ = [
    # CPU-bound functions
    "calculate_loss_function",
    "normalize_tensor_data",
    "validate_model_configuration",
    "compute_gradient_descent",
    "process_batch_data",
    
    # I/O-bound functions
    "load_training_data_async",
    "save_model_checkpoint_async",
    "fetch_external_api_data",
    "download_model_weights_async",
    "upload_results_to_storage",
    
    # Mixed operations
    "train_model_with_async_data_loading",
    "evaluate_model_with_async_metrics",
]
```

## Performance Considerations

### 1. CPU-Bound Operations

- Use `def` for pure functions
- Avoid blocking I/O in CPU-bound functions
- Consider multiprocessing for heavy computations
- Profile performance for optimization

### 2. I/O-Bound Operations

- Use `async def` for all I/O operations
- Implement proper timeouts
- Use connection pooling
- Handle retries appropriately

### 3. Mixed Operations

- Separate CPU and I/O concerns
- Use async/await properly
- Avoid blocking in async functions
- Consider parallel execution where appropriate

## Error Handling

### Consistent Error Patterns

```python
# CPU-bound error handling
def cpu_function(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # CPU operations
        result = heavy_computation(params)
        return {"is_successful": True, "result": result, "error": None}
    except ValueError as ve:
        return {"is_successful": False, "result": None, "error": f"Validation error: {ve}"}
    except Exception as exc:
        return {"is_successful": False, "result": None, "error": f"Computation error: {exc}"}

# I/O-bound error handling
async def io_function(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # I/O operations
        result = await network_request(params)
        return {"is_successful": True, "result": result, "error": None}
    except asyncio.TimeoutError:
        return {"is_successful": False, "result": None, "error": "Request timeout"}
    except aiohttp.ClientError as ce:
        return {"is_successful": False, "result": None, "error": f"Network error: {ce}"}
    except Exception as exc:
        return {"is_successful": False, "result": None, "error": f"I/O error: {exc}"}
```

## Examples from Current Codebase

### Files Following Good Patterns

1. **`evaluation_metrics.py`** - CPU-bound metric calculations
2. **`data_splitting_validation.py`** - CPU-bound data processing
3. **`gradient_clipping_nan_handling.py`** - CPU-bound mathematical operations
4. **`training_evaluation.py`** - Mixed CPU and I/O operations
5. **`efficient_data_loading.py`** - I/O-bound operations

### Recommended Refactoring

**Current Pattern (Good):**
```python
# evaluation_metrics.py - CPU-bound
def calculate_accuracy(predictions, targets):
    return (predictions.argmax(dim=1) == targets).float().mean()

# efficient_data_loading.py - I/O-bound
async def load_dataset_async(file_path):
    async with aiofiles.open(file_path, 'r') as f:
        return await f.read()
```

**Enhanced Pattern (Better):**
```python
# CPU-bound with RORO
def calculate_accuracy(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        predictions = params["predictions"]
        targets = params["targets"]
        accuracy = (predictions.argmax(dim=1) == targets).float().mean()
        return {"is_successful": True, "result": accuracy.item(), "error": None}
    except Exception as exc:
        return {"is_successful": False, "result": None, "error": str(exc)}

# I/O-bound with RORO
async def load_dataset_async(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        file_path = params["file_path"]
        async with aiofiles.open(file_path, 'r') as f:
            data = await f.read()
        return {"is_successful": True, "result": data, "error": None}
    except Exception as exc:
        return {"is_successful": False, "result": None, "error": str(exc)}
```

## Summary

### Key Principles

1. **Use `def` for CPU-bound operations:**
   - Mathematical computations
   - Data transformations
   - Validation logic
   - Algorithm implementations

2. **Use `async def` for I/O-bound operations:**
   - File operations
   - Network requests
   - Database queries
   - API calls

3. **Follow RORO pattern:**
   - Receive object (dict)
   - Return object (dict with `is_successful`, `result`, `error`)

4. **Use named exports:**
   - Explicit `__all__` declarations
   - Clear function naming
   - Proper import statements

5. **Separate concerns:**
   - Keep CPU and I/O operations separate
   - Combine them in higher-level async functions
   - Maintain clear error handling

### Benefits

- **Performance**: Proper async/sync usage improves performance
- **Maintainability**: Clear separation of concerns
- **Testability**: Pure functions are easier to test
- **Scalability**: Async operations handle concurrency better
- **Reliability**: Consistent error handling patterns

The existing codebase already demonstrates good patterns, and these guidelines help maintain consistency and improve performance across the entire system. 