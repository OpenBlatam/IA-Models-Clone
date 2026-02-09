# Error Handling Guide - Addiction Recovery AI

## ✅ Error Handling Structure

### Error Components

```
core/errors/
├── custom_exceptions.py  # ✅ Custom exception classes
└── error_handler.py      # ✅ Error handler
```

## 📦 Error Classes

### `core/errors/custom_exceptions.py` - Custom Exceptions
- **Status**: ✅ Active
- **Purpose**: Custom exception hierarchy
- **Exceptions**:
  - `RecoveryAIError` - Base exception
  - `ModelError` - Model-related errors
    - `ModelLoadError` - Model loading errors
    - `ModelInferenceError` - Inference errors
    - `ModelTrainingError` - Training errors
  - `DataError` - Data-related errors
    - `DataValidationError` - Validation errors
    - `DataProcessingError` - Processing errors
  - `ConfigurationError` - Configuration errors
  - `InferenceError` - Inference errors
    - `CUDAOutOfMemoryError` - CUDA OOM errors
  - `ValidationError` - Validation errors

**Usage:**
```python
from core.errors.custom_exceptions import (
    RecoveryAIError,
    ModelError,
    DataValidationError
)

try:
    # Your code
    pass
except ModelError as e:
    # Handle model error
    pass
except DataValidationError as e:
    # Handle validation error
    pass
```

### `core/errors/error_handler.py` - Error Handler
- **Status**: ✅ Active
- **Purpose**: Centralized error handling
- **Features**: Error logging, response formatting, error recovery

**Usage:**
```python
from core.errors.error_handler import ErrorHandler

handler = ErrorHandler()

try:
    # Your code
    pass
except Exception as e:
    handler.handle_error(e, context={})
```

## 📝 Error Handling Patterns

### Using Custom Exceptions
```python
from core.errors.custom_exceptions import ModelLoadError

def load_model(path):
    try:
        # Load model
        pass
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {path}: {e}")
```

### Using Error Handler
```python
from core.errors.error_handler import ErrorHandler
from fastapi import HTTPException

handler = ErrorHandler()

@router.get("/endpoint")
async def endpoint():
    try:
        # Your code
        pass
    except RecoveryAIError as e:
        raise HTTPException(
            status_code=500,
            detail=handler.format_error(e)
        )
```

## 🎯 Error Hierarchy

```
RecoveryAIError (base)
├── ModelError
│   ├── ModelLoadError
│   ├── ModelInferenceError
│   └── ModelTrainingError
├── DataError
│   ├── DataValidationError
│   └── DataProcessingError
├── ConfigurationError
├── InferenceError
│   └── CUDAOutOfMemoryError
└── ValidationError
```

## 📚 Additional Resources

- See `MIDDLEWARE_GUIDE.md` for error handling middleware
- See `API_GUIDE.md` for API error responses
- See `CORE_GUIDE.md` for core components






