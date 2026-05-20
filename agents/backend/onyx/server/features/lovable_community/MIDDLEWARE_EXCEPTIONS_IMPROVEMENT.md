# Middleware and Exceptions Improvement

## Overview

Improved error handling middleware with better logging context and enhanced exception classes with additional context fields.

## Changes Made

### 1. Enhanced Error Handler Middleware
- **Better Attribute Access**: Uses `getattr()` to safely access exception attributes
- **Improved Logging**: Added query parameters to log context for better debugging
- **Consistent Variable Usage**: Uses extracted `status_code` and `error_detail` consistently
- **Benefits**:
  - More robust error handling
  - Better debugging information
  - Prevents AttributeError when exception doesn't have expected attributes

### 2. Enhanced Exception Classes
- **ChatNotFoundError**: Added optional `additional_context` parameter and `chat_id` attribute
- **DatabaseError**: Added optional `operation` parameter and `operation` attribute
- **Benefits**:
  - More context in error messages
  - Easier debugging with stored attributes
  - More flexible error creation

## Before vs After

### Before - Error Handler Middleware
```python
except BaseCommunityException as exc:
    user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
    
    if exc.status_code < 500:
        logger.warning(
            "API exception",
            path=request.url.path,
            method=request.method,
            status_code=exc.status_code,
            exception_type=type(exc).__name__,
            message=exc.detail,
            user_id=user_id
        )
    
    error_response = ErrorResponse(
        error=type(exc).__name__,
        message=exc.detail,
        path=request.url.path,
        timestamp=datetime.utcnow()
    )
    
    return ORJSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )
```

### After - Error Handler Middleware
```python
except BaseCommunityException as exc:
    user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
    
    # Extraer detalles del error de forma segura
    error_detail = getattr(exc, 'detail', str(exc))
    status_code = getattr(exc, 'status_code', status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    if status_code < 500:
        logger.warning(
            "API exception",
            path=request.url.path,
            method=request.method,
            status_code=status_code,
            exception_type=type(exc).__name__,
            message=error_detail,
            user_id=user_id,
            query_params=dict(request.query_params) if request.query_params else None  # Added
        )
    
    error_response = ErrorResponse(
        error=type(exc).__name__,
        message=error_detail,  # Use extracted variable
        path=request.url.path,
        timestamp=datetime.utcnow()
    )
    
    return ORJSONResponse(
        status_code=status_code,  # Use extracted variable
        content=error_response.model_dump()
    )
```

### Before - Exceptions
```python
class ChatNotFoundError(BaseCommunityException):
    def __init__(self, chat_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat with ID '{chat_id}' not found"
        )

class DatabaseError(BaseCommunityException):
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {message}"
        )
```

### After - Exceptions
```python
class ChatNotFoundError(BaseCommunityException):
    def __init__(self, chat_id: str, additional_context: Optional[str] = None):
        detail = f"Chat with ID '{chat_id}' not found"
        if additional_context:
            detail += f". {additional_context}"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )
        self.chat_id = chat_id  # Store for debugging

class DatabaseError(BaseCommunityException):
    def __init__(self, message: str, operation: Optional[str] = None):
        detail = f"Database error: {message}"
        if operation:
            detail = f"Database error during {operation}: {message}"
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )
        self.operation = operation  # Store for debugging
```

## Files Modified

1. **`middleware/error_handler.py`**
   - Enhanced `BaseCommunityException` handling with safe attribute access
   - Added query parameters to logging context
   - Consistent use of extracted variables

2. **`exceptions.py`**
   - Enhanced `ChatNotFoundError` with optional context and stored `chat_id`
   - Enhanced `DatabaseError` with optional operation and stored `operation`

## Benefits

1. **Better Debugging**: Query parameters in logs help identify problematic requests
2. **Robust Error Handling**: Safe attribute access prevents AttributeError
3. **More Context**: Exception classes store additional context for debugging
4. **Flexible Error Messages**: Optional parameters allow more descriptive errors
5. **Consistent Code**: Uses extracted variables consistently throughout

## Improvements Details

### Middleware Improvements
- **Safe Attribute Access**: `getattr()` prevents AttributeError
- **Better Logging**: Query parameters included in log context
- **Consistent Variables**: Extracted `status_code` and `error_detail` used throughout

### Exception Improvements
- **ChatNotFoundError**: 
  - Optional `additional_context` for more descriptive messages
  - Stored `chat_id` attribute for debugging
- **DatabaseError**:
  - Optional `operation` parameter for context
  - Stored `operation` attribute for debugging

## Verification

- ✅ No linter errors
- ✅ All imports resolve correctly
- ✅ Safe attribute access prevents errors
- ✅ Better logging context
- ✅ More flexible exception creation
- ✅ Backward compatible

## Usage Examples

### Enhanced ChatNotFoundError
```python
# Before
raise ChatNotFoundError(chat_id)

# After - with additional context
raise ChatNotFoundError(chat_id, "Chat may have been deleted or is private")
```

### Enhanced DatabaseError
```python
# Before
raise DatabaseError("Connection failed")

# After - with operation context
raise DatabaseError("Connection failed", operation="create_chat")
```

## Testing Recommendations

1. Test middleware with exceptions missing attributes
2. Verify query parameters appear in logs
3. Test enhanced exceptions with optional parameters
4. Verify stored attributes are accessible
5. Test backward compatibility with existing exception usage



