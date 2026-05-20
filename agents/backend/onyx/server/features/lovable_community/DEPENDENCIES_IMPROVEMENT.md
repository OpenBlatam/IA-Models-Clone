# Dependencies Improvement

## Overview

Improved `dependencies.py` with better error handling, validation, and documentation for user ID extraction functions.

## Changes Made

### 1. Enhanced `get_user_id()` Function
- **Better Validation**: Now checks if user_id is not empty after stripping
- **Improved Error Messages**: More descriptive error message for authentication failures
- **Better Logging**: Includes headers in warning logs for debugging
- **Clearer Documentation**: Better docstring explaining the order of user_id extraction
- **Benefits**:
  - Prevents empty user_ids from being accepted
  - Better debugging information
  - More user-friendly error messages

### 2. Enhanced `get_optional_user_id()` Function
- **Better Exception Handling**: Specifically handles `HTTPException` separately
- **Improved Logging**: Logs unexpected errors with context
- **Better Documentation**: Clearer explanation of behavior
- **Benefits**:
  - More robust error handling
  - Better debugging information
  - Clearer intent

## Before vs After

### Before - get_user_id
```python
def get_user_id(request: Request) -> str:
    # Intentar obtener de header
    user_id: Optional[str] = request.headers.get("X-User-ID")
    if user_id:
        return user_id.strip()
    
    # ... rest of code
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )
```

### After - get_user_id
```python
def get_user_id(request: Request) -> str:
    """
    Intenta obtener el user_id en el siguiente orden:
    1. Header `X-User-ID`
    2. Query parameter `user_id`
    3. En modo debug: valor por defecto
    4. En producción: lanza HTTPException 401
    """
    # Intentar obtener de header (preferido)
    user_id: Optional[str] = request.headers.get("X-User-ID")
    if user_id and user_id.strip():  # Check not empty
        return user_id.strip()
    
    # ... rest of code
    logger.warning(
        "No user_id found in request",
        path=request.url.path,
        method=request.method,
        headers=dict(request.headers)  # Include headers for debugging
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide X-User-ID header or user_id query parameter."  # More descriptive
    )
```

### Before - get_optional_user_id
```python
def get_optional_user_id(request: Request) -> Optional[str]:
    try:
        return get_user_id(request)
    except (HTTPException, Exception):
        return None
```

### After - get_optional_user_id
```python
def get_optional_user_id(request: Request) -> Optional[str]:
    """
    Similar a `get_user_id()` pero nunca lanza excepciones.
    Retorna None si no se puede obtener el user_id.
    """
    try:
        return get_user_id(request)
    except HTTPException:
        # En producción, si no hay user_id, retornar None en lugar de lanzar
        return None
    except Exception as e:
        # Log unexpected errors but don't fail the request
        logger.warning(
            "Unexpected error getting optional user_id",
            error=str(e),
            error_type=type(e).__name__,
            path=request.url.path
        )
        return None
```

## Files Modified

1. **`dependencies.py`**
   - Enhanced `get_user_id()` with better validation and error messages
   - Enhanced `get_optional_user_id()` with better exception handling
   - Improved documentation

## Benefits

1. **Better Validation**: Empty user_ids are now rejected
2. **Improved Error Messages**: More descriptive and helpful error messages
3. **Better Debugging**: More context in logs (headers, path, method)
4. **Robust Error Handling**: Better exception handling in optional function
5. **Clearer Documentation**: Better docstrings explaining behavior
6. **User Experience**: More helpful error messages for API consumers

## Improvements Details

### Validation Improvements
- **Before**: `if user_id:` - Could accept empty strings
- **After**: `if user_id and user_id.strip():` - Rejects empty strings

### Error Message Improvements
- **Before**: "Authentication required"
- **After**: "Authentication required. Please provide X-User-ID header or user_id query parameter."

### Logging Improvements
- **Before**: Basic warning with path and method
- **After**: Includes headers dictionary for better debugging

### Exception Handling Improvements
- **Before**: Catches all exceptions together
- **After**: Handles `HTTPException` and other exceptions separately with appropriate logging

## Verification

- ✅ No linter errors
- ✅ All imports resolve correctly
- ✅ Better validation prevents empty user_ids
- ✅ Improved error messages
- ✅ Better logging for debugging
- ✅ Backward compatible

## Testing Recommendations

1. Test with valid user_id in header
2. Test with valid user_id in query params
3. Test with empty user_id strings
4. Test with missing user_id in production mode
5. Test with missing user_id in debug mode
6. Test `get_optional_user_id()` with various scenarios



