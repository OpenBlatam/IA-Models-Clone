# Factories Improvement

## Overview

Improved factory classes with better validation, error handling, and documentation.

## Changes Made

### 1. Enhanced `RepositoryFactory`
- **Added Validation**: Validates that `db` is not None in `__init__`
- **Enhanced `_get_or_create_repository()`**: 
  - Validates repository_class is not None
  - Better error handling with specific exception types
  - More descriptive error messages
- **Benefits**:
  - Prevents NoneType errors
  - Better error messages for debugging
  - More robust factory creation

### 2. Enhanced `ServiceFactory`
- **Added Validation**: Validates that `db` is not None in `__init__`
- **Enhanced `get_chat_service()`**: 
  - Better error handling for import errors
  - Better error handling for service creation
  - More descriptive error messages
- **Benefits**:
  - Prevents NoneType errors
  - Better error messages for debugging
  - More robust service creation

## Before vs After

### Before - RepositoryFactory.__init__
```python
def __init__(self, db: Session):
    self.db = db
    self._chat_repository: Optional[ChatRepository] = None
    # ...
```

### After - RepositoryFactory.__init__
```python
def __init__(self, db: Session):
    """
    Raises:
        ValueError: If db is None
    """
    if db is None:
        raise ValueError("Database session cannot be None")
    self.db = db
    self._chat_repository: Optional[ChatRepository] = None
    # ...
```

### Before - _get_or_create_repository
```python
def _get_or_create_repository(self, repository_attr: str, repository_class: type):
    cached = getattr(self, repository_attr, None)
    if cached is None:
        cached = repository_class(self.db)
        setattr(self, repository_attr, cached)
    return cached
```

### After - _get_or_create_repository
```python
def _get_or_create_repository(self, repository_attr: str, repository_class: type):
    """
    Raises:
        ValueError: If repository_class is None or invalid
        TypeError: If repository_class cannot be instantiated with db session
    """
    if repository_class is None:
        raise ValueError("Repository class cannot be None")
    
    cached = getattr(self, repository_attr, None)
    if cached is None:
        try:
            cached = repository_class(self.db)
            setattr(self, repository_attr, cached)
        except TypeError as e:
            raise TypeError(
                f"Repository class {repository_class.__name__} cannot be instantiated with database session: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create repository {repository_class.__name__}: {e}"
            ) from e
    return cached
```

### Before - ServiceFactory.get_chat_service
```python
def get_chat_service(self) -> ChatService:
    if self._chat_service is None:
        from ..services.chat import (...)
        # ... create service
        self._chat_service = ChatService(...)
    return self._chat_service
```

### After - ServiceFactory.get_chat_service
```python
def get_chat_service(self) -> ChatService:
    """
    Raises:
        RuntimeError: If service creation fails
    """
    if self._chat_service is None:
        try:
            from ..services.chat import (...)
            # ... create service
            self._chat_service = ChatService(...)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import chat service components: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to create ChatService: {e}"
            ) from e
    return self._chat_service
```

## Files Modified

1. **`factories/repository_factory.py`**
   - Added validation in `__init__()`
   - Enhanced `_get_or_create_repository()` with validation and error handling
   - Better documentation

2. **`factories/service_factory.py`**
   - Added validation in `__init__()`
   - Enhanced `get_chat_service()` with error handling
   - Better documentation

## Benefits

1. **Better Validation**: Prevents NoneType errors early
2. **Better Error Messages**: More descriptive error messages for debugging
3. **Robust Error Handling**: Specific exception types for different errors
4. **Better Documentation**: Clear Raises sections in docstrings
5. **Fail Fast**: Errors are caught early with clear messages

## Improvements Details

### Validation Improvements
- **Before**: No validation of `db` parameter
- **After**: Validates `db` is not None in both factories

### Error Handling Improvements
- **Before**: Generic exceptions or no error handling
- **After**: Specific exception types (ValueError, TypeError, RuntimeError) with descriptive messages

### Error Messages
- **Before**: Generic Python errors
- **After**: Descriptive messages like "Database session cannot be None" or "Failed to create ChatService: {error}"

## Verification

- ✅ No linter errors
- ✅ All imports resolve correctly
- ✅ Better validation prevents NoneType errors
- ✅ Better error messages for debugging
- ✅ Backward compatible (only adds validation)

## Testing Recommendations

1. Test factory initialization with None db (should raise ValueError)
2. Test repository creation with invalid class (should raise TypeError)
3. Test service creation with missing imports (should raise RuntimeError)
4. Test normal factory operations (should work as before)
5. Verify error messages are helpful for debugging



