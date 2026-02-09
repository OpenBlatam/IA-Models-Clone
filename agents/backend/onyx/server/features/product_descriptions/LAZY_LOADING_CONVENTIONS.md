# Lazy Loading System Conventions & Best Practices

## Table of Contents

1. [Naming Conventions](#naming-conventions)
2. [Code Structure](#code-structure)
3. [Error Handling](#error-handling)
4. [Performance Conventions](#performance-conventions)
5. [Documentation Standards](#documentation-standards)
6. [Testing Conventions](#testing-conventions)
7. [Configuration Standards](#configuration-standards)
8. [Integration Patterns](#integration-patterns)

## Naming Conventions

### Class Names

**✅ Good:**
```python
class LazyLoadingManager:      # Clear, descriptive
class OnDemandLoader:          # Strategy-specific
class PaginatedLoader:         # Strategy-specific
class LazyCache:               # Purpose-specific
class MemoryManager:           # Responsibility-specific
```

**❌ Avoid:**
```python
class Manager:                 # Too generic
class Loader:                  # Too generic
class Cache:                   # Too generic
class Lazy:                    # Too short
```

### Method Names

**✅ Good:**
```python
async def get_item(self, key: Any) -> T:           # Action-oriented
async def load_batch(self, keys: List[Any]) -> List[T]:  # Clear action
async def start_streaming(self) -> None:           # Command pattern
async def stop_streaming(self) -> None:            # Command pattern
def get_memory_usage(self) -> Dict[str, Any]:      # Descriptive
def can_allocate(self, size: int) -> bool:         # Boolean convention
```

**❌ Avoid:**
```python
async def item(self, key: Any) -> T:               # Not descriptive
async def batch(self, keys: List[Any]) -> List[T]: # Not descriptive
async def stream(self) -> None:                    # Ambiguous
def memory(self) -> Dict[str, Any]:                # Not descriptive
def allocate(self, size: int) -> bool:             # Should be can_allocate
```

### Variable Names

**✅ Good:**
```python
self._cache: Dict[Any, Tuple[LazyItem, float]]     # Type hint included
self._access_order: deque                          # Clear purpose
self._streaming: bool                              # Boolean flag
self._background_task: Optional[asyncio.Task]      # Optional type
self._window_start: int                            # Descriptive
self._window_end: int                              # Descriptive
```

**❌ Avoid:**
```python
self.cache: Dict                                   # No type hint
self.order: deque                                  # Unclear purpose
self.stream: bool                                  # Ambiguous
self.task: asyncio.Task                           # No optional hint
self.start: int                                   # Too generic
self.end: int                                     # Too generic
```

### Configuration Names

**✅ Good:**
```python
LazyLoadingConfig(
    strategy=LoadingStrategy.ON_DEMAND,
    batch_size=100,
    max_memory=1024 * 1024 * 100,
    prefetch_size=50,
    window_size=200,
    cache_ttl=300,
    enable_monitoring=True,
    enable_cleanup=True,
    cleanup_interval=60,
    retry_attempts=3,
    retry_delay=1.0
)
```

**❌ Avoid:**
```python
LazyLoadingConfig(
    strat=LoadingStrategy.ON_DEMAND,    # Abbreviated
    batch=100,                          # Too short
    memory=1024 * 1024 * 100,          # Unclear unit
    prefetch=50,                        # Too short
    window=200,                         # Too short
    ttl=300,                           # Abbreviated
    monitor=True,                      # Abbreviated
    cleanup=True,                      # Abbreviated
    interval=60,                       # Unclear purpose
    retries=3,                         # Abbreviated
    delay=1.0                          # Unclear purpose
)
```

## Code Structure

### Class Organization

**✅ Good Structure:**
```python
class LazyLoader(Generic[T]):
    """Base lazy loader class with common functionality."""
    
    # 1. Class-level constants
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_CACHE_TTL = 300
    
    # 2. Type hints and class variables
    T = TypeVar('T')
    
    # 3. __init__ method
    def __init__(self, config: LazyLoadingConfig):
        self.config = config
        self.cache = LazyCache(config.batch_size, config.cache_ttl)
        self.stats = LoadingStats()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        if config.enable_cleanup:
            self._start_cleanup_task()
    
    # 4. Public methods (alphabetical order)
    async def close(self) -> None:
        """Close the lazy loader and cleanup resources."""
        pass
    
    async def get_item(self, key: Any) -> T:
        """Get item by key."""
        pass
    
    # 5. Protected methods (alphabetical order)
    async def _cleanup(self) -> None:
        """Cleanup expired items."""
        pass
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        pass
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        pass
    
    # 6. Private methods (alphabetical order)
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.create_task(self.close())
```

### Method Organization

**✅ Good Method Structure:**
```python
async def get_item(self, key: Any) -> T:
    """
    Get item by key with caching and error handling.
    
    Args:
        key: The key to lookup
        
    Returns:
        The item associated with the key
        
    Raises:
        ItemNotFoundError: If item doesn't exist
        LoadingError: If loading fails
    """
    # 1. Input validation
    if key is None:
        raise ValueError("Key cannot be None")
    
    # 2. Early returns
    if key in self._local_cache:
        return self._local_cache[key]
    
    # 3. Main logic
    try:
        # Check cache first
        cached_item = self.cache.get(key)
        if cached_item:
            self.stats.cache_hits += 1
            return await cached_item.get_value()
        
        self.stats.cache_misses += 1
        
        # Load from data source
        value = await self._load_from_source(key)
        
        # Cache the result
        self.cache.set(key, LazyItem(key, lambda: value, self.config))
        
        return value
        
    except Exception as e:
        # 4. Error handling
        self.stats.errors += 1
        logger.error(f"Failed to load item {key}: {e}")
        raise LoadingError(f"Failed to load item {key}") from e
    
    finally:
        # 5. Cleanup (if needed)
        self._update_stats()
```

### Import Organization

**✅ Good Import Structure:**
```python
# 1. Standard library imports (alphabetical)
import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Dict, Generic, Iterator, List, Optional, Set, TypeVar, Union
from typing_extensions import Protocol

# 2. Third-party imports (alphabetical)
import aiofiles
from pydantic import BaseModel, Field

# 3. Local imports (alphabetical)
from .exceptions import LoadingError, ItemNotFoundError
from .models import LazyItem, LoadingStats
from .utils import timing_decorator, memory_tracker
```

## Error Handling

### Exception Hierarchy

**✅ Good Exception Structure:**
```python
class LazyLoadingError(Exception):
    """Base exception for lazy loading errors."""
    pass

class LoadingError(LazyLoadingError):
    """Raised when loading fails."""
    pass

class ItemNotFoundError(LazyLoadingError):
    """Raised when item is not found."""
    pass

class MemoryError(LazyLoadingError):
    """Raised when memory allocation fails."""
    pass

class ConfigurationError(LazyLoadingError):
    """Raised when configuration is invalid."""
    pass
```

### Error Handling Patterns

**✅ Good Error Handling:**
```python
async def get_item(self, key: Any) -> T:
    """Get item with comprehensive error handling."""
    try:
        # Validate input
        if not self._is_valid_key(key):
            raise ValueError(f"Invalid key: {key}")
        
        # Check cache
        cached_item = self.cache.get(key)
        if cached_item:
            return await cached_item.get_value()
        
        # Load from source
        value = await self._load_from_source(key)
        if value is None:
            raise ItemNotFoundError(f"Item not found: {key}")
        
        # Cache result
        self.cache.set(key, LazyItem(key, lambda: value, self.config))
        return value
        
    except ValueError as e:
        # Input validation errors
        logger.warning(f"Invalid input: {e}")
        raise
        
    except ItemNotFoundError:
        # Item not found - don't log, just re-raise
        raise
        
    except asyncio.TimeoutError as e:
        # Timeout errors
        logger.error(f"Timeout loading item {key}: {e}")
        raise LoadingError(f"Timeout loading item {key}") from e
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error loading item {key}: {e}", exc_info=True)
        raise LoadingError(f"Failed to load item {key}") from e
```

### Logging Conventions

**✅ Good Logging:**
```python
import logging

logger = logging.getLogger(__name__)

class LazyLoader:
    def __init__(self, config: LazyLoadingConfig):
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    async def get_item(self, key: Any) -> T:
        logger.debug(f"Loading item: {key}")
        try:
            # ... loading logic
            logger.debug(f"Successfully loaded item: {key}")
            return value
        except Exception as e:
            logger.error(f"Failed to load item {key}: {e}", exc_info=True)
            raise
```

**❌ Avoid:**
```python
# Don't use print statements
print(f"Loading item: {key}")

# Don't log sensitive information
logger.info(f"User password: {password}")

# Don't use generic exception handling without logging
try:
    # ... code
except Exception:
    pass
```

## Performance Conventions

### Async/Await Patterns

**✅ Good Async Patterns:**
```python
class LazyLoader:
    async def get_item(self, key: Any) -> T:
        """Async method with proper error handling."""
        return await self._load_item_async(key)
    
    async def get_batch(self, keys: List[Any]) -> List[T]:
        """Async batch loading with concurrency."""
        tasks = [self.get_item(key) for key in keys]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _load_item_async(self, key: Any) -> T:
        """Private async method."""
        async with self._semaphore:  # Rate limiting
            return await self._data_source.get_item(key)
```

**❌ Avoid:**
```python
# Don't block in async methods
def get_item(self, key: Any) -> T:
    return self._data_source.get_item(key)  # Blocking call

# Don't use asyncio.run in async context
async def get_item(self, key: Any) -> T:
    return asyncio.run(self._data_source.get_item(key))  # Wrong!
```

### Caching Patterns

**✅ Good Caching:**
```python
class LazyCache:
    def get(self, key: Any) -> Optional[LazyItem]:
        """Get item with TTL check."""
        if key in self._cache:
            item, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                self._remove(key)
                return None
            
            # Update access order
            self._update_access_order(key)
            return item
        
        return None
    
    def set(self, key: Any, item: LazyItem) -> None:
        """Set item with eviction policy."""
        # Evict if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = (item, time.time())
        self._update_access_order(key)
```

### Memory Management

**✅ Good Memory Management:**
```python
class MemoryManager:
    def allocate(self, key: Any, size: int) -> bool:
        """Allocate memory with bounds checking."""
        if not self.can_allocate(size):
            return False
        
        self._items[key] = size
        self.current_memory += size
        
        # Log memory usage
        if self.current_memory > self.max_memory * 0.8:
            logger.warning(f"Memory usage high: {self.current_memory}/{self.max_memory}")
        
        return True
    
    def deallocate(self, key: Any) -> None:
        """Deallocate memory safely."""
        if key in self._items:
            self.current_memory -= self._items[key]
            del self._items[key]
```

## Documentation Standards

### Docstring Conventions

**✅ Good Docstrings:**
```python
class LazyLoader(Generic[T]):
    """
    Base lazy loader class providing common functionality.
    
    This class implements the base lazy loading pattern with caching,
    memory management, and performance monitoring.
    
    Attributes:
        config: Configuration for the lazy loader
        cache: LRU cache for loaded items
        stats: Performance statistics
        _cleanup_task: Background cleanup task
    
    Example:
        >>> config = LazyLoadingConfig(strategy=LoadingStrategy.ON_DEMAND)
        >>> loader = LazyLoader(config)
        >>> item = await loader.get_item("key")
    """
    
    async def get_item(self, key: Any) -> T:
        """
        Get item by key with caching and error handling.
        
        This method implements the core lazy loading logic:
        1. Check cache for existing item
        2. Load from data source if not cached
        3. Cache the result for future access
        4. Handle errors gracefully
        
        Args:
            key: The key to lookup. Must be hashable.
            
        Returns:
            The item associated with the key.
            
        Raises:
            ValueError: If key is None or invalid
            ItemNotFoundError: If item doesn't exist in data source
            LoadingError: If loading fails due to network or other issues
            
        Example:
            >>> item = await loader.get_item("product_123")
            >>> print(item.title)
        """
        pass
```

### Type Hints

**✅ Good Type Hints:**
```python
from typing import Any, Dict, List, Optional, TypeVar, Union
from typing_extensions import Protocol

T = TypeVar('T')
K = TypeVar('K')

class DataSource(Protocol[T]):
    """Protocol for data sources."""
    
    async def get_item(self, key: Any) -> Optional[T]:
        """Get single item."""
        ...
    
    async def get_batch(self, keys: List[Any]) -> List[T]:
        """Get batch of items."""
        ...

class LazyLoader(Generic[T]):
    def __init__(self, data_source: DataSource[T], config: LazyLoadingConfig) -> None:
        self.data_source: DataSource[T] = data_source
        self.config: LazyLoadingConfig = config
        self._cache: Dict[Any, LazyItem[T]] = {}
    
    async def get_item(self, key: Any) -> T:
        """Get item by key."""
        pass
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics."""
        pass
```

## Testing Conventions

### Test Structure

**✅ Good Test Structure:**
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from lazy_loading_manager import LazyLoader, LazyLoadingConfig, LoadingStrategy

class TestLazyLoader:
    """Test suite for LazyLoader class."""
    
    @pytest.fixture
    def mock_data_source(self):
        """Create mock data source for testing."""
        source = AsyncMock()
        source.get_item.return_value = {"id": "test", "name": "Test Item"}
        return source
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LazyLoadingConfig(
            strategy=LoadingStrategy.ON_DEMAND,
            batch_size=50,
            cache_ttl=300
        )
    
    @pytest.fixture
    def loader(self, mock_data_source, config):
        """Create test loader instance."""
        return LazyLoader(mock_data_source, config)
    
    @pytest.mark.asyncio
    async def test_get_item_success(self, loader, mock_data_source):
        """Test successful item retrieval."""
        # Arrange
        key = "test_key"
        expected_item = {"id": "test", "name": "Test Item"}
        
        # Act
        result = await loader.get_item(key)
        
        # Assert
        assert result == expected_item
        mock_data_source.get_item.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_get_item_cache_hit(self, loader, mock_data_source):
        """Test cache hit behavior."""
        # Arrange
        key = "test_key"
        
        # Act - First call
        await loader.get_item(key)
        # Act - Second call (should hit cache)
        await loader.get_item(key)
        
        # Assert - Data source should only be called once
        mock_data_source.get_item.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_get_item_not_found(self, loader, mock_data_source):
        """Test item not found scenario."""
        # Arrange
        key = "nonexistent_key"
        mock_data_source.get_item.return_value = None
        
        # Act & Assert
        with pytest.raises(ItemNotFoundError):
            await loader.get_item(key)
```

### Test Naming

**✅ Good Test Names:**
```python
def test_get_item_success():                    # Clear success case
def test_get_item_cache_hit():                  # Specific behavior
def test_get_item_not_found():                  # Error case
def test_get_item_invalid_key():                # Input validation
def test_get_item_timeout():                    # Timeout scenario
def test_get_item_memory_limit_exceeded():      # Memory constraint
def test_get_item_concurrent_access():          # Concurrency
def test_get_item_cleanup_on_close():           # Resource cleanup
```

**❌ Avoid:**
```python
def test_1():                                   # No description
def test_get_item():                            # Too generic
def test_works():                               # Unclear what works
def test_bug_fix():                             # Not descriptive
```

## Configuration Standards

### Configuration Validation

**✅ Good Configuration:**
```python
from pydantic import BaseModel, Field, validator
from typing import Optional

class LazyLoadingConfig(BaseModel):
    """Configuration for lazy loading with validation."""
    
    strategy: LoadingStrategy = Field(
        default=LoadingStrategy.ON_DEMAND,
        description="Loading strategy to use"
    )
    
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Batch size for operations"
    )
    
    max_memory: int = Field(
        default=1024 * 1024 * 100,  # 100MB
        ge=1024 * 1024,  # At least 1MB
        description="Maximum memory usage in bytes"
    )
    
    cache_ttl: int = Field(
        default=300,
        ge=1,
        description="Cache TTL in seconds"
    )
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size is reasonable."""
        if v > 10000:
            raise ValueError("Batch size too large")
        return v
    
    @validator('max_memory')
    def validate_memory(cls, v):
        """Validate memory limit is reasonable."""
        if v > 1024 * 1024 * 1024:  # 1GB
            raise ValueError("Memory limit too high")
        return v
```

### Environment Configuration

**✅ Good Environment Config:**
```python
import os
from typing import Optional

class EnvironmentConfig:
    """Environment-specific configuration."""
    
    @staticmethod
    def get_lazy_loading_config() -> LazyLoadingConfig:
        """Get configuration from environment variables."""
        return LazyLoadingConfig(
            strategy=LoadingStrategy(os.getenv("LAZY_LOADING_STRATEGY", "on_demand")),
            batch_size=int(os.getenv("LAZY_LOADING_BATCH_SIZE", "100")),
            max_memory=int(os.getenv("LAZY_LOADING_MAX_MEMORY", str(1024 * 1024 * 100))),
            cache_ttl=int(os.getenv("LAZY_LOADING_CACHE_TTL", "300")),
            enable_monitoring=os.getenv("LAZY_LOADING_MONITORING", "true").lower() == "true",
            enable_cleanup=os.getenv("LAZY_LOADING_CLEANUP", "true").lower() == "true"
        )
```

## Integration Patterns

### FastAPI Integration

**✅ Good FastAPI Integration:**
```python
from fastapi import FastAPI, HTTPException, Depends
from typing import Optional

app = FastAPI()

# Dependency injection
def get_lazy_loader() -> LazyLoader:
    """Get lazy loader instance."""
    config = EnvironmentConfig.get_lazy_loading_config()
    return LazyLoader(data_source, config)

@app.get("/items/{item_id}")
async def get_item(
    item_id: str,
    loader: LazyLoader = Depends(get_lazy_loader)
):
    """Get item by ID with lazy loading."""
    try:
        item = await loader.get_item(item_id)
        return {"item": item, "cached": loader.stats.cache_hits > 0}
    except ItemNotFoundError:
        raise HTTPException(status_code=404, detail="Item not found")
    except LoadingError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(loader: LazyLoader = Depends(get_lazy_loader)):
    """Get lazy loading statistics."""
    return loader.get_stats()
```

### Service Layer Integration

**✅ Good Service Integration:**
```python
class ProductService:
    """Service layer with lazy loading."""
    
    def __init__(self):
        self.lazy_manager = get_lazy_loading_manager()
        self.product_loader = self.lazy_manager.get_loader("products")
    
    async def get_product(self, product_id: str) -> Product:
        """Get product with lazy loading."""
        try:
            data = await self.product_loader.get_item(product_id)
            return Product(**data)
        except ItemNotFoundError:
            raise ProductNotFoundError(f"Product {product_id} not found")
    
    async def get_products_batch(self, product_ids: List[str]) -> List[Product]:
        """Get multiple products with background loading."""
        loader = BackgroundLoader(self.data_source, self.config)
        await loader.start_background_loading(product_ids)
        
        products = []
        for product_id in product_ids:
            try:
                data = await loader.get_item(product_id)
                products.append(Product(**data))
            except Exception as e:
                logger.warning(f"Failed to load product {product_id}: {e}")
        
        return products
```

## Summary

These conventions ensure:

1. **Consistency**: All code follows the same patterns and naming
2. **Readability**: Clear, descriptive names and structure
3. **Maintainability**: Well-organized code that's easy to modify
4. **Reliability**: Proper error handling and validation
5. **Performance**: Optimized patterns and best practices
6. **Testability**: Code that's easy to test and verify

Following these conventions will make the lazy loading system more professional, maintainable, and reliable. 