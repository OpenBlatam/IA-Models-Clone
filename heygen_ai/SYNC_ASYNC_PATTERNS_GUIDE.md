# Synchronous and Asynchronous Patterns Guide

A comprehensive guide for implementing proper synchronous and asynchronous operations in FastAPI applications with clear guidelines on when to use `def` vs `async def`.

## 🚀 Overview

This guide covers:
- **When to use `def`**: Synchronous operations and pure functions
- **When to use `async def`**: Asynchronous operations and I/O-bound tasks
- **Performance Considerations**: CPU-bound vs I/O-bound operations
- **Patterns and Best Practices**: Proper sync/async usage patterns
- **Error Handling**: Error handling for both sync and async operations
- **Integration**: How to mix sync and async code effectively

## 📋 Table of Contents

1. [When to Use `def` (Synchronous)](#when-to-use-def-synchronous)
2. [When to Use `async def` (Asynchronous)](#when-to-use-async-def-asynchronous)
3. [Performance Considerations](#performance-considerations)
4. [Patterns and Best Practices](#patterns-and-best-practices)
5. [Error Handling](#error-handling)
6. [Integration Patterns](#integration-patterns)
7. [Testing Sync/Async Code](#testing-syncasync-code)
8. [Common Anti-Patterns](#common-anti-patterns)

## 🎯 When to Use `def` (Synchronous)

### Pure Functions and Data Processing

```python
def validate_username_sync(username: str) -> str:
    """
    Synchronous username validation.
    
    Use def for:
    - Pure functions with no I/O
    - CPU-bound operations
    - Simple data validation
    - Utility functions
    """
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters long")
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        raise ValueError("Username can only contain letters, numbers, and underscores")
    return username.lower()

def validate_password_strength_sync(password: str) -> str:
    """Synchronous password strength validation."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    if not re.search(r'[A-Z]', password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not re.search(r'[a-z]', password):
        raise ValueError("Password must contain at least one lowercase letter")
    if not re.search(r'\d', password):
        raise ValueError("Password must contain at least one digit")
    return password

def hash_password_sync(password: str) -> str:
    """Synchronous password hashing."""
    return hashlib.sha256(password.encode()).hexdigest()

def calculate_processing_efficiency_sync(processing_time: float, file_size: int) -> float:
    """Synchronous efficiency calculation."""
    if processing_time <= 0 or file_size <= 0:
        return 0.0
    return min(file_size / processing_time, 100.0)
```

### CPU-Bound Operations

```python
def cpu_bound_operation_sync(data: List[int]) -> List[int]:
    """
    CPU-bound synchronous operation.
    
    Use def for CPU-intensive operations that don't involve I/O.
    """
    return [x * x for x in data]

def complex_calculation_sync(n: int) -> int:
    """Complex mathematical calculation."""
    result = 0
    for i in range(n):
        result += i * i
    return result

@lru_cache(maxsize=128)
def expensive_calculation_sync(n: int) -> int:
    """
    Expensive synchronous calculation with caching.
    
    Use def with @lru_cache for expensive pure functions.
    """
    # Simulate expensive calculation
    time.sleep(0.1)
    return n * n
```

### Data Transformation and Validation

```python
def transform_user_data_sync(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous data transformation."""
    return {
        'id': user_data.get('id'),
        'username': user_data.get('username', '').lower(),
        'email': user_data.get('email', '').lower(),
        'created_at': user_data.get('created_at'),
        'display_name': user_data.get('full_name') or user_data.get('username')
    }

def validate_video_data_sync(video_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Synchronous video data validation."""
    try:
        # Validate required fields
        required_fields = ['script', 'voice_id']
        for field in required_fields:
            if not video_data.get(field):
                return False, f"Missing required field: {field}"
        
        # Validate script length
        script = video_data['script'].strip()
        if not script:
            return False, "Script cannot be empty"
        if len(script) > 1000:
            return False, "Script too long (max 1000 characters)"
        
        return True, None
    except Exception as e:
        return False, str(e)
```

## ⚡ When to Use `async def` (Asynchronous)

### Database Operations

```python
async def create_user_async(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronous user creation.
    
    Use async def for database operations.
    """
    try:
        # Simulate async database operation
        await asyncio.sleep(0.2)  # Simulate database write
        
        # Hash password
        hashed_password = hash_password_sync(user_data['password'])
        
        # Generate API key
        api_key = generate_api_key_sync()
        
        # Create user object
        user = {
            **user_data,
            'hashed_password': hashed_password,
            'api_key': api_key,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
        
        logger.info("User created asynchronously", username=user_data.get('username'))
        return user
        
    except Exception as e:
        logger.error("Error creating user asynchronously", error=str(e))
        raise

async def fetch_user_data_async(user_id: int) -> Optional[Dict[str, Any]]:
    """Asynchronous user data fetching."""
    try:
        # Simulate async database query
        await asyncio.sleep(0.1)  # Simulate database read
        
        # Simulate user data
        user_data = {
            'id': user_id,
            'username': f'user_{user_id}',
            'email': f'user_{user_id}@example.com',
            'created_at': datetime.now(timezone.utc)
        }
        
        return user_data
        
    except Exception as e:
        logger.error("Error fetching user data asynchronously", user_id=user_id, error=str(e))
        return None
```

### External API Calls

```python
async def send_notification_async(user_id: int, message: str) -> bool:
    """
    Asynchronous notification sending.
    
    Use async def for external API calls.
    """
    try:
        # Simulate async notification service
        await asyncio.sleep(0.3)  # Simulate network request
        
        logger.info("Notification sent asynchronously", user_id=user_id, message=message)
        return True
        
    except Exception as e:
        logger.error("Error sending notification asynchronously", user_id=user_id, error=str(e))
        return False

async def call_external_api_async(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Asynchronous external API call."""
    try:
        # Simulate async HTTP request
        await asyncio.sleep(0.5)  # Simulate network delay
        
        # Simulate API response
        response = {
            'status': 'success',
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error("Error calling external API", url=url, error=str(e))
        raise
```

### File I/O Operations

```python
async def read_file_async(file_path: str) -> str:
    """Asynchronous file reading."""
    try:
        # Simulate async file read
        await asyncio.sleep(0.1)  # Simulate I/O operation
        
        with open(file_path, 'r') as file:
            content = file.read()
        
        return content
        
    except Exception as e:
        logger.error("Error reading file asynchronously", file_path=file_path, error=str(e))
        raise

async def write_file_async(file_path: str, content: str) -> bool:
    """Asynchronous file writing."""
    try:
        # Simulate async file write
        await asyncio.sleep(0.1)  # Simulate I/O operation
        
        with open(file_path, 'w') as file:
            file.write(content)
        
        return True
        
    except Exception as e:
        logger.error("Error writing file asynchronously", file_path=file_path, error=str(e))
        return False
```

### Parallel Operations

```python
async def process_multiple_users_async(user_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Asynchronous parallel user processing.
    
    Use async def for parallel operations.
    """
    # Create tasks for all user IDs
    tasks = [fetch_user_data_async(user_id) for user_id in user_ids]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and collect results
    users = []
    for result in results:
        if isinstance(result, Exception):
            logger.error("Error processing user", error=str(result))
        elif result is not None:
            users.append(result)
    
    return users

async def parallel_io_operations_async(urls: List[str]) -> List[str]:
    """Parallel I/O operations."""
    async def fetch_url(url: str) -> str:
        # Simulate async HTTP request
        await asyncio.sleep(0.1)
        return f"Response from {url}"
    
    # Create tasks for all URLs
    tasks = [fetch_url(url) for url in urls]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    return [result for result in results if not isinstance(result, Exception)]
```

## ⚡ Performance Considerations

### CPU-Bound vs I/O-Bound Operations

```python
# CPU-Bound Operations (use def)
def cpu_intensive_calculation(data: List[int]) -> List[int]:
    """CPU-bound operation - use def."""
    return [x * x for x in data]

def complex_algorithm(data: List[float]) -> float:
    """Complex algorithm - use def."""
    result = 0.0
    for i, value in enumerate(data):
        result += value * (i + 1) ** 2
    return result

# I/O-Bound Operations (use async def)
async def database_query_async(query: str) -> List[Dict[str, Any]]:
    """I/O-bound operation - use async def."""
    await asyncio.sleep(0.1)  # Simulate database query
    return [{'id': 1, 'name': 'test'}]

async def http_request_async(url: str) -> Dict[str, Any]:
    """I/O-bound operation - use async def."""
    await asyncio.sleep(0.2)  # Simulate HTTP request
    return {'status': 'success', 'data': 'response'}
```

### When to Use ThreadPoolExecutor

```python
async def cpu_bound_in_async_context(data: List[int]) -> List[int]:
    """
    CPU-bound operation in async context.
    
    Use ThreadPoolExecutor for CPU-bound operations in async functions.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # Use default executor
        cpu_intensive_calculation,  # Sync function
        data
    )

async def process_large_dataset_async(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process large dataset with mixed sync/async operations."""
    processed_data = []
    
    # Process items in chunks to avoid blocking
    chunk_size = 100
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        
        # CPU-bound processing in thread pool
        processed_chunk = await asyncio.get_event_loop().run_in_executor(
            None,
            process_chunk_sync,  # Sync function
            chunk
        )
        
        # Async I/O operations
        for item in processed_chunk:
            # Send notification asynchronously
            asyncio.create_task(send_notification_async(item['id'], "Processed"))
        
        processed_data.extend(processed_chunk)
    
    return processed_data

def process_chunk_sync(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Synchronous chunk processing."""
    return [transform_item_sync(item) for item in chunk]

def transform_item_sync(item: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous item transformation."""
    return {
        **item,
        'processed_at': datetime.now(timezone.utc),
        'hash': hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
    }
```

## 🛠️ Patterns and Best Practices

### Service Layer Patterns

```python
class UserService:
    """Service class demonstrating sync/async patterns."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def validate_user_sync(self, user_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Synchronous user validation.
        
        Use def for validation that doesn't require I/O.
        """
        try:
            # Validate required fields
            required_fields = ['username', 'email', 'password']
            for field in required_fields:
                if not user_data.get(field):
                    return False, f"Missing required field: {field}"
            
            # Validate username
            username = validate_username_sync(user_data['username'])
            
            # Validate email
            email = validate_email_format_sync(user_data['email'])
            
            # Validate password
            password = validate_password_strength_sync(user_data['password'])
            
            return True, None
            
        except ValueError as e:
            return False, str(e)
    
    async def validate_user_async(self, user_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Asynchronous user validation.
        
        Use async def for validation that requires I/O.
        """
        try:
            # Basic sync validation
            is_valid, error = self.validate_user_sync(user_data)
            if not is_valid:
                return False, error
            
            # Async validation (e.g., checking against database)
            await asyncio.sleep(0.1)  # Simulate async operation
            
            # Check if username already exists
            # existing_user = await self.get_user_by_username_async(user_data['username'])
            # if existing_user:
            #     return False, "Username already exists"
            
            return True, None
            
        except Exception as e:
            logger.error("Error in async user validation", error=str(e))
            return False, "Validation error occurred"
    
    def create_user_sync(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous user creation.
        
        Use def when you need to block until operation completes.
        """
        # Validate user data
        is_valid, error = self.validate_user_sync(user_data)
        if not is_valid:
            raise ValueError(error)
        
        # Hash password
        hashed_password = hash_password_sync(user_data['password'])
        
        # Generate API key
        api_key = generate_api_key_sync()
        
        # Create user object
        user = {
            **user_data,
            'hashed_password': hashed_password,
            'api_key': api_key,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
        
        logger.info("User created synchronously", username=user_data.get('username'))
        return user
    
    async def create_user_async(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous user creation.
        
        Use async def for operations that involve I/O.
        """
        # Validate user data asynchronously
        is_valid, error = await self.validate_user_async(user_data)
        if not is_valid:
            raise ValueError(error)
        
        # Create user asynchronously
        user = await create_user_async(user_data)
        
        # Send notification asynchronously (fire and forget)
        asyncio.create_task(
            send_notification_async(user['id'], "Welcome to HeyGen AI!")
        )
        
        return user
```

### Utility Function Patterns

```python
class SyncAsyncUtils:
    """Utility class demonstrating proper sync/async usage."""
    
    @staticmethod
    def validate_input_sync(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Synchronous input validation.
        
        Use def for pure validation logic.
        """
        try:
            # Validate required fields
            required_fields = ['username', 'email', 'password']
            for field in required_fields:
                if not data.get(field):
                    return False, f"Missing required field: {field}"
            
            # Validate username
            username = validate_username_sync(data['username'])
            
            # Validate email
            email = validate_email_format_sync(data['email'])
            
            # Validate password
            password = validate_password_strength_sync(data['password'])
            
            return True, None
            
        except ValueError as e:
            return False, str(e)
    
    @staticmethod
    async def validate_input_async(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Asynchronous input validation.
        
        Use async def when validation involves I/O operations.
        """
        try:
            # Basic sync validation
            is_valid, error = SyncAsyncUtils.validate_input_sync(data)
            if not is_valid:
                return False, error
            
            # Async validation (e.g., checking against database)
            await asyncio.sleep(0.1)  # Simulate async operation
            
            return True, None
            
        except Exception as e:
            logger.error("Error in async input validation", error=str(e))
            return False, "Validation error occurred"
    
    @staticmethod
    def process_data_sync(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Synchronous data processing.
        
        Use def for CPU-bound data processing.
        """
        processed_data = []
        for item in data:
            # Process each item synchronously
            processed_item = {
                **item,
                'processed_at': datetime.now(timezone.utc),
                'hash': hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    @staticmethod
    async def process_data_async(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Asynchronous data processing.
        
        Use async def when processing involves I/O operations.
        """
        processed_data = []
        
        # Process items concurrently
        tasks = []
        for item in data:
            task = SyncAsyncUtils._process_item_async(item)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and collect results
        for result in results:
            if isinstance(result, Exception):
                logger.error("Error processing item", error=str(result))
            else:
                processed_data.append(result)
        
        return processed_data
    
    @staticmethod
    async def _process_item_async(item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous item processing.
        
        Private method for processing individual items.
        """
        try:
            # Simulate async processing
            await asyncio.sleep(0.1)
            
            processed_item = {
                **item,
                'processed_at': datetime.now(timezone.utc),
                'hash': hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
            }
            
            return processed_item
            
        except Exception as e:
            logger.error("Error processing item asynchronously", error=str(e))
            raise
```

## 🛡️ Error Handling

### Synchronous Error Handling

```python
def safe_sync_operation(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """
    Decorator for safe synchronous operations.
    
    Args:
        func: Synchronous function to wrap
        
    Returns:
        Wrapped function that returns None on error
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in sync operation {func.__name__}", error=str(e))
            return None
    return wrapper

@safe_sync_operation
def risky_sync_operation(data: str) -> str:
    """Risky synchronous operation with error handling."""
    if not data:
        raise ValueError("Data cannot be empty")
    return data.upper()

# Usage
result = risky_sync_operation("hello")  # Returns "HELLO"
result = risky_sync_operation("")       # Returns None, logs error
```

### Asynchronous Error Handling

```python
def safe_async_operation(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[Optional[T]]]:
    """
    Decorator for safe asynchronous operations.
    
    Args:
        func: Asynchronous function to wrap
        
    Returns:
        Wrapped function that returns None on error
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in async operation {func.__name__}", error=str(e))
            return None
    return wrapper

@safe_async_operation
async def risky_async_operation(data: str) -> str:
    """Risky asynchronous operation with error handling."""
    await asyncio.sleep(0.1)  # Simulate async operation
    if not data:
        raise ValueError("Data cannot be empty")
    return data.upper()

# Usage
result = await risky_async_operation("hello")  # Returns "HELLO"
result = await risky_async_operation("")       # Returns None, logs error
```

### Error Handling in Service Classes

```python
class ErrorHandler:
    """Error handling patterns for sync/async operations."""
    
    @staticmethod
    def handle_sync_error(func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for synchronous error handling."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                logger.warning(f"Validation error in {func.__name__}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise
        return wrapper
    
    @staticmethod
    def handle_async_error(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator for asynchronous error handling."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ValueError as e:
                logger.warning(f"Validation error in {func.__name__}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                raise
        return wrapper

# Usage
@ErrorHandler.handle_sync_error
def validate_user_sync(user_data: Dict[str, Any]) -> bool:
    """Synchronous user validation with error handling."""
    if not user_data.get('username'):
        raise ValueError("Username is required")
    return True

@ErrorHandler.handle_async_error
async def create_user_async(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Asynchronous user creation with error handling."""
    await asyncio.sleep(0.1)  # Simulate async operation
    if not user_data.get('username'):
        raise ValueError("Username is required")
    return user_data
```

## 🔄 Integration Patterns

### Converting Between Sync and Async

```python
def sync_to_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to convert synchronous function to asynchronous.
    
    Use when you need to call a sync function from async context.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper

def async_to_sync(func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """
    Decorator to convert asynchronous function to synchronous.
    
    Use when you need to call an async function from sync context.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

# Usage examples
@sync_to_async
def cpu_intensive_sync(data: List[int]) -> List[int]:
    """CPU-intensive synchronous function."""
    return [x * x for x in data]

@async_to_sync
async def io_intensive_async(url: str) -> str:
    """I/O-intensive asynchronous function."""
    await asyncio.sleep(0.1)
    return f"Response from {url}"

# Now you can use them in different contexts
async def async_context():
    result = await cpu_intensive_sync([1, 2, 3])  # Sync function in async context
    return result

def sync_context():
    result = io_intensive_async("https://example.com")  # Async function in sync context
    return result
```

### Mixed Sync/Async Operations

```python
class MixedOperations:
    """Class demonstrating mixed sync/async operations."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def process_data_sync(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous data processing."""
        return [self._transform_item_sync(item) for item in data]
    
    def _transform_item_sync(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous item transformation."""
        return {
            **item,
            'processed_at': datetime.now(timezone.utc),
            'hash': hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
        }
    
    async def process_data_async(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Asynchronous data processing with mixed operations."""
        processed_data = []
        
        for item in data:
            # CPU-bound transformation in thread pool
            transformed_item = await asyncio.get_event_loop().run_in_executor(
                None,
                self._transform_item_sync,
                item
            )
            
            # Async I/O operation
            notification_sent = await send_notification_async(
                transformed_item['id'], 
                "Item processed"
            )
            
            if notification_sent:
                transformed_item['notification_sent'] = True
            
            processed_data.append(transformed_item)
        
        return processed_data
    
    async def batch_process_async(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch processing with mixed operations."""
        # Process data in chunks
        chunk_size = 10
        all_results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            # Process chunk asynchronously
            chunk_results = await self.process_data_async(chunk)
            all_results.extend(chunk_results)
            
            # Small delay between chunks
            await asyncio.sleep(0.1)
        
        return all_results
```

## 🧪 Testing Sync/Async Code

### Testing Synchronous Functions

```python
import pytest

class TestSyncOperations:
    """Test synchronous operations."""
    
    def test_validate_username_sync_success(self):
        """Test successful username validation."""
        result = validate_username_sync("john_doe")
        assert result == "john_doe"
    
    def test_validate_username_sync_too_short(self):
        """Test username validation with short username."""
        with pytest.raises(ValueError, match="Username must be at least 3 characters"):
            validate_username_sync("jo")
    
    def test_validate_username_sync_invalid_characters(self):
        """Test username validation with invalid characters."""
        with pytest.raises(ValueError, match="Username can only contain letters"):
            validate_username_sync("john@doe")
    
    def test_hash_password_sync(self):
        """Test password hashing."""
        password = "SecurePass123"
        hashed = hash_password_sync(password)
        
        assert hashed != password
        assert len(hashed) == 64  # SHA-256 hash length
        assert hashed == hash_password_sync(password)  # Same input, same output
    
    def test_calculate_processing_efficiency_sync(self):
        """Test processing efficiency calculation."""
        efficiency = calculate_processing_efficiency_sync(10.0, 1000000)  # 1MB in 10s
        assert efficiency == 100000.0  # 100KB/s
    
    def test_calculate_processing_efficiency_sync_zero_time(self):
        """Test processing efficiency with zero time."""
        efficiency = calculate_processing_efficiency_sync(0.0, 1000000)
        assert efficiency == 0.0
```

### Testing Asynchronous Functions

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestAsyncOperations:
    """Test asynchronous operations."""
    
    @pytest.mark.asyncio
    async def test_validate_user_async_success(self):
        """Test successful async user validation."""
        user_data = {
            'username': 'john_doe',
            'email': 'john@example.com',
            'password': 'SecurePass123'
        }
        
        is_valid, error = await validate_user_async(user_data)
        assert is_valid is True
        assert error is None
    
    @pytest.mark.asyncio
    async def test_validate_user_async_invalid_data(self):
        """Test async user validation with invalid data."""
        user_data = {
            'username': 'jo',  # Too short
            'email': 'invalid-email',
            'password': 'weak'
        }
        
        is_valid, error = await validate_user_async(user_data)
        assert is_valid is False
        assert error is not None
    
    @pytest.mark.asyncio
    async def test_create_user_async(self):
        """Test async user creation."""
        user_data = {
            'username': 'john_doe',
            'email': 'john@example.com',
            'password': 'SecurePass123'
        }
        
        user = await create_user_async(user_data)
        
        assert user['username'] == 'john_doe'
        assert user['email'] == 'john@example.com'
        assert 'hashed_password' in user
        assert 'api_key' in user
        assert 'created_at' in user
    
    @pytest.mark.asyncio
    async def test_fetch_user_data_async_success(self):
        """Test successful async user data fetching."""
        user = await fetch_user_data_async(1)
        
        assert user is not None
        assert user['id'] == 1
        assert user['username'] == 'user_1'
        assert user['email'] == 'user_1@example.com'
    
    @pytest.mark.asyncio
    async def test_fetch_user_data_async_error(self):
        """Test async user data fetching with error."""
        with patch('asyncio.sleep', side_effect=Exception("Database error")):
            user = await fetch_user_data_async(999)
            assert user is None
    
    @pytest.mark.asyncio
    async def test_parallel_operations_async(self):
        """Test parallel async operations."""
        urls = ['https://api1.com', 'https://api2.com', 'https://api3.com']
        
        results = await parallel_io_operations_async(urls)
        
        assert len(results) == 3
        assert all('Response from' in result for result in results)
```

### Testing Mixed Operations

```python
class TestMixedOperations:
    """Test mixed sync/async operations."""
    
    @pytest.mark.asyncio
    async def test_mixed_operations_async(self):
        """Test mixed sync/async operations."""
        mixed_ops = MixedOperations()
        
        data = [
            {'id': 1, 'name': 'Item 1'},
            {'id': 2, 'name': 'Item 2'},
            {'id': 3, 'name': 'Item 3'}
        ]
        
        # Test sync processing
        sync_results = mixed_ops.process_data_sync(data)
        assert len(sync_results) == 3
        assert all('processed_at' in item for item in sync_results)
        assert all('hash' in item for item in sync_results)
        
        # Test async processing
        async_results = await mixed_ops.process_data_async(data)
        assert len(async_results) == 3
        assert all('processed_at' in item for item in async_results)
        assert all('hash' in item for item in async_results)
        assert all('notification_sent' in item for item in async_results)
    
    @pytest.mark.asyncio
    async def test_sync_to_async_conversion(self):
        """Test sync to async conversion."""
        @sync_to_async
        def cpu_intensive_sync(data: List[int]) -> List[int]:
            return [x * x for x in data]
        
        result = await cpu_intensive_sync([1, 2, 3, 4, 5])
        assert result == [1, 4, 9, 16, 25]
    
    @pytest.mark.asyncio
    async def test_async_to_sync_conversion(self):
        """Test async to sync conversion."""
        @async_to_sync
        async def io_intensive_async(url: str) -> str:
            await asyncio.sleep(0.01)  # Small delay for testing
            return f"Response from {url}"
        
        result = io_intensive_async("https://example.com")
        assert result == "Response from https://example.com"
```

## ❌ Common Anti-Patterns

### Don't Use `async def` for CPU-Bound Operations

```python
# ❌ Bad: Using async def for CPU-bound operations
async def cpu_intensive_async(data: List[int]) -> List[int]:
    """CPU-intensive operation - should use def, not async def."""
    return [x * x for x in data]  # No I/O, no benefit from async

# ✅ Good: Using def for CPU-bound operations
def cpu_intensive_sync(data: List[int]) -> List[int]:
    """CPU-intensive operation - use def."""
    return [x * x for x in data]

# ✅ Good: Using async def with ThreadPoolExecutor for CPU-bound in async context
async def cpu_intensive_in_async_context(data: List[int]) -> List[int]:
    """CPU-bound operation in async context."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, cpu_intensive_sync, data)
```

### Don't Use `def` for I/O-Bound Operations

```python
# ❌ Bad: Using def for I/O-bound operations
def database_query_sync(query: str) -> List[Dict[str, Any]]:
    """I/O-bound operation - should use async def, not def."""
    time.sleep(0.1)  # Blocking operation
    return [{'id': 1, 'name': 'test'}]

# ✅ Good: Using async def for I/O-bound operations
async def database_query_async(query: str) -> List[Dict[str, Any]]:
    """I/O-bound operation - use async def."""
    await asyncio.sleep(0.1)  # Non-blocking operation
    return [{'id': 1, 'name': 'test'}]
```

### Don't Block the Event Loop

```python
# ❌ Bad: Blocking the event loop
async def bad_async_function():
    """Bad: Blocking the event loop."""
    time.sleep(1)  # This blocks the entire event loop!
    return "done"

# ✅ Good: Non-blocking async operation
async def good_async_function():
    """Good: Non-blocking async operation."""
    await asyncio.sleep(1)  # This doesn't block the event loop
    return "done"

# ✅ Good: CPU-bound operation in thread pool
async def cpu_bound_async():
    """Good: CPU-bound operation in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, cpu_intensive_sync, [1, 2, 3])
```

### Don't Mix Sync and Async Without Proper Conversion

```python
# ❌ Bad: Calling sync function directly in async context
async def bad_mixed_function():
    """Bad: Calling sync function directly in async context."""
    result = cpu_intensive_sync([1, 2, 3])  # This blocks the event loop!
    return result

# ✅ Good: Using proper conversion
async def good_mixed_function():
    """Good: Using proper conversion."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, cpu_intensive_sync, [1, 2, 3])
    return result

# ✅ Good: Using decorator
@sync_to_async
def cpu_intensive_sync(data: List[int]) -> List[int]:
    return [x * x for x in data]

async def good_mixed_function_with_decorator():
    """Good: Using decorator for conversion."""
    result = await cpu_intensive_sync([1, 2, 3])
    return result
```

## 📚 Additional Resources

- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [FastAPI Async/Await](https://fastapi.tiangolo.com/async/)
- [Python Concurrency Patterns](https://docs.python.org/3/library/concurrent.futures.html)
- [AsyncIO Best Practices](https://docs.python.org/3/library/asyncio-dev.html)

## 🚀 Next Steps

1. **Review existing code** and identify sync/async usage patterns
2. **Refactor CPU-bound operations** to use `def` instead of `async def`
3. **Refactor I/O-bound operations** to use `async def` instead of `def`
4. **Implement proper error handling** for both sync and async operations
5. **Add comprehensive tests** for sync/async code
6. **Use ThreadPoolExecutor** for CPU-bound operations in async context
7. **Follow the patterns** outlined in this guide for consistent code

This sync/async patterns guide provides a comprehensive framework for implementing proper synchronous and asynchronous operations in your HeyGen AI FastAPI application with clear guidelines on when to use `def` vs `async def` for optimal performance and maintainability. 