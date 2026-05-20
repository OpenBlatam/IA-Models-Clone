from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import hashlib
import json
import re
from typing import (
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import structlog
from datetime import datetime, timezone
    import secrets
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Synchronous and Asynchronous Patterns for HeyGen AI API
Clear guidelines for when to use def vs async def with practical examples.
"""

    Any, Dict, List, Optional, Union, Callable, Awaitable,
    TypeVar, Generic, Tuple
)

logger = structlog.get_logger()

# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar('T')
R = TypeVar('R')

# =============================================================================
# Synchronous Operations (def)
# =============================================================================

def validate_username_sync(username: str) -> str:
    """
    Synchronous username validation.
    
    Use def for:
    - Pure functions with no I/O
    - CPU-bound operations
    - Simple data validation
    - Utility functions
    
    Args:
        username: Username to validate
        
    Returns:
        Validated username
        
    Raises:
        ValueError: If username is invalid
    """
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters long")
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        raise ValueError("Username can only contain letters, numbers, and underscores")
    return username.lower()

def validate_password_strength_sync(password: str) -> str:
    """
    Synchronous password strength validation.
    
    Args:
        password: Password to validate
        
    Returns:
        Validated password
        
    Raises:
        ValueError: If password is too weak
    """
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
    """
    Synchronous password hashing.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()

def calculate_processing_efficiency_sync(processing_time: float, file_size: int) -> float:
    """
    Synchronous efficiency calculation.
    
    Args:
        processing_time: Processing time in seconds
        file_size: File size in bytes
        
    Returns:
        Efficiency score (0-100)
    """
    if processing_time <= 0 or file_size <= 0:
        return 0.0
    return min(file_size / processing_time, 100.0)

def format_file_size_sync(size_bytes: int) -> str:
    """
    Synchronous file size formatting.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def validate_email_format_sync(email: str) -> str:
    """
    Synchronous email format validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        Validated email address
        
    Raises:
        ValueError: If email format is invalid
    """
    if not email or '@' not in email:
        raise ValueError("Invalid email format")
    return email.lower()

async def generate_api_key_sync() -> str:
    """
    Synchronous API key generation.
    
    Returns:
        Generated API key
    """
    return secrets.token_urlsafe(32)

def calculate_age_sync(birth_date: datetime) -> int:
    """
    Synchronous age calculation.
    
    Args:
        birth_date: Birth date
        
    Returns:
        Age in days
    """
    return (datetime.now(timezone.utc) - birth_date).days

def calculate_success_rate_sync(successful: int, total: int) -> float:
    """
    Synchronous success rate calculation.
    
    Args:
        successful: Number of successful operations
        total: Total number of operations
        
    Returns:
        Success rate percentage
    """
    if total == 0:
        return 0.0
    return round((successful / total) * 100, 2)

# =============================================================================
# Asynchronous Operations (async def)
# =============================================================================

async def validate_user_async(user_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Asynchronous user validation.
    
    Use async def for:
    - Database operations
    - Network requests
    - File I/O operations
    - External API calls
    - Operations that involve waiting
    
    Args:
        user_data: User data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Simulate async validation (e.g., checking against database)
        await asyncio.sleep(0.1)  # Simulate async operation
        
        # Validate username
        username = user_data.get('username', '')
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        # Validate email
        email = user_data.get('email', '')
        if not email or '@' not in email:
            return False, "Invalid email format"
        
        # Validate password
        password = user_data.get('password', '')
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        return True, None
        
    except Exception as e:
        logger.error("Error in async user validation", error=str(e))
        return False, "Validation error occurred"

async def create_user_async(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronous user creation.
    
    Args:
        user_data: User data to create
        
    Returns:
        Created user data
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

async def process_video_async(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronous video processing.
    
    Args:
        video_data: Video data to process
        
    Returns:
        Processed video data
    """
    try:
        # Simulate async video processing
        await asyncio.sleep(1.0)  # Simulate processing time
        
        # Update video status
        video_data['status'] = 'completed'
        video_data['processed_at'] = datetime.now(timezone.utc)
        
        logger.info("Video processed asynchronously", video_id=video_data.get('id'))
        return video_data
        
    except Exception as e:
        logger.error("Error processing video asynchronously", error=str(e))
        raise

async async def fetch_user_data_async(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Asynchronous user data fetching.
    
    Args:
        user_id: User ID to fetch
        
    Returns:
        User data or None if not found
    """
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

async def send_notification_async(user_id: int, message: str) -> bool:
    """
    Asynchronous notification sending.
    
    Args:
        user_id: User ID to notify
        message: Notification message
        
    Returns:
        True if notification sent successfully
    """
    try:
        # Simulate async notification service
        await asyncio.sleep(0.3)  # Simulate network request
        
        logger.info("Notification sent asynchronously", user_id=user_id, message=message)
        return True
        
    except Exception as e:
        logger.error("Error sending notification asynchronously", user_id=user_id, error=str(e))
        return False

# =============================================================================
# Mixed Sync/Async Patterns
# =============================================================================

def sync_to_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to convert synchronous function to asynchronous.
    
    Use when you need to call a sync function from async context.
    
    Args:
        func: Synchronous function to convert
        
    Returns:
        Asynchronous wrapper function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper

def async_to_sync(func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """
    Decorator to convert asynchronous function to synchronous.
    
    Use when you need to call an async function from sync context.
    
    Args:
        func: Asynchronous function to convert
        
    Returns:
        Synchronous wrapper function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

# =============================================================================
# Utility Functions with Proper Sync/Async Usage
# =============================================================================

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
            
            # Check if username already exists
            # await check_username_exists_async(data['username'])
            
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

# =============================================================================
# Service Patterns with Sync/Async
# =============================================================================

class UserService:
    """Service class demonstrating sync/async patterns."""
    
    def __init__(self) -> Any:
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def validate_user_sync(self, user_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Synchronous user validation.
        
        Use def for validation that doesn't require I/O.
        """
        return SyncAsyncUtils.validate_input_sync(user_data)
    
    async def validate_user_async(self, user_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Asynchronous user validation.
        
        Use async def for validation that requires I/O.
        """
        return await SyncAsyncUtils.validate_input_async(user_data)
    
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
        
        # Send notification asynchronously
        asyncio.create_task(
            send_notification_async(user['id'], "Welcome to HeyGen AI!")
        )
        
        return user
    
    def get_user_sync(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Synchronous user retrieval.
        
        Use def when you need to block for the result.
        """
        # Simulate synchronous database query
        time.sleep(0.1)  # Simulate blocking operation
        
        user_data = {
            'id': user_id,
            'username': f'user_{user_id}',
            'email': f'user_{user_id}@example.com',
            'created_at': datetime.now(timezone.utc)
        }
        
        return user_data
    
    async def get_user_async(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Asynchronous user retrieval.
        
        Use async def for non-blocking database queries.
        """
        return await fetch_user_data_async(user_id)
    
    def process_users_sync(self, users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Synchronous batch user processing.
        
        Use def for CPU-bound batch processing.
        """
        return SyncAsyncUtils.process_data_sync(users)
    
    async def process_users_async(self, users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Asynchronous batch user processing.
        
        Use async def for I/O-bound batch processing.
        """
        return await SyncAsyncUtils.process_data_async(users)

# =============================================================================
# Performance Optimization Patterns
# =============================================================================

class PerformanceOptimizer:
    """Performance optimization patterns for sync/async operations."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def expensive_calculation_sync(n: int) -> int:
        """
        Expensive synchronous calculation with caching.
        
        Use def with @lru_cache for expensive pure functions.
        """
        # Simulate expensive calculation
        time.sleep(0.1)
        return n * n
    
    @staticmethod
    async def expensive_calculation_async(n: int) -> int:
        """
        Expensive asynchronous calculation.
        
        Use async def for expensive I/O operations.
        """
        # Simulate expensive async operation
        await asyncio.sleep(0.1)
        return n * n
    
    @staticmethod
    def cpu_bound_operation_sync(data: List[int]) -> List[int]:
        """
        CPU-bound synchronous operation.
        
        Use def for CPU-intensive operations.
        """
        return [x * x for x in data]
    
    @staticmethod
    async def cpu_bound_operation_async(data: List[int]) -> List[int]:
        """
        CPU-bound asynchronous operation using ThreadPoolExecutor.
        
        Use async def with executor for CPU-bound operations in async context.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            PerformanceOptimizer.cpu_bound_operation_sync, 
            data
        )
    
    @staticmethod
    async def parallel_io_operations_async(urls: List[str]) -> List[str]:
        """
        Parallel I/O operations.
        
        Use async def for parallel I/O operations.
        """
        async async def fetch_url(url: str) -> str:
            # Simulate async HTTP request
            await asyncio.sleep(0.1)
            return f"Response from {url}"
        
        # Create tasks for all URLs
        tasks = [fetch_url(url) for url in urls]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [result for result in results if not isinstance(result, Exception)]

# =============================================================================
# Error Handling Patterns
# =============================================================================

class ErrorHandler:
    """Error handling patterns for sync/async operations."""
    
    @staticmethod
    def safe_sync_operation(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        """
        Decorator for safe synchronous operations.
        
        Args:
            func: Synchronous function to wrap
            
        Returns:
            Wrapped function that returns None on error
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in sync operation {func.__name__}", error=str(e))
                return None
        return wrapper
    
    @staticmethod
    def safe_async_operation(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[Optional[T]]]:
        """
        Decorator for safe asynchronous operations.
        
        Args:
            func: Asynchronous function to wrap
            
        Returns:
            Wrapped function that returns None on error
        """
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in async operation {func.__name__}", error=str(e))
                return None
        return wrapper

# =============================================================================
# Usage Examples
# =============================================================================

def demonstrate_sync_async_patterns():
    """Demonstrate proper sync/async usage patterns."""
    
    # Synchronous operations (use def)
    username = validate_username_sync("john_doe")
    password = validate_password_strength_sync("SecurePass123")
    hashed_password = hash_password_sync(password)
    efficiency = calculate_processing_efficiency_sync(10.0, 1000000)
    
    print(f"Sync operations completed: {username}, {hashed_password[:10]}..., {efficiency}")
    
    # Asynchronous operations (use async def)
    async def async_demo():
        
    """async_demo function."""
# Validate user asynchronously
        is_valid, error = await validate_user_async({
            'username': 'john_doe',
            'email': 'john@example.com',
            'password': 'SecurePass123'
        })
        
        if is_valid:
            # Create user asynchronously
            user = await create_user_async({
                'username': 'john_doe',
                'email': 'john@example.com',
                'password': 'SecurePass123'
            })
            
            # Process video asynchronously
            video = await process_video_async({
                'id': 1,
                'script': 'Hello world',
                'status': 'pending'
            })
            
            print(f"Async operations completed: {user['username']}, {video['status']}")
    
    # Run async demo
    asyncio.run(async_demo())

# =============================================================================
# Export all functions and classes
# =============================================================================

__all__ = [
    # Synchronous functions
    'validate_username_sync',
    'validate_password_strength_sync',
    'hash_password_sync',
    'calculate_processing_efficiency_sync',
    'format_file_size_sync',
    'validate_email_format_sync',
    'generate_api_key_sync',
    'calculate_age_sync',
    'calculate_success_rate_sync',
    
    # Asynchronous functions
    'validate_user_async',
    'create_user_async',
    'process_video_async',
    'fetch_user_data_async',
    'send_notification_async',
    
    # Utility classes
    'SyncAsyncUtils',
    'UserService',
    'PerformanceOptimizer',
    'ErrorHandler',
    
    # Decorators
    'sync_to_async',
    'async_to_sync',
    
    # Demo function
    'demonstrate_sync_async_patterns',
] 