"""
Utility Functions for OpusClip Improved
======================================

Common utility functions and helpers for the application.
"""

import asyncio
import logging
import hashlib
import secrets
import string
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import base64
import mimetypes
import aiofiles
import aiohttp
from functools import wraps
import structlog

from .schemas import get_settings
from .exceptions import UtilityError, create_utility_error

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Retry configuration"""
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.exceptions = exceptions


def retry_async(config: RetryConfig = None):
    """Async retry decorator"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.delay
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        break
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def retry_sync(config: RetryConfig = None):
    """Sync retry decorator"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.delay
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        break
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class AsyncContextManager:
    """Async context manager for resource management"""
    
    def __init__(self, resource_factory: Callable, cleanup_func: Callable = None):
        self.resource_factory = resource_factory
        self.cleanup_func = cleanup_func
        self.resource = None
    
    async def __aenter__(self):
        self.resource = await self.resource_factory()
        return self.resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_func and self.resource:
            await self.cleanup_func(self.resource)


class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self) -> bool:
        """Acquire rate limit permit"""
        now = time.time()
        
        # Remove old calls
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            return False
        
        self.calls.append(now)
        return True
    
    async def wait(self):
        """Wait for rate limit permit"""
        while not await self.acquire():
            await asyncio.sleep(0.1)


class BatchProcessor:
    """Batch processor for handling multiple items"""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batch = []
        self.last_batch_time = time.time()
        self.processing = False
    
    async def add_item(self, item: Any):
        """Add item to batch"""
        self.batch.append(item)
        
        if len(self.batch) >= self.batch_size:
            await self._process_batch()
        elif time.time() - self.last_batch_time >= self.max_wait_time:
            await self._process_batch()
    
    async def _process_batch(self):
        """Process current batch"""
        if not self.batch or self.processing:
            return
        
        self.processing = True
        try:
            batch_to_process = self.batch.copy()
            self.batch.clear()
            self.last_batch_time = time.time()
            
            # Process batch (placeholder)
            await self._process_items(batch_to_process)
        finally:
            self.processing = False
    
    async def _process_items(self, items: List[Any]):
        """Process items (to be implemented by subclass)"""
        pass


class FileUtils:
    """File utility functions"""
    
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
        """Get file hash"""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            raise create_utility_error("file_hash", file_path, e)
    
    @staticmethod
    async def get_file_hash_async(file_path: str, algorithm: str = "md5") -> str:
        """Get file hash asynchronously"""
        try:
            hash_obj = hashlib.new(algorithm)
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(4096):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            raise create_utility_error("file_hash_async", file_path, e)
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size"""
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            raise create_utility_error("file_size", file_path, e)
    
    @staticmethod
    def get_file_mime_type(file_path: str) -> str:
        """Get file MIME type"""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type or "application/octet-stream"
        except Exception as e:
            raise create_utility_error("file_mime_type", file_path, e)
    
    @staticmethod
    def ensure_directory(directory_path: str):
        """Ensure directory exists"""
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise create_utility_error("ensure_directory", directory_path, e)
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """Create safe filename"""
        try:
            # Remove or replace unsafe characters
            safe_chars = string.ascii_letters + string.digits + "._-"
            safe_filename = "".join(c for c in filename if c in safe_chars)
            
            # Ensure filename is not empty
            if not safe_filename:
                safe_filename = "file"
            
            # Limit length
            if len(safe_filename) > 255:
                name, ext = safe_filename.rsplit(".", 1) if "." in safe_filename else (safe_filename, "")
                safe_filename = name[:255 - len(ext) - 1] + (f".{ext}" if ext else "")
            
            return safe_filename
        except Exception as e:
            raise create_utility_error("safe_filename", filename, e)


class StringUtils:
    """String utility functions"""
    
    @staticmethod
    def generate_random_string(length: int = 32, include_symbols: bool = False) -> str:
        """Generate random string"""
        try:
            chars = string.ascii_letters + string.digits
            if include_symbols:
                chars += "!@#$%^&*"
            
            return "".join(secrets.choice(chars) for _ in range(length))
        except Exception as e:
            raise create_utility_error("generate_random_string", str(length), e)
    
    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID string"""
        return str(uuid.uuid4())
    
    @staticmethod
    def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate string to max length"""
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def slugify(text: str) -> str:
        """Convert text to URL-friendly slug"""
        try:
            import re
            # Convert to lowercase and replace spaces with hyphens
            slug = re.sub(r'[^\w\s-]', '', text.lower())
            slug = re.sub(r'[-\s]+', '-', slug)
            return slug.strip('-')
        except Exception as e:
            raise create_utility_error("slugify", text, e)
    
    @staticmethod
    def mask_sensitive_data(text: str, visible_chars: int = 4) -> str:
        """Mask sensitive data"""
        if len(text) <= visible_chars:
            return "*" * len(text)
        
        return text[:visible_chars] + "*" * (len(text) - visible_chars)


class DateTimeUtils:
    """DateTime utility functions"""
    
    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC datetime"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        """Convert datetime to UTC"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        try:
            if seconds < 60:
                return f"{seconds:.1f}s"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.1f}m"
            elif seconds < 86400:
                hours = seconds / 3600
                return f"{hours:.1f}h"
            else:
                days = seconds / 86400
                return f"{days:.1f}d"
        except Exception as e:
            raise create_utility_error("format_duration", str(seconds), e)
    
    @staticmethod
    def parse_duration(duration_str: str) -> float:
        """Parse duration string to seconds"""
        try:
            duration_str = duration_str.strip().lower()
            
            if duration_str.endswith('s'):
                return float(duration_str[:-1])
            elif duration_str.endswith('m'):
                return float(duration_str[:-1]) * 60
            elif duration_str.endswith('h'):
                return float(duration_str[:-1]) * 3600
            elif duration_str.endswith('d'):
                return float(duration_str[:-1]) * 86400
            else:
                return float(duration_str)
        except Exception as e:
            raise create_utility_error("parse_duration", duration_str, e)


class ValidationUtils:
    """Validation utility functions"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email address"""
        try:
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return re.match(pattern, email) is not None
        except Exception:
            return False
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL"""
        try:
            import re
            pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            return re.match(pattern, url) is not None
        except Exception:
            return False
    
    @staticmethod
    def is_valid_uuid(uuid_str: str) -> bool:
        """Validate UUID string"""
        try:
            uuid.UUID(uuid_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_valid_json(json_str: str) -> bool:
        """Validate JSON string"""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False


class HTTPUtils:
    """HTTP utility functions"""
    
    @staticmethod
    async def make_request(
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        data: Any = None,
        timeout: int = 30,
        retries: int = 3
    ) -> aiohttp.ClientResponse:
        """Make HTTP request with retries"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                for attempt in range(retries):
                    try:
                        async with session.request(
                            method=method,
                            url=url,
                            headers=headers,
                            data=data
                        ) as response:
                            return response
                    except Exception as e:
                        if attempt == retries - 1:
                            raise
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            raise create_utility_error("make_request", url, e)
    
    @staticmethod
    async def download_file(url: str, file_path: str, chunk_size: int = 8192) -> str:
        """Download file from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            await f.write(chunk)
            
            return file_path
        except Exception as e:
            raise create_utility_error("download_file", url, e)
    
    @staticmethod
    async def upload_file(
        url: str,
        file_path: str,
        field_name: str = "file",
        additional_fields: Dict[str, str] = None
    ) -> aiohttp.ClientResponse:
        """Upload file to URL"""
        try:
            data = aiohttp.FormData()
            
            # Add file
            data.add_field(
                field_name,
                open(file_path, 'rb'),
                filename=Path(file_path).name,
                content_type=mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            )
            
            # Add additional fields
            if additional_fields:
                for key, value in additional_fields.items():
                    data.add_field(key, value)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    return response
        except Exception as e:
            raise create_utility_error("upload_file", url, e)


class CryptoUtils:
    """Cryptographic utility functions"""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        try:
            if salt is None:
                salt = secrets.token_hex(32)
            
            # Use PBKDF2 for password hashing
            import hashlib
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000  # iterations
            )
            
            return base64.b64encode(password_hash).decode('utf-8'), salt
        except Exception as e:
            raise create_utility_error("hash_password", "password", e)
    
    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            computed_hash, _ = CryptoUtils.hash_password(password, salt)
            return computed_hash == password_hash
        except Exception as e:
            raise create_utility_error("verify_password", "password", e)
    
    @staticmethod
    def encrypt_data(data: str, key: str) -> str:
        """Encrypt data with key"""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'opus_clip_salt',  # In production, use random salt
                iterations=100000,
            )
            key_bytes = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            
            # Encrypt data
            fernet = Fernet(key_bytes)
            encrypted_data = fernet.encrypt(data.encode())
            
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            raise create_utility_error("encrypt_data", "data", e)
    
    @staticmethod
    def decrypt_data(encrypted_data: str, key: str) -> str:
        """Decrypt data with key"""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'opus_clip_salt',  # In production, use random salt
                iterations=100000,
            )
            key_bytes = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            
            # Decrypt data
            fernet = Fernet(key_bytes)
            decrypted_data = fernet.decrypt(base64.urlsafe_b64decode(encrypted_data))
            
            return decrypted_data.decode()
        except Exception as e:
            raise create_utility_error("decrypt_data", "encrypted_data", e)


class ConfigUtils:
    """Configuration utility functions"""
    
    @staticmethod
    def load_config_file(file_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    import yaml
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
        except Exception as e:
            raise create_utility_error("load_config_file", file_path, e)
    
    @staticmethod
    def save_config_file(config: Dict[str, Any], file_path: str):
        """Save configuration to file"""
        try:
            with open(file_path, 'w') as f:
                if file_path.endswith('.json'):
                    json.dump(config, f, indent=2)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    import yaml
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")
        except Exception as e:
            raise create_utility_error("save_config_file", file_path, e)
    
    @staticmethod
    def get_env_variable(key: str, default: Any = None, required: bool = False) -> Any:
        """Get environment variable with type conversion"""
        import os
        
        value = os.getenv(key, default)
        
        if value is None and required:
            raise ValueError(f"Required environment variable {key} not set")
        
        return value


class LoggingUtils:
    """Logging utility functions"""
    
    @staticmethod
    def setup_structured_logging(level: str = "INFO", service_name: str = "opus-clip"):
        """Setup structured logging"""
        try:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            
            logging.basicConfig(
                format="%(message)s",
                stream=sys.stdout,
                level=getattr(logging, level.upper()),
            )
            
            return structlog.get_logger(service_name)
        except Exception as e:
            raise create_utility_error("setup_structured_logging", level, e)
    
    @staticmethod
    def get_logger(name: str) -> structlog.BoundLogger:
        """Get structured logger"""
        return structlog.get_logger(name)


class PerformanceUtils:
    """Performance utility functions"""
    
    @staticmethod
    def measure_time(func: Callable) -> Callable:
        """Decorator to measure function execution time"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger.info(f"Function {func.__name__} executed in {execution_time:.2f}s")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger.info(f"Function {func.__name__} executed in {execution_time:.2f}s")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @staticmethod
    def measure_memory(func: Callable) -> Callable:
        """Decorator to measure function memory usage"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_memory = process.memory_info().rss
                memory_delta = end_memory - start_memory
                logger.info(f"Function {func.__name__} memory delta: {memory_delta / 1024 / 1024:.2f} MB")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_memory = process.memory_info().rss
                memory_delta = end_memory - start_memory
                logger.info(f"Function {func.__name__} memory delta: {memory_delta / 1024 / 1024:.2f} MB")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# Import sys for logging setup
import sys





























