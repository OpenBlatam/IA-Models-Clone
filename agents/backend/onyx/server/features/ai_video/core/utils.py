from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import re
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from urllib.parse import urlparse, urljoin
import asyncio
import time
from datetime import datetime, timedelta
import functools
import threading
from .exceptions import ValidationError, StorageError
from .constants import VALIDATION_RULES, MAX_FILE_SIZE, DEFAULT_TIMEOUT
    from .constants import LOG_FORMATS, LOG_LEVELS
        import psutil
        import psutil
        import psutil
        import platform
        import json
    import ipaddress
        import requests
        import psutil
    import re
    import re
    import uuid
    import secrets
    import hashlib
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import os
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Any, List, Dict, Optional
"""
AI Video System - Utilities

Production-ready utility functions for the AI Video System.
"""



logger = logging.getLogger(__name__)


# URL validation and processing
def validate_url(url: str) -> bool:
    """
    Validate URL format and security.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if URL is valid
        
    Raises:
        ValidationError: If URL is invalid
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string", field="url", value=url)
    
    if len(url) > VALIDATION_RULES["url"]["max_length"]:
        raise ValidationError(
            f"URL too long (max {VALIDATION_RULES['url']['max_length']} characters)",
            field="url",
            value=url
        )
    
    try:
        parsed = urlparse(url)
        
        # Check required fields
        if not parsed.scheme or not parsed.netloc:
            raise ValidationError("URL must have scheme and netloc", field="url", value=url)
        
        # Check allowed schemes
        if parsed.scheme not in VALIDATION_RULES["url"]["allowed_schemes"]:
            raise ValidationError(
                f"URL scheme must be one of {VALIDATION_RULES['url']['allowed_schemes']}",
                field="url",
                value=url
            )
        
        return True
        
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {e}", field="url", value=url)


def sanitize_url(url: str) -> str:
    """
    Sanitize URL for safe use.
    
    Args:
        url: URL to sanitize
        
    Returns:
        str: Sanitized URL
    """
    # Remove whitespace
    url = url.strip()
    
    # Ensure proper scheme
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    return url


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: URL to extract domain from
        
    Returns:
        str: Domain name
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return "unknown"


# File and path utilities
def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system use.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    if not filename:
        return "unnamed"
    
    # Remove forbidden characters
    for char in VALIDATION_RULES["filename"]["forbidden_chars"]:
        filename = filename.replace(char, "_")
    
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    
    # Limit length
    if len(filename) > VALIDATION_RULES["filename"]["max_length"]:
        name, ext = os.path.splitext(filename)
        max_name_length = VALIDATION_RULES["filename"]["max_length"] - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename


def safe_filename(filename: str) -> str:
    """
    Alias for sanitize_filename for backward compatibility.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    return sanitize_filename(filename)


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get file extension from file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        str: File extension (including dot)
    """
    path = Path(file_path)
    return path.suffix.lower()


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        int: File size in bytes
        
    Raises:
        StorageError: If file doesn't exist or can't be accessed
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise StorageError(f"File does not exist: {file_path}", storage_path=str(path))
        
        return path.stat().st_size
        
    except Exception as e:
        if isinstance(e, StorageError):
            raise
        raise StorageError(f"Error getting file size: {e}", storage_path=str(file_path))


def validate_file_size(file_path: Union[str, Path], max_size: Optional[int] = None) -> bool:
    """
    Validate file size.
    
    Args:
        file_path: Path to file
        max_size: Maximum allowed size (defaults to MAX_FILE_SIZE)
        
    Returns:
        bool: True if file size is valid
        
    Raises:
        ValidationError: If file is too large
    """
    max_size = max_size or MAX_FILE_SIZE
    file_size = get_file_size(file_path)
    
    if file_size > max_size:
        raise ValidationError(
            f"File too large ({file_size} bytes, max {max_size} bytes)",
            field="file_size",
            value=file_size
        )
    
    return True


def create_temp_file(prefix: str = "ai_video_", suffix: str = ".tmp") -> Path:
    """
    Create a temporary file.
    
    Args:
        prefix: File prefix
        suffix: File suffix
        
    Returns:
        Path: Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=suffix,
        delete=False
    )
    temp_file.close()
    return Path(temp_file.name)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path: Path object for directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def cleanup_old_files(directory: Union[str, Path], max_age_days: int = 7) -> int:
    """
    Clean up old files in directory.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
        
    Returns:
        int: Number of files cleaned up
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    cleaned_count = 0
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to clean up file {file_path}: {e}")
    
    return cleaned_count


# ID and hash utilities
def generate_workflow_id() -> str:
    """
    Generate unique workflow ID.
    
    Returns:
        str: Unique workflow ID
    """
    timestamp = int(time.time() * 1000)
    random_part = os.urandom(4).hex()
    return f"wf_{timestamp}_{random_part}"


def generate_video_id() -> str:
    """
    Generate unique video ID.
    
    Returns:
        str: Unique video ID
    """
    timestamp = int(time.time() * 1000)
    random_part = os.urandom(4).hex()
    return f"vid_{timestamp}_{random_part}"


def hash_content(content: str) -> str:
    """
    Generate hash for content.
    
    Args:
        content: Content to hash
        
    Returns:
        str: SHA-256 hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def hash_url(url: str) -> str:
    """
    Generate hash for URL.
    
    Args:
        url: URL to hash
        
    Returns:
        str: SHA-256 hash
    """
    return hash_content(url)


# Validation utilities
def validate_workflow_id(workflow_id: str) -> bool:
    """
    Validate workflow ID format.
    
    Args:
        workflow_id: Workflow ID to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If workflow ID is invalid
    """
    if not workflow_id or not isinstance(workflow_id, str):
        raise ValidationError("Workflow ID must be a non-empty string", field="workflow_id", value=workflow_id)
    
    if len(workflow_id) > VALIDATION_RULES["workflow_id"]["max_length"]:
        raise ValidationError(
            f"Workflow ID too long (max {VALIDATION_RULES['workflow_id']['max_length']} characters)",
            field="workflow_id",
            value=workflow_id
        )
    
    if not re.match(VALIDATION_RULES["workflow_id"]["pattern"], workflow_id):
        raise ValidationError(
            "Workflow ID contains invalid characters",
            field="workflow_id",
            value=workflow_id
        )
    
    return True


def validate_plugin_name(plugin_name: str) -> bool:
    """
    Validate plugin name format.
    
    Args:
        plugin_name: Plugin name to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If plugin name is invalid
    """
    if not plugin_name or not isinstance(plugin_name, str):
        raise ValidationError("Plugin name must be a non-empty string", field="plugin_name", value=plugin_name)
    
    if len(plugin_name) > VALIDATION_RULES["plugin_name"]["max_length"]:
        raise ValidationError(
            f"Plugin name too long (max {VALIDATION_RULES['plugin_name']['max_length']} characters)",
            field="plugin_name",
            value=plugin_name
        )
    
    if not re.match(VALIDATION_RULES["plugin_name"]["pattern"], plugin_name):
        raise ValidationError(
            "Plugin name contains invalid characters",
            field="plugin_name",
            value=plugin_name
        )
    
    return True


# Time and duration utilities
def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def parse_duration(duration_str: str) -> float:
    """
    Parse duration string to seconds.
    
    Args:
        duration_str: Duration string (e.g., "30s", "2.5m", "1h")
        
    Returns:
        float: Duration in seconds
    """
    duration_str = duration_str.lower().strip()
    
    if duration_str.endswith('s'):
        return float(duration_str[:-1])
    elif duration_str.endswith('m'):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith('h'):
        return float(duration_str[:-1]) * 3600
    else:
        return float(duration_str)


# Async utilities
async def with_timeout(coro, timeout: float = DEFAULT_TIMEOUT):
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        
    Returns:
        Result of coroutine
        
    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    return await asyncio.wait_for(coro, timeout=timeout)


async def retry_async(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry
        
    Returns:
        Result of function
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
    
    raise last_exception


# Configuration utilities
def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration dictionaries recursively.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_env_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dict[str, Any]: Configuration from environment
    """
    config = {}
    
    # Map environment variables to config structure
    env_mappings = {
        "AI_VIDEO_LOG_LEVEL": ("monitoring", "log_level"),
        "AI_VIDEO_ENVIRONMENT": ("environment",),
        "AI_VIDEO_DEBUG": ("debug",),
        "AI_VIDEO_STORAGE_PATH": ("storage", "local_storage_path"),
        "AI_VIDEO_TEMP_DIR": ("storage", "temp_directory"),
        "AI_VIDEO_OUTPUT_DIR": ("storage", "output_directory"),
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the correct location in config
            current = config
            for path_part in config_path[:-1]:
                if path_part not in current:
                    current[path_part] = {}
                current = current[path_part]
            
            # Set the value
            current[config_path[-1]] = value
    
    return config


# Logging utilities
def setup_logging(
    level: str = "INFO",
    format_str: str = "detailed",
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level
        format_str: Log format
        log_file: Log file path
    """
    
    # Get log level
    log_level = LOG_LEVELS.get(level.upper(), LOG_LEVELS["INFO"])
    
    # Get format
    log_format = LOG_FORMATS.get(format_str, LOG_FORMATS["detailed"])
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        logging.getLogger().addHandler(file_handler)


# Performance utilities
def measure_time(func) -> Any:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function
    """
    async def async_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.3f}s")
    
    def sync_wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.3f}s")
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dict[str, float]: Memory usage information
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"rss": 0, "vms": 0, "percent": 0}


def get_cpu_usage() -> float:
    """
    Get current CPU usage percentage.
    
    Returns:
        float: CPU usage percentage
    """
    try:
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dict[str, Any]: System information
    """
    try:
        
        # Memory info
        memory = psutil.virtual_memory()
        
        # Disk info
        disk = psutil.disk_usage('/')
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # System info
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": cpu_count,
            "cpu_usage": cpu_percent,
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_free": disk.free,
            "disk_percent": disk.percent,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
        return system_info
        
    except ImportError:
        logger.warning("psutil not available, returning basic system info")
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "error": "psutil not available for detailed system info"
        }


def validate_json(data: str) -> bool:
    """
    Validate JSON string.
    
    Args:
        data: JSON string to validate
        
    Returns:
        bool: True if valid JSON
    """
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def validate_ip_address(ip: str) -> bool:
    """
    Validate IP address format.
    
    Args:
        ip: IP address to validate
        
    Returns:
        bool: True if valid IP address
    """
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def check_connectivity(url: str, timeout: float = 5.0) -> bool:
    """
    Check network connectivity to URL.
    
    Args:
        url: URL to check
        timeout: Timeout in seconds
        
    Returns:
        bool: True if accessible
    """
    try:
        response = requests.head(url, timeout=timeout)
        return response.status_code < 400
    except Exception:
        return False


def is_valid_file_path(file_path: Union[str, Path]) -> bool:
    """
    Check if file path is valid.
    
    Args:
        file_path: Path to check
        
    Returns:
        bool: True if valid file path
    """
    try:
        path = Path(file_path)
        # Check if path is absolute or can be resolved
        if path.is_absolute() or path.resolve():
            return True
        return False
    except Exception:
        return False


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Dict[str, Any]: Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten a dictionary with dot notation keys.
    
    Args:
        d: Flattened dictionary
        sep: Separator for nested keys
        
    Returns:
        Dict[str, Any]: Unflattened dictionary
    """
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


def sanitize_data(data: Any) -> Any:
    """
    Sanitize data for safe storage and transmission.
    
    Args:
        data: Data to sanitize
        
    Returns:
        Any: Sanitized data
    """
    if isinstance(data, str):
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', 'script', 'javascript']
        for char in dangerous_chars:
            data = data.replace(char, '')
        return data.strip()
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    else:
        return data


def validate_data_types(data: Dict[str, Any], schema: Dict[str, type]) -> bool:
    """
    Validate data types against a schema.
    
    Args:
        data: Data to validate
        schema: Schema with expected types
        
    Returns:
        bool: True if all types match
    """
    for key, expected_type in schema.items():
        if key in data:
            if not isinstance(data[key], expected_type):
                return False
    return True


def get_timestamp() -> str:
    """
    Get current UTC timestamp as ISO string.
    
    Returns:
        str: Current UTC timestamp
    """
    return datetime.utcnow().isoformat() + 'Z'


def is_expired(timestamp: float, ttl: float) -> bool:
    """
    Check if a timestamp is expired given a TTL (time-to-live).
    
    Args:
        timestamp: Unix timestamp (seconds)
        ttl: Time-to-live in seconds
        
    Returns:
        bool: True if expired
    """
    return (time.time() - timestamp) > ttl


def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        url: URL string to check
        
    Returns:
        bool: True if valid URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_domain(url: str) -> str:
    """
    Extract the domain from a URL string.
    
    Args:
        url: URL string
        
    Returns:
        str: Domain name
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return "unknown"


def get_disk_usage(path: Union[str, Path] = "/") -> Dict[str, Any]:
    """
    Get disk usage statistics for the given path.
    
    Args:
        path: Path to check disk usage (default: "/")
        
    Returns:
        Dict[str, Any]: Disk usage information
    """
    try:
        usage = psutil.disk_usage(str(path))
        return {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": usage.percent
        }
    except ImportError:
        return {"total": 0, "used": 0, "free": 0, "percent": 0}
    except Exception:
        return {"total": 0, "used": 0, "free": 0, "percent": 0}


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if valid email
    """
    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(email_regex, email) is not None


def validate_phone(phone: str) -> bool:
    """
    Validate phone number format (basic international check).
    
    Args:
        phone: Phone number to validate
        
    Returns:
        bool: True if valid phone number
    """
    phone_regex = r"^\+?\d{7,15}$"
    return re.match(phone_regex, phone) is not None


def validate_uuid(uuid_str: str) -> bool:
    """
    Validate if a string is a valid UUID.
    
    Args:
        uuid_str: String to validate
        
    Returns:
        bool: True if valid UUID
    """
    try:
        uuid.UUID(uuid_str)
        return True
    except Exception:
        return False


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token (hex string).
    
    Args:
        length: Number of bytes (default: 32)
        
    Returns:
        str: Secure random token as hex string
    """
    return secrets.token_hex(length)


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """
    Hash a password with optional salt using SHA-256.
    
    Args:
        password: Password string
        salt: Optional salt string
        
    Returns:
        str: Hexadecimal hash
    """
    if salt is None:
        salt = ''
    hash_obj = hashlib.sha256((password + salt).encode('utf-8'))
    return hash_obj.hexdigest()


def verify_password(password: str, hashed: str, salt: Optional[str] = None) -> bool:
    """
    Verify a password against a hash with optional salt.
    
    Args:
        password: Password string
        hashed: Hashed password
        salt: Optional salt string
        
    Returns:
        bool: True if password matches hash
    """
    return hash_password(password, salt) == hashed


def encrypt_data(data: bytes, key: bytes) -> bytes:
    """
    Encrypt data using AES-256-GCM.
    
    Args:
        data: Data to encrypt (bytes)
        key: Encryption key (32 bytes)
        
    Returns:
        bytes: Encrypted data (nonce + tag + ciphertext)
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256-GCM.")
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, data, None)
    return nonce + ct


def decrypt_data(token: bytes, key: bytes) -> bytes:
    """
    Decrypt data using AES-256-GCM.
    
    Args:
        token: Encrypted data (nonce + tag + ciphertext)
        key: Encryption key (32 bytes)
        
    Returns:
        bytes: Decrypted data
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes for AES-256-GCM.")
    nonce = token[:12]
    ct = token[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, None)


def cache_result(ttl: int = 60):
    """
    Simple in-memory cache decorator with TTL (time-to-live in seconds).
    
    Args:
        ttl: Time-to-live in seconds
        
    Returns:
        Decorator for caching function results
    """
    def decorator(func) -> Any:
        cache = {}
        timestamps = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache and (now - timestamps[key]) < ttl:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = now
            return result
        return wrapper
    return decorator


def retry_operation(func, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Retry a synchronous operation with exponential backoff.
    
    Args:
        func: Function to retry (no-arg)
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry
        
    Returns:
        Result of function
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
    raise last_exception


def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator to rate limit a function (thread-safe, per-process).
    
    Args:
        calls_per_second: Allowed calls per second
        
    Returns:
        Decorator for rate limiting
    """
    min_interval = 1.0 / calls_per_second
    lock = threading.Lock()
    last_time = [0.0]
    
    def decorator(func) -> Any:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with lock:
                now = time.time()
                elapsed = now - last_time[0]
                wait = min_interval - elapsed
                if wait > 0:
                    time.sleep(wait)
                last_time[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """
    Record a metric (stub for integration with monitoring system).
    
    Args:
        name: Metric name
        value: Metric value
        labels: Optional labels/tags
    """
    # In production, integrate with Prometheus, StatsD, etc.
    logger.info(f"Metric recorded: {name}={value} labels={labels}")


def log_event(event: str, details: Optional[Dict[str, Any]] = None, level: str = "info"):
    """
    Log an event with optional details and log level.
    
    Args:
        event: Event name or message
        details: Optional dictionary with event details
        level: Log level ("info", "warning", "error", etc.)
    """
    msg = f"EVENT: {event} | DETAILS: {details}"
    if level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)


def create_alert(message: str, severity: str = "info", context: Optional[Dict[str, Any]] = None):
    """
    Create and log an alert event.
    
    Args:
        message: Alert message
        severity: Severity level ("info", "warning", "error", "critical")
        context: Optional context dictionary
    """
    alert = {
        "message": message,
        "severity": severity,
        "context": context or {},
        "timestamp": get_timestamp()
    }
    logger.warning(f"ALERT: {alert}")
    return alert


def check_health() -> Dict[str, Any]:
    """
    Perform a basic system health check.
    
    Returns:
        Dict[str, Any]: Health status and details
    """
    try:
        mem = get_memory_usage()
        cpu = get_cpu_usage()
        disk = get_disk_usage()
        status = "healthy" if cpu < 90 and mem["percent"] < 90 and disk["percent"] < 90 else "degraded"
        return {
            "status": status,
            "cpu": cpu,
            "memory": mem,
            "disk": disk
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)} 