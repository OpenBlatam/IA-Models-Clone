from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import hashlib
import secrets
import hmac
import base64
import re
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import asyncio
import functools
    import socket
            import time
from typing import Any, List, Dict, Optional
import logging
"""
Cybersecurity Core - Proper separation of CPU-bound and I/O-bound operations
Uses def for pure CPU operations, async def for I/O operations
"""


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class SecurityRequest:
    """Immutable security operation request"""
    operation: str
    parameters: Dict[str, Any]
    security_level: str = "standard"
    metadata: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class SecurityResponse:
    """Immutable security operation response"""
    is_successful: bool
    encrypted_data: Optional[bytes] = None
    decrypted_data: Optional[str] = None
    hash_value: Optional[str] = None
    signature: Optional[str] = None
    error_message: Optional[str] = None
    security_metrics: Optional[Dict[str, Any]] = None

# ============================================================================
# CPU-BOUND OPERATIONS (def functions)
# ============================================================================

def generate_secure_random_bytes(length: int = 32) -> bytes:
    """Generate cryptographically secure random bytes - CPU-bound"""
    return secrets.token_bytes(length)

def generate_secure_random_string(length: int = 32) -> str:
    """Generate cryptographically secure random string - CPU-bound"""
    return secrets.token_urlsafe(length)

def calculate_sha256_hash(data: str) -> str:
    """Calculate SHA-256 hash of input data - CPU-bound"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def calculate_hmac_signature(data: str, secret_key: str) -> str:
    """Calculate HMAC signature for data integrity - CPU-bound"""
    return hmac.new(
        secret_key.encode('utf-8'),
        data.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def sanitize_input_string(input_string: str) -> str:
    """Sanitize input string to prevent injection attacks - CPU-bound"""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', input_string)
    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized.strip())
    # Limit length
    return sanitized[:1000]

def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """Validate password strength and return issues - CPU-bound"""
    issues = []
    
    if len(password) < 12:
        issues.append("Password must be at least 12 characters long")
    
    if not re.search(r'[A-Z]', password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        issues.append("Password must contain at least one digit")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Password must contain at least one special character")
    
    return len(issues) == 0, issues

def is_valid_email_address(email: str) -> bool:
    """Validate email address format - CPU-bound"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def is_valid_ip_address(ip_address: str) -> bool:
    """Validate IP address format - CPU-bound"""
    ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return bool(re.match(ip_pattern, ip_address))

def derive_encryption_key(password: str, salt: bytes) -> bytes:
    """Derive encryption key from password using PBKDF2 - CPU-bound"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_data_sync(data: str, password: str) -> Tuple[bytes, bytes]:
    """Synchronously encrypt data with password - CPU-bound"""
    salt = generate_secure_random_bytes(16)
    key = derive_encryption_key(password, salt)
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode('utf-8'))
    return encrypted_data, salt

def decrypt_data_sync(encrypted_data: bytes, password: str, salt: bytes) -> str:
    """Synchronously decrypt data with password - CPU-bound"""
    key = derive_encryption_key(password, salt)
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data)
    return decrypted_data.decode('utf-8')

def filter_successful_responses(responses: List['SecurityResponse']) -> List['SecurityResponse']:
    """Filter only successful security responses - CPU-bound"""
    return [r for r in responses if r.is_successful]

def extract_security_metrics(responses: List['SecurityResponse']) -> List[Dict[str, Any]]:
    """Extract security metrics from responses - CPU-bound"""
    return [r.security_metrics for r in responses if r.security_metrics]

def calculate_security_score(responses: List['SecurityResponse']) -> float:
    """Calculate overall security score from responses - CPU-bound"""
    if not responses:
        return 0.0
    
    successful_count = len(filter_successful_responses(responses))
    return (successful_count / len(responses)) * 100.0

# ============================================================================
# I/O-BOUND OPERATIONS (async def functions)
# ============================================================================

async def encrypt_data_async(data: str, password: str) -> Tuple[bytes, bytes]:
    """Asynchronously encrypt data with password - I/O-bound (crypto operations)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, encrypt_data_sync, data, password)

async def decrypt_data_async(encrypted_data: bytes, password: str, salt: bytes) -> str:
    """Asynchronously decrypt data with password - I/O-bound (crypto operations)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, decrypt_data_sync, encrypted_data, password, salt)

async def encrypt_data_roro(request: SecurityRequest) -> SecurityResponse:
    """Encrypt data using RORO pattern - I/O-bound"""
    try:
        data = request.parameters.get("data", "")
        password = request.parameters.get("password", "")
        
        if not data or not password:
            return SecurityResponse(
                is_successful=False,
                error_message="Data and password are required"
            )
        
        # Sanitize inputs (CPU-bound)
        sanitized_data = sanitize_input_string(data)
        
        # Encrypt data (I/O-bound)
        encrypted_data, salt = await encrypt_data_async(sanitized_data, password)
        
        # Calculate hash for integrity (CPU-bound)
        data_hash = calculate_sha256_hash(sanitized_data)
        
        return SecurityResponse(
            is_successful=True,
            encrypted_data=encrypted_data,
            hash_value=data_hash,
            security_metrics={
                "encryption_algorithm": "AES-256",
                "key_derivation": "PBKDF2-HMAC-SHA256",
                "iterations": 100000,
                "salt_length": len(salt)
            }
        )
        
    except Exception as e:
        return SecurityResponse(
            is_successful=False,
            error_message=str(e)
        )

async def decrypt_data_roro(request: SecurityRequest) -> SecurityResponse:
    """Decrypt data using RORO pattern - I/O-bound"""
    try:
        encrypted_data = request.parameters.get("encrypted_data")
        password = request.parameters.get("password", "")
        salt = request.parameters.get("salt")
        
        if not encrypted_data or not password or not salt:
            return SecurityResponse(
                is_successful=False,
                error_message="Encrypted data, password, and salt are required"
            )
        
        # Decrypt data (I/O-bound)
        decrypted_data = await decrypt_data_async(encrypted_data, password, salt)
        
        # Verify hash if provided (CPU-bound)
        expected_hash = request.parameters.get("expected_hash")
        if expected_hash:
            actual_hash = calculate_sha256_hash(decrypted_data)
            if actual_hash != expected_hash:
                return SecurityResponse(
                    is_successful=False,
                    error_message="Data integrity check failed"
                )
        
        return SecurityResponse(
            is_successful=True,
            decrypted_data=decrypted_data
        )
        
    except Exception as e:
        return SecurityResponse(
            is_successful=False,
            error_message=str(e)
        )

async def validate_password_roro(request: SecurityRequest) -> SecurityResponse:
    """Validate password strength using RORO pattern - I/O-bound (for consistency)"""
    try:
        password = request.parameters.get("password", "")
        
        if not password:
            return SecurityResponse(
                is_successful=False,
                error_message="Password is required"
            )
        
        # Validate password (CPU-bound)
        is_strong, issues = validate_password_strength(password)
        
        return SecurityResponse(
            is_successful=is_strong,
            error_message="; ".join(issues) if issues else None,
            security_metrics={
                "password_length": len(password),
                "has_uppercase": bool(re.search(r'[A-Z]', password)),
                "has_lowercase": bool(re.search(r'[a-z]', password)),
                "has_digit": bool(re.search(r'\d', password)),
                "has_special": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
            }
        )
        
    except Exception as e:
        return SecurityResponse(
            is_successful=False,
            error_message=str(e)
        )

async def scan_port_async(host: str, port: int, timeout: int) -> bool:
    """Asynchronously scan a single port - I/O-bound (network operations)"""
    
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: scan_port_sync(host, port, timeout)
        )
    except:
        return False

def scan_port_sync(host: str, port: int, timeout: int) -> bool:
    """Synchronously scan a single port - CPU-bound (but wrapped for I/O)"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

async def scan_port_roro(request: SecurityRequest) -> SecurityResponse:
    """Scan port using RORO pattern - I/O-bound"""
    try:
        host = request.parameters.get("host", "")
        port = request.parameters.get("port", 80)
        timeout = request.parameters.get("timeout", 5)
        
        if not host:
            return SecurityResponse(
                is_successful=False,
                error_message="Host is required"
            )
        
        # Validate IP address (CPU-bound)
        if not is_valid_ip_address(host):
            return SecurityResponse(
                is_successful=False,
                error_message="Invalid IP address format"
            )
        
        # Scan port (I/O-bound)
        is_open = await scan_port_async(host, port, timeout)
        
        return SecurityResponse(
            is_successful=True,
            decrypted_data=str(is_open),  # Reusing field for result
            security_metrics={
                "host": host,
                "port": port,
                "timeout": timeout,
                "is_open": is_open
            }
        )
        
    except Exception as e:
        return SecurityResponse(
            is_successful=False,
            error_message=str(e)
        )

async def batch_scan_ports_roro(request: SecurityRequest) -> SecurityResponse:
    """Batch scan multiple ports - I/O-bound"""
    try:
        host = request.parameters.get("host", "")
        ports = request.parameters.get("ports", [80, 443, 22, 21])
        timeout = request.parameters.get("timeout", 5)
        max_concurrent = request.parameters.get("max_concurrent", 10)
        
        if not host:
            return SecurityResponse(
                is_successful=False,
                error_message="Host is required"
            )
        
        # Validate IP address (CPU-bound)
        if not is_valid_ip_address(host):
            return SecurityResponse(
                is_successful=False,
                error_message="Invalid IP address format"
            )
        
        # Scan ports with concurrency control (I/O-bound)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scan_with_semaphore(port: int) -> Tuple[int, bool]:
            async with semaphore:
                is_open = await scan_port_async(host, port, timeout)
                return port, is_open
        
        tasks = [scan_with_semaphore(port) for port in ports]
        results = await asyncio.gather(*tasks)
        
        open_ports = [port for port, is_open in results if is_open]
        
        return SecurityResponse(
            is_successful=True,
            decrypted_data=str(open_ports),  # Reusing field for result
            security_metrics={
                "host": host,
                "total_ports": len(ports),
                "open_ports": len(open_ports),
                "open_ports_list": open_ports,
                "timeout": timeout,
                "max_concurrent": max_concurrent
            }
        )
        
    except Exception as e:
        return SecurityResponse(
            is_successful=False,
            error_message=str(e)
        )

# ============================================================================
# HIGHER-ORDER FUNCTIONS
# ============================================================================

def with_rate_limiting(max_requests: int = 100, window_seconds: int = 60):
    """Decorator to add rate limiting to security operations - CPU-bound"""
    request_counts = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request: SecurityRequest) -> SecurityResponse:
            current_time = time.time()
            client_id = request.metadata.get("client_id", "default")
            
            # Clean old entries (CPU-bound)
            request_counts[client_id] = [
                req_time for req_time in request_counts.get(client_id, [])
                if current_time - req_time < window_seconds
            ]
            
            # Check rate limit (CPU-bound)
            if len(request_counts[client_id]) >= max_requests:
                return SecurityResponse(
                    is_successful=False,
                    error_message="Rate limit exceeded"
                )
            
            # Add current request (CPU-bound)
            request_counts[client_id].append(current_time)
            
            return await func(request)
        return wrapper
    return decorator

def with_input_validation(required_fields: List[str]):
    """Decorator to validate required input fields - CPU-bound"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(request: SecurityRequest) -> SecurityResponse:
            missing_fields = [
                field for field in required_fields
                if field not in request.parameters
            ]
            
            if missing_fields:
                return SecurityResponse(
                    is_successful=False,
                    error_message=f"Missing required fields: {missing_fields}"
                )
            
            return await func(request)
        return wrapper
    return decorator

# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Data structures
    "SecurityRequest",
    "SecurityResponse",
    
    # CPU-bound functions (def)
    "generate_secure_random_bytes",
    "generate_secure_random_string",
    "calculate_sha256_hash",
    "calculate_hmac_signature",
    "sanitize_input_string",
    "validate_password_strength",
    "is_valid_email_address",
    "is_valid_ip_address",
    "derive_encryption_key",
    "encrypt_data_sync",
    "decrypt_data_sync",
    "filter_successful_responses",
    "extract_security_metrics",
    "calculate_security_score",
    
    # I/O-bound functions (async def)
    "encrypt_data_async",
    "decrypt_data_async",
    "encrypt_data_roro",
    "decrypt_data_roro",
    "validate_password_roro",
    "scan_port_async",
    "scan_port_sync",
    "scan_port_roro",
    "batch_scan_ports_roro",
    
    # Higher-order functions
    "with_rate_limiting",
    "with_input_validation"
] 