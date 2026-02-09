from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import hashlib
import hmac
import secrets
import base64
import json
import re
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from ..core.exceptions import SecurityError, ValidationError
            import os
                    from onyx.core.functions import validate_user_access
from typing import Any, List, Dict, Optional
import asyncio
"""
Onyx AI Video System - Security Manager

Security utilities for the Onyx AI Video system with integration
with Onyx's security patterns and access control.
"""





@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None
    validate_input: bool = True
    max_input_length: int = 10000
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    use_onyx_security: bool = True
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.mp4', '.avi', '.mov', '.webm', '.mkv'])
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    session_timeout: int = 3600  # 1 hour


@dataclass
class AccessControl:
    """Access control configuration."""
    user_id: str
    resource_id: str
    permissions: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """
    Security manager for AI Video system.
    
    Provides encryption, validation, access control, and security
    features with Onyx integration.
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        
    """__init__ function."""
self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Encryption
        self._fernet: Optional[Fernet] = None
        self._setup_encryption()
        
        # Rate limiting
        self._rate_limit_store: Dict[str, List[datetime]] = {}
        self._rate_limit_lock = None
        
        # Access control
        self._access_control_store: Dict[str, AccessControl] = {}
        
        # Input validation patterns
        self._validation_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'filename': r'^[a-zA-Z0-9._-]+$',
            'user_id': r'^[a-zA-Z0-9_-]+$',
            'request_id': r'^[a-zA-Z0-9_-]+$'
        }
    
    def _setup_encryption(self) -> Any:
        """Setup encryption with Fernet."""
        if not self.config.encryption_enabled:
            return
        
        try:
            if self.config.encryption_key:
                # Use provided key
                key = base64.urlsafe_b64encode(
                    hashlib.sha256(self.config.encryption_key.encode()).digest()
                )
            else:
                # Generate new key
                key = Fernet.generate_key()
            
            self._fernet = Fernet(key)
            self.logger.info("Encryption setup completed")
            
        except Exception as e:
            self.logger.error(f"Encryption setup failed: {e}")
            self.config.encryption_enabled = False
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data using Fernet encryption.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        if not self.config.encryption_enabled or not self._fernet:
            return data if isinstance(data, str) else data.decode('utf-8')
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self._fernet.encrypt(data)
            return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise SecurityError(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt data using Fernet encryption.
        
        Args:
            encrypted_data: Encrypted data as base64 string
            
        Returns:
            Decrypted data
        """
        if not self.config.encryption_enabled or not self._fernet:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self._fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise SecurityError(f"Decryption failed: {e}")
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash data with salt using PBKDF2.
        
        Args:
            data: Data to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(data.encode()))
            return key.decode(), salt
            
        except Exception as e:
            self.logger.error(f"Hashing failed: {e}")
            raise SecurityError(f"Hashing failed: {e}")
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """
        Verify hash against data.
        
        Args:
            data: Original data
            hash_value: Hash to verify
            salt: Salt used for hashing
            
        Returns:
            True if hash matches
        """
        try:
            expected_hash, _ = self.hash_data(data, salt)
            return hmac.compare_digest(hash_value, expected_hash)
            
        except Exception as e:
            self.logger.error(f"Hash verification failed: {e}")
            return False
    
    def validate_input(self, input_text: str, max_length: Optional[int] = None) -> Tuple[bool, str]:
        """
        Validate input text.
        
        Args:
            input_text: Text to validate
            max_length: Maximum length (uses config default if None)
            
        Returns:
            Tuple of (is_valid, cleaned_text_or_error)
        """
        if not self.config.validate_input:
            return True, input_text
        
        try:
            # Check length
            max_len = max_length or self.config.max_input_length
            if len(input_text) > max_len:
                return False, f"Input too long (max {max_len} characters)"
            
            # Check for empty input
            if not input_text.strip():
                return False, "Input cannot be empty"
            
            # Check for potentially dangerous patterns
            dangerous_patterns = [
                r'<script[^>]*>.*?</script>',  # Script tags
                r'javascript:',  # JavaScript protocol
                r'data:text/html',  # Data URLs
                r'vbscript:',  # VBScript
                r'on\w+\s*=',  # Event handlers
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, input_text, re.IGNORECASE):
                    return False, f"Input contains potentially dangerous content: {pattern}"
            
            # Clean and return
            cleaned_text = input_text.strip()
            return True, cleaned_text
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False, f"Validation error: {e}"
    
    def validate_pattern(self, value: str, pattern_name: str) -> bool:
        """
        Validate value against a pattern.
        
        Args:
            value: Value to validate
            pattern_name: Name of pattern to use
            
        Returns:
            True if value matches pattern
        """
        pattern = self._validation_patterns.get(pattern_name)
        if not pattern:
            return False
        
        return bool(re.match(pattern, value))
    
    def validate_file(self, file_path: str, allowed_extensions: Optional[List[str]] = None, max_size: Optional[int] = None) -> Tuple[bool, str]:
        """
        Validate file.
        
        Args:
            file_path: Path to file
            allowed_extensions: Allowed file extensions
            max_size: Maximum file size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            
            # Check if file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            max_file_size = max_size or self.config.max_file_size
            
            if file_size > max_file_size:
                return False, f"File too large (max {max_file_size} bytes)"
            
            # Check file extension
            ext = os.path.splitext(file_path)[1].lower()
            allowed_exts = allowed_extensions or self.config.allowed_file_extensions
            
            if ext not in allowed_exts:
                return False, f"File extension {ext} not allowed"
            
            return True, "File valid"
            
        except Exception as e:
            self.logger.error(f"File validation failed: {e}")
            return False, f"Validation error: {e}"
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        if not self.config.rate_limit_enabled:
            return True, {"allowed": True, "remaining": -1, "reset_time": None}
        
        try:
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=self.config.rate_limit_window)
            
            # Get user's request history
            user_requests = self._rate_limit_store.get(user_id, [])
            
            # Remove old requests outside window
            user_requests = [req_time for req_time in user_requests if req_time > window_start]
            
            # Check if limit exceeded
            if len(user_requests) >= self.config.rate_limit_requests:
                reset_time = user_requests[0] + timedelta(seconds=self.config.rate_limit_window)
                return False, {
                    "allowed": False,
                    "remaining": 0,
                    "reset_time": reset_time,
                    "limit": self.config.rate_limit_requests,
                    "window": self.config.rate_limit_window
                }
            
            # Update request history
            user_requests.append(current_time)
            self._rate_limit_store[user_id] = user_requests
            
            remaining = self.config.rate_limit_requests - len(user_requests)
            return True, {
                "allowed": True,
                "remaining": remaining,
                "reset_time": current_time + timedelta(seconds=self.config.rate_limit_window),
                "limit": self.config.rate_limit_requests,
                "window": self.config.rate_limit_window
            }
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True, {"allowed": True, "remaining": -1, "reset_time": None}
    
    def grant_access(self, user_id: str, resource_id: str, permissions: List[str], expires_at: Optional[datetime] = None) -> str:
        """
        Grant access to resource.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            permissions: List of permissions
            expires_at: Expiration time
            
        Returns:
            Access token
        """
        try:
            access_control = AccessControl(
                user_id=user_id,
                resource_id=resource_id,
                permissions=permissions,
                expires_at=expires_at
            )
            
            # Generate access token
            token_data = {
                "user_id": user_id,
                "resource_id": resource_id,
                "permissions": permissions,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "created_at": datetime.now().isoformat()
            }
            
            token = self.encrypt_data(json.dumps(token_data))
            self._access_control_store[token] = access_control
            
            return token
            
        except Exception as e:
            self.logger.error(f"Access grant failed: {e}")
            raise SecurityError(f"Access grant failed: {e}")
    
    def validate_access(self, user_id: str, resource_id: str, required_permissions: Optional[List[str]] = None) -> bool:
        """
        Validate user access to resource.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            required_permissions: Required permissions
            
        Returns:
            True if access is granted
        """
        try:
            # Check if Onyx security is available
            if self.config.use_onyx_security:
                try:
                    return validate_user_access(user_id, resource_id)
                except ImportError:
                    self.logger.warning("Onyx security not available, using local validation")
            
            # Local validation
            for access_control in self._access_control_store.values():
                if (access_control.user_id == user_id and 
                    access_control.resource_id == resource_id):
                    
                    # Check expiration
                    if access_control.expires_at and access_control.expires_at < datetime.now():
                        continue
                    
                    # Check permissions
                    if required_permissions:
                        if not all(perm in access_control.permissions for perm in required_permissions):
                            return False
                    
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Access validation failed: {e}")
            return False
    
    def revoke_access(self, token: str) -> bool:
        """
        Revoke access token.
        
        Args:
            token: Access token to revoke
            
        Returns:
            True if token was revoked
        """
        try:
            if token in self._access_control_store:
                del self._access_control_store[token]
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Access revocation failed: {e}")
            return False
    
    def cleanup_expired_access(self) -> Any:
        """Clean up expired access controls."""
        try:
            current_time = datetime.now()
            expired_tokens = []
            
            for token, access_control in self._access_control_store.items():
                if access_control.expires_at and access_control.expires_at < current_time:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                del self._access_control_store[token]
            
            if expired_tokens:
                self.logger.info(f"Cleaned up {len(expired_tokens)} expired access tokens")
                
        except Exception as e:
            self.logger.error(f"Access cleanup failed: {e}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate secure random token.
        
        Args:
            length: Token length
            
        Returns:
            Secure token
        """
        return secrets.token_urlsafe(length)
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for security.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized or "unnamed"
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status."""
        return {
            "encryption_enabled": self.config.encryption_enabled,
            "validate_input": self.config.validate_input,
            "rate_limit_enabled": self.config.rate_limit_enabled,
            "use_onyx_security": self.config.use_onyx_security,
            "active_access_tokens": len(self._access_control_store),
            "rate_limit_store_size": len(self._rate_limit_store),
            "timestamp": datetime.now().isoformat()
        }


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(config)
    return _security_manager


def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive data."""
    manager = get_security_manager()
    return manager.encrypt_data(data)


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data."""
    manager = get_security_manager()
    return manager.decrypt_data(encrypted_data)


def validate_input(input_text: str, max_length: Optional[int] = None) -> Tuple[bool, str]:
    """Validate input text."""
    manager = get_security_manager()
    return manager.validate_input(input_text, max_length)


def check_rate_limit(user_id: str) -> Tuple[bool, Dict[str, Any]]:
    """Check rate limit for user."""
    manager = get_security_manager()
    return manager.check_rate_limit(user_id)


def validate_access(user_id: str, resource_id: str, required_permissions: Optional[List[str]] = None) -> bool:
    """Validate user access to resource."""
    manager = get_security_manager()
    return manager.validate_access(user_id, resource_id, required_permissions)


def grant_access(user_id: str, resource_id: str, permissions: List[str], expires_at: Optional[datetime] = None) -> str:
    """Grant access to resource."""
    manager = get_security_manager()
    return manager.grant_access(user_id, resource_id, permissions, expires_at)


def revoke_access(token: str) -> bool:
    """Revoke access token."""
    manager = get_security_manager()
    return manager.revoke_access(token)


def generate_secure_token(length: int = 32) -> str:
    """Generate secure random token."""
    manager = get_security_manager()
    return manager.generate_secure_token(length)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for security."""
    manager = get_security_manager()
    return manager.sanitize_filename(filename)


def get_security_status() -> Dict[str, Any]:
    """Get security status."""
    manager = get_security_manager()
    return manager.get_security_status()


# Security decorators
def require_access(required_permissions: Optional[List[str]] = None):
    """Decorator to require access validation."""
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            # Extract user_id and resource_id from function arguments
            # This is a simplified implementation
            user_id = kwargs.get('user_id') or args[0] if args else None
            resource_id = kwargs.get('resource_id') or args[1] if len(args) > 1 else None
            
            if not user_id or not resource_id:
                raise SecurityError("user_id and resource_id required for access validation")
            
            if not validate_access(user_id, resource_id, required_permissions):
                raise SecurityError("Access denied")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limited(max_requests: int = 100, window_seconds: int = 60):
    """Decorator to apply rate limiting."""
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            user_id = kwargs.get('user_id') or args[0] if args else 'anonymous'
            
            allowed, rate_info = check_rate_limit(user_id)
            if not allowed:
                raise SecurityError(f"Rate limit exceeded. Reset at {rate_info['reset_time']}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input_decorator(max_length: Optional[int] = None):
    """Decorator to validate input."""
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            # Validate input_text parameter
            input_text = kwargs.get('input_text') or args[0] if args else None
            
            if input_text:
                is_valid, result = validate_input(input_text, max_length)
                if not is_valid:
                    raise ValidationError(result)
                # Update with cleaned text
                if 'input_text' in kwargs:
                    kwargs['input_text'] = result
                elif args:
                    args = (result,) + args[1:]
            
            return func(*args, **kwargs)
        return wrapper
    return decorator 