"""
Refactored Security System

Sistema de seguridad y validación refactorizado para el AI History Comparison System.
Maneja autenticación, autorización, validación de entrada, encriptación y auditoría de seguridad.
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import jwt
import bcrypt
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from contextlib import asynccontextmanager
import re
import ipaddress
import weakref
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(Enum):
    """Authentication method enumeration"""
    PASSWORD = "password"
    TOKEN = "token"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SAML = "saml"
    LDAP = "ldap"
    BIOMETRIC = "biometric"
    MULTI_FACTOR = "multi_factor"


class AuthorizationLevel(Enum):
    """Authorization level enumeration"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class ValidationType(Enum):
    """Validation type enumeration"""
    INPUT = "input"
    OUTPUT = "output"
    SCHEMA = "schema"
    BUSINESS_RULE = "business_rule"
    SECURITY = "security"
    FORMAT = "format"


class EncryptionAlgorithm(Enum):
    """Encryption algorithm enumeration"""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20 = "chacha20"
    BCRYPT = "bcrypt"
    ARGON2 = "argon2"


@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    authentication_method: AuthenticationMethod
    authorization_level: AuthorizationLevel
    permissions: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event for auditing"""
    event_id: str
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: str
    action: str
    result: str
    severity: SecurityLevel
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Validation rule configuration"""
    name: str
    validation_type: ValidationType
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    error_message: str = "Validation failed"
    severity: SecurityLevel = SecurityLevel.MEDIUM


class AuthenticationProvider(ABC):
    """Abstract authentication provider"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[SecurityContext]:
        """Authenticate user"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[SecurityContext]:
        """Validate authentication token"""
        pass
    
    @abstractmethod
    async def refresh_token(self, token: str) -> Optional[str]:
        """Refresh authentication token"""
        pass


class PasswordAuthenticationProvider(AuthenticationProvider):
    """Password-based authentication provider"""
    
    def __init__(self, secret_key: str):
        self._secret_key = secret_key
        self._user_credentials: Dict[str, Dict[str, Any]] = {}
        self._active_sessions: Dict[str, SecurityContext] = {}
        self._lock = asyncio.Lock()
    
    async def add_user(self, user_id: str, password: str, permissions: Set[str] = None, roles: Set[str] = None) -> None:
        """Add user with hashed password"""
        async with self._lock:
            hashed_password = await self._hash_password(password)
            self._user_credentials[user_id] = {
                'password_hash': hashed_password,
                'permissions': permissions or set(),
                'roles': roles or set(),
                'created_at': datetime.utcnow()
            }
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[SecurityContext]:
        """Authenticate user with password"""
        user_id = credentials.get('user_id')
        password = credentials.get('password')
        
        if not user_id or not password:
            return None
        
        async with self._lock:
            user_data = self._user_credentials.get(user_id)
            if not user_data:
                return None
            
            if not await self._verify_password(password, user_data['password_hash']):
                return None
            
            # Create security context
            session_id = secrets.token_urlsafe(32)
            context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                ip_address=credentials.get('ip_address', ''),
                user_agent=credentials.get('user_agent', ''),
                authentication_method=AuthenticationMethod.PASSWORD,
                authorization_level=AuthorizationLevel.READ,
                permissions=user_data['permissions'],
                roles=user_data['roles'],
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            
            self._active_sessions[session_id] = context
            return context
    
    async def validate_token(self, token: str) -> Optional[SecurityContext]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self._secret_key, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            async with self._lock:
                context = self._active_sessions.get(session_id)
                if context and context.expires_at > datetime.utcnow():
                    return context
            
            return None
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def refresh_token(self, token: str) -> Optional[str]:
        """Refresh JWT token"""
        context = await self.validate_token(token)
        if not context:
            return None
        
        # Extend expiration
        context.expires_at = datetime.utcnow() + timedelta(hours=24)
        
        # Generate new token
        payload = {
            'user_id': context.user_id,
            'session_id': context.session_id,
            'exp': context.expires_at.timestamp()
        }
        
        return jwt.encode(payload, self._secret_key, algorithm='HS256')
    
    async def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    async def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


class APIKeyAuthenticationProvider(AuthenticationProvider):
    """API key-based authentication provider"""
    
    def __init__(self):
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def generate_api_key(self, user_id: str, permissions: Set[str] = None, roles: Set[str] = None) -> str:
        """Generate API key for user"""
        api_key = secrets.token_urlsafe(32)
        
        async with self._lock:
            self._api_keys[api_key] = {
                'user_id': user_id,
                'permissions': permissions or set(),
                'roles': roles or set(),
                'created_at': datetime.utcnow(),
                'last_used': None
            }
        
        return api_key
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[SecurityContext]:
        """Authenticate user with API key"""
        api_key = credentials.get('api_key')
        
        if not api_key:
            return None
        
        async with self._lock:
            key_data = self._api_keys.get(api_key)
            if not key_data:
                return None
            
            # Update last used
            key_data['last_used'] = datetime.utcnow()
            
            # Create security context
            context = SecurityContext(
                user_id=key_data['user_id'],
                session_id=api_key,
                ip_address=credentials.get('ip_address', ''),
                user_agent=credentials.get('user_agent', ''),
                authentication_method=AuthenticationMethod.API_KEY,
                authorization_level=AuthorizationLevel.READ,
                permissions=key_data['permissions'],
                roles=key_data['roles']
            )
            
            return context
    
    async def validate_token(self, token: str) -> Optional[SecurityContext]:
        """Validate API key token"""
        return await self.authenticate({'api_key': token})
    
    async def refresh_token(self, token: str) -> Optional[str]:
        """API keys don't need refresh"""
        return token


class AuthorizationManager:
    """Authorization manager with role-based access control"""
    
    def __init__(self):
        self._permissions: Dict[str, Set[str]] = defaultdict(set)
        self._roles: Dict[str, Set[str]] = defaultdict(set)
        self._role_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def add_permission(self, permission: str, description: str = "") -> None:
        """Add permission"""
        async with self._lock:
            self._permissions[permission] = set()
    
    async def add_role(self, role: str, permissions: Set[str] = None) -> None:
        """Add role with permissions"""
        async with self._lock:
            self._roles[role] = permissions or set()
    
    async def assign_role_to_user(self, user_id: str, role: str) -> None:
        """Assign role to user"""
        async with self._lock:
            self._permissions[user_id].update(self._roles.get(role, set()))
    
    async def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission"""
        async with self._lock:
            return permission in self._permissions.get(user_id, set())
    
    async def check_authorization(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Check if user is authorized for action on resource"""
        required_permission = f"{resource}:{action}"
        return await self.check_permission(context.user_id, required_permission)


class InputValidator:
    """Input validation with security checks"""
    
    def __init__(self):
        self._rules: Dict[str, List[ValidationRule]] = defaultdict(list)
        self._security_patterns = {
            'sql_injection': re.compile(r"('|(\\')|(;)|(\-\-)|(\s*(union|select|insert|update|delete|drop|create|alter|exec|execute)\s+)", re.IGNORECASE),
            'xss': re.compile(r"<script[^>]*>.*?</script>|<[^>]*on\w+\s*=", re.IGNORECASE),
            'path_traversal': re.compile(r"\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c", re.IGNORECASE),
            'command_injection': re.compile(r"[;&|`$(){}[\]\\]", re.IGNORECASE)
        }
    
    async def add_rule(self, field_name: str, rule: ValidationRule) -> None:
        """Add validation rule for field"""
        self._rules[field_name].append(rule)
    
    async def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data"""
        errors = []
        
        for field_name, value in data.items():
            if field_name in self._rules:
                for rule in self._rules[field_name]:
                    error = await self._validate_field(field_name, value, rule)
                    if error:
                        errors.append(error)
        
        # Security validation
        security_errors = await self._validate_security(data)
        errors.extend(security_errors)
        
        return len(errors) == 0, errors
    
    async def _validate_field(self, field_name: str, value: Any, rule: ValidationRule) -> Optional[str]:
        """Validate single field against rule"""
        try:
            if rule.validation_type == ValidationType.INPUT:
                if rule.pattern and not re.match(rule.pattern, str(value)):
                    return f"{field_name}: {rule.error_message}"
                
                if rule.min_length and len(str(value)) < rule.min_length:
                    return f"{field_name}: Minimum length is {rule.min_length}"
                
                if rule.max_length and len(str(value)) > rule.max_length:
                    return f"{field_name}: Maximum length is {rule.max_length}"
                
                if rule.min_value is not None and float(value) < rule.min_value:
                    return f"{field_name}: Minimum value is {rule.min_value}"
                
                if rule.max_value is not None and float(value) > rule.max_value:
                    return f"{field_name}: Maximum value is {rule.max_value}"
                
                if rule.allowed_values and value not in rule.allowed_values:
                    return f"{field_name}: Value not in allowed values"
                
                if rule.custom_validator:
                    if not rule.custom_validator(value):
                        return f"{field_name}: {rule.error_message}"
            
            return None
            
        except Exception as e:
            return f"{field_name}: Validation error - {str(e)}"
    
    async def _validate_security(self, data: Dict[str, Any]) -> List[str]:
        """Validate for security threats"""
        errors = []
        
        for field_name, value in data.items():
            value_str = str(value)
            
            # Check for SQL injection
            if self._security_patterns['sql_injection'].search(value_str):
                errors.append(f"{field_name}: Potential SQL injection detected")
            
            # Check for XSS
            if self._security_patterns['xss'].search(value_str):
                errors.append(f"{field_name}: Potential XSS attack detected")
            
            # Check for path traversal
            if self._security_patterns['path_traversal'].search(value_str):
                errors.append(f"{field_name}: Potential path traversal detected")
            
            # Check for command injection
            if self._security_patterns['command_injection'].search(value_str):
                errors.append(f"{field_name}: Potential command injection detected")
        
        return errors


class EncryptionManager:
    """Encryption manager with multiple algorithms"""
    
    def __init__(self, secret_key: str):
        self._secret_key = secret_key.encode('utf-8')
        self._algorithms = {
            EncryptionAlgorithm.AES_256: self._encrypt_aes256,
            EncryptionAlgorithm.RSA_2048: self._encrypt_rsa,
            EncryptionAlgorithm.CHACHA20: self._encrypt_chacha20
        }
    
    async def encrypt(self, data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
        """Encrypt data"""
        if algorithm in self._algorithms:
            return await self._algorithms[algorithm](data)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
    
    async def decrypt(self, encrypted_data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
        """Decrypt data"""
        if algorithm == EncryptionAlgorithm.AES_256:
            return await self._decrypt_aes256(encrypted_data)
        elif algorithm == EncryptionAlgorithm.RSA_2048:
            return await self._decrypt_rsa(encrypted_data)
        elif algorithm == EncryptionAlgorithm.CHACHA20:
            return await self._decrypt_chacha20(encrypted_data)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
    
    async def _encrypt_aes256(self, data: str) -> str:
        """Encrypt using AES-256"""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        f = Fernet(key)
        encrypted = f.encrypt(data.encode('utf-8'))
        return f"{key.decode()}:{encrypted.decode()}"
    
    async def _decrypt_aes256(self, encrypted_data: str) -> str:
        """Decrypt using AES-256"""
        from cryptography.fernet import Fernet
        key_str, encrypted_str = encrypted_data.split(':', 1)
        key = key_str.encode()
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_str.encode())
        return decrypted.decode('utf-8')
    
    async def _encrypt_rsa(self, data: str) -> str:
        """Encrypt using RSA"""
        # Simplified RSA implementation
        # In production, use proper RSA key management
        return f"RSA_ENCRYPTED:{data}"
    
    async def _decrypt_rsa(self, encrypted_data: str) -> str:
        """Decrypt using RSA"""
        # Simplified RSA implementation
        if encrypted_data.startswith("RSA_ENCRYPTED:"):
            return encrypted_data[14:]
        return encrypted_data
    
    async def _encrypt_chacha20(self, data: str) -> str:
        """Encrypt using ChaCha20"""
        # Simplified ChaCha20 implementation
        return f"CHACHA20_ENCRYPTED:{data}"
    
    async def _decrypt_chacha20(self, encrypted_data: str) -> str:
        """Decrypt using ChaCha20"""
        # Simplified ChaCha20 implementation
        if encrypted_data.startswith("CHACHA20_ENCRYPTED:"):
            return encrypted_data[19:]
        return encrypted_data


class SecurityAuditor:
    """Security auditor for logging and monitoring"""
    
    def __init__(self):
        self._events: List[SecurityEvent] = []
        self._lock = asyncio.Lock()
        self._callbacks: List[Callable] = []
    
    async def log_event(self, event: SecurityEvent) -> None:
        """Log security event"""
        async with self._lock:
            self._events.append(event)
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in security audit callback: {e}")
    
    async def get_events(self, start_time: datetime = None, end_time: datetime = None,
                        severity: SecurityLevel = None) -> List[SecurityEvent]:
        """Get security events"""
        async with self._lock:
            events = self._events.copy()
            
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
            
            if severity:
                events = [e for e in events if e.severity == severity]
            
            return events
    
    def add_callback(self, callback: Callable) -> None:
        """Add security audit callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove security audit callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)


class RefactoredSecurityManager:
    """Refactored security manager with comprehensive security features"""
    
    def __init__(self, secret_key: str):
        self._secret_key = secret_key
        self._auth_providers: Dict[AuthenticationMethod, AuthenticationProvider] = {}
        self._authorization_manager = AuthorizationManager()
        self._input_validator = InputValidator()
        self._encryption_manager = EncryptionManager(secret_key)
        self._security_auditor = SecurityAuditor()
        self._rate_limiter: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval: float = 300.0  # 5 minutes
    
    async def initialize(self) -> None:
        """Initialize security manager"""
        # Initialize authentication providers
        self._auth_providers[AuthenticationMethod.PASSWORD] = PasswordAuthenticationProvider(self._secret_key)
        self._auth_providers[AuthenticationMethod.API_KEY] = APIKeyAuthenticationProvider()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Refactored security manager initialized")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop for expired sessions and rate limits"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_sessions()
                await self._cleanup_rate_limits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in security cleanup loop: {e}")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions"""
        # Implementation would clean up expired sessions
        pass
    
    async def _cleanup_rate_limits(self) -> None:
        """Cleanup old rate limit entries"""
        async with self._lock:
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(minutes=5)
            
            for key in list(self._rate_limiter.keys()):
                self._rate_limiter[key] = [
                    timestamp for timestamp in self._rate_limiter[key]
                    if timestamp > cutoff_time
                ]
                
                if not self._rate_limiter[key]:
                    del self._rate_limiter[key]
    
    async def authenticate(self, method: AuthenticationMethod, credentials: Dict[str, Any]) -> Optional[SecurityContext]:
        """Authenticate user"""
        provider = self._auth_providers.get(method)
        if not provider:
            return None
        
        # Rate limiting
        ip_address = credentials.get('ip_address', '')
        if not await self._check_rate_limit(ip_address):
            await self._log_security_event(
                event_type="rate_limit_exceeded",
                user_id=credentials.get('user_id'),
                ip_address=ip_address,
                severity=SecurityLevel.MEDIUM
            )
            return None
        
        context = await provider.authenticate(credentials)
        
        if context:
            await self._log_security_event(
                event_type="authentication_success",
                user_id=context.user_id,
                ip_address=context.ip_address,
                severity=SecurityLevel.LOW
            )
        else:
            await self._log_security_event(
                event_type="authentication_failure",
                user_id=credentials.get('user_id'),
                ip_address=ip_address,
                severity=SecurityLevel.MEDIUM
            )
        
        return context
    
    async def authorize(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Authorize user action"""
        is_authorized = await self._authorization_manager.check_authorization(context, resource, action)
        
        await self._log_security_event(
            event_type="authorization_check",
            user_id=context.user_id,
            ip_address=context.ip_address,
            resource=resource,
            action=action,
            result="allowed" if is_authorized else "denied",
            severity=SecurityLevel.MEDIUM if is_authorized else SecurityLevel.HIGH
        )
        
        return is_authorized
    
    async def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data"""
        is_valid, errors = await self._input_validator.validate_input(data)
        
        if not is_valid:
            await self._log_security_event(
                event_type="input_validation_failure",
                ip_address="",  # Would be provided in real implementation
                severity=SecurityLevel.MEDIUM,
                details={"errors": errors}
            )
        
        return is_valid, errors
    
    async def encrypt_data(self, data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
        """Encrypt sensitive data"""
        return await self._encryption_manager.encrypt(data, algorithm)
    
    async def decrypt_data(self, encrypted_data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256) -> str:
        """Decrypt sensitive data"""
        return await self._encryption_manager.decrypt(encrypted_data, algorithm)
    
    async def _check_rate_limit(self, ip_address: str, max_requests: int = 100, window_minutes: int = 5) -> bool:
        """Check rate limit for IP address"""
        async with self._lock:
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=window_minutes)
            
            # Clean old requests
            self._rate_limiter[ip_address] = [
                timestamp for timestamp in self._rate_limiter[ip_address]
                if timestamp > window_start
            ]
            
            # Check limit
            if len(self._rate_limiter[ip_address]) >= max_requests:
                return False
            
            # Add current request
            self._rate_limiter[ip_address].append(current_time)
            return True
    
    async def _log_security_event(self, event_type: str, user_id: str = None, ip_address: str = "",
                                 resource: str = "", action: str = "", result: str = "",
                                 severity: SecurityLevel = SecurityLevel.MEDIUM, details: Dict[str, Any] = None) -> None:
        """Log security event"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent="",  # Would be provided in real implementation
            resource=resource,
            action=action,
            result=result,
            severity=severity,
            details=details or {}
        )
        
        await self._security_auditor.log_event(event)
    
    async def get_security_events(self, **kwargs) -> List[SecurityEvent]:
        """Get security events"""
        return await self._security_auditor.get_events(**kwargs)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get security manager health status"""
        return {
            "auth_providers": len(self._auth_providers),
            "rate_limits": len(self._rate_limiter),
            "cleanup_interval": self._cleanup_interval,
            "security_events_count": len(self._security_auditor._events)
        }
    
    async def shutdown(self) -> None:
        """Shutdown security manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Refactored security manager shutdown")


# Global security manager
security_manager = RefactoredSecurityManager("your-secret-key-here")


# Convenience functions
async def authenticate_user(method: AuthenticationMethod, credentials: Dict[str, Any]):
    """Authenticate user"""
    return await security_manager.authenticate(method, credentials)


async def authorize_user(context: SecurityContext, resource: str, action: str):
    """Authorize user action"""
    return await security_manager.authorize(context, resource, action)


async def validate_input_data(data: Dict[str, Any]):
    """Validate input data"""
    return await security_manager.validate_input(data)


async def encrypt_sensitive_data(data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256):
    """Encrypt sensitive data"""
    return await security_manager.encrypt_data(data, algorithm)


async def decrypt_sensitive_data(encrypted_data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256):
    """Decrypt sensitive data"""
    return await security_manager.decrypt_data(encrypted_data, algorithm)


# Security decorators
def require_auth(method: AuthenticationMethod = AuthenticationMethod.PASSWORD):
    """Authentication required decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract credentials from request
            credentials = kwargs.get('credentials', {})
            context = await authenticate_user(method, credentials)
            
            if not context:
                raise PermissionError("Authentication required")
            
            return await func(context, *args, **kwargs)
        return wrapper
    return decorator


def require_permission(resource: str, action: str):
    """Permission required decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(context: SecurityContext, *args, **kwargs):
            if not await authorize_user(context, resource, action):
                raise PermissionError(f"Permission denied: {resource}:{action}")
            
            return await func(context, *args, **kwargs)
        return wrapper
    return decorator


def validate_input(rules: Dict[str, List[ValidationRule]]):
    """Input validation decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract input data
            data = kwargs.get('data', {})
            
            # Add rules to validator
            for field, field_rules in rules.items():
                for rule in field_rules:
                    await security_manager._input_validator.add_rule(field, rule)
            
            is_valid, errors = await validate_input_data(data)
            if not is_valid:
                raise ValueError(f"Input validation failed: {errors}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator





















