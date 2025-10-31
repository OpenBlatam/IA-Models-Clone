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

import ssl
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import json
import yaml
from datetime import datetime, timedelta
import os
            import ipaddress
from typing import Any, List, Dict, Optional
import asyncio
"""
Secure Configuration Module
==========================

Secure configuration management with secure defaults:
- TLS 1.2+ configuration
- Strong cipher suites
- Secure defaults for all settings
- Configuration validation
- Security parameter management
"""


# Get logger
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Custom configuration error."""
    def __init__(self, error_type: str, message: str, context: Optional[Dict] = None):
        
    """__init__ function."""
self.error_type = error_type
        self.message = message
        self.context = context or {}
        super().__init__(message)

class InvalidConfigurationError(Exception):
    """Custom invalid configuration error."""
    def __init__(self, field: str, value: Any, message: str, context: Optional[Dict] = None):
        
    """__init__ function."""
self.field = field
        self.value = value
        self.message = message
        self.context = context or {}
        super().__init__(message)

class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, field: str, value: Any, error_type: str, message: str, context: Optional[Dict] = None):
        
    """__init__ function."""
self.field = field
        self.value = value
        self.error_type = error_type
        self.message = message
        self.context = context or {}
        super().__init__(message)

class SecureConfig:
    """
    Secure configuration management with secure defaults.
    """
    
    def __init__(self) -> Any:
        """Initialize secure configuration with secure defaults."""
        # Secure TLS configuration
        self.tls_config = {
            'min_version': ssl.TLSVersion.TLSv1_2,
            'max_version': ssl.TLSVersion.TLSv1_3,
            'verify_mode': ssl.CERT_REQUIRED,
            'check_hostname': True,
            'ciphers': ':'.join([
                'ECDHE-ECDSA-AES256-GCM-SHA384',
                'ECDHE-RSA-AES256-GCM-SHA384',
                'ECDHE-ECDSA-CHACHA20-POLY1305',
                'ECDHE-RSA-CHACHA20-POLY1305',
                'ECDHE-ECDSA-AES128-GCM-SHA256',
                'ECDHE-RSA-AES128-GCM-SHA256',
                'ECDHE-ECDSA-AES256-SHA384',
                'ECDHE-RSA-AES256-SHA384',
                'ECDHE-ECDSA-AES128-SHA256',
                'ECDHE-RSA-AES128-SHA256'
            ]),
            'options': (
                ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1 |
                ssl.OP_NO_COMPRESSION | ssl.OP_SINGLE_DH_USE | ssl.OP_SINGLE_ECDH_USE |
                ssl.OP_NO_RENEGOTIATION
            )
        }
        
        # Secure HTTP headers
        self.security_headers = {
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
            'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        # Secure session configuration
        self.session_config = {
            'secret_key': secrets.token_urlsafe(32),
            'session_timeout': 3600,  # 1 hour
            'max_session_age': 86400,  # 24 hours
            'secure_cookies': True,
            'http_only_cookies': True,
            'same_site': 'Strict',
            'session_cookie_name': 'secure_session'
        }
        
        # Secure password policy
        self.password_policy = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_digits': True,
            'require_special_chars': True,
            'max_age_days': 90,
            'prevent_reuse_count': 5,
            'lockout_attempts': 5,
            'lockout_duration': 900,  # 15 minutes
            'password_history_size': 10
        }
        
        # Secure encryption settings
        self.encryption_config = {
            'algorithm': 'AES-256-GCM',
            'key_derivation': 'PBKDF2',
            'iterations': 100000,
            'salt_length': 32,
            'key_length': 32,
            'nonce_length': 12,
            'tag_length': 16
        }
        
        # Secure logging configuration
        self.logging_config = {
            'log_level': 'INFO',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': 'secure_app.log',
            'max_log_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'log_sensitive_data': False,
            'log_encryption': True
        }
        
        # Secure network configuration
        self.network_config = {
            'bind_address': '127.0.0.1',
            'port': 8443,
            'max_connections': 1000,
            'connection_timeout': 30,
            'keep_alive_timeout': 5,
            'max_request_size': 10 * 1024 * 1024,  # 10MB
            'rate_limit_requests': 100,
            'rate_limit_window': 60,  # 1 minute
            'enable_cors': False,
            'allowed_origins': []
        }
        
        # Secure file upload configuration
        self.upload_config = {
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'allowed_extensions': ['.txt', '.pdf', '.doc', '.docx', '.jpg', '.png', '.gif'],
            'allowed_mime_types': [
                'text/plain', 'application/pdf', 'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'image/jpeg', 'image/png', 'image/gif'
            ],
            'upload_directory': '/tmp/secure_uploads',
            'scan_uploads': True,
            'quarantine_suspicious': True
        }
    
    def get_tls_context(self) -> ssl.SSLContext:
        """
        Create a secure TLS context with secure defaults.
        
        Returns:
            Configured SSL context
        """
        context = ssl.create_default_context()
        
        # Set minimum and maximum TLS versions
        context.minimum_version = self.tls_config['min_version']
        context.maximum_version = self.tls_config['max_version']
        
        # Set verification mode
        context.verify_mode = self.tls_config['verify_mode']
        context.check_hostname = self.tls_config['check_hostname']
        
        # Set cipher suites
        context.set_ciphers(self.tls_config['ciphers'])
        
        # Set SSL options
        context.options = self.tls_config['options']
        
        logger.info("Secure TLS context created", extra={
            "min_version": str(self.tls_config['min_version']),
            "max_version": str(self.tls_config['max_version']),
            "verify_mode": self.tls_config['verify_mode'],
            "cipher_count": len(self.tls_config['ciphers'].split(':'))
        })
        
        return context
    
    def get_security_headers(self) -> Dict[str, str]:
        """
        Get secure HTTP headers.
        
        Returns:
            Dictionary of security headers
        """
        return self.security_headers.copy()
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """
        Validate password against security policy.
        
        Args:
            password: Password to validate
            
        Returns:
            Validation result dictionary
        """
        if not isinstance(password, str):
            raise ValidationError(
                "password",
                password,
                "invalid_type",
                "Password must be a string",
                context={"operation": "password_validation"}
            )
        
        errors = []
        
        # Check minimum length
        if len(password) < self.password_policy['min_length']:
            errors.append(f"Password must be at least {self.password_policy['min_length']} characters long")
        
        # Check character requirements
        if self.password_policy['require_uppercase'] and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.password_policy['require_lowercase'] and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.password_policy['require_digits'] and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.password_policy['require_special_chars'] and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            errors.append("Password must contain at least one special character")
        
        # Check for common patterns
        common_patterns = ['password', '123456', 'qwerty', 'admin', 'user']
        if password.lower() in common_patterns:
            errors.append("Password cannot be a common pattern")
        
        # Check for sequential characters
        if any(password[i:i+3] in 'abcdefghijklmnopqrstuvwxyz' for i in range(len(password)-2)):
            errors.append("Password cannot contain sequential characters")
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'strength_score': self._calculate_password_strength(password)
        }
    
    def _calculate_password_strength(self, password: str) -> int:
        """
        Calculate password strength score (0-100).
        
        Args:
            password: Password to score
            
        Returns:
            Strength score (0-100)
        """
        score = 0
        
        # Length contribution
        score += min(len(password) * 4, 40)
        
        # Character variety contribution
        if any(c.isupper() for c in password):
            score += 10
        if any(c.islower() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            score += 20
        
        # Complexity bonus
        unique_chars = len(set(password))
        score += min(unique_chars * 2, 20)
        
        return min(score, 100)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Secure token
        """
        return secrets.token_urlsafe(length)
    
    def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a secure random password.
        
        Args:
            length: Password length
            
        Returns:
            Secure password
        """
        # Define character sets
        lowercase = 'abcdefghijklmnopqrstuvwxyz'
        uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        digits = '0123456789'
        special = '!@#$%^&*()_+-=[]{}|;:,.<>?'
        
        # Ensure at least one character from each set
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # Fill remaining length with random characters
        all_chars = lowercase + uppercase + digits + special
        password.extend(secrets.choice(all_chars) for _ in range(length - 4))
        
        # Shuffle the password
        password_list = list(password)
        secrets.SystemRandom().shuffle(password_list)
        
        return ''.join(password_list)
    
    async def validate_file_upload_config(self, filename: str, content_type: str, file_size: int) -> Dict[str, Any]:
        """
        Validate file upload against security configuration.
        
        Args:
            filename: Uploaded filename
            content_type: File content type
            file_size: File size in bytes
            
        Returns:
            Validation result dictionary
        """
        errors = []
        
        # Check file size
        if file_size > self.upload_config['max_file_size']:
            errors.append(f"File size exceeds maximum allowed size of {self.upload_config['max_file_size']} bytes")
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.upload_config['allowed_extensions']:
            errors.append(f"File extension '{file_ext}' is not allowed")
        
        # Check MIME type
        if content_type not in self.upload_config['allowed_mime_types']:
            errors.append(f"Content type '{content_type}' is not allowed")
        
        # Check for suspicious patterns in filename
        suspicious_patterns = ['..', 'cmd', 'exe', 'bat', 'sh', 'php', 'asp', 'jsp']
        filename_lower = filename.lower()
        for pattern in suspicious_patterns:
            if pattern in filename_lower:
                errors.append(f"Filename contains suspicious pattern: {pattern}")
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'file_size': file_size,
            'content_type': content_type,
            'filename': filename
        }
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """
        Get rate limiting configuration.
        
        Returns:
            Rate limiting configuration
        """
        return {
            'requests_per_window': self.network_config['rate_limit_requests'],
            'window_seconds': self.network_config['rate_limit_window'],
            'enable_rate_limiting': True
        }
    
    def validate_network_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate network configuration.
        
        Args:
            config: Network configuration to validate
            
        Returns:
            Validation result dictionary
        """
        errors = []
        
        # Validate bind address
        bind_address = config.get('bind_address', self.network_config['bind_address'])
        if not self._is_valid_ip_address(bind_address):
            errors.append(f"Invalid bind address: {bind_address}")
        
        # Validate port
        port = config.get('port', self.network_config['port'])
        if not isinstance(port, int) or port < 1 or port > 65535:
            errors.append(f"Invalid port number: {port}")
        
        # Validate connection limits
        max_connections = config.get('max_connections', self.network_config['max_connections'])
        if not isinstance(max_connections, int) or max_connections < 1:
            errors.append(f"Invalid max connections: {max_connections}")
        
        # Validate timeouts
        connection_timeout = config.get('connection_timeout', self.network_config['connection_timeout'])
        if not isinstance(connection_timeout, (int, float)) or connection_timeout < 1:
            errors.append(f"Invalid connection timeout: {connection_timeout}")
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'validated_config': config
        }
    
    def _is_valid_ip_address(self, ip_address: str) -> bool:
        """
        Check if IP address is valid.
        
        Args:
            ip_address: IP address to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            ipaddress.ip_address(ip_address)
            return True
        except ValueError:
            return False
    
    def get_secure_defaults(self) -> Dict[str, Any]:
        """
        Get all secure default configurations.
        
        Returns:
            Dictionary of secure defaults
        """
        return {
            'tls_config': self.tls_config,
            'security_headers': self.security_headers,
            'session_config': self.session_config,
            'password_policy': self.password_policy,
            'encryption_config': self.encryption_config,
            'logging_config': self.logging_config,
            'network_config': self.network_config,
            'upload_config': self.upload_config
        }
    
    def load_config_from_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file with validation.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded and validated configuration
        """
        # Basic path validation
        if not os.path.exists(config_path):
            raise ConfigurationError(
                "config_file_not_found",
                f"Configuration file not found: {config_path}"
            )
        
        try:
            with open(config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith(('.yml', '.yaml')):
                    config = yaml.safe_load(f)
                else:
                    raise ConfigurationError(
                        "unsupported_config_format",
                        f"Unsupported configuration file format: {config_path}"
                    )
            
            # Validate loaded configuration
            return self.validate_config(config)
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(
                "config_parsing_error",
                f"Error parsing configuration file: {str(e)}"
            )
        except Exception as e:
            raise ConfigurationError(
                "config_loading_error",
                f"Error loading configuration: {str(e)}"
            )
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against security requirements.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated configuration
        """
        # Basic validation
        if not isinstance(config, dict):
            raise InvalidConfigurationError(
                "config",
                config,
                "Configuration must be a dictionary"
            )
        
        # Validate TLS configuration
        if 'tls_config' in config:
            tls_errors = self._validate_tls_config(config['tls_config'])
            if tls_errors:
                raise InvalidConfigurationError(
                    "tls_config",
                    config['tls_config'],
                    f"TLS configuration errors: {', '.join(tls_errors)}"
                )
        
        # Validate password policy
        if 'password_policy' in config:
            policy_errors = self._validate_password_policy(config['password_policy'])
            if policy_errors:
                raise InvalidConfigurationError(
                    "password_policy",
                    config['password_policy'],
                    f"Password policy errors: {', '.join(policy_errors)}"
                )
        
        # Validate network configuration
        if 'network_config' in config:
            network_validation = self.validate_network_config(config['network_config'])
            if not network_validation['is_valid']:
                raise InvalidConfigurationError(
                    "network_config",
                    config['network_config'],
                    f"Network configuration errors: {', '.join(network_validation['errors'])}"
                )
        
        logger.info("Configuration validated successfully", extra={
            "config_keys": list(config.keys())
        })
        
        return config
    
    def _validate_tls_config(self, tls_config: Dict[str, Any]) -> List[str]:
        """Validate TLS configuration."""
        errors = []
        
        # Check minimum version
        if 'min_version' in tls_config:
            min_version = tls_config['min_version']
            if min_version not in [ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3]:
                errors.append("Minimum TLS version must be TLSv1.2 or higher")
        
        # Check cipher suites
        if 'ciphers' in tls_config:
            ciphers = tls_config['ciphers']
            if not isinstance(ciphers, str) or len(ciphers.strip()) == 0:
                errors.append("Cipher suites must be specified")
        
        return errors
    
    def _validate_password_policy(self, policy: Dict[str, Any]) -> List[str]:
        """Validate password policy."""
        errors = []
        
        # Check minimum length
        if 'min_length' in policy:
            min_length = policy['min_length']
            if not isinstance(min_length, int) or min_length < 8:
                errors.append("Minimum password length must be at least 8 characters")
        
        # Check lockout settings
        if 'lockout_attempts' in policy:
            lockout_attempts = policy['lockout_attempts']
            if not isinstance(lockout_attempts, int) or lockout_attempts < 1:
                errors.append("Lockout attempts must be at least 1")
        
        return errors

# Global secure config instance
_secure_config = SecureConfig()

def get_secure_config() -> SecureConfig:
    """Get global secure configuration instance."""
    return _secure_config

def get_tls_context() -> ssl.SSLContext:
    """Get secure TLS context."""
    return _secure_config.get_tls_context()

def get_security_headers() -> Dict[str, str]:
    """Get secure HTTP headers."""
    return _secure_config.get_security_headers()

def validate_password(password: str) -> Dict[str, Any]:
    """Validate password against security policy."""
    return _secure_config.validate_password(password)

def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return _secure_config.generate_secure_token(length)

def generate_secure_password(length: int = 16) -> str:
    """Generate a secure random password."""
    return _secure_config.generate_secure_password(length)

async def validate_file_upload_config(filename: str, content_type: str, file_size: int) -> Dict[str, Any]:
    """Validate file upload against security configuration."""
    return _secure_config.validate_file_upload_config(filename, content_type, file_size)

def get_rate_limit_config() -> Dict[str, Any]:
    """Get rate limiting configuration."""
    return _secure_config.get_rate_limit_config()

def get_secure_defaults() -> Dict[str, Any]:
    """Get all secure default configurations."""
    return _secure_config.get_secure_defaults()

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from file with validation."""
    return _secure_config.load_config_from_file(config_path)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration against security requirements."""
    return _secure_config.validate_config(config)

# --- Named Exports ---

__all__ = [
    'SecureConfig',
    'ConfigurationError',
    'InvalidConfigurationError',
    'ValidationError',
    'get_secure_config',
    'get_tls_context',
    'get_security_headers',
    'validate_password',
    'generate_secure_token',
    'generate_secure_password',
    'validate_file_upload_config',
    'get_rate_limit_config',
    'get_secure_defaults',
    'load_config_from_file',
    'validate_config'
] 