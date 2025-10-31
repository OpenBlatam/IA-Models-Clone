from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import hashlib
import hmac
import os
import re
import time
import uuid
import ssl
import socket
import json
import base64
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from typing import Any, List, Dict, Optional
"""
Security Implementation - Practical application of security guidelines.
Implements secure coding patterns, validation, and ethical usage practices.
"""


# Configure secure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityError(Exception):
    """Custom security exception."""
    message: str
    code: str = "SECURITY_ERROR"
    details: Optional[Dict[str, Any]] = None

class SecureSecretManager:
    """Secure secret management with environment variables and secure stores."""
    
    def __init__(self) -> Any:
        self.secrets_cache: Dict[str, Any] = {}
        self.encrypted_secrets: Dict[str, bytes] = {}
        self.secret_sources = {
            'env': self._load_from_env,
            'file': self._load_from_secure_file,
            'vault': self._load_from_vault,
            'aws': self._load_from_aws_secrets,
            'azure': self._load_from_azure_keyvault,
            'gcp': self._load_from_gcp_secretmanager
        }
    
    def get_secret(self, secret_name: str, source: str = 'env', 
                   default: Optional[str] = None, required: bool = True) -> Optional[str]:
        """Get secret from specified source."""
        try:
            if source in self.secret_sources:
                secret = self.secret_sources[source](secret_name)
                if secret:
                    return secret
                elif required and default is None:
                    raise SecurityError(f"Required secret '{secret_name}' not found in {source}", 
                                     "SECRET_NOT_FOUND")
                else:
                    return default
            else:
                raise SecurityError(f"Unknown secret source: {source}", "INVALID_SECRET_SOURCE")
        except Exception as e:
            if required:
                raise SecurityError(f"Failed to load secret '{secret_name}': {str(e)}", 
                                 "SECRET_LOAD_ERROR")
            return default
    
    def _load_from_env(self, secret_name: str) -> Optional[str]:
        """Load secret from environment variable."""
        # Try different environment variable naming conventions
        env_vars = [
            secret_name,
            secret_name.upper(),
            secret_name.upper().replace('-', '_'),
            f'SECURITY_{secret_name.upper()}',
            f'CYBERSECURITY_{secret_name.upper()}',
            f'API_{secret_name.upper()}'
        ]
        
        for env_var in env_vars:
            value = os.getenv(env_var)
            if value:
                return value
        
        return None
    
    def _load_from_secure_file(self, secret_name: str) -> Optional[str]:
        """Load secret from secure file."""
        try:
            # Try common secure file locations
            file_paths = [
                f"/etc/security/{secret_name}",
                f"/opt/secrets/{secret_name}",
                f"./secrets/{secret_name}",
                f"./config/secrets/{secret_name}",
                os.path.expanduser(f"~/.security/{secret_name}")
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        return f.read().strip()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            return None
        except Exception:
            return None
    
    def _load_from_vault(self, secret_name: str) -> Optional[str]:
        """Load secret from HashiCorp Vault."""
        try:
            # This would integrate with HashiCorp Vault
            # For demo purposes, we'll simulate vault access
            vault_addr = os.getenv('VAULT_ADDR')
            vault_token = os.getenv('VAULT_TOKEN')
            
            if vault_addr and vault_token:
                # In a real implementation, this would use the vault client
                # import hvac
                # client = hvac.Client(url=vault_addr, token=vault_token)
                # secret = client.secrets.kv.v2.read_secret_version(path=secret_name)
                # return secret['data']['data']['value']
                return None
            
            return None
        except Exception:
            return None
    
    def _load_from_aws_secrets(self, secret_name: str) -> Optional[str]:
        """Load secret from AWS Secrets Manager."""
        try:
            # This would integrate with AWS Secrets Manager
            # For demo purposes, we'll simulate AWS access
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if aws_access_key and aws_secret_key:
                # In a real implementation, this would use boto3
                # import boto3
                # client = boto3.client('secretsmanager')
                # response = client.get_secret_value(SecretId=secret_name)
                # return response['SecretString']
                return None
            
            return None
        except Exception:
            return None
    
    def _load_from_azure_keyvault(self, secret_name: str) -> Optional[str]:
        """Load secret from Azure Key Vault."""
        try:
            # This would integrate with Azure Key Vault
            # For demo purposes, we'll simulate Azure access
            azure_tenant_id = os.getenv('AZURE_TENANT_ID')
            azure_client_id = os.getenv('AZURE_CLIENT_ID')
            azure_client_secret = os.getenv('AZURE_CLIENT_SECRET')
            
            if azure_tenant_id and azure_client_id and azure_client_secret:
                # In a real implementation, this would use azure-identity and azure-keyvault-secrets
                # from azure.identity import DefaultAzureCredential
                # from azure.keyvault.secrets import SecretClient
                # credential = DefaultAzureCredential()
                # client = SecretClient(vault_url="https://your-vault.vault.azure.net/", credential=credential)
                # secret = client.get_secret(secret_name)
                # return secret.value
                return None
            
            return None
        except Exception:
            return None
    
    def _load_from_gcp_secretmanager(self, secret_name: str) -> Optional[str]:
        """Load secret from Google Cloud Secret Manager."""
        try:
            # This would integrate with Google Cloud Secret Manager
            # For demo purposes, we'll simulate GCP access
            gcp_project = os.getenv('GOOGLE_CLOUD_PROJECT')
            gcp_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            
            if gcp_project and gcp_credentials:
                # In a real implementation, this would use google-cloud-secret-manager
                # from google.cloud import secretmanager
                # client = secretmanager.SecretManagerServiceClient()
                # name = f"projects/{gcp_project}/secrets/{secret_name}/versions/latest"
                # response = client.access_secret_version(request={"name": name})
                # return response.payload.data.decode("UTF-8")
                return None
            
            return None
        except Exception:
            return None
    
    def encrypt_secret(self, secret: str, key: Optional[bytes] = None) -> bytes:
        """Encrypt a secret for secure storage."""
        if not key:
            key = Fernet.generate_key()
        
        cipher_suite = Fernet(key)
        return cipher_suite.encrypt(secret.encode())
    
    def decrypt_secret(self, encrypted_secret: bytes, key: bytes) -> str:
        """Decrypt a secret."""
        cipher_suite = Fernet(key)
        return cipher_suite.decrypt(encrypted_secret).decode()
    
    def validate_secret_strength(self, secret: str, secret_type: str = "password") -> Dict[str, Any]:
        """Validate secret strength."""
        score = 0
        feedback = []
        
        # Length check
        if len(secret) >= 12:
            score += 2
        elif len(secret) >= 8:
            score += 1
        else:
            feedback.append("Secret should be at least 8 characters long")
        
        # Complexity checks
        if re.search(r'[a-z]', secret):
            score += 1
        else:
            feedback.append("Include lowercase letters")
        
        if re.search(r'[A-Z]', secret):
            score += 1
        else:
            feedback.append("Include uppercase letters")
        
        if re.search(r'\d', secret):
            score += 1
        else:
            feedback.append("Include numbers")
        
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', secret):
            score += 1
        else:
            feedback.append("Include special characters")
        
        # Strength assessment
        if score >= 5:
            strength = "STRONG"
        elif score >= 3:
            strength = "MEDIUM"
        else:
            strength = "WEAK"
        
        return {
            "score": score,
            "strength": strength,
            "feedback": feedback,
            "length": len(secret)
        }

@dataclass
class SecurityConfig:
    """Secure configuration management with TLS defaults and secret management."""
    api_key: Optional[str] = None
    encryption_key: Optional[bytes] = None
    max_retries: int = 3
    timeout: float = 10.0
    rate_limit: int = 100
    session_timeout: int = 3600
    
    # Secure TLS defaults
    tls_version: str = "TLSv1.2"
    min_tls_version: str = "TLSv1.2"
    cipher_suites: List[str] = None
    verify_ssl: bool = True
    cert_verify_mode: str = "CERT_REQUIRED"
    
    # Secret management
    secret_manager: Optional[SecureSecretManager] = None
    
    def __post_init__(self) -> Any:
        """Initialize secure defaults with secret management."""
        # Initialize secret manager
        if not self.secret_manager:
            self.secret_manager = SecureSecretManager()
        
        # Load API key from secure sources
        if not self.api_key:
            self.api_key = self.secret_manager.get_secret('api_key', 'env', required=False)
            if not self.api_key:
                self.api_key = self.secret_manager.get_secret('security_api_key', 'env', required=False)
            if not self.api_key:
                self.api_key = self.secret_manager.get_secret('cybersecurity_api_key', 'env', required=False)
        
        # Generate encryption key if not provided
        if not self.encryption_key:
            # Try to load from secure source first
            encryption_key_str = self.secret_manager.get_secret('encryption_key', 'env', required=False)
            if encryption_key_str:
                try:
                    self.encryption_key = base64.urlsafe_b64decode(encryption_key_str)
                except Exception:
                    self.encryption_key = Fernet.generate_key()
            else:
                self.encryption_key = Fernet.generate_key()
        
        # Initialize secure cipher suites
        if self.cipher_suites is None:
            self.cipher_suites = [
                'ECDHE-RSA-AES256-GCM-SHA384',
                'ECDHE-RSA-AES128-GCM-SHA256',
                'ECDHE-RSA-AES256-SHA384',
                'ECDHE-RSA-AES128-SHA256',
                'DHE-RSA-AES256-GCM-SHA384',
                'DHE-RSA-AES128-GCM-SHA256'
            ]
    
    def validate(self) -> bool:
        """Validate security configuration."""
        # Validate API key
        if not self.api_key:
            raise SecurityError("API key not configured", "MISSING_API_KEY")
        
        # Validate encryption key
        if not self.encryption_key:
            raise SecurityError("Encryption key not configured", "MISSING_ENCRYPTION_KEY")
        
        # Validate other parameters
        if self.max_retries < 0 or self.max_retries > 10:
            raise SecurityError("Invalid retry count", "INVALID_RETRY_COUNT")
        if self.timeout <= 0 or self.timeout > 300:
            raise SecurityError("Invalid timeout value", "INVALID_TIMEOUT")
        
        # Validate TLS configuration
        if self.tls_version not in ["TLSv1.2", "TLSv1.3"]:
            raise SecurityError("Invalid TLS version", "INVALID_TLS_VERSION")
        if self.min_tls_version not in ["TLSv1.2", "TLSv1.3"]:
            raise SecurityError("Invalid minimum TLS version", "INVALID_MIN_TLS_VERSION")
        
        return True
    
    def get_secret(self, secret_name: str, source: str = 'env', 
                   default: Optional[str] = None, required: bool = True) -> Optional[str]:
        """Get secret using the secret manager."""
        return self.secret_manager.get_secret(secret_name, source, default, required)
    
    def validate_secret(self, secret_name: str, secret_value: str) -> Dict[str, Any]:
        """Validate secret strength."""
        return self.secret_manager.validate_secret_strength(secret_value, secret_name)

class SecureTLSConfig:
    """Secure TLS configuration with strong defaults."""
    
    def __init__(self) -> Any:
        # Strong cipher suites (TLS 1.2+)
        self.strong_cipher_suites = [
            'ECDHE-RSA-AES256-GCM-SHA384',
            'ECDHE-RSA-AES128-GCM-SHA256',
            'ECDHE-RSA-AES256-SHA384',
            'ECDHE-RSA-AES128-SHA256',
            'DHE-RSA-AES256-GCM-SHA384',
            'DHE-RSA-AES128-GCM-SHA256',
            'ECDHE-ECDSA-AES256-GCM-SHA384',
            'ECDHE-ECDSA-AES128-GCM-SHA256'
        ]
        
        # TLS 1.3 cipher suites
        self.tls13_cipher_suites = [
            'TLS_AES_256_GCM_SHA384',
            'TLS_CHACHA20_POLY1305_SHA256',
            'TLS_AES_128_GCM_SHA256'
        ]
        
        # Disallowed weak cipher suites
        self.weak_cipher_suites = [
            'NULL', 'aNULL', 'eNULL', 'EXPORT', 'LOW', 'MEDIUM',
            'DES', '3DES', 'RC4', 'MD5', 'SHA1'
        ]
    
    def create_secure_ssl_context(self, tls_version: str = "TLSv1.2") -> ssl.SSLContext:
        """Create secure SSL context with strong defaults."""
        if tls_version == "TLSv1.3":
            context = ssl.SSLContext(ssl.PROTOCOL_TLS)
            context.minimum_version = ssl.TLSVersion.TLSv1_3
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Set secure defaults
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        
        # Set strong cipher suites
        if tls_version == "TLSv1.3":
            context.set_ciphers(':'.join(self.tls13_cipher_suites))
        else:
            # Filter out weak ciphers
            strong_ciphers = []
            for cipher in self.strong_cipher_suites:
                if not any(weak in cipher for weak in self.weak_cipher_suites):
                    strong_ciphers.append(cipher)
            context.set_ciphers(':'.join(strong_ciphers))
        
        # Set secure options
        context.options |= (
            ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1 |
            ssl.OP_NO_COMPRESSION | ssl.OP_NO_RENEGOTIATION
        )
        
        return context
    
    def validate_cipher_suite(self, cipher_suite: str) -> bool:
        """Validate cipher suite strength."""
        # Check for weak ciphers
        if any(weak in cipher_suite.upper() for weak in self.weak_cipher_suites):
            return False
        
        # Check for strong ciphers
        if any(strong in cipher_suite for strong in self.strong_cipher_suites):
            return True
        
        # Check for TLS 1.3 ciphers
        if any(tls13 in cipher_suite for tls13 in self.tls13_cipher_suites):
            return True
        
        return False
    
    def get_secure_cipher_list(self, tls_version: str = "TLSv1.2") -> List[str]:
        """Get list of secure cipher suites for specified TLS version."""
        if tls_version == "TLSv1.3":
            return self.tls13_cipher_suites
        else:
            return [cipher for cipher in self.strong_cipher_suites 
                   if self.validate_cipher_suite(cipher)]

class SecureInputValidator:
    """Secure input validation and sanitization."""
    
    def __init__(self) -> Any:
        self.dangerous_patterns = [
            r'\.\./', r'\.\.\\', r'script:', r'javascript:', r'data:',
            r'vbscript:', r'onload=', r'onerror=', r'onclick=',
            r'<script', r'</script>', r'<iframe', r'</iframe>'
        ]
        self.max_length = 1000
        self.allowed_schemes = {'http', 'https', 'ftp'}
    
    def validate_target(self, target: str) -> bool:
        """Comprehensive target validation."""
        if not target or not isinstance(target, str):
            return False
        
        # Check length
        if len(target) > self.max_length:
            return False
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, target, re.IGNORECASE):
                return False
        
        # Validate URL format
        try:
            parsed = urlparse(target)
            if parsed.scheme and parsed.scheme not in self.allowed_schemes:
                return False
            if not parsed.netloc:
                return False
        except Exception:
            return False
        
        return True
    
    def sanitize_input(self, data: Any) -> str:
        """Sanitize input data."""
        if not isinstance(data, str):
            data = str(data)
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>"\']', '', data)
        
        # Limit length
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length]
        
        return sanitized
    
    def validate_port_range(self, port_range: str) -> bool:
        """Validate port range specification."""
        try:
            if '-' in port_range:
                start, end = map(int, port_range.split('-'))
                return 1 <= start <= end <= 65535
            else:
                port = int(port_range)
                return 1 <= port <= 65535
        except (ValueError, TypeError):
            return False

class SecureDataHandler:
    """Secure data handling and encryption."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        
    """__init__ function."""
if encryption_key:
            self.key = encryption_key
        else:
            self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data."""
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash password with salt using PBKDF2."""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def verify_password(self, password: str, hashed: bytes, salt: bytes) -> bool:
        """Verify password against hash."""
        try:
            new_hash, _ = self.hash_password(password, salt)
            return hmac.compare_digest(hashed, new_hash)
        except Exception:
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return base64.urlsafe_b64encode(os.urandom(length)).decode('ascii')

class RateLimiter:
    """Advanced rate limiting with back-off for network scans."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60, 
                 backoff_multiplier: float = 2.0, max_backoff: int = 3600):
        
    """__init__ function."""
self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff = max_backoff
        self.requests: Dict[str, List[float]] = {}
        self.backoff_timers: Dict[str, float] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_request_times: Dict[str, float] = {}
    
    async def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limit with back-off."""
        now = time.time()
        
        # Check if in back-off period
        if identifier in self.backoff_timers:
            backoff_until = self.backoff_timers[identifier]
            if now < backoff_until:
                remaining_backoff = backoff_until - now
                raise SecurityError(
                    f"Rate limit back-off active for {remaining_backoff:.1f}s",
                    "RATE_LIMIT_BACKOFF"
                )
            else:
                # Back-off period expired
                del self.backoff_timers[identifier]
        
        # Check sliding window rate limit
        window_start = now - self.window_seconds
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.max_requests:
            # Calculate back-off period
            failure_count = self.failure_counts.get(identifier, 0) + 1
            self.failure_counts[identifier] = failure_count
            
            backoff_duration = min(
                self.window_seconds * (self.backoff_multiplier ** failure_count),
                self.max_backoff
            )
            
            self.backoff_timers[identifier] = now + backoff_duration
            
            raise SecurityError(
                f"Rate limit exceeded. Back-off for {backoff_duration:.1f}s",
                "RATE_LIMIT_EXCEEDED"
            )
        
        # Add request and update last request time
        self.requests[identifier].append(now)
        self.last_request_times[identifier] = now
        
        return True
    
    async def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        if identifier in self.requests:
            recent_requests = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
            return max(0, self.max_requests - len(recent_requests))
        
        return self.max_requests
    
    def get_backoff_status(self, identifier: str) -> Dict[str, Any]:
        """Get back-off status for identifier."""
        now = time.time()
        status = {
            "in_backoff": False,
            "remaining_backoff": 0.0,
            "failure_count": self.failure_counts.get(identifier, 0),
            "last_request": self.last_request_times.get(identifier, 0)
        }
        
        if identifier in self.backoff_timers:
            backoff_until = self.backoff_timers[identifier]
            if now < backoff_until:
                status["in_backoff"] = True
                status["remaining_backoff"] = backoff_until - now
        
        return status
    
    def reset_backoff(self, identifier: str):
        """Reset back-off for identifier."""
        if identifier in self.backoff_timers:
            del self.backoff_timers[identifier]
        if identifier in self.failure_counts:
            del self.failure_counts[identifier]
    
    def get_rate_limit_stats(self, identifier: str) -> Dict[str, Any]:
        """Get rate limit statistics for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        recent_requests = []
        if identifier in self.requests:
            recent_requests = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        
        return {
            "requests_in_window": len(recent_requests),
            "max_requests": self.max_requests,
            "remaining_requests": max(0, self.max_requests - len(recent_requests)),
            "window_seconds": self.window_seconds,
            "failure_count": self.failure_counts.get(identifier, 0),
            "last_request": self.last_request_times.get(identifier, 0),
            "backoff_status": self.get_backoff_status(identifier)
        }

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on target response."""
    
    def __init__(self, base_max_requests: int = 100, base_window_seconds: int = 60):
        
    """__init__ function."""
self.base_max_requests = base_max_requests
        self.base_window_seconds = base_window_seconds
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.target_responses: Dict[str, List[Dict[str, Any]]] = {}
        self.adaptive_configs: Dict[str, Dict[str, Any]] = {}
    
    def _create_adaptive_config(self, target: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Create adaptive configuration based on target response."""
        status_code = response.get('status_code', 200)
        response_time = response.get('response_time', 1.0)
        
        # Adjust rate limits based on response
        if status_code == 429:  # Too Many Requests
            return {
                'max_requests': max(10, self.base_max_requests // 4),
                'window_seconds': self.base_window_seconds * 2,
                'backoff_multiplier': 3.0
            }
        elif status_code >= 500:  # Server errors
            return {
                'max_requests': max(20, self.base_max_requests // 2),
                'window_seconds': self.base_window_seconds * 1.5,
                'backoff_multiplier': 2.0
            }
        elif response_time > 5.0:  # Slow response
            return {
                'max_requests': max(30, self.base_max_requests // 2),
                'window_seconds': self.base_window_seconds * 1.2,
                'backoff_multiplier': 1.5
            }
        else:  # Normal response
            return {
                'max_requests': self.base_max_requests,
                'window_seconds': self.base_window_seconds,
                'backoff_multiplier': 2.0
            }
    
    async def check_rate_limit(self, target: str, response: Dict[str, Any] = None) -> bool:
        """Check rate limit with adaptive adjustment."""
        # Update adaptive configuration based on response
        if response:
            self.adaptive_configs[target] = self._create_adaptive_config(target, response)
            if target not in self.target_responses:
                self.target_responses[target] = []
            self.target_responses[target].append(response)
            
            # Keep only recent responses
            if len(self.target_responses[target]) > 10:
                self.target_responses[target] = self.target_responses[target][-10:]
        
        # Get or create rate limiter for target
        if target not in self.rate_limiters:
            config = self.adaptive_configs.get(target, {
                'max_requests': self.base_max_requests,
                'window_seconds': self.base_window_seconds,
                'backoff_multiplier': 2.0
            })
            
            self.rate_limiters[target] = RateLimiter(
                max_requests=config['max_requests'],
                window_seconds=config['window_seconds'],
                backoff_multiplier=config['backoff_multiplier']
            )
        
        return await self.rate_limiters[target].check_rate_limit(target)
    
    def get_adaptive_stats(self, target: str) -> Dict[str, Any]:
        """Get adaptive rate limiting statistics."""
        if target not in self.rate_limiters:
            return {"error": "No rate limiter for target"}
        
        base_stats = self.rate_limiters[target].get_rate_limit_stats(target)
        adaptive_config = self.adaptive_configs.get(target, {})
        
        return {
            **base_stats,
            "adaptive_config": adaptive_config,
            "recent_responses": len(self.target_responses.get(target, [])),
            "target": target
        }

class NetworkScanRateLimiter:
    """Specialized rate limiter for network scanning operations."""
    
    def __init__(self) -> Any:
        self.scan_limits = {
            'port_scan': {'max_requests': 50, 'window_seconds': 300},  # 5 minutes
            'vulnerability_scan': {'max_requests': 20, 'window_seconds': 600},  # 10 minutes
            'web_scan': {'max_requests': 30, 'window_seconds': 300},  # 5 minutes
            'network_discovery': {'max_requests': 100, 'window_seconds': 1800},  # 30 minutes
        }
        self.scan_rate_limiters: Dict[str, RateLimiter] = {}
        self.scan_histories: Dict[str, List[Dict[str, Any]]] = {}
    
    async def check_scan_rate_limit(self, scan_type: str, target: str) -> bool:
        """Check rate limit for specific scan type."""
        if scan_type not in self.scan_limits:
            raise SecurityError(f"Unknown scan type: {scan_type}", "UNKNOWN_SCAN_TYPE")
        
        # Create rate limiter for scan type if not exists
        if scan_type not in self.scan_rate_limiters:
            limits = self.scan_limits[scan_type]
            self.scan_rate_limiters[scan_type] = RateLimiter(
                max_requests=limits['max_requests'],
                window_seconds=limits['window_seconds'],
                backoff_multiplier=2.5,
                max_backoff=7200  # 2 hours max back-off
            )
        
        # Check rate limit
        return await self.scan_rate_limiters[scan_type].check_rate_limit(target)
    
    def record_scan_result(self, scan_type: str, target: str, result: Dict[str, Any]):
        """Record scan result for adaptive learning."""
        if scan_type not in self.scan_histories:
            self.scan_histories[scan_type] = []
        
        self.scan_histories[scan_type].append({
            'target': target,
            'timestamp': time.time(),
            'result': result
        })
        
        # Keep only recent history
        if len(self.scan_histories[scan_type]) > 50:
            self.scan_histories[scan_type] = self.scan_histories[scan_type][-50:]
    
    def get_scan_stats(self, scan_type: str = None) -> Dict[str, Any]:
        """Get scanning statistics."""
        if scan_type:
            if scan_type not in self.scan_rate_limiters:
                return {"error": f"No rate limiter for scan type: {scan_type}"}
            
            return {
                "scan_type": scan_type,
                "limits": self.scan_limits[scan_type],
                "rate_limiter_stats": self.scan_rate_limiters[scan_type].get_rate_limit_stats(scan_type),
                "history_count": len(self.scan_histories.get(scan_type, []))
            }
        else:
            return {
                "scan_types": list(self.scan_limits.keys()),
                "active_rate_limiters": len(self.scan_rate_limiters),
                "total_history": sum(len(history) for history in self.scan_histories.values())
            }

class SecureSessionManager:
    """Secure session management."""
    
    def __init__(self, session_timeout: int = 3600):
        
    """__init__ function."""
self.session_timeout = session_timeout
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, user_id: str, permissions: List[str]) -> str:
        """Create secure session."""
        session_id = self._generate_session_token()
        
        self.sessions[session_id] = {
            'user_id': user_id,
            'permissions': permissions,
            'created_at': time.time(),
            'expires_at': time.time() + self.session_timeout
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check expiry
        if time.time() > session['expires_at']:
            del self.sessions[session_id]
            return False
        
        return True
    
    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get session data."""
        if not self.validate_session(session_id):
            return {}
        
        return self.sessions[session_id].copy()
    
    def revoke_session(self, session_id: str):
        """Revoke session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def _generate_session_token(self) -> str:
        """Generate secure session token."""
        return base64.urlsafe_b64encode(os.urandom(32)).decode('ascii')

class SecureLogger:
    """Secure logging with sensitive data redaction."""
    
    def __init__(self) -> Any:
        self.sensitive_patterns = [
            r'password["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            r'token["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            r'secret["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
            r'key["\']?\s*[:=]\s*["\']?[^"\']+["\']?',
        ]
    
    def redact_sensitive_data(self, message: str) -> str:
        """Redact sensitive data from log messages."""
        redacted = message
        
        for pattern in self.sensitive_patterns:
            redacted = re.sub(pattern, r'\1=***REDACTED***', redacted, flags=re.IGNORECASE)
        
        return redacted
    
    def secure_log(self, level: int, message: str, *args, **kwargs):
        """Secure logging with redaction."""
        redacted_message = self.redact_sensitive_data(message)
        logger.log(level, redacted_message, *args, **kwargs)

class AuthorizationChecker:
    """Authorization and consent management."""
    
    def __init__(self) -> Any:
        self.authorized_targets: Dict[str, Dict[str, Any]] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
    
    def add_authorized_target(self, target: str, owner: str, expiry: int, scope: List[str]):
        """Add authorized target."""
        self.authorized_targets[target] = {
            'owner': owner,
            'expiry': expiry,
            'scope': scope
        }
    
    def is_authorized(self, target: str, user: str, operation: str) -> bool:
        """Check if user is authorized for operation on target."""
        if target not in self.authorized_targets:
            return False
        
        permission = self.authorized_targets[target]
        
        # Check expiry
        if time.time() > permission['expiry']:
            self.remove_authorized_target(target)
            return False
        
        # Check ownership and scope
        if permission['owner'] != user or operation not in permission['scope']:
            return False
        
        return True
    
    def remove_authorized_target(self, target: str):
        """Remove authorized target."""
        if target in self.authorized_targets:
            del self.authorized_targets[target]
    
    def record_consent(self, user_id: str, consent_given: bool, purpose: str):
        """Record user consent."""
        self.consent_records[user_id] = {
            'consent_given': consent_given,
            'purpose': purpose,
            'timestamp': time.time(),
            'expiry': time.time() + 86400  # 24 hours
        }
    
    def has_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has consent for purpose."""
        if user_id not in self.consent_records:
            return False
        
        record = self.consent_records[user_id]
        
        # Check expiry
        if time.time() > record['expiry']:
            del self.consent_records[user_id]
            return False
        
        # Check purpose
        if record['purpose'] != purpose:
            return False
        
        return record['consent_given']

class SecureNetworkScanner:
    """Secure network scanning implementation with rate limiting and back-off."""
    
    def __init__(self, config: SecurityConfig):
        
    """__init__ function."""
self.config = config
        self.validator = SecureInputValidator()
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.adaptive_limiter = AdaptiveRateLimiter()
        self.scan_limiter = NetworkScanRateLimiter()
        self.logger = SecureLogger()
        self.auth_checker = AuthorizationChecker()
        self.tls_config = SecureTLSConfig()
    
    async def secure_scan(self, target: str, user: str, session_id: str, 
                         scan_type: str = "port_scan") -> Dict[str, Any]:
        """Perform secure network scan with rate limiting and back-off."""
        # Validate inputs
        if not self.validator.validate_target(target):
            raise SecurityError("Invalid target specification", "INVALID_TARGET")
        
        # Check authorization
        if not self.auth_checker.is_authorized(target, user, "scan"):
            raise SecurityError("Not authorized to scan target", "UNAUTHORIZED")
        
        # Check scan-specific rate limit
        try:
            await self.scan_limiter.check_scan_rate_limit(scan_type, target)
        except SecurityError as e:
            self.logger.secure_log(logging.WARNING, 
                                 f"Scan rate limit exceeded for {target}: {e.message}")
            raise SecurityError(f"Scan rate limit exceeded: {e.message}", e.code)
        
        # Check adaptive rate limit
        try:
            await self.adaptive_limiter.check_rate_limit(target)
        except SecurityError as e:
            self.logger.secure_log(logging.WARNING, 
                                 f"Adaptive rate limit exceeded for {target}: {e.message}")
            raise SecurityError(f"Adaptive rate limit exceeded: {e.message}", e.code)
        
        # Check consent
        if not self.auth_checker.has_consent(user, "network_scanning"):
            raise SecurityError("Consent required for network scanning", "CONSENT_REQUIRED")
        
        try:
            # Perform scan with proper error handling
            result = await self._perform_scan_with_backoff(target, scan_type)
            
            # Record scan result for adaptive learning
            self.scan_limiter.record_scan_result(scan_type, target, result)
            
            # Log securely
            self.logger.secure_log(logging.INFO, f"Scan completed for {target} by {user}")
            
            return {
                "success": True,
                "target": target,
                "scan_type": scan_type,
                "data": result,
                "timestamp": time.time(),
                "rate_limit_info": self._get_rate_limit_info(target, scan_type)
            }
            
        except Exception as e:
            # Log error securely
            self.logger.secure_log(logging.ERROR, f"Scan failed for {target}: {str(e)}")
            
            return {
                "success": False,
                "target": target,
                "scan_type": scan_type,
                "error": "Scan operation failed",
                "timestamp": time.time(),
                "rate_limit_info": self._get_rate_limit_info(target, scan_type)
            }
    
    async def _perform_scan_with_backoff(self, target: str, scan_type: str) -> Dict[str, Any]:
        """Perform scan with exponential back-off on failures."""
        max_retries = self.config.max_retries
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                result = await self._perform_scan(target, scan_type)
                
                # Update adaptive rate limiter with response
                response_info = {
                    'status_code': result.get('status_code', 200),
                    'response_time': result.get('response_time', 1.0),
                    'success': result.get('success', True)
                }
                
                await self.adaptive_limiter.check_rate_limit(target, response_info)
                
                return result
                
            except SecurityError as e:
                if e.code == "RATE_LIMIT_BACKOFF" or e.code == "RATE_LIMIT_EXCEEDED":
                    # Don't retry on rate limit errors
                    raise e
                
                if attempt < max_retries:
                    # Calculate back-off delay
                    delay = base_delay * (2 ** attempt)  # Exponential back-off
                    self.logger.secure_log(logging.WARNING, 
                                         f"Scan attempt {attempt + 1} failed for {target}, "
                                         f"retrying in {delay:.1f}s")
                    
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    raise e
    
    async def _perform_scan(self, target: str, scan_type: str) -> Dict[str, Any]:
        """Perform actual network scan based on type."""
        start_time = time.time()
        
        try:
            if scan_type == "port_scan":
                result = await self._port_scan(target)
            elif scan_type == "vulnerability_scan":
                result = await self._vulnerability_scan(target)
            elif scan_type == "web_scan":
                result = await self._web_scan(target)
            elif scan_type == "network_discovery":
                result = await self._network_discovery(target)
            else:
                raise SecurityError(f"Unknown scan type: {scan_type}", "UNKNOWN_SCAN_TYPE")
            
            result['response_time'] = time.time() - start_time
            result['status_code'] = 200
            result['success'] = True
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time,
                'status_code': 500
            }
    
    async def _port_scan(self, target: str) -> Dict[str, Any]:
        """Perform port scan with rate limiting."""
        # Simulate port scan with delays
        await asyncio.sleep(0.1)  # Simulate scan time
        
        return {
            "scan_type": "port_scan",
            "ports_scanned": [22, 80, 443, 8080, 3306],
            "open_ports": [80, 443],
            "scan_duration": 2.5,
            "target": target
        }
    
    async def _vulnerability_scan(self, target: str) -> Dict[str, Any]:
        """Perform vulnerability scan with rate limiting."""
        # Simulate vulnerability scan with longer delays
        await asyncio.sleep(0.5)  # Simulate scan time
        
        return {
            "scan_type": "vulnerability_scan",
            "vulnerabilities_found": 0,
            "scan_duration": 10.0,
            "target": target
        }
    
    async def _web_scan(self, target: str) -> Dict[str, Any]:
        """Perform web security scan with rate limiting."""
        # Simulate web scan
        await asyncio.sleep(0.2)  # Simulate scan time
        
        return {
            "scan_type": "web_scan",
            "security_headers": ["HSTS", "CSP"],
            "ssl_info": {"version": "TLSv1.2", "cipher": "ECDHE-RSA-AES256-GCM-SHA384"},
            "scan_duration": 5.0,
            "target": target
        }
    
    async def _network_discovery(self, target: str) -> Dict[str, Any]:
        """Perform network discovery with rate limiting."""
        # Simulate network discovery
        await asyncio.sleep(0.3)  # Simulate scan time
        
        return {
            "scan_type": "network_discovery",
            "hosts_found": 5,
            "services_discovered": ["HTTP", "HTTPS", "SSH"],
            "scan_duration": 15.0,
            "target": target
        }
    
    def _get_rate_limit_info(self, target: str, scan_type: str) -> Dict[str, Any]:
        """Get rate limiting information for target."""
        return {
            "adaptive_stats": self.adaptive_limiter.get_adaptive_stats(target),
            "scan_stats": self.scan_limiter.get_scan_stats(scan_type),
            "backoff_status": self.rate_limiter.get_backoff_status(target)
        }
    
    def get_scan_capabilities(self) -> Dict[str, Any]:
        """Get available scan capabilities and their rate limits."""
        return {
            "scan_types": {
                "port_scan": {
                    "description": "Port scanning with service detection",
                    "rate_limit": self.scan_limiter.scan_limits["port_scan"]
                },
                "vulnerability_scan": {
                    "description": "Vulnerability assessment",
                    "rate_limit": self.scan_limiter.scan_limits["vulnerability_scan"]
                },
                "web_scan": {
                    "description": "Web security assessment",
                    "rate_limit": self.scan_limiter.scan_limits["web_scan"]
                },
                "network_discovery": {
                    "description": "Network topology discovery",
                    "rate_limit": self.scan_limiter.scan_limits["network_discovery"]
                }
            },
            "rate_limiting": {
                "adaptive": True,
                "backoff": True,
                "scan_specific": True
            }
        }

# Utility functions for security operations
def create_secure_config() -> SecurityConfig:
    """Create secure configuration."""
    config = SecurityConfig()
    config.validate()
    return config

def validate_and_sanitize_input(data: Any) -> str:
    """Validate and sanitize input data."""
    validator = SecureInputValidator()
    return validator.sanitize_input(data)

def encrypt_sensitive_data(data: str, key: Optional[bytes] = None) -> bytes:
    """Encrypt sensitive data."""
    handler = SecureDataHandler(key)
    return handler.encrypt_data(data)

def verify_user_consent(user_id: str, purpose: str) -> bool:
    """Verify user consent for specific purpose."""
    auth_checker = AuthorizationChecker()
    return auth_checker.has_consent(user_id, purpose)

def create_secure_ssl_context(tls_version: str = "TLSv1.2") -> ssl.SSLContext:
    """Create secure SSL context with strong defaults."""
    tls_config = SecureTLSConfig()
    return tls_config.create_secure_ssl_context(tls_version)

def validate_cipher_suite(cipher_suite: str) -> bool:
    """Validate cipher suite strength."""
    tls_config = SecureTLSConfig()
    return tls_config.validate_cipher_suite(cipher_suite) 