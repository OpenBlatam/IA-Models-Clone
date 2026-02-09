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
import socket
import hashlib
import secrets
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import yaml
from pathlib import Path
import tempfile
import os
from datetime import datetime, timedelta
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID
import certifi
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
        import string
from fastapi import APIRouter
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Secure Defaults System for Cybersecurity Tools
Implements TLSv1.2+, strong cipher suites, and security best practices
"""


class SecurityLevel(Enum):
    """Security levels for different environments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CipherStrength(Enum):
    """Cipher strength levels"""
    WEAK = "weak"      # < 128 bits
    MEDIUM = "medium"  # 128-256 bits
    STRONG = "strong"  # > 256 bits
    MAXIMUM = "maximum" # Maximum available

@dataclass
class TLSSecurityConfig:
    """TLS security configuration"""
    min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2
    max_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_3
    cipher_suites: List[str] = None
    cert_reqs: int = ssl.CERT_REQUIRED
    verify_mode: int = ssl.CERT_REQUIRED
    check_hostname: bool = True
    session_tickets: bool = False
    session_cache_size: int = 0
    session_timeout: int = 300
    
    def __post_init__(self) -> Any:
        if self.cipher_suites is None:
            self.cipher_suites = self._get_strong_cipher_suites()

    def _get_strong_cipher_suites(self) -> List[str]:
        """Get strong cipher suites"""
        return [
            # TLS 1.3 cipher suites (strongest)
            'TLS_AES_256_GCM_SHA384',
            'TLS_CHACHA20_POLY1305_SHA256',
            'TLS_AES_128_GCM_SHA256',
            
            # TLS 1.2 strong cipher suites
            'ECDHE-RSA-AES256-GCM-SHA384',
            'ECDHE-RSA-AES128-GCM-SHA256',
            'ECDHE-RSA-CHACHA20-POLY1305',
            'ECDHE-ECDSA-AES256-GCM-SHA384',
            'ECDHE-ECDSA-AES128-GCM-SHA256',
            'ECDHE-ECDSA-CHACHA20-POLY1305',
            'DHE-RSA-AES256-GCM-SHA384',
            'DHE-RSA-AES128-GCM-SHA256',
            'DHE-RSA-CHACHA20-POLY1305'
        ]

@dataclass
class CryptoConfig:
    """Cryptographic configuration"""
    hash_algorithm: str = "sha256"
    key_size: int = 4096  # RSA key size
    curve: str = "secp384r1"  # ECC curve
    encryption_algorithm: str = "AES-256-GCM"
    pbkdf2_iterations: int = 100000
    salt_length: int = 32
    iv_length: int = 16
    tag_length: int = 16

@dataclass
class SecurityDefaults:
    """Security defaults configuration"""
    tls_config: TLSSecurityConfig = None
    crypto_config: CryptoConfig = None
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    password_min_length: int = 12
    require_special_chars: bool = True
    require_numbers: bool = True
    require_uppercase: bool = True
    require_lowercase: bool = True
    max_session_age: int = 86400  # 24 hours
    secure_cookies: bool = True
    http_only_cookies: bool = True
    same_site_cookies: str = "strict"
    csrf_protection: bool = True
    rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = None
    max_file_size: int = 5 * 1024 * 1024  # 5MB
    
    def __post_init__(self) -> Any:
        if self.tls_config is None:
            self.tls_config = TLSSecurityConfig()
        if self.crypto_config is None:
            self.crypto_config = CryptoConfig()
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.txt', '.log', '.csv', '.json', '.xml']

class SecureDefaultsManager:
    """Manager for secure defaults configuration"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        
    """__init__ function."""
self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        self.defaults = self._get_defaults_for_level(security_level)
        
    def _get_defaults_for_level(self, level: SecurityLevel) -> SecurityDefaults:
        """Get security defaults for specified level"""
        if level == SecurityLevel.LOW:
            return SecurityDefaults(
                tls_config=TLSSecurityConfig(
                    min_version=ssl.TLSVersion.TLSv1_1,
                    cipher_suites=['ECDHE-RSA-AES128-GCM-SHA256', 'DHE-RSA-AES128-GCM-SHA256'],
                    cert_reqs=ssl.CERT_NONE,
                    verify_mode=ssl.CERT_NONE,
                    check_hostname=False
                ),
                crypto_config=CryptoConfig(
                    hash_algorithm="sha1",
                    key_size=2048,
                    curve="secp256r1",
                    encryption_algorithm="AES-128-CBC",
                    pbkdf2_iterations=10000
                ),
                session_timeout=7200,
                max_login_attempts=10,
                lockout_duration=300,
                password_min_length=8,
                require_special_chars=False,
                require_numbers=False,
                require_uppercase=False,
                require_lowercase=False,
                secure_cookies=False,
                http_only_cookies=False,
                same_site_cookies="lax",
                csrf_protection=False,
                rate_limiting=False,
                rate_limit_per_minute=1000,
                max_request_size=100 * 1024 * 1024,
                max_file_size=50 * 1024 * 1024
            )
        
        elif level == SecurityLevel.MEDIUM:
            return SecurityDefaults(
                tls_config=TLSSecurityConfig(
                    min_version=ssl.TLSVersion.TLSv1_2,
                    cipher_suites=['ECDHE-RSA-AES256-GCM-SHA384', 'ECDHE-RSA-AES128-GCM-SHA256'],
                    cert_reqs=ssl.CERT_OPTIONAL,
                    verify_mode=ssl.CERT_OPTIONAL,
                    check_hostname=True
                ),
                crypto_config=CryptoConfig(
                    hash_algorithm="sha256",
                    key_size=3072,
                    curve="secp384r1",
                    encryption_algorithm="AES-256-CBC",
                    pbkdf2_iterations=50000
                ),
                session_timeout=3600,
                max_login_attempts=7,
                lockout_duration=600,
                password_min_length=10,
                require_special_chars=True,
                require_numbers=True,
                require_uppercase=True,
                require_lowercase=True,
                secure_cookies=True,
                http_only_cookies=True,
                same_site_cookies="strict",
                csrf_protection=True,
                rate_limiting=True,
                rate_limit_per_minute=120,
                max_request_size=50 * 1024 * 1024,
                max_file_size=25 * 1024 * 1024
            )
        
        elif level == SecurityLevel.HIGH:
            return SecurityDefaults(
                tls_config=TLSSecurityConfig(
                    min_version=ssl.TLSVersion.TLSv1_2,
                    max_version=ssl.TLSVersion.TLSv1_3,
                    cipher_suites=[
                        'TLS_AES_256_GCM_SHA384',
                        'TLS_CHACHA20_POLY1305_SHA256',
                        'ECDHE-RSA-AES256-GCM-SHA384',
                        'ECDHE-RSA-CHACHA20-POLY1305'
                    ],
                    cert_reqs=ssl.CERT_REQUIRED,
                    verify_mode=ssl.CERT_REQUIRED,
                    check_hostname=True,
                    session_tickets=False,
                    session_cache_size=0
                ),
                crypto_config=CryptoConfig(
                    hash_algorithm="sha384",
                    key_size=4096,
                    curve="secp384r1",
                    encryption_algorithm="AES-256-GCM",
                    pbkdf2_iterations=100000,
                    salt_length=32,
                    iv_length=16,
                    tag_length=16
                ),
                session_timeout=1800,
                max_login_attempts=5,
                lockout_duration=900,
                password_min_length=12,
                require_special_chars=True,
                require_numbers=True,
                require_uppercase=True,
                require_lowercase=True,
                max_session_age=3600,
                secure_cookies=True,
                http_only_cookies=True,
                same_site_cookies="strict",
                csrf_protection=True,
                rate_limiting=True,
                rate_limit_per_minute=60,
                max_request_size=10 * 1024 * 1024,
                max_file_size=5 * 1024 * 1024
            )
        
        else:  # SecurityLevel.CRITICAL
            return SecurityDefaults(
                tls_config=TLSSecurityConfig(
                    min_version=ssl.TLSVersion.TLSv1_3,
                    max_version=ssl.TLSVersion.TLSv1_3,
                    cipher_suites=[
                        'TLS_AES_256_GCM_SHA384',
                        'TLS_CHACHA20_POLY1305_SHA256'
                    ],
                    cert_reqs=ssl.CERT_REQUIRED,
                    verify_mode=ssl.CERT_REQUIRED,
                    check_hostname=True,
                    session_tickets=False,
                    session_cache_size=0,
                    session_timeout=300
                ),
                crypto_config=CryptoConfig(
                    hash_algorithm="sha512",
                    key_size=8192,
                    curve="secp521r1",
                    encryption_algorithm="AES-256-GCM",
                    pbkdf2_iterations=200000,
                    salt_length=64,
                    iv_length=16,
                    tag_length=16
                ),
                session_timeout=900,
                max_login_attempts=3,
                lockout_duration=1800,
                password_min_length=16,
                require_special_chars=True,
                require_numbers=True,
                require_uppercase=True,
                require_lowercase=True,
                max_session_age=1800,
                secure_cookies=True,
                http_only_cookies=True,
                same_site_cookies="strict",
                csrf_protection=True,
                rate_limiting=True,
                rate_limit_per_minute=30,
                max_request_size=5 * 1024 * 1024,
                max_file_size=1 * 1024 * 1024
            )
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with secure defaults"""
        context = ssl.create_default_context()
        
        # Set TLS version requirements
        context.minimum_version = self.defaults.tls_config.min_version
        context.maximum_version = self.defaults.tls_config.max_version
        
        # Set cipher suites
        context.set_ciphers(':'.join(self.defaults.tls_config.cipher_suites))
        
        # Set certificate requirements
        context.verify_mode = self.defaults.tls_config.verify_mode
        context.check_hostname = self.defaults.tls_config.check_hostname
        
        # Disable session tickets for security
        if not self.defaults.tls_config.session_tickets:
            context.options |= ssl.OP_NO_TICKET
        
        # Disable session cache
        if self.defaults.tls_config.session_cache_size == 0:
            context.options |= ssl.OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION
        
        # Set session timeout
        context.session_timeout = self.defaults.tls_config.session_timeout
        
        # Load CA certificates
        context.load_verify_locations(cafile=certifi.where())
        
        return context
    
    def create_secure_socket(self, host: str, port: int) -> socket.socket:
        """Create secure socket with TLS"""
        context = self.create_ssl_context()
        sock = socket.create_connection((host, port))
        secure_sock = context.wrap_socket(sock, server_hostname=host)
        return secure_sock
    
    def generate_secure_key_pair(self) -> Tuple[bytes, bytes]:
        """Generate secure key pair"""
        if self.defaults.crypto_config.key_size >= 4096:
            # Use RSA for large keys
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.defaults.crypto_config.key_size,
                backend=default_backend()
            )
        else:
            # Use ECC for smaller keys
            curve = getattr(ec, self.defaults.crypto_config.curve.upper())
            private_key = ec.generate_private_key(curve, default_backend())
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(
                self._generate_secure_password().encode()
            )
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def generate_self_signed_certificate(self, common_name: str) -> Tuple[bytes, bytes]:
        """Generate self-signed certificate"""
        # Generate key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.defaults.crypto_config.key_size,
            backend=default_backend()
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Secure Defaults"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Security"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(common_name),
                x509.IPAddress(socket.inet_aton("127.0.0.1"))
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        # Serialize certificate and key
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return cert_pem, key_pem
    
    def _generate_secure_password(self) -> str:
        """Generate secure password according to defaults"""
        
        # Define character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Build character pool based on requirements
        pool = ""
        if self.defaults.require_lowercase:
            pool += lowercase
        if self.defaults.require_uppercase:
            pool += uppercase
        if self.defaults.require_numbers:
            pool += digits
        if self.defaults.require_special_chars:
            pool += special
        
        if not pool:
            pool = lowercase + uppercase + digits + special
        
        # Generate password
        password = []
        
        # Ensure at least one character from each required set
        if self.defaults.require_lowercase:
            password.append(secrets.choice(lowercase))
        if self.defaults.require_uppercase:
            password.append(secrets.choice(uppercase))
        if self.defaults.require_numbers:
            password.append(secrets.choice(digits))
        if self.defaults.require_special_chars:
            password.append(secrets.choice(special))
        
        # Fill remaining length
        remaining_length = self.defaults.password_min_length - len(password)
        password.extend(secrets.choice(pool) for _ in range(remaining_length))
        
        # Shuffle password
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength according to defaults"""
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "strength_score": 0
        }
        
        # Check length
        if len(password) < self.defaults.password_min_length:
            result["is_valid"] = False
            result["errors"].append(f"Password must be at least {self.defaults.password_min_length} characters")
        
        # Check character requirements
        if self.defaults.require_lowercase and not any(c.islower() for c in password):
            result["is_valid"] = False
            result["errors"].append("Password must contain lowercase letters")
        
        if self.defaults.require_uppercase and not any(c.isupper() for c in password):
            result["is_valid"] = False
            result["errors"].append("Password must contain uppercase letters")
        
        if self.defaults.require_numbers and not any(c.isdigit() for c in password):
            result["is_valid"] = False
            result["errors"].append("Password must contain numbers")
        
        if self.defaults.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            result["is_valid"] = False
            result["errors"].append("Password must contain special characters")
        
        # Calculate strength score
        score = 0
        score += len(password) * 4  # Length bonus
        score += len(set(password)) * 2  # Character variety bonus
        
        if any(c.islower() for c in password):
            score += 10
        if any(c.isupper() for c in password):
            score += 10
        if any(c.isdigit() for c in password):
            score += 10
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 20
        
        result["strength_score"] = min(score, 100)
        
        # Add warnings for weak passwords
        if result["strength_score"] < 50:
            result["warnings"].append("Password strength is weak")
        elif result["strength_score"] < 70:
            result["warnings"].append("Password strength could be improved")
        
        return result
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers based on defaults"""
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': f'max-age={self.defaults.max_session_age}; includeSubDomains; preload',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        }
        
        if self.defaults.csrf_protection:
            headers['X-CSRF-Protection'] = '1'
        
        return headers
    
    def get_cookie_settings(self) -> Dict[str, Any]:
        """Get secure cookie settings"""
        return {
            'secure': self.defaults.secure_cookies,
            'httponly': self.defaults.http_only_cookies,
            'samesite': self.defaults.same_site_cookies,
            'max_age': self.defaults.session_timeout,
            'path': '/',
            'domain': None
        }

# Pydantic models for API
class SecurityDefaultsRequest(BaseModel):
    security_level: SecurityLevel = Field(default=SecurityLevel.HIGH)
    custom_config: Optional[Dict[str, Any]] = None

class SecurityDefaultsResponse(BaseModel):
    security_level: SecurityLevel
    tls_config: Dict[str, Any]
    crypto_config: Dict[str, Any]
    session_config: Dict[str, Any]
    password_config: Dict[str, Any]
    security_headers: Dict[str, str]
    cookie_settings: Dict[str, Any]

class PasswordValidationRequest(BaseModel):
    password: str = Field(..., min_length=1)

class PasswordValidationResponse(BaseModel):
    is_valid: bool
    strength_score: int
    errors: List[str]
    warnings: List[str]

class CertificateGenerationRequest(BaseModel):
    common_name: str = Field(..., min_length=1)
    organization: str = Field(default="Secure Defaults")
    country: str = Field(default="US")

class CertificateGenerationResponse(BaseModel):
    certificate: str
    private_key: str
    common_name: str
    valid_from: datetime
    valid_until: datetime

# FastAPI router

router = APIRouter(prefix="/secure-defaults", tags=["Secure Defaults"])

@router.post("/configure", response_model=SecurityDefaultsResponse)
async def configure_secure_defaults(request: SecurityDefaultsRequest) -> SecurityDefaultsResponse:
    """Configure secure defaults for specified security level"""
    
    manager = SecureDefaultsManager(request.security_level)
    
    return SecurityDefaultsResponse(
        security_level=request.security_level,
        tls_config={
            "min_version": manager.defaults.tls_config.min_version.name,
            "max_version": manager.defaults.tls_config.max_version.name,
            "cipher_suites": manager.defaults.tls_config.cipher_suites,
            "cert_required": manager.defaults.tls_config.cert_reqs == ssl.CERT_REQUIRED,
            "check_hostname": manager.defaults.tls_config.check_hostname
        },
        crypto_config={
            "hash_algorithm": manager.defaults.crypto_config.hash_algorithm,
            "key_size": manager.defaults.crypto_config.key_size,
            "curve": manager.defaults.crypto_config.curve,
            "encryption_algorithm": manager.defaults.crypto_config.encryption_algorithm,
            "pbkdf2_iterations": manager.defaults.crypto_config.pbkdf2_iterations
        },
        session_config={
            "session_timeout": manager.defaults.session_timeout,
            "max_session_age": manager.defaults.max_session_age,
            "max_login_attempts": manager.defaults.max_login_attempts,
            "lockout_duration": manager.defaults.lockout_duration
        },
        password_config={
            "min_length": manager.defaults.password_min_length,
            "require_special_chars": manager.defaults.require_special_chars,
            "require_numbers": manager.defaults.require_numbers,
            "require_uppercase": manager.defaults.require_uppercase,
            "require_lowercase": manager.defaults.require_lowercase
        },
        security_headers=manager.get_security_headers(),
        cookie_settings=manager.get_cookie_settings()
    )

@router.post("/validate-password", response_model=PasswordValidationResponse)
async def validate_password_strength(request: PasswordValidationRequest) -> PasswordValidationResponse:
    """Validate password strength according to secure defaults"""
    
    manager = SecureDefaultsManager(SecurityLevel.HIGH)
    result = manager.validate_password_strength(request.password)
    
    return PasswordValidationResponse(
        is_valid=result["is_valid"],
        strength_score=result["strength_score"],
        errors=result["errors"],
        warnings=result["warnings"]
    )

@router.post("/generate-certificate", response_model=CertificateGenerationResponse)
async def generate_secure_certificate(request: CertificateGenerationRequest) -> CertificateGenerationResponse:
    """Generate secure self-signed certificate"""
    
    manager = SecureDefaultsManager(SecurityLevel.HIGH)
    cert_pem, key_pem = manager.generate_self_signed_certificate(request.common_name)
    
    # Parse certificate to get validity dates
    cert = x509.load_pem_x509_certificate(cert_pem, default_backend())
    
    return CertificateGenerationResponse(
        certificate=cert_pem.decode(),
        private_key=key_pem.decode(),
        common_name=request.common_name,
        valid_from=cert.not_valid_before,
        valid_until=cert.not_valid_after
    )

@router.get("/generate-password")
async def generate_secure_password() -> Dict[str, str]:
    """Generate secure password according to defaults"""
    
    manager = SecureDefaultsManager(SecurityLevel.HIGH)
    password = manager._generate_secure_password()
    
    return {"password": password}

@router.get("/security-levels")
async def get_security_levels() -> Dict[str, List[str]]:
    """Get available security levels"""
    
    return {
        "security_levels": [level.value for level in SecurityLevel],
        "cipher_strengths": [strength.value for strength in CipherStrength]
    }

# Demo function
async def demo_secure_defaults():
    """Demonstrate secure defaults features"""
    print("=== Secure Defaults Demo ===\n")
    
    # Test different security levels
    print("1. Testing Different Security Levels...")
    for level in SecurityLevel:
        manager = SecureDefaultsManager(level)
        print(f"   Level: {level.value}")
        print(f"   TLS Min Version: {manager.defaults.tls_config.min_version.name}")
        print(f"   Key Size: {manager.defaults.crypto_config.key_size} bits")
        print(f"   Password Min Length: {manager.defaults.password_min_length}")
        print(f"   Session Timeout: {manager.defaults.session_timeout} seconds")
        print()
    
    # Test password generation and validation
    print("2. Testing Password Generation and Validation...")
    manager = SecureDefaultsManager(SecurityLevel.HIGH)
    
    # Generate secure password
    password = manager._generate_secure_password()
    print(f"   Generated Password: {password}")
    
    # Validate password
    validation = manager.validate_password_strength(password)
    print(f"   Is Valid: {validation['is_valid']}")
    print(f"   Strength Score: {validation['strength_score']}/100")
    print(f"   Errors: {validation['errors']}")
    print(f"   Warnings: {validation['warnings']}")
    print()
    
    # Test weak password
    weak_password = "password123"
    weak_validation = manager.validate_password_strength(weak_password)
    print(f"   Weak Password: {weak_password}")
    print(f"   Is Valid: {weak_validation['is_valid']}")
    print(f"   Strength Score: {weak_validation['strength_score']}/100")
    print(f"   Errors: {weak_validation['errors']}")
    print()
    
    # Test certificate generation
    print("3. Testing Certificate Generation...")
    cert_pem, key_pem = manager.generate_self_signed_certificate("example.com")
    print(f"   Certificate Length: {len(cert_pem)} bytes")
    print(f"   Private Key Length: {len(key_pem)} bytes")
    print(f"   Certificate starts with: {cert_pem[:50].decode()}...")
    print()
    
    # Test security headers
    print("4. Testing Security Headers...")
    headers = manager.get_security_headers()
    for header, value in headers.items():
        print(f"   {header}: {value}")
    print()
    
    # Test cookie settings
    print("5. Testing Cookie Settings...")
    cookie_settings = manager.get_cookie_settings()
    for setting, value in cookie_settings.items():
        print(f"   {setting}: {value}")
    print()
    
    print("=== Secure Defaults Demo Completed! ===")

# FastAPI app
app = FastAPI(
    title="Secure Defaults Demo",
    description="Demonstration of secure defaults for cybersecurity tools",
    version="1.0.0"
)

# Include router
app.include_router(router)

if __name__ == "__main__":
    print("Secure Defaults Demo")
    print("Access API at: http://localhost:8000")
    print("API Documentation at: http://localhost:8000/docs")
    
    # Run demo
    asyncio.run(demo_secure_defaults())
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000) 