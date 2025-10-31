from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import os
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from enum import Enum
import hashlib
import hmac
import secrets
import socket
import threading
from contextlib import contextmanager
    import cryptography
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.hmac import HMAC
    from cryptography.hazmat.primitives.constant_time import bytes_eq
    from cryptography.x509 import load_pem_x509_certificate, load_der_x509_certificate
    from cryptography.x509.oid import NameOID
    import OpenSSL
    from OpenSSL import SSL, crypto
    from OpenSSL.crypto import X509, X509Req, PKey
from typing import Any, List, Dict, Optional
"""
Secure Cipher Suites and Cryptographic Configuration Examples
============================================================

This module provides comprehensive secure cipher suite and cryptographic
configuration capabilities for TLS/SSL connections and general cryptography.

Features:
- Secure cipher suite selection and validation
- TLS/SSL configuration with security best practices
- Cryptographic algorithm recommendations
- Certificate validation and management
- Key exchange protocol security
- Hash function security validation
- Random number generation security
- Cryptographic protocol version management
- Security level configuration
- Cryptographic compliance checking

Author: AI Assistant
License: MIT
"""


try:
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    OPENSSL_AVAILABLE = True
except ImportError:
    OPENSSL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for cryptographic operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CipherSuiteCategory(Enum):
    """Categories of cipher suites."""
    MODERN = "modern"
    INTERMEDIATE = "intermediate"
    LEGACY = "legacy"
    DEPRECATED = "deprecated"
    INSECURE = "insecure"


class HashAlgorithm(Enum):
    """Secure hash algorithms."""
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_384 = "sha3_384"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class KeyExchangeAlgorithm(Enum):
    """Secure key exchange algorithms."""
    ECDHE_RSA = "ECDHE-RSA"
    ECDHE_ECDSA = "ECDHE-ECDSA"
    DHE_RSA = "DHE-RSA"
    DHE_DSS = "DHE-DSS"
    RSA = "RSA"
    ECDSA = "ECDSA"


@dataclass
class CipherSuiteInfo:
    """Information about a cipher suite."""
    name: str
    category: CipherSuiteCategory
    security_level: SecurityLevel
    key_exchange: str
    authentication: str
    encryption: str
    hash_function: str
    key_size: int
    block_size: int
    tls_version: str
    recommended: bool = False
    deprecated: bool = False
    insecure: bool = False


@dataclass
class TLSSecurityConfig:
    """TLS security configuration."""
    min_tls_version: str = "TLSv1.2"
    max_tls_version: str = "TLSv1.3"
    cipher_suites: List[str] = field(default_factory=list)
    allowed_curves: List[str] = field(default_factory=list)
    allowed_signature_algorithms: List[str] = field(default_factory=list)
    require_forward_secrecy: bool = True
    require_strong_crypto: bool = True
    verify_mode: int = ssl.CERT_REQUIRED
    check_hostname: bool = True
    security_level: SecurityLevel = SecurityLevel.HIGH


@dataclass
class CryptographicValidationResult:
    """Result of cryptographic validation."""
    valid: bool
    security_level: SecurityLevel
    cipher_suite_info: Optional[CipherSuiteInfo] = None
    vulnerabilities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    risk_score: float = 0.0


@dataclass
class CertificateValidationResult:
    """Result of certificate validation."""
    valid: bool
    certificate_info: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    expiry_info: Dict[str, Any] = field(default_factory=dict)
    trust_chain: List[str] = field(default_factory=list)


class CipherSuiteError(Exception):
    """Custom exception for cipher suite errors."""
    pass


class CryptographicError(Exception):
    """Custom exception for cryptographic errors."""
    pass


class CertificateError(Exception):
    """Custom exception for certificate errors."""
    pass


class SecurityConfigurationError(Exception):
    """Custom exception for security configuration errors."""
    pass


class SecureCipherSuiteManager:
    """Manager for secure cipher suite configuration and validation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize cipher suite manager."""
        self.config = config or {}
        
        # Define secure cipher suites by category
        self.cipher_suites = {
            CipherSuiteCategory.MODERN: [
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
                "TLS_AES_128_GCM_SHA256",
                "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
                "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
                "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
                "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256"
            ],
            CipherSuiteCategory.INTERMEDIATE: [
                "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
                "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
                "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384",
                "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384",
                "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384",
                "TLS_DHE_RSA_WITH_AES_128_GCM_SHA256"
            ],
            CipherSuiteCategory.LEGACY: [
                "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
                "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256",
                "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
                "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA"
            ],
            CipherSuiteCategory.DEPRECATED: [
                "TLS_RSA_WITH_AES_256_GCM_SHA384",
                "TLS_RSA_WITH_AES_128_GCM_SHA256",
                "TLS_RSA_WITH_AES_256_CBC_SHA256",
                "TLS_RSA_WITH_AES_128_CBC_SHA256"
            ],
            CipherSuiteCategory.INSECURE: [
                "TLS_RSA_WITH_AES_256_CBC_SHA",
                "TLS_RSA_WITH_AES_128_CBC_SHA",
                "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
                "TLS_RSA_WITH_RC4_128_SHA",
                "TLS_RSA_WITH_RC4_128_MD5",
                "TLS_NULL_WITH_NULL_NULL",
                "TLS_RSA_WITH_NULL_MD5",
                "TLS_RSA_WITH_NULL_SHA"
            ]
        }
        
        # Security level mappings
        self.security_levels = {
            SecurityLevel.CRITICAL: [CipherSuiteCategory.MODERN],
            SecurityLevel.HIGH: [CipherSuiteCategory.MODERN, CipherSuiteCategory.INTERMEDIATE],
            SecurityLevel.MEDIUM: [CipherSuiteCategory.MODERN, CipherSuiteCategory.INTERMEDIATE, CipherSuiteCategory.LEGACY],
            SecurityLevel.LOW: [CipherSuiteCategory.MODERN, CipherSuiteCategory.INTERMEDIATE, CipherSuiteCategory.LEGACY, CipherSuiteCategory.DEPRECATED]
        }
        
        # Cipher suite information database
        self.cipher_suite_info = self._build_cipher_suite_database()
    
    def _build_cipher_suite_database(self) -> Dict[str, CipherSuiteInfo]:
        """Build comprehensive cipher suite information database."""
        database = {}
        
        # Modern cipher suites (TLS 1.3)
        modern_suites = [
            ("TLS_AES_256_GCM_SHA384", "AES-256-GCM", "SHA384", 256, 128, "TLSv1.3"),
            ("TLS_CHACHA20_POLY1305_SHA256", "ChaCha20-Poly1305", "SHA256", 256, 128, "TLSv1.3"),
            ("TLS_AES_128_GCM_SHA256", "AES-128-GCM", "SHA256", 128, 128, "TLSv1.3")
        ]
        
        for name, encryption, hash_func, key_size, block_size, tls_ver in modern_suites:
            database[name] = CipherSuiteInfo(
                name=name,
                category=CipherSuiteCategory.MODERN,
                security_level=SecurityLevel.CRITICAL,
                key_exchange="ECDHE",
                authentication="RSA/ECDSA",
                encryption=encryption,
                hash_function=hash_func,
                key_size=key_size,
                block_size=block_size,
                tls_version=tls_ver,
                recommended=True
            )
        
        # TLS 1.2 cipher suites
        tls12_suites = [
            ("TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", "ECDHE-RSA", "AES-256-GCM", "SHA384", 256, 128, "TLSv1.2"),
            ("TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256", "ECDHE-RSA", "ChaCha20-Poly1305", "SHA256", 256, 128, "TLSv1.2"),
            ("TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", "ECDHE-ECDSA", "AES-256-GCM", "SHA384", 256, 128, "TLSv1.2"),
            ("TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", "ECDHE-RSA", "AES-128-GCM", "SHA256", 128, 128, "TLSv1.2"),
            ("TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", "ECDHE-ECDSA", "AES-128-GCM", "SHA256", 128, 128, "TLSv1.2"),
            ("TLS_DHE_RSA_WITH_AES_256_GCM_SHA384", "DHE-RSA", "AES-256-GCM", "SHA384", 256, 128, "TLSv1.2"),
            ("TLS_DHE_RSA_WITH_AES_128_GCM_SHA256", "DHE-RSA", "AES-128-GCM", "SHA256", 128, 128, "TLSv1.2")
        ]
        
        for name, kex, encryption, hash_func, key_size, block_size, tls_ver in tls12_suites:
            database[name] = CipherSuiteInfo(
                name=name,
                category=CipherSuiteCategory.INTERMEDIATE,
                security_level=SecurityLevel.HIGH,
                key_exchange=kex,
                authentication="RSA/ECDSA",
                encryption=encryption,
                hash_function=hash_func,
                key_size=key_size,
                block_size=block_size,
                tls_version=tls_ver,
                recommended=True
            )
        
        # Legacy cipher suites
        legacy_suites = [
            ("TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384", "ECDHE-RSA", "AES-256-CBC", "SHA384", 256, 128, "TLSv1.2"),
            ("TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384", "ECDHE-ECDSA", "AES-256-CBC", "SHA384", 256, 128, "TLSv1.2"),
            ("TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256", "ECDHE-RSA", "AES-128-CBC", "SHA256", 128, 128, "TLSv1.2"),
            ("TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256", "ECDHE-ECDSA", "AES-128-CBC", "SHA256", 128, 128, "TLSv1.2")
        ]
        
        for name, kex, encryption, hash_func, key_size, block_size, tls_ver in legacy_suites:
            database[name] = CipherSuiteInfo(
                name=name,
                category=CipherSuiteCategory.LEGACY,
                security_level=SecurityLevel.MEDIUM,
                key_exchange=kex,
                authentication="RSA/ECDSA",
                encryption=encryption,
                hash_function=hash_func,
                key_size=key_size,
                block_size=block_size,
                tls_version=tls_ver,
                recommended=False
            )
        
        # Deprecated cipher suites
        deprecated_suites = [
            ("TLS_RSA_WITH_AES_256_GCM_SHA384", "RSA", "AES-256-GCM", "SHA384", 256, 128, "TLSv1.2"),
            ("TLS_RSA_WITH_AES_128_GCM_SHA256", "RSA", "AES-128-GCM", "SHA256", 128, 128, "TLSv1.2"),
            ("TLS_RSA_WITH_AES_256_CBC_SHA256", "RSA", "AES-256-CBC", "SHA256", 256, 128, "TLSv1.2"),
            ("TLS_RSA_WITH_AES_128_CBC_SHA256", "RSA", "AES-128-CBC", "SHA256", 128, 128, "TLSv1.2")
        ]
        
        for name, kex, encryption, hash_func, key_size, block_size, tls_ver in deprecated_suites:
            database[name] = CipherSuiteInfo(
                name=name,
                category=CipherSuiteCategory.DEPRECATED,
                security_level=SecurityLevel.LOW,
                key_exchange=kex,
                authentication="RSA",
                encryption=encryption,
                hash_function=hash_func,
                key_size=key_size,
                block_size=block_size,
                tls_version=tls_ver,
                recommended=False,
                deprecated=True
            )
        
        # Insecure cipher suites
        insecure_suites = [
            ("TLS_RSA_WITH_AES_256_CBC_SHA", "RSA", "AES-256-CBC", "SHA1", 256, 128, "TLSv1.0"),
            ("TLS_RSA_WITH_AES_128_CBC_SHA", "RSA", "AES-128-CBC", "SHA1", 128, 128, "TLSv1.0"),
            ("TLS_RSA_WITH_3DES_EDE_CBC_SHA", "RSA", "3DES", "SHA1", 168, 64, "TLSv1.0"),
            ("TLS_RSA_WITH_RC4_128_SHA", "RSA", "RC4", "SHA1", 128, 0, "TLSv1.0"),
            ("TLS_RSA_WITH_RC4_128_MD5", "RSA", "RC4", "MD5", 128, 0, "TLSv1.0"),
            ("TLS_NULL_WITH_NULL_NULL", "NULL", "NULL", "NULL", 0, 0, "TLSv1.0"),
            ("TLS_RSA_WITH_NULL_MD5", "RSA", "NULL", "MD5", 0, 0, "TLSv1.0"),
            ("TLS_RSA_WITH_NULL_SHA", "RSA", "NULL", "SHA1", 0, 0, "TLSv1.0")
        ]
        
        for name, kex, encryption, hash_func, key_size, block_size, tls_ver in insecure_suites:
            database[name] = CipherSuiteInfo(
                name=name,
                category=CipherSuiteCategory.INSECURE,
                security_level=SecurityLevel.LOW,
                key_exchange=kex,
                authentication="RSA",
                encryption=encryption,
                hash_function=hash_func,
                key_size=key_size,
                block_size=block_size,
                tls_version=tls_ver,
                recommended=False,
                deprecated=True,
                insecure=True
            )
        
        return database
    
    def get_recommended_cipher_suites(self, security_level: SecurityLevel = SecurityLevel.HIGH) -> List[str]:
        """Get recommended cipher suites for specified security level."""
        if security_level not in self.security_levels:
            raise SecurityConfigurationError(f"Invalid security level: {security_level}")
        
        recommended_suites = []
        allowed_categories = self.security_levels[security_level]
        
        for category in allowed_categories:
            if category in self.cipher_suites:
                recommended_suites.extend(self.cipher_suites[category])
        
        return recommended_suites
    
    def validate_cipher_suite(self, cipher_suite: str) -> CryptographicValidationResult:
        """Validate a cipher suite for security."""
        if not cipher_suite:
            return CryptographicValidationResult(
                valid=False,
                security_level=SecurityLevel.LOW,
                vulnerabilities=["empty_cipher_suite"]
            )
        
        # Get cipher suite information
        suite_info = self.cipher_suite_info.get(cipher_suite)
        if not suite_info:
            return CryptographicValidationResult(
                valid=False,
                security_level=SecurityLevel.LOW,
                vulnerabilities=["unknown_cipher_suite"]
            )
        
        vulnerabilities = []
        recommendations = []
        compliance_status = {}
        
        # Check for insecure cipher suites
        if suite_info.insecure:
            vulnerabilities.append("insecure_cipher_suite")
            recommendations.append("Replace with a secure cipher suite")
        
        # Check for deprecated cipher suites
        if suite_info.deprecated:
            vulnerabilities.append("deprecated_cipher_suite")
            recommendations.append("Consider upgrading to a modern cipher suite")
        
        # Check key size
        if suite_info.key_size < 128:
            vulnerabilities.append("weak_key_size")
            recommendations.append("Use cipher suites with key size >= 128 bits")
        
        # Check hash function
        weak_hashes = ["MD5", "SHA1", "NULL"]
        if suite_info.hash_function in weak_hashes:
            vulnerabilities.append("weak_hash_function")
            recommendations.append("Use strong hash functions (SHA256, SHA384, SHA512)")
        
        # Check encryption algorithm
        weak_encryption = ["RC4", "3DES", "NULL"]
        if suite_info.encryption in weak_encryption:
            vulnerabilities.append("weak_encryption")
            recommendations.append("Use strong encryption algorithms (AES-GCM, ChaCha20-Poly1305)")
        
        # Check key exchange
        if "RSA" in suite_info.key_exchange and "ECDHE" not in suite_info.key_exchange:
            vulnerabilities.append("no_forward_secrecy")
            recommendations.append("Use cipher suites with forward secrecy (ECDHE)")
        
        # Compliance checks
        compliance_status = {
            "tls_1.3_compliant": suite_info.tls_version == "TLSv1.3",
            "forward_secrecy": "ECDHE" in suite_info.key_exchange or "DHE" in suite_info.key_exchange,
            "strong_crypto": suite_info.key_size >= 128 and suite_info.hash_function not in weak_hashes,
            "modern_encryption": suite_info.encryption not in weak_encryption
        }
        
        # Calculate risk score
        risk_score = len(vulnerabilities) * 0.25
        if risk_score > 1.0:
            risk_score = 1.0
        
        valid = len(vulnerabilities) == 0
        
        return CryptographicValidationResult(
            valid=valid,
            security_level=suite_info.security_level,
            cipher_suite_info=suite_info,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            compliance_status=compliance_status,
            risk_score=risk_score
        )
    
    def create_secure_ssl_context(self, security_level: SecurityLevel = SecurityLevel.HIGH) -> ssl.SSLContext:
        """Create a secure SSL context with recommended settings."""
        if not hasattr(ssl, 'PROTOCOL_TLS'):
            raise SecurityConfigurationError("SSL module does not support modern TLS")
        
        # Create SSL context
        if hasattr(ssl, 'PROTOCOL_TLS_CLIENT'):
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS)
        
        # Set minimum and maximum TLS versions
        if hasattr(ssl, 'TLSVersion'):
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_3
        else:
            # Fallback for older Python versions
            context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
            if hasattr(ssl, 'OP_NO_TLSv1_3'):
                context.options &= ~ssl.OP_NO_TLSv1_3
        
        # Set cipher suites
        recommended_suites = self.get_recommended_cipher_suites(security_level)
        if recommended_suites:
            context.set_ciphers(':'.join(recommended_suites))
        
        # Set security options
        context.options |= (
            ssl.OP_NO_COMPRESSION |
            ssl.OP_NO_RENEGOTIATION |
            ssl.OP_SINGLE_DH_USE |
            ssl.OP_SINGLE_ECDH_USE
        )
        
        # Set verification mode
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        
        # Set default verify paths
        context.load_default_certs()
        
        return context
    
    def validate_ssl_connection(self, hostname: str, port: int = 443, 
                              timeout: int = 10) -> CryptographicValidationResult:
        """Validate SSL/TLS connection security."""
        if not hostname:
            return CryptographicValidationResult(
                valid=False,
                security_level=SecurityLevel.LOW,
                vulnerabilities=["empty_hostname"]
            )
        
        try:
            # Create secure context
            context = self.create_secure_ssl_context(SecurityLevel.HIGH)
            
            # Connect and get cipher info
            with socket.create_connection((hostname, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cipher_info = ssock.cipher()
                    version = ssock.version()
                    cert = ssock.getpeercert()
                    
                    # Validate cipher suite
                    cipher_suite = cipher_info[0]
                    validation_result = self.validate_cipher_suite(cipher_suite)
                    
                    # Add connection-specific vulnerabilities
                    if version < "TLSv1.2":
                        validation_result.vulnerabilities.append("weak_tls_version")
                        validation_result.recommendations.append("Upgrade to TLS 1.2 or higher")
                    
                    if not cert:
                        validation_result.vulnerabilities.append("no_certificate")
                        validation_result.recommendations.append("Server should provide a valid certificate")
                    
                    return validation_result
        
        except ssl.SSLError as e:
            return CryptographicValidationResult(
                valid=False,
                security_level=SecurityLevel.LOW,
                vulnerabilities=["ssl_error"],
                recommendations=[f"SSL error: {e}"]
            )
        except socket.timeout:
            return CryptographicValidationResult(
                valid=False,
                security_level=SecurityLevel.LOW,
                vulnerabilities=["connection_timeout"]
            )
        except Exception as e:
            return CryptographicValidationResult(
                valid=False,
                security_level=SecurityLevel.LOW,
                vulnerabilities=["connection_error"],
                recommendations=[f"Connection error: {e}"]
            )


class CertificateValidator:
    """Certificate validation and security assessment."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize certificate validator."""
        self.config = config or {}
        
        # Security requirements
        self.min_key_size = self.config.get('min_key_size', 2048)
        self.min_signature_algorithm = self.config.get('min_signature_algorithm', 'sha256')
        self.max_validity_days = self.config.get('max_validity_days', 825)  # ~2.25 years
        self.require_san = self.config.get('require_san', True)
        self.require_ocsp = self.config.get('require_ocsp', False)
    
    def validate_certificate(self, cert_data: Union[str, bytes], 
                           cert_format: str = 'pem') -> CertificateValidationResult:
        """Validate certificate for security compliance."""
        if not cert_data:
            return CertificateValidationResult(
                valid=False,
                validation_errors=["empty_certificate"]
            )
        
        try:
            # Load certificate
            if cert_format.lower() == 'pem':
                if isinstance(cert_data, bytes):
                    cert_data = cert_data.decode('utf-8')
                cert = load_pem_x509_certificate(cert_data.encode('utf-8'))
            else:
                if isinstance(cert_data, str):
                    cert_data = cert_data.encode('utf-8')
                cert = load_der_x509_certificate(cert_data)
            
            validation_errors = []
            security_issues = []
            certificate_info = {}
            
            # Extract certificate information
            certificate_info['subject'] = str(cert.subject)
            certificate_info['issuer'] = str(cert.issuer)
            certificate_info['serial_number'] = str(cert.serial_number)
            certificate_info['version'] = cert.version
            
            # Check validity period
            not_before = cert.not_valid_before
            not_after = cert.not_valid_after
            now = time.time()
            
            certificate_info['validity'] = {
                'not_before': not_before.isoformat(),
                'not_after': not_after.isoformat(),
                'days_remaining': (not_after.timestamp() - now) / 86400
            }
            
            # Check if certificate is expired
            if now < not_before.timestamp():
                validation_errors.append("certificate_not_yet_valid")
            
            if now > not_after.timestamp():
                validation_errors.append("certificate_expired")
            
            # Check validity period length
            validity_days = (not_after - not_before).days
            if validity_days > self.max_validity_days:
                security_issues.append("long_validity_period")
            
            # Check public key
            public_key = cert.public_key()
            if hasattr(public_key, 'key_size'):
                key_size = public_key.key_size
                certificate_info['key_size'] = key_size
                
                if key_size < self.min_key_size:
                    security_issues.append(f"weak_key_size_{key_size}")
            
            # Check signature algorithm
            signature_algorithm = cert.signature_algorithm_oid
            certificate_info['signature_algorithm'] = str(signature_algorithm)
            
            if 'md5' in str(signature_algorithm).lower() or 'sha1' in str(signature_algorithm).lower():
                security_issues.append("weak_signature_algorithm")
            
            # Check extensions
            extensions = {}
            for ext in cert.extensions:
                extensions[ext.oid.dotted_string] = str(ext.value)
            
            certificate_info['extensions'] = extensions
            
            # Check Subject Alternative Names
            if self.require_san:
                san_extension = cert.extensions.get_extension_for_oid(
                    cryptography.x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                )
                if not san_extension:
                    security_issues.append("missing_san_extension")
            
            # Check key usage
            try:
                key_usage = cert.extensions.get_extension_for_oid(
                    cryptography.x509.oid.ExtensionOID.KEY_USAGE
                )
                certificate_info['key_usage'] = str(key_usage.value)
            except cryptography.x509.extensions.ExtensionNotFound:
                pass
            
            # Check extended key usage
            try:
                extended_key_usage = cert.extensions.get_extension_for_oid(
                    cryptography.x509.oid.ExtensionOID.EXTENDED_KEY_USAGE
                )
                certificate_info['extended_key_usage'] = str(extended_key_usage.value)
            except cryptography.x509.extensions.ExtensionNotFound:
                pass
            
            valid = len(validation_errors) == 0
            
            return CertificateValidationResult(
                valid=valid,
                certificate_info=certificate_info,
                validation_errors=validation_errors,
                security_issues=security_issues,
                expiry_info=certificate_info.get('validity', {})
            )
        
        except Exception as e:
            return CertificateValidationResult(
                valid=False,
                validation_errors=[f"certificate_parsing_error: {e}"]
            )


class CryptographicSecurityManager:
    """Comprehensive cryptographic security management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize cryptographic security manager."""
        self.config = config or {}
        self.cipher_manager = SecureCipherSuiteManager(config)
        self.cert_validator = CertificateValidator(config)
        
        # Security settings
        self.min_hash_length = self.config.get('min_hash_length', 256)
        self.require_secure_random = self.config.get('require_secure_random', True)
        self.key_rotation_days = self.config.get('key_rotation_days', 90)
        
        # Security events
        self.security_events: List[str] = []
    
    def _log_security_event(self, event: str):
        """Log security event."""
        self.security_events.append(event)
        logger.warning(f"Cryptographic Security Event: {event}")
    
    def validate_cryptographic_configuration(self, config: Dict[str, Any]) -> CryptographicValidationResult:
        """Validate cryptographic configuration for security."""
        vulnerabilities = []
        recommendations = []
        compliance_status = {}
        
        # Check TLS configuration
        if 'tls' in config:
            tls_config = config['tls']
            
            # Check TLS version
            min_version = tls_config.get('min_version', 'TLSv1.0')
            if min_version < 'TLSv1.2':
                vulnerabilities.append("weak_tls_version")
                recommendations.append("Use TLS 1.2 or higher")
            
            # Check cipher suites
            cipher_suites = tls_config.get('cipher_suites', [])
            for suite in cipher_suites:
                validation = self.cipher_manager.validate_cipher_suite(suite)
                if not validation.valid:
                    vulnerabilities.extend(validation.vulnerabilities)
                    recommendations.extend(validation.recommendations)
        
        # Check hash algorithms
        if 'hash_algorithms' in config:
            hash_algs = config['hash_algorithms']
            weak_hashes = ['md5', 'sha1']
            for alg in hash_algs:
                if alg.lower() in weak_hashes:
                    vulnerabilities.append("weak_hash_algorithm")
                    recommendations.append(f"Replace {alg} with SHA256 or stronger")
        
        # Check key sizes
        if 'key_sizes' in config:
            key_sizes = config['key_sizes']
            for key_type, size in key_sizes.items():
                if size < 2048:
                    vulnerabilities.append("weak_key_size")
                    recommendations.append(f"Increase {key_type} key size to 2048 bits or more")
        
        # Check random number generation
        if 'random_source' in config:
            random_source = config['random_source']
            if random_source != 'secure' and self.require_secure_random:
                vulnerabilities.append("insecure_random_source")
                recommendations.append("Use cryptographically secure random number generator")
        
        # Compliance checks
        compliance_status = {
            "tls_1.2_plus": min_version >= 'TLSv1.2' if 'min_version' in locals() else False,
            "strong_ciphers": len(vulnerabilities) == 0,
            "secure_hashes": not any('weak_hash' in v for v in vulnerabilities),
            "adequate_key_sizes": not any('weak_key' in v for v in vulnerabilities)
        }
        
        valid = len(vulnerabilities) == 0
        risk_score = len(vulnerabilities) * 0.2
        
        return CryptographicValidationResult(
            valid=valid,
            security_level=SecurityLevel.HIGH if valid else SecurityLevel.LOW,
            vulnerabilities=vulnerabilities,
            recommendations=recommendations,
            compliance_status=compliance_status,
            risk_score=risk_score
        )
    
    def generate_secure_random(self, length: int = 32) -> bytes:
        """Generate cryptographically secure random bytes."""
        if length <= 0:
            raise CryptographicError("Length must be positive")
        
        if length > 1024:
            self._log_security_event(f"Large random generation requested: {length} bytes")
        
        return secrets.token_bytes(length)
    
    def create_secure_hash(self, data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bytes:
        """Create secure hash of data."""
        if not data:
            raise CryptographicError("Data cannot be empty")
        
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).digest()
        elif algorithm == HashAlgorithm.SHA384:
            return hashlib.sha384(data).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).digest()
        elif algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).digest()
        elif algorithm == HashAlgorithm.SHA3_384:
            return hashlib.sha3_384(data).digest()
        elif algorithm == HashAlgorithm.SHA3_512:
            return hashlib.sha3_512(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s(data).digest()
        else:
            raise CryptographicError(f"Unsupported hash algorithm: {algorithm}")


# Example usage functions
def demonstrate_cipher_suite_validation():
    """Demonstrate cipher suite validation."""
    manager = SecureCipherSuiteManager()
    
    test_cipher_suites = [
        "TLS_AES_256_GCM_SHA384",  # Modern, secure
        "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",  # High security
        "TLS_RSA_WITH_AES_256_CBC_SHA",  # Deprecated
        "TLS_RSA_WITH_RC4_128_SHA",  # Insecure
        "TLS_NULL_WITH_NULL_NULL"  # Extremely insecure
    ]
    
    for cipher_suite in test_cipher_suites:
        result = manager.validate_cipher_suite(cipher_suite)
        print(f"\nCipher Suite: {cipher_suite}")
        print(f"Valid: {result.valid}")
        print(f"Security Level: {result.security_level.value}")
        print(f"Risk Score: {result.risk_score:.2f}")
        print(f"Vulnerabilities: {result.vulnerabilities}")
        print(f"Recommendations: {result.recommendations}")
        print(f"Compliance: {result.compliance_status}")


def demonstrate_secure_ssl_context():
    """Demonstrate secure SSL context creation."""
    manager = SecureCipherSuiteManager()
    
    # Create secure SSL context
    context = manager.create_secure_ssl_context(SecurityLevel.HIGH)
    
    print("Secure SSL Context Configuration:")
    print(f"Protocol: {context.protocol}")
    print(f"Options: {context.options}")
    print(f"Verify Mode: {context.verify_mode}")
    print(f"Check Hostname: {context.check_hostname}")
    
    # Get recommended cipher suites
    recommended = manager.get_recommended_cipher_suites(SecurityLevel.HIGH)
    print(f"\nRecommended Cipher Suites ({len(recommended)}):")
    for suite in recommended[:5]:  # Show first 5
        print(f"  - {suite}")
    if len(recommended) > 5:
        print(f"  ... and {len(recommended) - 5} more")


def demonstrate_certificate_validation():
    """Demonstrate certificate validation."""
    validator = CertificateValidator()
    
    # This would normally be a real certificate
    # For demonstration, we'll show the validation structure
    print("Certificate Validation Example:")
    print("(In a real scenario, this would validate an actual certificate)")
    
    # Show validation requirements
    print(f"Minimum Key Size: {validator.min_key_size} bits")
    print(f"Minimum Signature Algorithm: {validator.min_signature_algorithm}")
    print(f"Maximum Validity Days: {validator.max_validity_days}")
    print(f"Require SAN: {validator.require_san}")
    print(f"Require OCSP: {validator.require_ocsp}")


def demonstrate_cryptographic_security():
    """Demonstrate cryptographic security management."""
    manager = CryptographicSecurityManager()
    
    # Test configuration validation
    test_config = {
        'tls': {
            'min_version': 'TLSv1.0',
            'cipher_suites': ['TLS_RSA_WITH_RC4_128_SHA', 'TLS_RSA_WITH_3DES_EDE_CBC_SHA']
        },
        'hash_algorithms': ['md5', 'sha1', 'sha256'],
        'key_sizes': {'rsa': 1024, 'ec': 256},
        'random_source': 'insecure'
    }
    
    result = manager.validate_cryptographic_configuration(test_config)
    
    print("Cryptographic Configuration Validation:")
    print(f"Valid: {result.valid}")
    print(f"Security Level: {result.security_level.value}")
    print(f"Risk Score: {result.risk_score:.2f}")
    print(f"Vulnerabilities: {result.vulnerabilities}")
    print(f"Recommendations: {result.recommendations}")
    print(f"Compliance Status: {result.compliance_status}")
    
    # Test secure random generation
    secure_random = manager.generate_secure_random(32)
    print(f"\nSecure Random (32 bytes): {secure_random.hex()}")
    
    # Test secure hashing
    test_data = b"Hello, World!"
    secure_hash = manager.create_secure_hash(test_data, HashAlgorithm.SHA256)
    print(f"Secure Hash (SHA256): {secure_hash.hex()}")


def main():
    """Main function demonstrating secure cipher suites and cryptographic security."""
    logger.info("Starting secure cipher suites and cryptographic security examples")
    
    # Demonstrate cipher suite validation
    try:
        demonstrate_cipher_suite_validation()
    except Exception as e:
        logger.error(f"Cipher suite validation demonstration failed: {e}")
    
    # Demonstrate secure SSL context
    try:
        demonstrate_secure_ssl_context()
    except Exception as e:
        logger.error(f"Secure SSL context demonstration failed: {e}")
    
    # Demonstrate certificate validation
    try:
        demonstrate_certificate_validation()
    except Exception as e:
        logger.error(f"Certificate validation demonstration failed: {e}")
    
    # Demonstrate cryptographic security
    try:
        demonstrate_cryptographic_security()
    except Exception as e:
        logger.error(f"Cryptographic security demonstration failed: {e}")
    
    logger.info("Secure cipher suites and cryptographic security examples completed")


match __name__:
    case "__main__":
    main() 