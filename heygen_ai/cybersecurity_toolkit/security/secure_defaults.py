from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import ssl
import socket
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from ..utils.structured_logger import get_logger, log_function_call
from typing import Any, List, Dict, Optional
import asyncio
"""
Secure Defaults Module
=====================

Secure default configurations for:
- TLS/SSL configurations
- Strong cipher suites
- Security headers
- Cryptographic defaults
- Network security settings
"""


# Import structured logger

# Get logger instance
logger = get_logger("secure_defaults")

class SecureDefaults:
    """
    Secure default configurations for cybersecurity operations.
    
    Provides secure defaults for TLS, cipher suites, and other
    security-related configurations.
    """
    
    def __init__(self) -> Any:
        """Initialize secure defaults with strong security configurations."""
        # Secure TLS versions (TLS 1.2 and above)
        self.secure_tls_versions = {
            "TLSv1.2": ssl.TLSVersion.TLSv1_2,
            "TLSv1.3": ssl.TLSVersion.TLSv1_3
        }
        
        # Strong cipher suites (prioritized by security)
        self.strong_cipher_suites = [
            # TLS 1.3 cipher suites (most secure)
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "TLS_AES_128_GCM_SHA256",
            
            # TLS 1.2 strong cipher suites
            "ECDHE-RSA-AES256-GCM-SHA384",
            "ECDHE-RSA-AES128-GCM-SHA256",
            "ECDHE-RSA-CHACHA20-POLY1305",
            "ECDHE-ECDSA-AES256-GCM-SHA384",
            "ECDHE-ECDSA-AES128-GCM-SHA256",
            "ECDHE-ECDSA-CHACHA20-POLY1305",
            "DHE-RSA-AES256-GCM-SHA384",
            "DHE-RSA-AES128-GCM-SHA256"
        ]
        
        # Weak cipher suites to avoid
        self.weak_cipher_suites = [
            # Null ciphers
            "NULL",
            "NULL-SHA",
            "NULL-MD5",
            
            # Export ciphers
            "EXP",
            "EXPORT",
            
            # RC4 ciphers
            "RC4",
            "ARCFOUR",
            
            # DES ciphers
            "DES",
            "3DES",
            "DES-CBC",
            "DES-CBC3",
            
            # MD5 ciphers
            "MD5",
            
            # CBC mode without proper padding
            "CBC",
            
            # Static RSA key exchange
            "RSA",
            "RSA-SHA",
            "RSA-MD5"
        ]
        
        # Secure security headers
        self.security_headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        # Cryptographic defaults
        self.crypto_defaults = {
            "hash_algorithm": "sha256",
            "key_size": 256,
            "salt_length": 32,
            "iterations": 100000,
            "block_size": 16,
            "iv_length": 16
        }
    
    @log_function_call
    def create_secure_ssl_context(self, 
                                min_tls_version: str = "TLSv1.2",
                                cipher_suites: Optional[List[str]] = None) -> ssl.SSLContext:
        """
        Create a secure SSL context with strong defaults.
        
        Args:
            min_tls_version: Minimum TLS version (TLSv1.2 or TLSv1.3)
            cipher_suites: List of cipher suites to use
            
        Returns:
            Configured SSL context
            
        Raises:
            ValueError: When TLS version is not supported
        """
        # Guard clause 1: Validate TLS version
        if min_tls_version not in self.secure_tls_versions:
            raise ValueError(f"Unsupported TLS version: {min_tls_version}")
        
        # Guard clause 2: Create SSL context
        ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        # Guard clause 3: Set minimum TLS version
        ssl_context.minimum_version = self.secure_tls_versions[min_tls_version]
        ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Guard clause 4: Set cipher suites
        if cipher_suites is None:
            cipher_suites = self.strong_cipher_suites
        
        # Filter out weak cipher suites
        secure_ciphers = [cipher for cipher in cipher_suites 
                         if not any(weak in cipher for weak in self.weak_cipher_suites)]
        
        if secure_ciphers:
            ssl_context.set_ciphers(':'.join(secure_ciphers))
        
        # Guard clause 5: Set additional security options
        ssl_context.options |= (
            ssl.OP_NO_SSLv2 | 
            ssl.OP_NO_SSLv3 | 
            ssl.OP_NO_TLSv1 | 
            ssl.OP_NO_TLSv1_1 |
            ssl.OP_NO_COMPRESSION |
            ssl.OP_NO_RENEGOTIATION
        )
        
        # Guard clause 6: Set verification options
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = True
        
        # Happy path: Return configured SSL context
        logger.log_function_exit(
            "create_secure_ssl_context",
            {
                "min_tls_version": min_tls_version,
                "cipher_count": len(secure_ciphers),
                "verification_mode": "CERT_REQUIRED"
            },
            context={"ssl_context_creation": "success"}
        )
        
        return ssl_context
    
    @log_function_call
    def create_client_ssl_context(self) -> ssl.SSLContext:
        """
        Create a secure SSL context for client connections.
        
        Returns:
            Configured SSL context for client use
        """
        return self.create_secure_ssl_context(min_tls_version="TLSv1.2")
    
    @log_function_call
    def create_server_ssl_context(self, 
                                cert_file: str,
                                key_file: str,
                                ca_certs: Optional[str] = None) -> ssl.SSLContext:
        """
        Create a secure SSL context for server connections.
        
        Args:
            cert_file: Path to certificate file
            key_file: Path to private key file
            ca_certs: Path to CA certificates file
            
        Returns:
            Configured SSL context for server use
            
        Raises:
            FileNotFoundError: When certificate files are not found
            ssl.SSLError: When SSL configuration fails
        """
        # Guard clause 1: Create base SSL context
        ssl_context = self.create_secure_ssl_context(min_tls_version="TLSv1.2")
        
        # Guard clause 2: Load certificate and key
        try:
            ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Certificate or key file not found: {e}")
        except ssl.SSLError as e:
            raise ssl.SSLError(f"SSL configuration failed: {e}")
        
        # Guard clause 3: Load CA certificates if provided
        if ca_certs:
            try:
                ssl_context.load_verify_locations(cafile=ca_certs)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"CA certificates file not found: {e}")
            except ssl.SSLError as e:
                raise ssl.SSLError(f"CA certificates loading failed: {e}")
        
        # Guard clause 4: Set server-specific options
        ssl_context.options |= ssl.OP_SINGLE_DH_USE | ssl.OP_SINGLE_ECDH_USE
        
        # Happy path: Return configured server SSL context
        logger.log_function_exit(
            "create_server_ssl_context",
            {
                "cert_file": cert_file,
                "key_file": key_file,
                "ca_certs": ca_certs,
                "server_context": True
            },
            context={"server_ssl_context_creation": "success"}
        )
        
        return ssl_context
    
    @log_function_call
    def get_secure_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get secure HTTP headers.
        
        Args:
            additional_headers: Additional headers to include
            
        Returns:
            Dictionary of secure headers
        """
        # Guard clause 1: Start with default secure headers
        headers = self.security_headers.copy()
        
        # Guard clause 2: Add additional headers if provided
        if additional_headers:
            if not isinstance(additional_headers, dict):
                raise ValueError("Additional headers must be a dictionary")
            
            headers.update(additional_headers)
        
        # Happy path: Return secure headers
        logger.log_function_exit(
            "get_secure_headers",
            {"header_count": len(headers)},
            context={"secure_headers_generation": "success"}
        )
        
        return headers
    
    @log_function_call
    def get_crypto_defaults(self) -> Dict[str, Any]:
        """
        Get secure cryptographic defaults.
        
        Returns:
            Dictionary of cryptographic defaults
        """
        return self.crypto_defaults.copy()
    
    @log_function_call
    def validate_tls_configuration(self, hostname: str, port: int = 443) -> Dict[str, Any]:
        """
        Validate TLS configuration of a remote host.
        
        Args:
            hostname: Target hostname
            port: Target port
            
        Returns:
            TLS configuration validation result
        """
        # Guard clause 1: Create secure SSL context
        ssl_context = self.create_client_ssl_context()
        
        # Guard clause 2: Connect and validate
        try:
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with ssl_context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    # Get connection info
                    cipher = ssock.cipher()
                    version = ssock.version()
                    cert = ssock.getpeercert()
                    
                    # Validate certificate
                    cert_valid = cert is not None
                    cert_subject = cert.get('subject', []) if cert else []
                    cert_issuer = cert.get('issuer', []) if cert else []
                    
                    # Check if TLS version is secure
                    secure_version = version in ['TLSv1.2', 'TLSv1.3']
                    
                    # Check if cipher is strong
                    strong_cipher = any(strong in cipher[0] for strong in self.strong_cipher_suites)
                    
                    result = {
                        "hostname": hostname,
                        "port": port,
                        "tls_version": version,
                        "cipher_suite": cipher[0],
                        "certificate_valid": cert_valid,
                        "certificate_subject": cert_subject,
                        "certificate_issuer": cert_issuer,
                        "secure_version": secure_version,
                        "strong_cipher": strong_cipher,
                        "validation_timestamp": datetime.utcnow().isoformat()
                    }
                    
                    logger.log_function_exit(
                        "validate_tls_configuration",
                        result,
                        context={"tls_validation": "success"}
                    )
                    
                    return result
                    
        except socket.timeout:
            return {
                "hostname": hostname,
                "port": port,
                "error": "Connection timeout",
                "validation_timestamp": datetime.utcnow().isoformat()
            }
        except ssl.SSLError as e:
            return {
                "hostname": hostname,
                "port": port,
                "error": f"SSL error: {str(e)}",
                "validation_timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "hostname": hostname,
                "port": port,
                "error": f"Connection error: {str(e)}",
                "validation_timestamp": datetime.utcnow().isoformat()
            }
    
    @log_function_call
    def get_recommended_cipher_suites(self) -> List[str]:
        """
        Get recommended cipher suites for maximum security.
        
        Returns:
            List of recommended cipher suites
        """
        return self.strong_cipher_suites.copy()
    
    @log_function_call
    def get_weak_cipher_suites(self) -> List[str]:
        """
        Get list of weak cipher suites to avoid.
        
        Returns:
            List of weak cipher suites
        """
        return self.weak_cipher_suites.copy()
    
    @log_function_call
    def is_cipher_secure(self, cipher_suite: str) -> bool:
        """
        Check if a cipher suite is considered secure.
        
        Args:
            cipher_suite: Cipher suite to check
            
        Returns:
            True if cipher suite is secure, False otherwise
        """
        # Guard clause 1: Check if cipher is in strong list
        if any(strong in cipher_suite for strong in self.strong_cipher_suites):
            return True
        
        # Guard clause 2: Check if cipher is in weak list
        if any(weak in cipher_suite for weak in self.weak_cipher_suites):
            return False
        
        # Guard clause 3: Default to secure for unknown ciphers
        return True

# Global secure defaults instance
_secure_defaults = SecureDefaults()

# Convenience functions
def create_secure_ssl_context(min_tls_version: str = "TLSv1.2") -> ssl.SSLContext:
    """Create a secure SSL context with strong defaults."""
    return _secure_defaults.create_secure_ssl_context(min_tls_version)

def create_client_ssl_context() -> ssl.SSLContext:
    """Create a secure SSL context for client connections."""
    return _secure_defaults.create_client_ssl_context()

def create_server_ssl_context(cert_file: str, key_file: str, ca_certs: Optional[str] = None) -> ssl.SSLContext:
    """Create a secure SSL context for server connections."""
    return _secure_defaults.create_server_ssl_context(cert_file, key_file, ca_certs)

def get_secure_headers(additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Get secure HTTP headers."""
    return _secure_defaults.get_secure_headers(additional_headers)

def get_crypto_defaults() -> Dict[str, Any]:
    """Get secure cryptographic defaults."""
    return _secure_defaults.get_crypto_defaults()

def validate_tls_configuration(hostname: str, port: int = 443) -> Dict[str, Any]:
    """Validate TLS configuration of a remote host."""
    return _secure_defaults.validate_tls_configuration(hostname, port)

# --- Named Exports ---

__all__ = [
    'SecureDefaults',
    'create_secure_ssl_context',
    'create_client_ssl_context',
    'create_server_ssl_context',
    'get_secure_headers',
    'get_crypto_defaults',
    'validate_tls_configuration'
] 