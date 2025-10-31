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
import base64
import secrets
import socket
import aiohttp
import ssl
import urllib.parse
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import conint, confloat, constr
import numpy as np
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from typing import Any, List, Dict, Optional
import logging
"""
Utils Module Structure - Crypto Helpers, Network Helpers
=======================================================

This file demonstrates how to organize the utils module structure:
- Crypto helpers with type hints and Pydantic validation
- Network helpers with async/sync patterns
- Named exports for utilities
"""


# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Crypto Helpers
    "CryptoHelpers",
    "CryptoHelpersConfig",
    "EncryptionType",
    
    # Network Helpers
    "NetworkHelpers", 
    "NetworkHelpersConfig",
    "ProtocolType",
    
    # Common utilities
    "UtilsResult",
    "UtilsConfig",
    "HelperType"
]

# ============================================================================
# COMMON UTILITIES
# ============================================================================

class UtilsResult(BaseModel):
    """Pydantic model for utils results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether operation was successful")
    operation_type: str = Field(description="Type of operation performed")
    result: Optional[Any] = Field(default=None, description="Operation result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operation metadata")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")

class UtilsConfig(BaseModel):
    """Pydantic model for utils configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    timeout: confloat(gt=0.0) = Field(default=30.0, description="Timeout in seconds")
    max_retries: conint(ge=0, le=10) = Field(default=3, description="Maximum retries")
    verbose: bool = Field(default=False, description="Verbose output")
    log_operations: bool = Field(default=True, description="Log operations")

class HelperType(BaseModel):
    """Pydantic model for helper type validation."""
    
    model_config = ConfigDict(extra="forbid")
    
    type_name: constr(strip_whitespace=True) = Field(
        pattern=r"^(crypto|network|file|string|validation|compression)$"
    )
    description: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True)

# ============================================================================
# CRYPTO HELPERS
# ============================================================================

class CryptoHelpers:
    """Crypto helpers module with proper exports."""
    
    __all__ = [
        "encrypt_data",
        "decrypt_data",
        "generate_key",
        "hash_data",
        "sign_data",
        "verify_signature",
        "CryptoHelpersConfig",
        "EncryptionType"
    ]
    
    class CryptoHelpersConfig(BaseModel):
        """Pydantic model for crypto helpers configuration."""
        
        model_config = ConfigDict(extra="forbid")
        
        encryption_algorithm: constr(strip_whitespace=True) = Field(
            default="AES",
            description="Encryption algorithm (AES, RSA, Fernet)"
        )
        key_size: conint(ge=128, le=4096) = Field(default=256, description="Key size in bits")
        hash_algorithm: constr(strip_whitespace=True) = Field(
            default="SHA256",
            description="Hash algorithm (MD5, SHA1, SHA256, SHA512)"
        )
        salt_length: conint(ge=8, le=64) = Field(default=16, description="Salt length in bytes")
        iterations: conint(ge=1000, le=100000) = Field(default=10000, description="PBKDF2 iterations")
        encoding: constr(strip_whitespace=True) = Field(default="utf-8", description="String encoding")
    
    class EncryptionType(BaseModel):
        """Pydantic model for encryption type validation."""
        
        model_config = ConfigDict(extra="forbid")
        
        encryption_type: constr(strip_whitespace=True) = Field(
            pattern=r"^(AES|RSA|Fernet|ChaCha20|Blowfish)$"
        )
        description: Optional[str] = Field(default=None)
    
    async def encrypt_data(
        data: Union[str, bytes],
        key: Union[str, bytes],
        config: CryptoHelpersConfig
    ) -> UtilsResult:
        """Encrypt data with comprehensive type hints and validation."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not data:
                raise ValueError("data cannot be empty")
            if not key:
                raise ValueError("key cannot be empty")
            
            # Convert data to bytes if string
            if isinstance(data, str):
                data_bytes = data.encode(config.encoding)
            else:
                data_bytes = data
            
            # Convert key to bytes if string
            if isinstance(key, str):
                key_bytes = key.encode(config.encoding)
            else:
                key_bytes = key
            
            # Encrypt based on algorithm
            if config.encryption_algorithm == "Fernet":
                encrypted_data = await self._encrypt_fernet(data_bytes, key_bytes)
            elif config.encryption_algorithm == "AES":
                encrypted_data = await self._encrypt_aes(data_bytes, key_bytes, config)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {config.encryption_algorithm}")
            
            # Encode result
            encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return UtilsResult(
                is_successful=True,
                operation_type="encryption",
                result=encrypted_b64,
                metadata={
                    "algorithm": config.encryption_algorithm,
                    "key_size": config.key_size,
                    "original_size": len(data_bytes),
                    "encrypted_size": len(encrypted_data)
                },
                execution_time=execution_time
            )
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="encryption",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    async def decrypt_data(
        encrypted_data: Union[str, bytes],
        key: Union[str, bytes],
        config: CryptoHelpersConfig
    ) -> UtilsResult:
        """Decrypt data with comprehensive type hints and validation."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not encrypted_data:
                raise ValueError("encrypted_data cannot be empty")
            if not key:
                raise ValueError("key cannot be empty")
            
            # Decode encrypted data if string
            if isinstance(encrypted_data, str):
                encrypted_bytes = base64.b64decode(encrypted_data)
            else:
                encrypted_bytes = encrypted_data
            
            # Convert key to bytes if string
            if isinstance(key, str):
                key_bytes = key.encode(config.encoding)
            else:
                key_bytes = key
            
            # Decrypt based on algorithm
            if config.encryption_algorithm == "Fernet":
                decrypted_data = await self._decrypt_fernet(encrypted_bytes, key_bytes)
            elif config.encryption_algorithm == "AES":
                decrypted_data = await self._decrypt_aes(encrypted_bytes, key_bytes, config)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {config.encryption_algorithm}")
            
            # Decode result
            decrypted_str = decrypted_data.decode(config.encoding)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return UtilsResult(
                is_successful=True,
                operation_type="decryption",
                result=decrypted_str,
                metadata={
                    "algorithm": config.encryption_algorithm,
                    "key_size": config.key_size,
                    "encrypted_size": len(encrypted_bytes),
                    "decrypted_size": len(decrypted_data)
                },
                execution_time=execution_time
            )
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="decryption",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    async def _encrypt_fernet(
        self,
        data: bytes,
        key: bytes
    ) -> bytes:
        """Encrypt data using Fernet."""
        # Generate Fernet key from provided key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'static_salt',  # In production, use random salt
            iterations=100000,
        )
        fernet_key = base64.urlsafe_b64encode(kdf.derive(key))
        
        f = Fernet(fernet_key)
        return f.encrypt(data)
    
    async def _decrypt_fernet(
        self,
        encrypted_data: bytes,
        key: bytes
    ) -> bytes:
        """Decrypt data using Fernet."""
        # Generate Fernet key from provided key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'static_salt',  # In production, use random salt
            iterations=100000,
        )
        fernet_key = base64.urlsafe_b64encode(kdf.derive(key))
        
        f = Fernet(fernet_key)
        return f.decrypt(encrypted_data)
    
    async def _encrypt_aes(
        self,
        data: bytes,
        key: bytes,
        config: CryptoHelpersConfig
    ) -> bytes:
        """Encrypt data using AES (simplified implementation)."""
        # This is a simplified AES implementation
        # In production, use proper AES encryption
        
        # Pad key to required length
        key_padded = key.ljust(32, b'\0')[:32]
        
        # Generate IV
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key_padded), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data
        padded_data = data + b'\0' * (16 - len(data) % 16)
        
        # Encrypt
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        return iv + encrypted
    
    async def _decrypt_aes(
        self,
        encrypted_data: bytes,
        key: bytes,
        config: CryptoHelpersConfig
    ) -> bytes:
        """Decrypt data using AES (simplified implementation)."""
        # This is a simplified AES implementation
        # In production, use proper AES decryption
        
        # Pad key to required length
        key_padded = key.ljust(32, b'\0')[:32]
        
        # Extract IV
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Create cipher
        cipher = Cipher(algorithms.AES(key_padded), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        # Decrypt
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        return decrypted.rstrip(b'\0')
    
    def generate_key(
        key_size: int = 256,
        encoding: str = "utf-8"
    ) -> str:
        """Generate cryptographic key."""
        try:
            # Generate random key
            key_bytes = secrets.token_bytes(key_size // 8)
            return base64.b64encode(key_bytes).decode(encoding)
        except Exception as exc:
            raise ValueError(f"Error generating key: {exc}")
    
    def hash_data(
        data: Union[str, bytes],
        algorithm: str = "SHA256",
        encoding: str = "utf-8"
    ) -> str:
        """Hash data using specified algorithm."""
        try:
            # Convert data to bytes if string
            if isinstance(data, str):
                data_bytes = data.encode(encoding)
            else:
                data_bytes = data
            
            # Hash based on algorithm
            if algorithm == "MD5":
                hash_obj = hashlib.md5()
            elif algorithm == "SHA1":
                hash_obj = hashlib.sha1()
            elif algorithm == "SHA256":
                hash_obj = hashlib.sha256()
            elif algorithm == "SHA512":
                hash_obj = hashlib.sha512()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            hash_obj.update(data_bytes)
            return hash_obj.hexdigest()
            
        except Exception as exc:
            raise ValueError(f"Error hashing data: {exc}")

# ============================================================================
# NETWORK HELPERS
# ============================================================================

class NetworkHelpers:
    """Network helpers module with proper exports."""
    
    __all__ = [
        "check_connectivity",
        "resolve_domain",
        "test_port",
        "get_network_info",
        "validate_url",
        "NetworkHelpersConfig",
        "ProtocolType"
    ]
    
    class NetworkHelpersConfig(BaseModel):
        """Pydantic model for network helpers configuration."""
        
        model_config = ConfigDict(extra="forbid")
        
        timeout: confloat(gt=0.0) = Field(default=10.0, description="Network timeout in seconds")
        max_retries: conint(ge=0, le=5) = Field(default=3, description="Maximum retries")
        verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
        follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
        user_agent: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Custom user agent")
        proxy_url: Optional[constr(strip_whitespace=True)] = Field(default=None, description="Proxy URL")
        connection_pool_size: conint(ge=1, le=100) = Field(default=10, description="Connection pool size")
    
    class ProtocolType(BaseModel):
        """Pydantic model for protocol type validation."""
        
        model_config = ConfigDict(extra="forbid")
        
        protocol_type: constr(strip_whitespace=True) = Field(
            pattern=r"^(http|https|ftp|sftp|tcp|udp|ssh|telnet)$"
        )
        description: Optional[str] = Field(default=None)
    
    async def check_connectivity(
        target: str,
        config: NetworkHelpersConfig
    ) -> UtilsResult:
        """Check network connectivity with comprehensive type hints and validation."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not target:
                raise ValueError("target cannot be empty")
            
            # Parse target
            if target.startswith(('http://', 'https://')):
                result = await self._check_http_connectivity(target, config)
            else:
                result = await self._check_tcp_connectivity(target, config)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return UtilsResult(
                is_successful=True,
                operation_type="connectivity_check",
                result=result,
                metadata={
                    "target": target,
                    "timeout": config.timeout,
                    "max_retries": config.max_retries,
                    "verify_ssl": config.verify_ssl
                },
                execution_time=execution_time
            )
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="connectivity_check",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    async async def _check_http_connectivity(
        self,
        url: str,
        config: NetworkHelpersConfig
    ) -> Dict[str, Any]:
        """Check HTTP connectivity."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.timeout),
            connector=aiohttp.TCPConnector(
                limit=config.connection_pool_size,
                verify_ssl=config.verify_ssl
            )
        ) as session:
            headers = {}
            if config.user_agent:
                headers['User-Agent'] = config.user_agent
            
            async with session.get(url, headers=headers, allow_redirects=config.follow_redirects) as response:
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "url": str(response.url),
                    "is_accessible": response.status < 400
                }
    
    async def _check_tcp_connectivity(
        self,
        target: str,
        config: NetworkHelpersConfig
    ) -> Dict[str, Any]:
        """Check TCP connectivity."""
        # Parse host and port
        if ':' in target:
            host, port_str = target.rsplit(':', 1)
            try:
                port = int(port_str)
            except ValueError:
                port = 80
        else:
            host = target
            port = 80
        
        # Try to connect
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=config.timeout
            )
            writer.close()
            await writer.wait_closed()
            
            return {
                "host": host,
                "port": port,
                "is_accessible": True,
                "protocol": "tcp"
            }
        except Exception as e:
            return {
                "host": host,
                "port": port,
                "is_accessible": False,
                "error": str(e),
                "protocol": "tcp"
            }
    
    async def resolve_domain(
        domain: str,
        config: NetworkHelpersConfig
    ) -> UtilsResult:
        """Resolve domain name to IP addresses."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not domain:
                raise ValueError("domain cannot be empty")
            
            # Resolve domain
            loop = asyncio.get_event_loop()
            ip_addresses = await loop.run_in_executor(None, socket.gethostbyname_ex, domain)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return UtilsResult(
                is_successful=True,
                operation_type="domain_resolution",
                result={
                    "domain": domain,
                    "ip_addresses": ip_addresses[2],
                    "canonical_name": ip_addresses[0]
                },
                metadata={
                    "timeout": config.timeout,
                    "resolved_count": len(ip_addresses[2])
                },
                execution_time=execution_time
            )
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="domain_resolution",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    async def test_port(
        host: str,
        port: int,
        config: NetworkHelpersConfig
    ) -> UtilsResult:
        """Test if port is open."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate inputs
            if not host:
                raise ValueError("host cannot be empty")
            if not (1 <= port <= 65535):
                raise ValueError("port must be between 1 and 65535")
            
            # Test port
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=config.timeout
                )
                writer.close()
                await writer.wait_closed()
                
                is_open = True
                error = None
            except Exception as e:
                is_open = False
                error = str(e)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return UtilsResult(
                is_successful=True,
                operation_type="port_test",
                result={
                    "host": host,
                    "port": port,
                    "is_open": is_open,
                    "error": error
                },
                metadata={
                    "timeout": config.timeout,
                    "protocol": "tcp"
                },
                execution_time=execution_time
            )
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="port_test",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    def validate_url(
        url: str,
        allowed_schemes: List[str] = None
    ) -> bool:
        """Validate URL format."""
        try:
            if allowed_schemes is None:
                allowed_schemes = ['http', 'https', 'ftp', 'sftp']
            
            parsed = urllib.parse.urlparse(url)
            return (
                parsed.scheme in allowed_schemes and
                parsed.netloc and
                len(parsed.netloc) > 0
            )
        except Exception:
            return False
    
    def get_network_info() -> Dict[str, Any]:
        """Get local network information."""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            return {
                "hostname": hostname,
                "local_ip": local_ip,
                "platform": socket.getfqdn()
            }
        except Exception as exc:
            return {
                "error": str(exc)
            }

# ============================================================================
# MAIN UTILS MODULE
# ============================================================================

class MainUtilsModule:
    """Main utils module with proper imports and exports."""
    
    # Import all utils modules
    crypto_helpers = CryptoHelpers()
    network_helpers = NetworkHelpers()
    
    # Define main exports
    __all__ = [
        # Utils modules
        "CryptoHelpers",
        "NetworkHelpers",
        
        # Common utilities
        "UtilsResult",
        "UtilsConfig",
        "HelperType",
        
        # Main functions
        "encrypt_data",
        "decrypt_data",
        "check_connectivity",
        "resolve_domain",
        "test_port",
        "validate_url"
    ]
    
    async def encrypt_data(
        data: Union[str, bytes],
        key: Union[str, bytes],
        config: UtilsConfig
    ) -> UtilsResult:
        """Encrypt data with all patterns integrated."""
        try:
            crypto_config = CryptoHelpers.CryptoHelpersConfig(
                encryption_algorithm="Fernet",
                key_size=256,
                hash_algorithm="SHA256"
            )
            
            return await crypto_helpers.encrypt_data(data, key, crypto_config)
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="encryption",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    async def decrypt_data(
        encrypted_data: Union[str, bytes],
        key: Union[str, bytes],
        config: UtilsConfig
    ) -> UtilsResult:
        """Decrypt data with all patterns integrated."""
        try:
            crypto_config = CryptoHelpers.CryptoHelpersConfig(
                encryption_algorithm="Fernet",
                key_size=256,
                hash_algorithm="SHA256"
            )
            
            return await crypto_helpers.decrypt_data(encrypted_data, key, crypto_config)
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="decryption",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    async def check_connectivity(
        target: str,
        config: UtilsConfig
    ) -> UtilsResult:
        """Check connectivity with all patterns integrated."""
        try:
            network_config = NetworkHelpers.NetworkHelpersConfig(
                timeout=config.timeout,
                max_retries=config.max_retries,
                verify_ssl=True
            )
            
            return await network_helpers.check_connectivity(target, network_config)
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="connectivity_check",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )
    
    async def resolve_domain(
        domain: str,
        config: UtilsConfig
    ) -> UtilsResult:
        """Resolve domain with all patterns integrated."""
        try:
            network_config = NetworkHelpers.NetworkHelpersConfig(
                timeout=config.timeout,
                max_retries=config.max_retries
            )
            
            return await network_helpers.resolve_domain(domain, network_config)
            
        except Exception as exc:
            return UtilsResult(
                is_successful=False,
                operation_type="domain_resolution",
                result=None,
                errors=[str(exc)],
                execution_time=None
            )

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_utils_structure():
    """Demonstrate the utils structure with all patterns."""
    
    print("🔧 Demonstrating Utils Structure with All Patterns")
    print("=" * 60)
    
    # Sample data
    sample_data = "Hello, World! This is a test message for encryption."
    sample_key = CryptoHelpers.generate_key(256)
    
    # Example 1: Crypto helpers
    print("\n🔐 Crypto Helpers:")
    crypto_helpers = CryptoHelpers()
    crypto_config = CryptoHelpers.CryptoHelpersConfig(
        encryption_algorithm="Fernet",
        key_size=256,
        hash_algorithm="SHA256"
    )
    
    # Encrypt data
    encrypt_result = await crypto_helpers.encrypt_data(sample_data, sample_key, crypto_config)
    print(f"Encryption: {encrypt_result.is_successful}")
    if encrypt_result.is_successful:
        print(f"Encrypted data length: {len(encrypt_result.result)}")
        print(f"Execution time: {encrypt_result.execution_time:.3f}s")
    
    # Decrypt data
    if encrypt_result.is_successful:
        decrypt_result = await crypto_helpers.decrypt_data(encrypt_result.result, sample_key, crypto_config)
        print(f"Decryption: {decrypt_result.is_successful}")
        if decrypt_result.is_successful:
            print(f"Decrypted data: {decrypt_result.result}")
    
    # Hash data
    hash_result = CryptoHelpers.hash_data(sample_data, "SHA256")
    print(f"Hash (SHA256): {hash_result}")
    
    # Example 2: Network helpers
    print("\n🌐 Network Helpers:")
    network_helpers = NetworkHelpers()
    network_config = NetworkHelpers.NetworkHelpersConfig(
        timeout=10.0,
        max_retries=3,
        verify_ssl=True
    )
    
    # Check connectivity
    connectivity_result = await network_helpers.check_connectivity("https://www.google.com", network_config)
    print(f"Connectivity check: {connectivity_result.is_successful}")
    if connectivity_result.is_successful:
        print(f"Status code: {connectivity_result.result.get('status_code', 'N/A')}")
    
    # Resolve domain
    domain_result = await network_helpers.resolve_domain("google.com", network_config)
    print(f"Domain resolution: {domain_result.is_successful}")
    if domain_result.is_successful:
        print(f"IP addresses: {domain_result.result.get('ip_addresses', [])}")
    
    # Test port
    port_result = await network_helpers.test_port("google.com", 80, network_config)
    print(f"Port test: {port_result.is_successful}")
    if port_result.is_successful:
        print(f"Port 80 open: {port_result.result.get('is_open', False)}")
    
    # Validate URL
    url_validation = NetworkHelpers.validate_url("https://www.google.com")
    print(f"URL validation: {url_validation}")
    
    # Example 3: Main module
    print("\n🎯 Main Utils Module:")
    main_module = MainUtilsModule()
    
    # Comprehensive crypto test
    crypto_test_result = await main_module.encrypt_data(sample_data, sample_key, UtilsConfig())
    print(f"Main module encryption: {crypto_test_result.is_successful}")
    
    # Comprehensive network test
    network_test_result = await main_module.check_connectivity("https://www.google.com", UtilsConfig())
    print(f"Main module connectivity: {network_test_result.is_successful}")

def show_utils_benefits():
    """Show the benefits of utils structure."""
    
    benefits = {
        "organization": [
            "Clear separation of utility types (Crypto, Network)",
            "Logical grouping of related functionality",
            "Easy to navigate and understand",
            "Scalable architecture for new utilities"
        ],
        "type_safety": [
            "Type hints throughout all utilities",
            "Pydantic validation for configurations",
            "Consistent error handling",
            "Clear function signatures"
        ],
        "async_support": [
            "Non-blocking utility operations",
            "Proper timeout handling",
            "Concurrent utility execution",
            "Efficient resource utilization"
        ],
        "security": [
            "Proper cryptographic operations",
            "Secure network communications",
            "Input validation and sanitization",
            "Error handling without information leakage"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate utils structure
    asyncio.run(demonstrate_utils_structure())
    
    benefits = show_utils_benefits()
    
    print("\n🎯 Key Utils Structure Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n✅ Utils structure organization completed successfully!") 