from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import hashlib
import secrets
import base64
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cryptographic utilities with proper async/def distinction.
Def for CPU-bound cryptographic operations.
"""


@dataclass
class SecurityConfig:
    """Security configuration for cryptographic operations."""
    key_length: int = 32
    salt_length: int = 16
    hash_algorithm: str = "sha256"
    iterations: int = 100000
    key_derivation_algorithm: str = "pbkdf2"

def hash_password(password: str, config: SecurityConfig) -> str:
    """Hash password with salt using PBKDF2."""
    salt = secrets.token_bytes(config.salt_length)
    hash_obj = hashlib.pbkdf2_hmac(
        config.hash_algorithm,
        password.encode('utf-8'),
        salt,
        config.iterations
    )
    return f"{salt.hex()}:{hash_obj.hex()}"

def verify_password(password: str, hashed: str, config: SecurityConfig) -> bool:
    """Verify password against stored hash."""
    try:
        salt_hex, hash_hex = hashed.split(':')
        salt = bytes.fromhex(salt_hex)
        hash_obj = hashlib.pbkdf2_hmac(
            config.hash_algorithm,
            password.encode('utf-8'),
            salt,
            config.iterations
        )
        return secrets.compare_digest(hash_obj.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False

def generate_secure_key(config: SecurityConfig) -> bytes:
    """Generate cryptographically secure random key."""
    return secrets.token_bytes(config.key_length)

def generate_rsa_keypair(key_size: int = 2048) -> tuple[bytes, bytes]:
    """Generate RSA key pair."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size
    )
    
    public_key = private_key.public_key()
    
    # Serialize keys
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem, public_pem

def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt data using Fernet (symmetric encryption)."""
    f = Fernet(base64.urlsafe_b64encode(key))
    return f.encrypt(data)

def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """Decrypt data using Fernet (symmetric encryption)."""
    f = Fernet(base64.urlsafe_b64encode(key))
    return f.decrypt(encrypted_data)

def encrypt_asymmetric(data: bytes, public_key_pem: bytes) -> bytes:
    """Encrypt data using RSA public key."""
    public_key = serialization.load_pem_public_key(public_key_pem)
    encrypted = public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted

def decrypt_asymmetric(encrypted_data: bytes, private_key_pem: bytes) -> bytes:
    """Decrypt data using RSA private key."""
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None
    )
    decrypted = private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return decrypted

def create_digital_signature(data: bytes, private_key_pem: bytes) -> bytes:
    """Create digital signature using RSA private key."""
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None
    )
    signature = private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature

def verify_digital_signature(data: bytes, signature: bytes, public_key_pem: bytes) -> bool:
    """Verify digital signature using RSA public key."""
    try:
        public_key = serialization.load_pem_public_key(public_key_pem)
        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False

def generate_secure_token(length: int = 32) -> str:
    """Generate secure random token."""
    return secrets.token_urlsafe(length)

def hash_file(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate hash of file content."""
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        for chunk in iter(lambda: f.read(4096), b""):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def derive_key_from_password(password: str, salt: bytes, config: SecurityConfig) -> bytes:
    """Derive encryption key from password using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=config.key_length,
        salt=salt,
        iterations=config.iterations
    )
    return kdf.derive(password.encode('utf-8'))

def verify_key_derivation(password: str, salt: bytes, key: bytes, config: SecurityConfig) -> bool:
    """Verify key derivation from password."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=config.key_length,
        salt=salt,
        iterations=config.iterations
    )
    try:
        kdf.verify(password.encode('utf-8'), key)
        return True
    except Exception:
        return False

def secure_compare(a: bytes, b: bytes) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return secrets.compare_digest(a, b)

def generate_nonce(length: int = 16) -> bytes:
    """Generate cryptographically secure nonce."""
    return secrets.token_bytes(length)

def calculate_hmac(data: bytes, key: bytes, algorithm: str = "sha256") -> bytes:
    """Calculate HMAC of data with key."""
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(key)
    hash_obj.update(data)
    return hash_obj.digest()

def verify_hmac(data: bytes, key: bytes, hmac_value: bytes, algorithm: str = "sha256") -> bool:
    """Verify HMAC of data with key."""
    expected_hmac = calculate_hmac(data, key, algorithm)
    return secure_compare(expected_hmac, hmac_value)

# Named exports for main functionality
__all__ = [
    'hash_password',
    'verify_password',
    'generate_secure_key',
    'generate_rsa_keypair',
    'encrypt_data',
    'decrypt_data',
    'encrypt_asymmetric',
    'decrypt_asymmetric',
    'create_digital_signature',
    'verify_digital_signature',
    'generate_secure_token',
    'hash_file',
    'derive_key_from_password',
    'verify_key_derivation',
    'secure_compare',
    'generate_nonce',
    'calculate_hmac',
    'verify_hmac',
    'SecurityConfig'
] 