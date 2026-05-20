from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from typing import Dict, Optional, Tuple
from pydantic import BaseModel, field_validator
import structlog
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519, x25519
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import os
import base64
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Key generation utilities for cybersecurity tools.
"""

logger = structlog.get_logger(__name__)

class KeyGenerationInput(BaseModel):
    """Input model for key generation."""
    key_type: str = "RSA"
    key_size: int = 2048
    password: Optional[str] = None
    encoding: str = "utf-8"
    
    @field_validator('key_type')
    def validate_key_type(cls, v) -> bool:
        valid_types = ["RSA", "Ed25519", "X25519"]
        if v not in valid_types:
            raise ValueError(f"Key type must be one of: {valid_types}")
        return v
    
    @field_validator('key_size')
    def validate_key_size(cls, v) -> bool:
        if v < 1024 or v > 4096:
            raise ValueError("Key size must be between 1024 and 4096 bits")
        return v

class KeyPairResult(BaseModel):
    """Result model for key pair generation."""
    public_key: str
    private_key: str
    key_type: str
    key_size: int
    fingerprint: str
    is_successful: bool
    error_message: Optional[str] = None

def generate_key_pair(input_data: KeyGenerationInput) -> KeyPairResult:
    """
    RORO: Receive KeyGenerationInput, return KeyPairResult
    
    Generate a cryptographic key pair.
    """
    try:
        if input_data.key_type == "RSA":
            return generate_rsa_key_pair(input_data)
        elif input_data.key_type == "Ed25519":
            return generate_ed25519_key_pair(input_data)
        elif input_data.key_type == "X25519":
            return generate_x25519_key_pair(input_data)
        else:
            raise ValueError(f"Unsupported key type: {input_data.key_type}")
            
    except Exception as e:
        logger.error("Key generation failed", error=str(e))
        return KeyPairResult(
            public_key="",
            private_key="",
            key_type=input_data.key_type,
            key_size=input_data.key_size,
            fingerprint="",
            is_successful=False,
            error_message=str(e)
        )

def generate_rsa_key_pair(input_data: KeyGenerationInput) -> KeyPairResult:
    """Generate RSA key pair."""
    try:
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=input_data.key_size,
            backend=default_backend()
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(
                input_data.password.encode(input_data.encoding)
            ) if input_data.password else serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Generate fingerprint
        fingerprint = generate_key_fingerprint(public_pem)
        
        return KeyPairResult(
            public_key=public_pem.decode(input_data.encoding),
            private_key=private_pem.decode(input_data.encoding),
            key_type="RSA",
            key_size=input_data.key_size,
            fingerprint=fingerprint,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("RSA key generation failed", error=str(e))
        raise

def generate_ed25519_key_pair(input_data: KeyGenerationInput) -> KeyPairResult:
    """Generate Ed25519 key pair."""
    try:
        # Generate private key
        private_key = ed25519.Ed25519PrivateKey.generate()
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(
                input_data.password.encode(input_data.encoding)
            ) if input_data.password else serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Generate fingerprint
        fingerprint = generate_key_fingerprint(public_pem)
        
        return KeyPairResult(
            public_key=public_pem.decode(input_data.encoding),
            private_key=private_pem.decode(input_data.encoding),
            key_type="Ed25519",
            key_size=256,  # Ed25519 is always 256 bits
            fingerprint=fingerprint,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Ed25519 key generation failed", error=str(e))
        raise

def generate_x25519_key_pair(input_data: KeyGenerationInput) -> KeyPairResult:
    """Generate X25519 key pair."""
    try:
        # Generate private key
        private_key = x25519.X25519PrivateKey.generate()
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(
                input_data.password.encode(input_data.encoding)
            ) if input_data.password else serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Generate fingerprint
        fingerprint = generate_key_fingerprint(public_pem)
        
        return KeyPairResult(
            public_key=public_pem.decode(input_data.encoding),
            private_key=private_pem.decode(input_data.encoding),
            key_type="X25519",
            key_size=256,  # X25519 is always 256 bits
            fingerprint=fingerprint,
            is_successful=True
        )
        
    except Exception as e:
        logger.error("X25519 key generation failed", error=str(e))
        raise

def generate_key_fingerprint(public_key_bytes: bytes) -> str:
    """Generate SHA-256 fingerprint of public key."""
    try:
        hash_obj = hashes.Hash(hashes.SHA256(), backend=default_backend())
        hash_obj.update(public_key_bytes)
        digest = hash_obj.finalize()
        
        # Format as colon-separated hex pairs
        fingerprint = ":".join(f"{b:02x}" for b in digest)
        return fingerprint
        
    except Exception as e:
        logger.error("Fingerprint generation failed", error=str(e))
        return ""

def derive_key_from_password(password: str, salt: Optional[bytes] = None, key_length: int = 32) -> Tuple[bytes, bytes]:
    """
    Derive a cryptographic key from a password using PBKDF2.
    
    Returns:
        Tuple of (derived_key, salt)
    """
    try:
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        derived_key = kdf.derive(password.encode('utf-8'))
        return derived_key, salt
        
    except Exception as e:
        logger.error("Key derivation failed", error=str(e))
        raise 