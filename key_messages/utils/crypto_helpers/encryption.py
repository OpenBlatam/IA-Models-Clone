from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Optional, Union
from pydantic import BaseModel, field_validator
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import base64
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Encryption utilities for cybersecurity tools.
"""

logger = structlog.get_logger(__name__)

class EncryptionInput(BaseModel):
    """Input model for data encryption."""
    data: Union[str, bytes]
    key: Optional[str] = None
    algorithm: str = "AES-256-GCM"
    encoding: str = "utf-8"
    
    @field_validator('data')
    def validate_data(cls, v) -> bool:
        if not v:
            raise ValueError("Data cannot be empty")
        return v
    
    @field_validator('algorithm')
    def validate_algorithm(cls, v) -> bool:
        valid_algorithms = ["AES-256-GCM", "AES-256-CBC", "Fernet"]
        if v not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of: {valid_algorithms}")
        return v

class DecryptionInput(BaseModel):
    """Input model for data decryption."""
    encrypted_data: Union[str, bytes]
    key: str
    algorithm: str = "AES-256-GCM"
    encoding: str = "utf-8"
    
    @field_validator('encrypted_data')
    def validate_encrypted_data(cls, v) -> bool:
        if not v:
            raise ValueError("Encrypted data cannot be empty")
        return v
    
    @field_validator('key')
    def validate_key(cls, v) -> bool:
        if not v:
            raise ValueError("Key cannot be empty")
        return v

class EncryptionResult(BaseModel):
    """Result model for encryption operations."""
    encrypted_data: str
    key: str
    algorithm: str
    iv: Optional[str] = None
    tag: Optional[str] = None
    is_successful: bool
    error_message: Optional[str] = None

class DecryptionResult(BaseModel):
    """Result model for decryption operations."""
    decrypted_data: str
    algorithm: str
    is_successful: bool
    error_message: Optional[str] = None

def encrypt_data(input_data: EncryptionInput) -> EncryptionResult:
    """
    RORO: Receive EncryptionInput, return EncryptionResult
    
    Encrypt data using specified algorithm.
    """
    try:
        # Convert data to bytes if it's a string
        if isinstance(input_data.data, str):
            data_bytes = input_data.data.encode(input_data.encoding)
        else:
            data_bytes = input_data.data
        
        if input_data.algorithm == "Fernet":
            return encrypt_with_fernet(data_bytes, input_data.key, input_data.encoding)
        elif input_data.algorithm == "AES-256-GCM":
            return encrypt_with_aes_gcm(data_bytes, input_data.key, input_data.encoding)
        elif input_data.algorithm == "AES-256-CBC":
            return encrypt_with_aes_cbc(data_bytes, input_data.key, input_data.encoding)
        else:
            raise ValueError(f"Unsupported algorithm: {input_data.algorithm}")
            
    except Exception as e:
        logger.error("Encryption failed", error=str(e))
        return EncryptionResult(
            encrypted_data="",
            key="",
            algorithm=input_data.algorithm,
            is_successful=False,
            error_message=str(e)
        )

def decrypt_data(input_data: DecryptionInput) -> DecryptionResult:
    """
    RORO: Receive DecryptionInput, return DecryptionResult
    
    Decrypt data using specified algorithm.
    """
    try:
        # Convert encrypted data to bytes if it's a string
        if isinstance(input_data.encrypted_data, str):
            encrypted_bytes = base64.b64decode(input_data.encrypted_data)
        else:
            encrypted_bytes = input_data.encrypted_data
        
        if input_data.algorithm == "Fernet":
            return decrypt_with_fernet(encrypted_bytes, input_data.key, input_data.encoding)
        elif input_data.algorithm == "AES-256-GCM":
            return decrypt_with_aes_gcm(encrypted_bytes, input_data.key, input_data.encoding)
        elif input_data.algorithm == "AES-256-CBC":
            return decrypt_with_aes_cbc(encrypted_bytes, input_data.key, input_data.encoding)
        else:
            raise ValueError(f"Unsupported algorithm: {input_data.algorithm}")
            
    except Exception as e:
        logger.error("Decryption failed", error=str(e))
        return DecryptionResult(
            decrypted_data="",
            algorithm=input_data.algorithm,
            is_successful=False,
            error_message=str(e)
        )

def encrypt_with_fernet(data_bytes: bytes, key: Optional[str], encoding: str) -> EncryptionResult:
    """Encrypt data using Fernet (symmetric encryption)."""
    try:
        if key is None:
            # Generate a new key
            key = Fernet.generate_key()
        elif isinstance(key, str):
            key = key.encode(encoding)
        
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data_bytes)
        
        return EncryptionResult(
            encrypted_data=base64.b64encode(encrypted_data).decode(encoding),
            key=base64.b64encode(key).decode(encoding),
            algorithm="Fernet",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Fernet encryption failed", error=str(e))
        raise

def decrypt_with_fernet(encrypted_bytes: bytes, key: str, encoding: str) -> DecryptionResult:
    """Decrypt data using Fernet."""
    try:
        if isinstance(key, str):
            key = base64.b64decode(key.encode(encoding))
        
        fernet = Fernet(key)
        decrypted_data = fernet.decrypt(encrypted_bytes)
        
        return DecryptionResult(
            decrypted_data=decrypted_data.decode(encoding),
            algorithm="Fernet",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("Fernet decryption failed", error=str(e))
        raise

def encrypt_with_aes_gcm(data_bytes: bytes, key: Optional[str], encoding: str) -> EncryptionResult:
    """Encrypt data using AES-256-GCM."""
    try:
        if key is None:
            # Generate a new key
            key = os.urandom(32)
        elif isinstance(key, str):
            key = key.encode(encoding)
        
        # Generate IV
        iv = os.urandom(12)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
        
        # Get tag
        tag = encryptor.tag
        
        # Combine IV + tag + ciphertext
        encrypted_data = iv + tag + ciphertext
        
        return EncryptionResult(
            encrypted_data=base64.b64encode(encrypted_data).decode(encoding),
            key=base64.b64encode(key).decode(encoding),
            algorithm="AES-256-GCM",
            iv=base64.b64encode(iv).decode(encoding),
            tag=base64.b64encode(tag).decode(encoding),
            is_successful=True
        )
        
    except Exception as e:
        logger.error("AES-GCM encryption failed", error=str(e))
        raise

def decrypt_with_aes_gcm(encrypted_bytes: bytes, key: str, encoding: str) -> DecryptionResult:
    """Decrypt data using AES-256-GCM."""
    try:
        if isinstance(key, str):
            key = base64.b64decode(key.encode(encoding))
        
        # Extract IV, tag, and ciphertext
        iv = encrypted_bytes[:12]
        tag = encrypted_bytes[12:28]
        ciphertext = encrypted_bytes[28:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt data
        decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        return DecryptionResult(
            decrypted_data=decrypted_data.decode(encoding),
            algorithm="AES-256-GCM",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("AES-GCM decryption failed", error=str(e))
        raise

def encrypt_with_aes_cbc(data_bytes: bytes, key: Optional[str], encoding: str) -> EncryptionResult:
    """Encrypt data using AES-256-CBC."""
    try:
        if key is None:
            # Generate a new key
            key = os.urandom(32)
        elif isinstance(key, str):
            key = key.encode(encoding)
        
        # Generate IV
        iv = os.urandom(16)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        block_size = 16
        padding_length = block_size - (len(data_bytes) % block_size)
        padded_data = data_bytes + bytes([padding_length] * padding_length)
        
        # Encrypt data
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine IV + ciphertext
        encrypted_data = iv + ciphertext
        
        return EncryptionResult(
            encrypted_data=base64.b64encode(encrypted_data).decode(encoding),
            key=base64.b64encode(key).decode(encoding),
            algorithm="AES-256-CBC",
            iv=base64.b64encode(iv).decode(encoding),
            is_successful=True
        )
        
    except Exception as e:
        logger.error("AES-CBC encryption failed", error=str(e))
        raise

def decrypt_with_aes_cbc(encrypted_bytes: bytes, key: str, encoding: str) -> DecryptionResult:
    """Decrypt data using AES-256-CBC."""
    try:
        if isinstance(key, str):
            key = base64.b64decode(key.encode(encoding))
        
        # Extract IV and ciphertext
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt data
        decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding_length = decrypted_data[-1]
        decrypted_data = decrypted_data[:-padding_length]
        
        return DecryptionResult(
            decrypted_data=decrypted_data.decode(encoding),
            algorithm="AES-256-CBC",
            is_successful=True
        )
        
    except Exception as e:
        logger.error("AES-CBC decryption failed", error=str(e))
        raise 