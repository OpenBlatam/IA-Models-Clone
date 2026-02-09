"""
Encryptor

Utilities for data encryption and decryption.
"""

import logging
from typing import Optional
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger(__name__)


class Encryptor:
    """Encrypt and decrypt data."""
    
    def __init__(
        self,
        key: Optional[bytes] = None,
        password: Optional[str] = None,
        salt: Optional[bytes] = None
    ):
        """
        Initialize encryptor.
        
        Args:
            key: Encryption key (Fernet key)
            password: Password for key derivation
            salt: Salt for key derivation
        """
        if key:
            self.key = key
        elif password:
            if salt is None:
                salt = b'salt_12345678'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key_material = kdf.derive(password.encode())
            self.key = base64.urlsafe_b64encode(key_material)
        else:
            # Generate new key
            self.key = Fernet.generate_key()
            logger.warning("Generated new encryption key. Save this key securely!")
        
        self.cipher = Fernet(self.key)
    
    def encrypt(
        self,
        data: bytes,
        file_path: Optional[str] = None
    ) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            file_path: Optional file path to save
            
        Returns:
            Encrypted bytes
        """
        encrypted = self.cipher.encrypt(data)
        
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(encrypted)
            logger.info(f"Encrypted data saved: {file_path}")
        
        return encrypted
    
    def decrypt(
        self,
        encrypted_data: bytes,
        file_path: Optional[str] = None
    ) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted bytes (or None if file_path provided)
            file_path: Optional file path to load from
            
        Returns:
            Decrypted bytes
        """
        if file_path:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
        
        decrypted = self.cipher.decrypt(encrypted_data)
        return decrypted
    
    def get_key(self) -> bytes:
        """Get encryption key."""
        return self.key


def encrypt_data(
    data: bytes,
    key: Optional[bytes] = None,
    password: Optional[str] = None,
    **kwargs
) -> bytes:
    """Encrypt data."""
    encryptor = Encryptor(key, password, **kwargs)
    return encryptor.encrypt(data)


def decrypt_data(
    encrypted_data: bytes,
    key: Optional[bytes] = None,
    password: Optional[str] = None,
    **kwargs
) -> bytes:
    """Decrypt data."""
    encryptor = Encryptor(key, password, **kwargs)
    return encryptor.decrypt(encrypted_data)


def encrypt_file(
    input_path: str,
    output_path: str,
    key: Optional[bytes] = None,
    password: Optional[str] = None,
    **kwargs
) -> str:
    """Encrypt file."""
    with open(input_path, 'rb') as f:
        data = f.read()
    
    encryptor = Encryptor(key, password, **kwargs)
    encryptor.encrypt(data, output_path)
    return output_path


def decrypt_file(
    input_path: str,
    output_path: str,
    key: Optional[bytes] = None,
    password: Optional[str] = None,
    **kwargs
) -> str:
    """Decrypt file."""
    encryptor = Encryptor(key, password, **kwargs)
    decrypted = encryptor.decrypt(None, input_path)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(decrypted)
    
    return output_path



