"""
Encryption Utilities
====================

Utilities for encryption and hashing.
"""

import hashlib
import secrets
import base64
import logging
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Encryption manager for secure data handling.
    
    Features:
    - Symmetric encryption (Fernet)
    - Password-based key derivation
    - Secure random generation
    - Hashing utilities
    """
    
    def __init__(self, key: Optional[bytes] = None, password: Optional[str] = None):
        """
        Initialize encryption manager.
        
        Args:
            key: Encryption key (bytes)
            password: Password for key derivation
        """
        if key:
            self.key = key
        elif password:
            self.key = self._derive_key_from_password(password)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
        self._logger = logger
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """
        Derive key from password.
        
        Args:
            password: Password string
            salt: Optional salt (generated if None)
        
        Returns:
            Derived key
        """
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt (str or bytes)
        
        Returns:
            Encrypted bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted bytes
        
        Returns:
            Decrypted bytes
        """
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_string(self, text: str) -> str:
        """
        Encrypt string and return base64 encoded.
        
        Args:
            text: Text to encrypt
        
        Returns:
            Base64 encoded encrypted string
        """
        encrypted = self.encrypt(text)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """
        Decrypt base64 encoded encrypted string.
        
        Args:
            encrypted_text: Base64 encoded encrypted string
        
        Returns:
            Decrypted string
        """
        encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
        decrypted = self.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def get_key(self) -> bytes:
        """Get encryption key."""
        return self.key
    
    def save_key(self, filepath: str) -> None:
        """
        Save key to file.
        
        Args:
            filepath: Path to save key
        """
        with open(filepath, 'wb') as f:
            f.write(self.key)
        self._logger.info(f"Key saved to {filepath}")
    
    @staticmethod
    def load_key(filepath: str) -> bytes:
        """
        Load key from file.
        
        Args:
            filepath: Path to key file
        
        Returns:
            Key bytes
        """
        with open(filepath, 'rb') as f:
            return f.read()


class HashManager:
    """Hash manager for secure hashing."""
    
    @staticmethod
    def hash_sha256(data: Union[str, bytes]) -> str:
        """
        Hash data using SHA256.
        
        Args:
            data: Data to hash
        
        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def hash_sha512(data: Union[str, bytes]) -> str:
        """
        Hash data using SHA512.
        
        Args:
            data: Data to hash
        
        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha512(data).hexdigest()
    
    @staticmethod
    def hash_md5(data: Union[str, bytes]) -> str:
        """
        Hash data using MD5.
        
        Args:
            data: Data to hash
        
        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate secure random token.
        
        Args:
            length: Token length in bytes
        
        Returns:
            Hexadecimal token string
        """
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_password(length: int = 16) -> str:
        """
        Generate secure random password.
        
        Args:
            length: Password length
        
        Returns:
            Random password string
        """
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def verify_hash(data: Union[str, bytes], hash_value: str, algorithm: str = "sha256") -> bool:
        """
        Verify data against hash.
        
        Args:
            data: Data to verify
            hash_value: Hash to compare against
            algorithm: Hash algorithm
        
        Returns:
            True if hash matches
        """
        if algorithm == "sha256":
            computed = HashManager.hash_sha256(data)
        elif algorithm == "sha512":
            computed = HashManager.hash_sha512(data)
        elif algorithm == "md5":
            computed = HashManager.hash_md5(data)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return secrets.compare_digest(computed, hash_value)




