"""
Encryption Utilities
====================

Advanced encryption utilities.
"""

import hashlib
import hmac
import secrets
import base64
import logging
from typing import Optional, bytes
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class EncryptionUtils:
    """Encryption utility functions."""
    
    @staticmethod
    def generate_key(password: Optional[bytes] = None, salt: Optional[bytes] = None) -> bytes:
        """
        Generate encryption key.
        
        Args:
            password: Optional password
            salt: Optional salt
            
        Returns:
            Encryption key
        """
        if password is None:
            return Fernet.generate_key()
        
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    @staticmethod
    def encrypt(data: bytes, key: bytes) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            key: Encryption key
            
        Returns:
            Encrypted data
        """
        fernet = Fernet(key)
        return fernet.encrypt(data)
    
    @staticmethod
    def decrypt(data: bytes, key: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            data: Encrypted data
            key: Encryption key
            
        Returns:
            Decrypted data
        """
        fernet = Fernet(key)
        return fernet.decrypt(data)
    
    @staticmethod
    def hash_data(data: bytes, algorithm: str = "sha256") -> str:
        """
        Hash data.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, sha512, md5)
            
        Returns:
            Hash string
        """
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def hmac_sign(data: bytes, key: bytes, algorithm: str = "sha256") -> str:
        """
        Generate HMAC signature.
        
        Args:
            data: Data to sign
            key: HMAC key
            algorithm: Hash algorithm
            
        Returns:
            HMAC signature
        """
        if algorithm == "sha256":
            return hmac.new(key, data, hashlib.sha256).hexdigest()
        elif algorithm == "sha512":
            return hmac.new(key, data, hashlib.sha512).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def verify_hmac(data: bytes, signature: str, key: bytes, algorithm: str = "sha256") -> bool:
        """
        Verify HMAC signature.
        
        Args:
            data: Data to verify
            signature: HMAC signature
            key: HMAC key
            algorithm: Hash algorithm
            
        Returns:
            True if signature is valid
        """
        expected = EncryptionUtils.hmac_sign(data, key, algorithm)
        return hmac.compare_digest(expected, signature)
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate random token.
        
        Args:
            length: Token length
            
        Returns:
            Random token
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_salt(length: int = 16) -> bytes:
        """
        Generate random salt.
        
        Args:
            length: Salt length
            
        Returns:
            Random salt
        """
        return secrets.token_bytes(length)




