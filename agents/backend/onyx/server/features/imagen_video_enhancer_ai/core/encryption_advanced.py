"""
Advanced Encryption System
==========================

Advanced encryption system with multiple algorithms and key management.
"""

import logging
import base64
import hashlib
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Encryption algorithms."""
    FERNET = "fernet"
    AES = "aes"


@dataclass
class EncryptionKey:
    """Encryption key."""
    key: bytes
    algorithm: EncryptionAlgorithm
    created_at: Optional[Any] = None


class AdvancedEncryptionManager:
    """Advanced encryption manager."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize advanced encryption manager.
        
        Args:
            master_key: Optional master key (generated if not provided)
        """
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        self.keys: Dict[str, EncryptionKey] = {}
    
    def generate_key(self, name: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET) -> bytes:
        """
        Generate encryption key.
        
        Args:
            name: Key name
            algorithm: Encryption algorithm
            
        Returns:
            Generated key bytes
        """
        if algorithm == EncryptionAlgorithm.FERNET:
            key = Fernet.generate_key()
        else:
            # For AES, generate a 32-byte key
            key = Fernet.generate_key()[:32]
        
        self.keys[name] = EncryptionKey(
            key=key,
            algorithm=algorithm
        )
        
        logger.info(f"Generated encryption key: {name}")
        return key
    
    def encrypt(
        self,
        data: str,
        key_name: Optional[str] = None,
        algorithm: Optional[EncryptionAlgorithm] = None
    ) -> str:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            key_name: Optional key name
            algorithm: Optional algorithm
            
        Returns:
            Encrypted data (base64 encoded)
        """
        if key_name and key_name in self.keys:
            key_obj = self.keys[key_name]
            if key_obj.algorithm == EncryptionAlgorithm.FERNET:
                fernet = Fernet(key_obj.key)
                encrypted = fernet.encrypt(data.encode('utf-8'))
                return base64.b64encode(encrypted).decode('utf-8')
        else:
            # Use default master key
            encrypted = self.fernet.encrypt(data.encode('utf-8'))
            return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt(
        self,
        encrypted_data: str,
        key_name: Optional[str] = None
    ) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data (base64 encoded)
            key_name: Optional key name
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            if key_name and key_name in self.keys:
                key_obj = self.keys[key_name]
                if key_obj.algorithm == EncryptionAlgorithm.FERNET:
                    fernet = Fernet(key_obj.key)
                    decrypted = fernet.decrypt(encrypted_bytes)
                    return decrypted.decode('utf-8')
            else:
                # Use default master key
                decrypted = self.fernet.decrypt(encrypted_bytes)
                return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError(f"Failed to decrypt data: {e}")
    
    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """
        Hash data.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm
            
        Returns:
            Hash string
        """
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        return hash_func(data.encode('utf-8')).hexdigest()
    
    def derive_key_from_password(
        self,
        password: str,
        salt: Optional[bytes] = None,
        length: int = 32
    ) -> Tuple[bytes, bytes]:
        """
        Derive key from password using PBKDF2.
        
        Args:
            password: Password
            salt: Optional salt
            length: Key length
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = Fernet.generate_key()[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode('utf-8'))
        return key, salt
    
    def get_key(self, name: str) -> Optional[EncryptionKey]:
        """
        Get encryption key by name.
        
        Args:
            name: Key name
            
        Returns:
            Encryption key or None
        """
        return self.keys.get(name)
    
    def delete_key(self, name: str):
        """
        Delete encryption key.
        
        Args:
            name: Key name
        """
        if name in self.keys:
            del self.keys[name]
            logger.info(f"Deleted encryption key: {name}")



