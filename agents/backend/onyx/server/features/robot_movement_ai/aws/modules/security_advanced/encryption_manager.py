"""
Encryption Manager
=================

Advanced encryption management.
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Advanced encryption manager."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self._master_key = master_key
        else:
            # Generate master key (in production, use secure key management)
            self._master_key = Fernet.generate_key()
        
        self._fernet = Fernet(self._master_key)
        self._keys: Dict[str, bytes] = {}
    
    def generate_key(self, key_id: str) -> bytes:
        """Generate encryption key."""
        key = Fernet.generate_key()
        self._keys[key_id] = key
        logger.info(f"Generated key: {key_id}")
        return key
    
    def encrypt(self, data: str, key_id: Optional[str] = None) -> str:
        """Encrypt data."""
        if key_id and key_id in self._keys:
            fernet = Fernet(self._keys[key_id])
        else:
            fernet = self._fernet
        
        encrypted = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str, key_id: Optional[str] = None) -> str:
        """Decrypt data."""
        if key_id and key_id in self._keys:
            fernet = Fernet(self._keys[key_id])
        else:
            fernet = self._fernet
        
        decoded = base64.b64decode(encrypted_data.encode())
        decrypted = fernet.decrypt(decoded)
        return decrypted.decode()
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode())
        return {
            "hash": base64.b64encode(key).decode(),
            "salt": base64.b64encode(salt).decode()
        }
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password."""
        salt_bytes = base64.b64decode(salt.encode())
        hash_bytes = base64.b64decode(password_hash.encode())
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=100000,
            backend=default_backend()
        )
        
        try:
            kdf.verify(password.encode(), hash_bytes)
            return True
        except Exception:
            return False
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """Hash data."""
        if algorithm == "sha256":
            return hashlib.sha256(data.encode()).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data.encode()).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")















