"""
Security Manager
================

Advanced security utilities for encryption, hashing, and validation.
"""

import hashlib
import hmac
import secrets
import base64
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class SecurityManager:
    """Advanced security manager."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self._fernet = None
    
    def _get_fernet(self) -> Fernet:
        """Get or create Fernet cipher."""
        if self._fernet is None:
            # Derive key from secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'bulk_truthgpt_salt',
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt(self, data: str) -> str:
        """Encrypt data."""
        try:
            fernet = self._get_fernet()
            encrypted = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        try:
            fernet = self._get_fernet()
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def hash_password(self, password: str) -> str:
        """Hash password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=secrets.token_bytes(16),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key.decode()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            # This is simplified - in production use proper password verification
            return self.hash_password(password) == hashed
        except:
            return False
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure token."""
        return secrets.token_urlsafe(length)
    
    def generate_hmac(self, data: str) -> str:
        """Generate HMAC for data."""
        return hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_hmac(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.generate_hmac(data)
        return hmac.compare_digest(expected, signature)
    
    def sanitize_input(self, data: str, max_length: Optional[int] = None) -> str:
        """Sanitize user input."""
        # Remove control characters
        sanitized = ''.join(c for c in data if ord(c) >= 32 or c in '\n\r\t')
        
        # Limit length
        if max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()

# Global instance
security_manager = SecurityManager()
































