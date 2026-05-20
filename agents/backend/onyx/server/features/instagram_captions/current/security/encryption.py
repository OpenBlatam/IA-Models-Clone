"""
Encryption Utilities for Instagram Captions API v10.0

Advanced encryption and cryptographic functions.
"""

import hashlib
import hmac
import secrets
import base64
from typing import Dict, Any, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class EncryptionUtils:
    """Advanced encryption utilities."""
    
    def __init__(self):
        self.encryption_key = None
        self.fernet = None
    
    def generate_encryption_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def initialize_encryption(self, key: bytes):
        """Initialize encryption with a key."""
        self.encryption_key = key
        self.fernet = Fernet(key)
    
    def encrypt_text(self, text: str) -> str:
        """Encrypt text using Fernet symmetric encryption."""
        if not self.fernet:
            raise ValueError("Encryption not initialized. Call initialize_encryption() first.")
        
        encrypted_data = self.fernet.encrypt(text.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """Decrypt text using Fernet symmetric encryption."""
        if not self.fernet:
            raise ValueError("Encryption not initialized. Call initialize_encryption() first.")
        
        encrypted_data = base64.urlsafe_b64decode(encrypted_text.encode())
        decrypted_data = self.fernet.decrypt(encrypted_data)
        return decrypted_data.decode()
    
    @staticmethod
    def hash_data(data: str, algorithm: str = "sha256") -> str:
        """Hash data using specified algorithm."""
        if algorithm == "sha256":
            return hashlib.sha256(data.encode()).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data.encode()).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data.encode()).hexdigest()
        elif algorithm == "blake2b":
            return hashlib.blake2b(data.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def create_hmac(data: str, key: str, algorithm: str = "sha256") -> str:
        """Create HMAC for data verification."""
        if algorithm == "sha256":
            return hmac.new(key.encode(), data.encode(), hashlib.sha256).hexdigest()
        elif algorithm == "sha512":
            return hmac.new(key.encode(), data.encode(), hashlib.sha512).hexdigest()
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
    
    @staticmethod
    def verify_hmac(data: str, key: str, expected_hmac: str, algorithm: str = "sha256") -> bool:
        """Verify HMAC for data integrity."""
        calculated_hmac = EncryptionUtils.create_hmac(data, key, algorithm)
        return hmac.compare_digest(calculated_hmac, expected_hmac)
    
    @staticmethod
    def generate_random_bytes(length: int = 32) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_random_hex(length: int = 32) -> str:
        """Generate cryptographically secure random hex string."""
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_random_urlsafe(length: int = 32) -> str:
        """Generate cryptographically secure random URL-safe string."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def secure_compare(a: str, b: str) -> bool:
        """Securely compare two strings to prevent timing attacks."""
        return hmac.compare_digest(a, b)
    
    def encrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: list) -> Dict[str, Any]:
        """Encrypt sensitive fields in a data dictionary."""
        if not self.fernet:
            raise ValueError("Encryption not initialized. Call initialize_encryption() first.")
        
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data and isinstance(encrypted_data[field], str):
                encrypted_data[field] = self.encrypt_text(encrypted_data[field])
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: list) -> Dict[str, Any]:
        """Decrypt sensitive fields in a data dictionary."""
        if not self.fernet:
            raise ValueError("Encryption not initialized. Call initialize_encryption() first.")
        
        decrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in decrypted_data and isinstance(decrypted_data[field], str):
                try:
                    decrypted_data[field] = self.decrypt_text(decrypted_data[field])
                except Exception:
                    # Field might not be encrypted, leave as is
                    pass
        
        return decrypted_data
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """Get information about current encryption setup."""
        return {
            'encryption_initialized': self.fernet is not None,
            'key_length': len(self.encryption_key) if self.encryption_key else 0,
            'algorithm': 'Fernet (AES-128-CBC)',
            'key_derivation': 'PBKDF2-HMAC-SHA256'
        }






