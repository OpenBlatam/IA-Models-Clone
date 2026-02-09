"""
Encryption Service
Data encryption and decryption
"""

from typing import Optional
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import logging

logger = logging.getLogger(__name__)


class EncryptionService:
    """Encryption service for sensitive data"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key:
            self.key = key
        else:
            # Generate or load key from environment
            key_str = os.getenv("ENCRYPTION_KEY")
            if key_str:
                self.key = key_str.encode()
            else:
                # Generate new key (in production, store securely)
                self.key = Fernet.generate_key()
                logger.warning("Generated new encryption key. Store ENCRYPTION_KEY in environment.")
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt data
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data (base64 encoded)
        """
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data
        
        Args:
            encrypted_data: Encrypted data (base64 encoded)
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt file"""
        from pathlib import Path
        
        input_path = Path(file_path)
        if output_path:
            output = Path(output_path)
        else:
            output = input_path.parent / f"{input_path.stem}.encrypted{input_path.suffix}"
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        encrypted = self.cipher.encrypt(data)
        
        with open(output, 'wb') as f:
            f.write(encrypted)
        
        return str(output)
    
    def decrypt_file(self, encrypted_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt file"""
        from pathlib import Path
        
        input_path = Path(encrypted_path)
        if output_path:
            output = Path(output_path)
        else:
            output = input_path.parent / input_path.stem.replace('.encrypted', '')
        
        with open(input_path, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.cipher.decrypt(encrypted)
        
        with open(output, 'wb') as f:
            f.write(decrypted)
        
        return str(output)


_encryption_service: Optional[EncryptionService] = None


def get_encryption_service(key: Optional[bytes] = None) -> EncryptionService:
    """Get encryption service instance (singleton)"""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService(key=key)
    return _encryption_service

