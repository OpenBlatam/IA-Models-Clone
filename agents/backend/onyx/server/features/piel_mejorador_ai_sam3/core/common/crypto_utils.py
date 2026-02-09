"""
Cryptographic Utilities for Piel Mejorador AI SAM3
==================================================

Unified cryptographic and hashing utilities.
"""

import hashlib
import hmac
import secrets
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


class CryptoUtils:
    """Unified cryptographic utilities."""
    
    @staticmethod
    def sha256(data: Union[str, bytes]) -> str:
        """
        Generate SHA256 hash.
        
        Args:
            data: Data to hash (string or bytes)
            
        Returns:
            Hex digest
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def sha256_bytes(data: Union[str, bytes]) -> bytes:
        """
        Generate SHA256 hash as bytes.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).digest()
    
    @staticmethod
    def md5(data: Union[str, bytes]) -> str:
        """
        Generate MD5 hash.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex digest
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def hmac_sha256(
        message: Union[str, bytes],
        secret: Union[str, bytes],
        digest: bool = True
    ) -> Union[str, bytes]:
        """
        Generate HMAC-SHA256 signature.
        
        Args:
            message: Message to sign
            secret: Secret key
            digest: Whether to return hex digest (True) or bytes (False)
            
        Returns:
            HMAC signature
        """
        if isinstance(message, str):
            message = message.encode('utf-8')
        if isinstance(secret, str):
            secret = secret.encode('utf-8')
        
        signature = hmac.new(secret, message, hashlib.sha256)
        
        if digest:
            return signature.hexdigest()
        return signature.digest()
    
    @staticmethod
    def verify_hmac(
        message: Union[str, bytes],
        signature: str,
        secret: Union[str, bytes]
    ) -> bool:
        """
        Verify HMAC signature.
        
        Args:
            message: Original message
            signature: Signature to verify
            secret: Secret key
            
        Returns:
            True if signature is valid
        """
        expected = CryptoUtils.hmac_sha256(message, secret)
        return hmac.compare_digest(expected, signature)
    
    @staticmethod
    def hash_data(data: Union[str, bytes, dict], algorithm: str = "sha256") -> str:
        """
        Hash data (supports dict by converting to JSON).
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, md5)
            
        Returns:
            Hash hex digest
        """
        import json
        
        # Convert dict to JSON string
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def generate_secret(length: int = 32) -> str:
        """
        Generate secure random secret.
        
        Args:
            length: Secret length in bytes
            
        Returns:
            URL-safe secret
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            URL-safe token
        """
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_hex_token(length: int = 32) -> str:
        """
        Generate hex token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Hex token
        """
        return secrets.token_hex(length)
    
    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """
        Constant-time string comparison (prevents timing attacks).
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            True if equal
        """
        return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


# Convenience functions
def sha256(data: Union[str, bytes]) -> str:
    """Generate SHA256 hash."""
    return CryptoUtils.sha256(data)


def hmac_sign(message: Union[str, bytes], secret: Union[str, bytes]) -> str:
    """Generate HMAC signature."""
    return CryptoUtils.hmac_sha256(message, secret)


def verify_signature(message: Union[str, bytes], signature: str, secret: Union[str, bytes]) -> bool:
    """Verify HMAC signature."""
    return CryptoUtils.verify_hmac(message, signature, secret)




