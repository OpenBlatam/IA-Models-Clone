"""
Security System
==============

Advanced security system with encryption, hashing, and token management.
"""

import logging
import hashlib
import hmac
import secrets
import base64
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SecurityManager:
    """Security manager for encryption and hashing."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize security manager.
        
        Args:
            secret_key: Secret key for operations (generated if not provided)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash a password.
        
        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        import hashlib
        import binascii
        
        # Simple implementation (in production, use proper PBKDF2)
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        hashed = binascii.hexlify(hash_obj).decode('utf-8')
        
        return hashed, salt
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """
        Verify a password.
        
        Args:
            password: Password to verify
            hashed: Hashed password
            salt: Salt used for hashing
            
        Returns:
            True if password matches
        """
        new_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)
    
    def generate_token(self, length: int = 32) -> str:
        """
        Generate a secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Secure random token
        """
        return secrets.token_urlsafe(length)
    
    def generate_api_key(self) -> str:
        """
        Generate an API key.
        
        Returns:
            API key string
        """
        return f"sk_{secrets.token_urlsafe(32)}"
    
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
    
    def create_signature(self, data: str) -> str:
        """
        Create HMAC signature.
        
        Args:
            data: Data to sign
            
        Returns:
            Signature string
        """
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """
        Verify HMAC signature.
        
        Args:
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if signature is valid
        """
        expected = self.create_signature(data)
        return hmac.compare_digest(expected, signature)
    
    def encrypt_simple(self, data: str) -> str:
        """
        Simple encryption (base64 encoding).
        
        Note: For production, use proper encryption like AES.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted string
        """
        encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
        return encoded
    
    def decrypt_simple(self, encrypted: str) -> str:
        """
        Simple decryption (base64 decoding).
        
        Args:
            encrypted: Encrypted string
            
        Returns:
            Decrypted string
        """
        decoded = base64.b64decode(encrypted.encode('utf-8')).decode('utf-8')
        return decoded


@dataclass
class Token:
    """Token data structure."""
    value: str
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class TokenManager:
    """Manager for token generation and validation."""
    
    def __init__(self, security_manager: SecurityManager):
        """
        Initialize token manager.
        
        Args:
            security_manager: Security manager instance
        """
        self.security_manager = security_manager
        self.tokens: Dict[str, Token] = {}
    
    def generate_token(
        self,
        expires_in_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Token:
        """
        Generate a new token.
        
        Args:
            expires_in_seconds: Optional expiration time
            metadata: Optional token metadata
            
        Returns:
            Token object
        """
        value = self.security_manager.generate_token()
        expires_at = None
        
        if expires_in_seconds:
            expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        token = Token(
            value=value,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        self.tokens[value] = token
        return token
    
    def validate_token(self, token_value: str) -> Tuple[bool, Optional[Token]]:
        """
        Validate a token.
        
        Args:
            token_value: Token value to validate
            
        Returns:
            Tuple of (is_valid, token_object)
        """
        if token_value not in self.tokens:
            return False, None
        
        token = self.tokens[token_value]
        
        if token.is_expired():
            del self.tokens[token_value]
            return False, None
        
        return True, token
    
    def revoke_token(self, token_value: str):
        """
        Revoke a token.
        
        Args:
            token_value: Token value to revoke
        """
        self.tokens.pop(token_value, None)
        logger.info(f"Token revoked: {token_value[:8]}...")




