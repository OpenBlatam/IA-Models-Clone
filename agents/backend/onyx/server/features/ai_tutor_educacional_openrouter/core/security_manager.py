"""
Security management system.
"""

import hashlib
import secrets
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Security management for authentication and data protection.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize security manager.
        
        Args:
            secret_key: Secret key for JWT tokens (auto-generated if not provided)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.token_expiry = timedelta(hours=24)
        self.password_min_length = 8
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using SHA-256.
        
        Args:
            password: Plain text password
        
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password
        
        Returns:
            True if password matches
        """
        return self.hash_password(password) == hashed
    
    def generate_token(self, user_id: str, payload: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a JWT token.
        
        Args:
            user_id: User identifier
            payload: Additional payload data
        
        Returns:
            JWT token string
        """
        token_payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow()
        }
        
        if payload:
            token_payload.update(payload)
        
        return jwt.encode(token_payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
        
        Returns:
            Validation result with suggestions
        """
        issues = []
        strength = "weak"
        
        if len(password) < self.password_min_length:
            issues.append(f"Password must be at least {self.password_min_length} characters")
        
        if not any(c.isupper() for c in password):
            issues.append("Password should contain uppercase letters")
        
        if not any(c.islower() for c in password):
            issues.append("Password should contain lowercase letters")
        
        if not any(c.isdigit() for c in password):
            issues.append("Password should contain numbers")
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password should contain special characters")
        
        if len(issues) == 0:
            strength = "strong"
        elif len(issues) <= 2:
            strength = "medium"
        
        return {
            "valid": len(issues) == 0,
            "strength": strength,
            "issues": issues
        }
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            text: Input text
        
        Returns:
            Sanitized text
        """
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$']
        sanitized = text
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()




