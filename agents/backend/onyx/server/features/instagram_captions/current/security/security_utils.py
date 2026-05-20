"""
Core Security Utilities for Instagram Captions API v10.0

Extracted and optimized from the main utils.py file.
"""

import hashlib
import secrets
import time
import re
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

class SecurityUtils:
    """Core security utility functions."""
    
    # Core security patterns
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>'
    ]
    
    SQL_PATTERNS = [
        r'(\b(union|select|insert|update|delete|drop)\b)',
        r'(\b(or|and)\b\s+\d+\s*[=<>])',
        r'(\b(exec|execute|sp_executesql)\b)'
    ]
    
    @staticmethod
    def generate_api_key(length: int = 32, complexity: str = "high") -> str:
        """Generate secure API key."""
        if complexity == "maximum":
            charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
        else:
            charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        return ''.join(secrets.choice(charset) for _ in range(length))
    
    @staticmethod
    def hash_password(password: str, algorithm: str = "sha256") -> str:
        """Hash password with specified algorithm."""
        salt = secrets.token_hex(16)
        
        if algorithm == "sha256":
            hash_obj = hashlib.sha256()
            hash_obj.update((password + salt).encode())
            return f"sha256:{salt}:{hash_obj.hexdigest()}"
        
        return f"sha256:{salt}:{hashlib.sha256((password + salt).encode()).hexdigest()}"
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            parts = hashed.split(":")
            if len(parts) != 3:
                return False
            
            algorithm, salt, hash_value = parts
            if algorithm == "sha256":
                return hashlib.sha256((password + salt).encode()).hexdigest() == hash_value
            
            return False
        except:
            return False
    
    @staticmethod
    def sanitize_input(text: str, strict: bool = False) -> Dict[str, Any]:
        """Sanitize input text."""
        threats = []
        sanitized = text
        
        # Check for XSS patterns
        for pattern in SecurityUtils.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(f"XSS pattern detected: {pattern}")
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Check for SQL injection patterns
        for pattern in SecurityUtils.SQL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(f"SQL injection pattern detected: {pattern}")
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        security_score = max(0, 100 - len(threats) * 20)
        
        return {
            "sanitized_text": sanitized,
            "threats_detected": threats,
            "security_score": security_score,
            "risk_level": "high" if threats else "low"
        }
    
    @staticmethod
    def verify_api_key(api_key: str) -> bool:
        """Verify API key format and strength."""
        if not api_key or len(api_key) < 32:
            return False
        
        # Check for weak patterns
        if re.search(r'(.)\1{3,}', api_key):  # Repeated characters
            return False
        
        if re.search(r'(123|abc|qwe)', api_key, re.IGNORECASE):  # Common sequences
            return False
        
        return True






