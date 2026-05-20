"""
Security Manager for Document Analyzer
=======================================

Advanced security features including encryption, sanitization, and access control.
"""

import logging
import hashlib
import hmac
import secrets
import base64
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class SecurityManager:
    """Advanced security manager"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        logger.info("SecurityManager initialized")
    
    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """Hash data"""
        if algorithm == "sha256":
            return hashlib.sha256(data.encode()).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data.encode()).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def generate_token(self, length: int = 32) -> str:
        """Generate secure token"""
        return secrets.token_urlsafe(length)
    
    def generate_hmac(self, data: str) -> str:
        """Generate HMAC signature"""
        return hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_hmac(self, data: str, signature: str) -> bool:
        """Verify HMAC signature"""
        expected = self.generate_hmac(data)
        return hmac.compare_digest(expected, signature)
    
    def sanitize_input(self, text: str, max_length: int = 10000) -> str:
        """Sanitize input text"""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove potentially dangerous characters
        text = re.sub(r'[^\w\s\-\.\,\!\?\;\:]', '', text)
        
        return text.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename"""
        # Remove path components
        filename = Path(filename).name
        
        # Remove dangerous characters
        filename = re.sub(r'[^\w\s\-\.]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255 - len(ext) - 1] + '.' + ext if ext else name[:255]
        
        return filename
    
    def mask_sensitive_data(self, text: str, pattern: str = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b') -> str:
        """Mask sensitive data like credit card numbers"""
        def mask_match(match):
            full = match.group(0)
            return full[:4] + '*' * (len(full) - 8) + full[-4:]
        
        return re.sub(pattern, mask_match, text)
    
    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """Check password strength"""
        score = 0
        feedback = []
        
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("Password should be at least 8 characters")
        
        if re.search(r'[a-z]', password):
            score += 1
        else:
            feedback.append("Password should contain lowercase letters")
        
        if re.search(r'[A-Z]', password):
            score += 1
        else:
            feedback.append("Password should contain uppercase letters")
        
        if re.search(r'\d', password):
            score += 1
        else:
            feedback.append("Password should contain numbers")
        
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 1
        else:
            feedback.append("Password should contain special characters")
        
        strength = {
            0: "very_weak",
            1: "weak",
            2: "weak",
            3: "medium",
            4: "strong",
            5: "very_strong"
        }.get(score, "weak")
        
        return {
            "strength": strength,
            "score": score,
            "max_score": 5,
            "feedback": feedback
        }
    
    def encrypt_simple(self, data: str) -> str:
        """Simple base64 encoding (not for production security)"""
        return base64.b64encode(data.encode()).decode()
    
    def decrypt_simple(self, encoded: str) -> str:
        """Simple base64 decoding"""
        return base64.b64decode(encoded.encode()).decode()
    
    def generate_api_key(self, prefix: str = "doc_analyzer") -> str:
        """Generate API key"""
        token = self.generate_token(32)
        return f"{prefix}_{token}"

# Global instance
security_manager = SecurityManager()

