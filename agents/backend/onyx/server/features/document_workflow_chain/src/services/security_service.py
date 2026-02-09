"""
Security Service - Advanced Implementation
=========================================

Advanced security service with comprehensive security features.
"""

from __future__ import annotations
import logging
import hashlib
import secrets
import jwt
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import bcrypt

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithm enumeration"""
    AES256 = "aes256"
    RSA2048 = "rsa2048"
    RSA4096 = "rsa4096"
    CHACHA20 = "chacha20"


class SecurityService:
    """Advanced security service with comprehensive security features"""
    
    def __init__(self):
        self.secret_key = secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.password_min_length = 8
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 15
        self.session_timeout_minutes = 60
        
        # Security policies
        self.policies = {
            "password": {
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special_chars": True,
                "max_age_days": 90
            },
            "session": {
                "timeout_minutes": 60,
                "max_concurrent": 5,
                "require_https": True
            },
            "api": {
                "rate_limit_per_minute": 100,
                "max_request_size_mb": 10,
                "require_authentication": True
            }
        }
        
        # Security statistics
        self.stats = {
            "total_logins": 0,
            "failed_logins": 0,
            "password_resets": 0,
            "security_events": 0,
            "blocked_ips": 0
        }
    
    async def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        
        except Exception as e:
            logger.error(f"Failed to hash password: {e}")
            raise
    
    async def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
        except Exception as e:
            logger.error(f"Failed to verify password: {e}")
            return False
    
    async def generate_jwt_token(
        self,
        user_id: int,
        username: str,
        roles: List[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> Dict[str, str]:
        """Generate JWT access and refresh tokens"""
        try:
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
            # Access token payload
            access_payload = {
                "user_id": user_id,
                "username": username,
                "roles": roles or [],
                "type": "access",
                "exp": expire,
                "iat": datetime.utcnow(),
                "jti": secrets.token_urlsafe(16)
            }
            
            # Refresh token payload
            refresh_expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
            refresh_payload = {
                "user_id": user_id,
                "username": username,
                "type": "refresh",
                "exp": refresh_expire,
                "iat": datetime.utcnow(),
                "jti": secrets.token_urlsafe(16)
            }
            
            # Generate tokens
            access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
            refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }
        
        except Exception as e:
            logger.error(f"Failed to generate JWT tokens: {e}")
            raise
    
    async def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            raise ValueError("Invalid token")
        except Exception as e:
            logger.error(f"Failed to verify JWT token: {e}")
            raise
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            payload = await self.verify_jwt_token(refresh_token)
            
            if payload.get("type") != "refresh":
                raise ValueError("Invalid token type")
            
            # Generate new access token
            new_tokens = await self.generate_jwt_token(
                user_id=payload["user_id"],
                username=payload["username"],
                roles=payload.get("roles", [])
            )
            
            return new_tokens
        
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            raise
    
    async def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength against security policies"""
        try:
            policy = self.policies["password"]
            issues = []
            score = 0
            
            # Length check
            if len(password) < policy["min_length"]:
                issues.append(f"Password must be at least {policy['min_length']} characters long")
            else:
                score += 1
            
            # Character requirements
            if policy["require_uppercase"] and not any(c.isupper() for c in password):
                issues.append("Password must contain at least one uppercase letter")
            else:
                score += 1
            
            if policy["require_lowercase"] and not any(c.islower() for c in password):
                issues.append("Password must contain at least one lowercase letter")
            else:
                score += 1
            
            if policy["require_numbers"] and not any(c.isdigit() for c in password):
                issues.append("Password must contain at least one number")
            else:
                score += 1
            
            if policy["require_special_chars"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                issues.append("Password must contain at least one special character")
            else:
                score += 1
            
            # Calculate strength level
            if score <= 2:
                strength = "weak"
            elif score <= 3:
                strength = "medium"
            elif score <= 4:
                strength = "strong"
            else:
                strength = "very_strong"
            
            return {
                "valid": len(issues) == 0,
                "strength": strength,
                "score": score,
                "issues": issues,
                "recommendations": self._get_password_recommendations(issues)
            }
        
        except Exception as e:
            logger.error(f"Failed to validate password strength: {e}")
            return {"valid": False, "error": str(e)}
    
    async def generate_api_key(
        self,
        user_id: int,
        name: str,
        permissions: List[str] = None,
        expires_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate API key for user"""
        try:
            # Generate API key
            api_key = secrets.token_urlsafe(32)
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Calculate expiration
            expires_at = None
            if expires_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_days)
            
            # Create API key record
            api_key_record = {
                "id": secrets.token_urlsafe(16),
                "user_id": user_id,
                "name": name,
                "key_hash": api_key_hash,
                "permissions": permissions or [],
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "last_used": None,
                "is_active": True
            }
            
            return {
                "api_key": api_key,
                "api_key_record": api_key_record
            }
        
        except Exception as e:
            logger.error(f"Failed to generate API key: {e}")
            raise
    
    async def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key and return user information"""
        try:
            # Hash the provided API key
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # In a real implementation, you would check against database
            # For now, we'll simulate verification
            if len(api_key) == 32:  # Simple validation
                return {
                    "valid": True,
                    "user_id": 1,
                    "permissions": ["read", "write"],
                    "expires_at": None
                }
            else:
                return {"valid": False, "error": "Invalid API key"}
        
        except Exception as e:
            logger.error(f"Failed to verify API key: {e}")
            return {"valid": False, "error": str(e)}
    
    async def encrypt_data(self, data: str, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES256) -> Dict[str, Any]:
        """Encrypt sensitive data"""
        try:
            # Simple encryption simulation
            # In production, use proper encryption libraries
            encrypted_data = hashlib.sha256(data.encode()).hexdigest()
            
            return {
                "encrypted_data": encrypted_data,
                "algorithm": algorithm.value,
                "key_id": secrets.token_urlsafe(16),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: str, key_id: str) -> str:
        """Decrypt sensitive data"""
        try:
            # Simple decryption simulation
            # In production, use proper decryption libraries
            return f"decrypted_{encrypted_data[:8]}"
        
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    async def audit_security_event(
        self,
        event_type: str,
        user_id: Optional[int] = None,
        details: Dict[str, Any] = None,
        severity: SecurityLevel = SecurityLevel.MEDIUM
    ) -> Dict[str, Any]:
        """Audit security event"""
        try:
            event = {
                "id": secrets.token_urlsafe(16),
                "event_type": event_type,
                "user_id": user_id,
                "details": details or {},
                "severity": severity.value,
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": "127.0.0.1",  # In production, get from request
                "user_agent": "DocumentWorkflowChain/3.0"
            }
            
            # Log security event
            logger.warning(f"Security event: {event_type} - {severity.value}")
            
            # Update statistics
            self.stats["security_events"] += 1
            
            return event
        
        except Exception as e:
            logger.error(f"Failed to audit security event: {e}")
            return {"error": str(e)}
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int = 100,
        window_minutes: int = 1
    ) -> Dict[str, Any]:
        """Check rate limit for identifier"""
        try:
            # Simple rate limiting simulation
            # In production, use Redis or similar
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=window_minutes)
            
            # Simulate rate limit check
            requests_count = 50  # Simulated count
            
            if requests_count >= limit:
                return {
                    "allowed": False,
                    "limit": limit,
                    "remaining": 0,
                    "reset_time": (current_time + timedelta(minutes=window_minutes)).isoformat()
                }
            else:
                return {
                    "allowed": True,
                    "limit": limit,
                    "remaining": limit - requests_count,
                    "reset_time": (current_time + timedelta(minutes=window_minutes)).isoformat()
                }
        
        except Exception as e:
            logger.error(f"Failed to check rate limit: {e}")
            return {"allowed": False, "error": str(e)}
    
    async def validate_input(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against schema"""
        try:
            errors = []
            
            for field, rules in schema.items():
                value = data.get(field)
                
                # Required field check
                if rules.get("required", False) and value is None:
                    errors.append(f"Field '{field}' is required")
                    continue
                
                # Type check
                if value is not None:
                    expected_type = rules.get("type")
                    if expected_type and not isinstance(value, expected_type):
                        errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
                    
                    # Length check
                    if isinstance(value, str):
                        min_length = rules.get("min_length")
                        max_length = rules.get("max_length")
                        
                        if min_length and len(value) < min_length:
                            errors.append(f"Field '{field}' must be at least {min_length} characters long")
                        
                        if max_length and len(value) > max_length:
                            errors.append(f"Field '{field}' must be at most {max_length} characters long")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
        
        except Exception as e:
            logger.error(f"Failed to validate input: {e}")
            return {"valid": False, "error": str(e)}
    
    def _get_password_recommendations(self, issues: List[str]) -> List[str]:
        """Get password improvement recommendations"""
        recommendations = []
        
        if any("length" in issue.lower() for issue in issues):
            recommendations.append("Use a longer password (12+ characters recommended)")
        
        if any("uppercase" in issue.lower() for issue in issues):
            recommendations.append("Include uppercase letters (A-Z)")
        
        if any("lowercase" in issue.lower() for issue in issues):
            recommendations.append("Include lowercase letters (a-z)")
        
        if any("number" in issue.lower() for issue in issues):
            recommendations.append("Include numbers (0-9)")
        
        if any("special" in issue.lower() for issue in issues):
            recommendations.append("Include special characters (!@#$%^&*)")
        
        if not recommendations:
            recommendations.append("Consider using a passphrase for better security")
        
        return recommendations
    
    async def get_security_stats(self) -> Dict[str, Any]:
        """Get security service statistics"""
        try:
            return {
                "total_logins": self.stats["total_logins"],
                "failed_logins": self.stats["failed_logins"],
                "password_resets": self.stats["password_resets"],
                "security_events": self.stats["security_events"],
                "blocked_ips": self.stats["blocked_ips"],
                "policies": self.policies,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get security stats: {e}")
            return {"error": str(e)}


# Global security service instance
security_service = SecurityService()


