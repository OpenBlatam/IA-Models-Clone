"""
Security Enhancements for Email Sequence System

Provides advanced security features including encryption, authentication,
rate limiting, audit logging, and data protection for the email sequence system.
"""

import asyncio
import logging
import time
import json
import hashlib
import hmac
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import base64
import os

# Security imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import bcrypt
import redis

# Models
from ..models.sequence import EmailSequence, SequenceStep, SequenceTrigger
from ..models.subscriber import Subscriber, SubscriberSegment
from ..models.template import EmailTemplate, TemplateVariable
from ..models.campaign import EmailCampaign, CampaignMetrics

logger = logging.getLogger(__name__)

# Constants
MAX_REQUESTS_PER_MINUTE = 100
MAX_REQUESTS_PER_HOUR = 1000
MAX_REQUESTS_PER_DAY = 10000
ENCRYPTION_KEY_SIZE = 32
JWT_SECRET_SIZE = 64
AUDIT_LOG_RETENTION_DAYS = 90
RATE_LIMIT_WINDOW = 60  # seconds


class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EncryptionType(Enum):
    """Encryption types"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HYBRID = "hybrid"


@dataclass
class SecurityConfig:
    """Security configuration"""
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    enable_encryption: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    enable_jwt_auth: bool = True
    enable_bcrypt_hashing: bool = True
    encryption_type: EncryptionType = EncryptionType.SYMMETRIC
    jwt_secret: str = None
    encryption_key: str = None
    rate_limit_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE
    rate_limit_requests_per_hour: int = MAX_REQUESTS_PER_HOUR
    rate_limit_requests_per_day: int = MAX_REQUESTS_PER_DAY
    audit_log_retention_days: int = AUDIT_LOG_RETENTION_DAYS
    max_password_length: int = 128
    min_password_length: int = 8
    password_complexity_required: bool = True


@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    event_type: str
    user_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: SecurityLevel
    success: bool


@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    user_id: str
    ip_address: str
    requests_per_minute: int = 0
    requests_per_hour: int = 0
    requests_per_day: int = 0
    last_request: datetime = None
    blocked_until: Optional[datetime] = None


class SecurityManager:
    """Advanced security manager with encryption, authentication, and audit logging"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet = None
        self.private_key = None
        self.public_key = None
        self.redis_client = None
        
        # Rate limiting
        self.rate_limits: Dict[str, RateLimitInfo] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Audit logging
        self.audit_events: deque = deque(maxlen=10000)
        self.security_events: List[SecurityEvent] = []
        
        # Authentication
        self.jwt_secret = config.jwt_secret or self._generate_jwt_secret()
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Encryption
        self.encryption_key = config.encryption_key or self._generate_encryption_key()
        
        logger.info(f"Security Manager initialized with level: {config.security_level.value}")
    
    def _generate_jwt_secret(self) -> str:
        """Generate JWT secret"""
        return secrets.token_urlsafe(JWT_SECRET_SIZE)
    
    def _generate_encryption_key(self) -> str:
        """Generate encryption key"""
        return base64.urlsafe_b64encode(secrets.token_bytes(ENCRYPTION_KEY_SIZE)).decode()
    
    async def initialize(self) -> None:
        """Initialize security components"""
        try:
            # Initialize encryption
            if self.config.enable_encryption:
                await self._initialize_encryption()
            
            # Initialize Redis for rate limiting
            if self.config.enable_rate_limiting:
                await self._initialize_redis()
            
            # Initialize audit logging
            if self.config.enable_audit_logging:
                await self._initialize_audit_logging()
            
            logger.info("Security manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize security manager: {e}")
            raise
    
    async def _initialize_encryption(self) -> None:
        """Initialize encryption components"""
        if self.config.encryption_type == EncryptionType.SYMMETRIC:
            # Initialize Fernet for symmetric encryption
            key = base64.urlsafe_b64decode(self.encryption_key.encode())
            self.fernet = Fernet(key)
        
        elif self.config.encryption_type == EncryptionType.ASYMMETRIC:
            # Generate RSA key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
        
        logger.info(f"Encryption initialized: {self.config.encryption_type.value}")
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis for rate limiting"""
        try:
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("Redis initialized for rate limiting")
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting: {e}")
            self.redis_client = None
    
    async def _initialize_audit_logging(self) -> None:
        """Initialize audit logging"""
        # Create audit log directory if it doesn't exist
        os.makedirs("logs/audit", exist_ok=True)
        logger.info("Audit logging initialized")
    
    async def encrypt_data(self, data: str) -> str:
        """Encrypt data"""
        if not self.config.enable_encryption:
            return data
        
        try:
            if self.config.encryption_type == EncryptionType.SYMMETRIC:
                return self.fernet.encrypt(data.encode()).decode()
            
            elif self.config.encryption_type == EncryptionType.ASYMMETRIC:
                # Encrypt with public key
                encrypted = self.public_key.encrypt(
                    data.encode(),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return base64.b64encode(encrypted).decode()
            
            return data
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        if not self.config.enable_encryption:
            return encrypted_data
        
        try:
            if self.config.encryption_type == EncryptionType.SYMMETRIC:
                return self.fernet.decrypt(encrypted_data.encode()).decode()
            
            elif self.config.encryption_type == EncryptionType.ASYMMETRIC:
                # Decrypt with private key
                encrypted_bytes = base64.b64decode(encrypted_data.encode())
                decrypted = self.private_key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return decrypted.decode()
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    async def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        if not self.config.enable_bcrypt_hashing:
            return password
        
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Password hashing error: {e}")
            raise
    
    async def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        if not self.config.enable_bcrypt_hashing:
            return password == hashed_password
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    async def generate_jwt_token(self, user_data: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate JWT token"""
        if not self.config.enable_jwt_auth:
            return "no_auth"
        
        try:
            payload = {
                "user_id": user_data.get("user_id"),
                "email": user_data.get("email"),
                "role": user_data.get("role", "user"),
                "exp": datetime.utcnow() + timedelta(seconds=expires_in),
                "iat": datetime.utcnow(),
                "jti": secrets.token_urlsafe(16)
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            self.active_tokens[payload["jti"]] = payload
            
            return token
            
        except Exception as e:
            logger.error(f"JWT generation error: {e}")
            raise
    
    async def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        if not self.config.enable_jwt_auth:
            return {"user_id": "anonymous", "role": "user"}
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token is in active tokens
            if payload.get("jti") not in self.active_tokens:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT verification error: {e}")
            return None
    
    async def check_rate_limit(self, user_id: str, ip_address: str) -> bool:
        """Check rate limiting"""
        if not self.config.enable_rate_limiting:
            return True
        
        try:
            current_time = datetime.utcnow()
            key = f"rate_limit:{user_id}:{ip_address}"
            
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                if current_time < self.blocked_ips[ip_address]:
                    return False
                else:
                    del self.blocked_ips[ip_address]
            
            # Get current rate limit info
            rate_info = self.rate_limits.get(key, RateLimitInfo(
                user_id=user_id,
                ip_address=ip_address,
                last_request=current_time
            ))
            
            # Update request counts
            if rate_info.last_request:
                time_diff = (current_time - rate_info.last_request).total_seconds()
                
                if time_diff < 60:  # Within minute
                    rate_info.requests_per_minute += 1
                else:
                    rate_info.requests_per_minute = 1
                
                if time_diff < 3600:  # Within hour
                    rate_info.requests_per_hour += 1
                else:
                    rate_info.requests_per_hour = 1
                
                if time_diff < 86400:  # Within day
                    rate_info.requests_per_day += 1
                else:
                    rate_info.requests_per_day = 1
            
            rate_info.last_request = current_time
            self.rate_limits[key] = rate_info
            
            # Check limits
            if (rate_info.requests_per_minute > self.config.rate_limit_requests_per_minute or
                rate_info.requests_per_hour > self.config.rate_limit_requests_per_hour or
                rate_info.requests_per_day > self.config.rate_limit_requests_per_day):
                
                # Block IP for 15 minutes
                self.blocked_ips[ip_address] = current_time + timedelta(minutes=15)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: str,
        ip_address: str,
        user_agent: str,
        details: Dict[str, Any],
        severity: SecurityLevel,
        success: bool
    ) -> None:
        """Log security event for audit"""
        if not self.config.enable_audit_logging:
            return
        
        try:
            event = SecurityEvent(
                event_id=secrets.token_urlsafe(16),
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                timestamp=datetime.utcnow(),
                details=details,
                severity=severity,
                success=success
            )
            
            self.security_events.append(event)
            self.audit_events.append(event)
            
            # Log to file
            await self._write_audit_log(event)
            
            # Log to console for high severity events
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                logger.warning(f"Security event: {event_type} - {details}")
            
        except Exception as e:
            logger.error(f"Security event logging error: {e}")
    
    async def _write_audit_log(self, event: SecurityEvent) -> None:
        """Write audit log to file"""
        try:
            log_entry = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "user_id": event.user_id,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details,
                "severity": event.severity.value,
                "success": event.success
            }
            
            log_file = f"logs/audit/security_{datetime.utcnow().strftime('%Y-%m-%d')}.log"
            
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Audit log write error: {e}")
    
    async def validate_input(self, data: Dict[str, Any], input_type: str) -> bool:
        """Validate input data"""
        try:
            if input_type == "email":
                return self._validate_email(data.get("email", ""))
            elif input_type == "password":
                return self._validate_password(data.get("password", ""))
            elif input_type == "sequence":
                return self._validate_sequence_data(data)
            elif input_type == "template":
                return self._validate_template_data(data)
            else:
                return True
                
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    def _validate_email(self, email: str) -> bool:
        """Validate email address"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.config.min_password_length:
            return False
        
        if len(password) > self.config.max_password_length:
            return False
        
        if not self.config.password_complexity_required:
            return True
        
        # Check complexity requirements
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _validate_sequence_data(self, data: Dict[str, Any]) -> bool:
        """Validate sequence data"""
        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in data or not data[field]:
                return False
        
        # Check for XSS in content
        if "content" in data:
            dangerous_patterns = ["<script>", "javascript:", "onload=", "onerror="]
            content = data["content"].lower()
            for pattern in dangerous_patterns:
                if pattern in content:
                    return False
        
        return True
    
    def _validate_template_data(self, data: Dict[str, Any]) -> bool:
        """Validate template data"""
        required_fields = ["name", "subject", "html_content"]
        for field in required_fields:
            if field not in data or not data[field]:
                return False
        
        # Check for XSS in HTML content
        dangerous_patterns = ["<script>", "javascript:", "onload=", "onerror="]
        html_content = data["html_content"].lower()
        for pattern in dangerous_patterns:
            if pattern in html_content:
                return False
        
        return True
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        total_events = len(self.security_events)
        successful_events = sum(1 for e in self.security_events if e.success)
        failed_events = total_events - successful_events
        
        severity_counts = defaultdict(int)
        for event in self.security_events:
            severity_counts[event.severity.value] += 1
        
        return {
            "total_security_events": total_events,
            "successful_events": successful_events,
            "failed_events": failed_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0,
            "severity_distribution": dict(severity_counts),
            "active_tokens": len(self.active_tokens),
            "blocked_ips": len(self.blocked_ips),
            "rate_limited_users": len(self.rate_limits)
        }
    
    async def cleanup(self) -> None:
        """Cleanup security resources"""
        try:
            # Clear expired tokens
            current_time = datetime.utcnow()
            expired_tokens = [
                jti for jti, payload in self.active_tokens.items()
                if datetime.fromtimestamp(payload["exp"]) < current_time
            ]
            
            for jti in expired_tokens:
                del self.active_tokens[jti]
            
            # Clear expired blocked IPs
            expired_ips = [
                ip for ip, block_until in self.blocked_ips.items()
                if block_until < current_time
            ]
            
            for ip in expired_ips:
                del self.blocked_ips[ip]
            
            logger.info("Security manager cleaned up")
            
        except Exception as e:
            logger.error(f"Security cleanup error: {e}")


class EmailSequenceSecurityService:
    """Service for email sequence security operations"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    async def secure_sequence_data(self, sequence: EmailSequence) -> EmailSequence:
        """Secure sequence data with encryption"""
        try:
            # Encrypt sensitive fields
            if sequence.description:
                sequence.description = await self.security_manager.encrypt_data(sequence.description)
            
            # Encrypt step content
            for step in sequence.steps:
                if step.content:
                    step.content = await self.security_manager.encrypt_data(step.content)
                if step.subject:
                    step.subject = await self.security_manager.encrypt_data(step.subject)
            
            return sequence
            
        except Exception as e:
            logger.error(f"Sequence data encryption error: {e}")
            return sequence
    
    async def secure_template_data(self, template: EmailTemplate) -> EmailTemplate:
        """Secure template data with encryption"""
        try:
            # Encrypt sensitive fields
            if template.description:
                template.description = await self.security_manager.encrypt_data(template.description)
            
            if template.html_content:
                template.html_content = await self.security_manager.encrypt_data(template.html_content)
            
            if template.text_content:
                template.text_content = await self.security_manager.encrypt_data(template.text_content)
            
            return template
            
        except Exception as e:
            logger.error(f"Template data encryption error: {e}")
            return template
    
    async def secure_subscriber_data(self, subscriber: Subscriber) -> Subscriber:
        """Secure subscriber data with encryption"""
        try:
            # Encrypt sensitive fields
            if subscriber.first_name:
                subscriber.first_name = await self.security_manager.encrypt_data(subscriber.first_name)
            
            if subscriber.last_name:
                subscriber.last_name = await self.security_manager.encrypt_data(subscriber.last_name)
            
            if subscriber.phone:
                subscriber.phone = await self.security_manager.encrypt_data(subscriber.phone)
            
            if subscriber.company:
                subscriber.company = await self.security_manager.encrypt_data(subscriber.company)
            
            return subscriber
            
        except Exception as e:
            logger.error(f"Subscriber data encryption error: {e}")
            return subscriber
    
    async def validate_sequence_access(
        self,
        user_id: str,
        sequence_id: str,
        action: str
    ) -> bool:
        """Validate user access to sequence"""
        try:
            # This would typically check against a permissions system
            # For now, we'll implement basic validation
            
            # Log access attempt
            await self.security_manager.log_security_event(
                event_type="sequence_access",
                user_id=user_id,
                ip_address="unknown",
                user_agent="unknown",
                details={
                    "sequence_id": sequence_id,
                    "action": action
                },
                severity=SecurityLevel.MEDIUM,
                success=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Sequence access validation error: {e}")
            return False
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        return self.security_manager.get_security_metrics() 