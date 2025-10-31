"""
Advanced Security Service for comprehensive security management
"""

import asyncio
import hashlib
import secrets
import jwt
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
from passlib.context import CryptContext
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import ipaddress
import re

from ..models.database import User, SecurityEvent, FailedLoginAttempt, SecurityPolicy
from ..core.exceptions import DatabaseError, ValidationError, SecurityError


class AdvancedSecurityService:
    """Service for advanced security operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Security policies
        self.security_policies = {
            "password_min_length": 12,
            "password_require_uppercase": True,
            "password_require_lowercase": True,
            "password_require_numbers": True,
            "password_require_special_chars": True,
            "max_login_attempts": 5,
            "lockout_duration_minutes": 30,
            "session_timeout_minutes": 60,
            "require_2fa": False,
            "ip_whitelist_enabled": False,
            "rate_limit_requests_per_minute": 100
        }
    
    async def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)
    
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    async def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength against security policies."""
        try:
            validation_results = {
                "is_valid": True,
                "score": 0,
                "issues": [],
                "suggestions": []
            }
            
            # Length check
            if len(password) < self.security_policies["password_min_length"]:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Password too short")
                validation_results["suggestions"].append(f"Use at least {self.security_policies['password_min_length']} characters")
            else:
                validation_results["score"] += 25
            
            # Uppercase check
            if self.security_policies["password_require_uppercase"] and not re.search(r'[A-Z]', password):
                validation_results["is_valid"] = False
                validation_results["issues"].append("Password must contain uppercase letters")
                validation_results["suggestions"].append("Add uppercase letters (A-Z)")
            else:
                validation_results["score"] += 25
            
            # Lowercase check
            if self.security_policies["password_require_lowercase"] and not re.search(r'[a-z]', password):
                validation_results["is_valid"] = False
                validation_results["issues"].append("Password must contain lowercase letters")
                validation_results["suggestions"].append("Add lowercase letters (a-z)")
            else:
                validation_results["score"] += 25
            
            # Numbers check
            if self.security_policies["password_require_numbers"] and not re.search(r'\d', password):
                validation_results["is_valid"] = False
                validation_results["issues"].append("Password must contain numbers")
                validation_results["suggestions"].append("Add numbers (0-9)")
            else:
                validation_results["score"] += 25
            
            # Special characters check
            if self.security_policies["password_require_special_chars"] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                validation_results["is_valid"] = False
                validation_results["issues"].append("Password must contain special characters")
                validation_results["suggestions"].append("Add special characters (!@#$%^&*)")
            else:
                validation_results["score"] += 25
            
            # Common password check
            common_passwords = ["password", "123456", "qwerty", "abc123", "password123"]
            if password.lower() in common_passwords:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Password is too common")
                validation_results["suggestions"].append("Use a unique password")
            
            # Entropy calculation
            entropy = self._calculate_entropy(password)
            validation_results["entropy"] = entropy
            
            if entropy < 50:
                validation_results["suggestions"].append("Use a more random password")
            
            return validation_results
            
        except Exception as e:
            raise SecurityError(f"Password validation failed: {str(e)}")
    
    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy."""
        import math
        
        # Count character types
        lowercase = sum(1 for c in password if c.islower())
        uppercase = sum(1 for c in password if c.isupper())
        digits = sum(1 for c in password if c.isdigit())
        special = sum(1 for c in password if not c.isalnum())
        
        # Calculate entropy
        charset_size = 0
        if lowercase > 0:
            charset_size += 26
        if uppercase > 0:
            charset_size += 26
        if digits > 0:
            charset_size += 10
        if special > 0:
            charset_size += 32  # Common special characters
        
        if charset_size == 0:
            return 0
        
        entropy = len(password) * math.log2(charset_size)
        return entropy
    
    async def generate_secure_token(self, user_id: str, token_type: str = "access") -> str:
        """Generate a secure JWT token."""
        try:
            payload = {
                "user_id": user_id,
                "token_type": token_type,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(minutes=self.security_policies["session_timeout_minutes"]),
                "jti": secrets.token_urlsafe(32)  # JWT ID for tracking
            }
            
            # Use a secure secret key (in production, this should be from environment)
            secret_key = "your-secret-key-here"  # Should be from environment
            
            token = jwt.encode(payload, secret_key, algorithm="HS256")
            return token
            
        except Exception as e:
            raise SecurityError(f"Token generation failed: {str(e)}")
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            secret_key = "your-secret-key-here"  # Should be from environment
            
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            
            return {
                "valid": True,
                "user_id": payload["user_id"],
                "token_type": payload["token_type"],
                "expires_at": payload["exp"],
                "issued_at": payload["iat"]
            }
            
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"valid": False, "error": "Invalid token"}
        except Exception as e:
            return {"valid": False, "error": f"Token verification failed: {str(e)}"}
    
    async def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            raise SecurityError(f"Data encryption failed: {str(e)}")
    
    async def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
            
        except Exception as e:
            raise SecurityError(f"Data decryption failed: {str(e)}")
    
    async def check_login_attempts(self, user_id: str, ip_address: str) -> Dict[str, Any]:
        """Check if user has exceeded login attempts."""
        try:
            # Get failed attempts in the last lockout period
            lockout_duration = timedelta(minutes=self.security_policies["lockout_duration_minutes"])
            since_time = datetime.utcnow() - lockout_duration
            
            failed_attempts_query = select(func.count(FailedLoginAttempt.id)).where(
                and_(
                    FailedLoginAttempt.user_id == user_id,
                    FailedLoginAttempt.ip_address == ip_address,
                    FailedLoginAttempt.attempted_at >= since_time
                )
            )
            
            failed_attempts_result = await self.session.execute(failed_attempts_query)
            failed_attempts = failed_attempts_result.scalar()
            
            is_locked = failed_attempts >= self.security_policies["max_login_attempts"]
            
            return {
                "is_locked": is_locked,
                "failed_attempts": failed_attempts,
                "max_attempts": self.security_policies["max_login_attempts"],
                "lockout_remaining": self._calculate_lockout_remaining(user_id, ip_address) if is_locked else 0
            }
            
        except Exception as e:
            raise DatabaseError(f"Login attempt check failed: {str(e)}")
    
    async def record_failed_login(self, user_id: str, ip_address: str, reason: str = "invalid_password"):
        """Record a failed login attempt."""
        try:
            failed_attempt = FailedLoginAttempt(
                user_id=user_id,
                ip_address=ip_address,
                reason=reason,
                attempted_at=datetime.utcnow()
            )
            
            self.session.add(failed_attempt)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to record login attempt: {str(e)}")
    
    async def clear_failed_logins(self, user_id: str, ip_address: str):
        """Clear failed login attempts for a user."""
        try:
            # Delete failed attempts for this user and IP
            delete_query = FailedLoginAttempt.__table__.delete().where(
                and_(
                    FailedLoginAttempt.user_id == user_id,
                    FailedLoginAttempt.ip_address == ip_address
                )
            )
            
            await self.session.execute(delete_query)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to clear login attempts: {str(e)}")
    
    def _calculate_lockout_remaining(self, user_id: str, ip_address: str) -> int:
        """Calculate remaining lockout time in minutes."""
        # This would typically query the database for the oldest failed attempt
        # For now, return a mock value
        return self.security_policies["lockout_duration_minutes"]
    
    async def validate_ip_address(self, ip_address: str) -> Dict[str, Any]:
        """Validate and analyze IP address."""
        try:
            # Parse IP address
            ip = ipaddress.ip_address(ip_address)
            
            # Check if IP is private
            is_private = ip.is_private
            
            # Check if IP is loopback
            is_loopback = ip.is_loopback
            
            # Check if IP is reserved
            is_reserved = ip.is_reserved
            
            # Check if IP is multicast
            is_multicast = ip.is_multicast
            
            # Get IP type
            ip_type = "IPv4" if ip.version == 4 else "IPv6"
            
            return {
                "valid": True,
                "ip_address": ip_address,
                "ip_type": ip_type,
                "is_private": is_private,
                "is_loopback": is_loopback,
                "is_reserved": is_reserved,
                "is_multicast": is_multicast,
                "risk_level": self._assess_ip_risk(ip_address, is_private, is_loopback)
            }
            
        except ValueError:
            return {
                "valid": False,
                "error": "Invalid IP address format"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"IP validation failed: {str(e)}"
            }
    
    def _assess_ip_risk(self, ip_address: str, is_private: bool, is_loopback: bool) -> str:
        """Assess IP address risk level."""
        if is_loopback:
            return "low"
        elif is_private:
            return "medium"
        else:
            # Could integrate with threat intelligence services
            return "medium"
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: str,
        details: Dict[str, Any],
        severity: str = "medium"
    ):
        """Log a security event."""
        try:
            security_event = SecurityEvent(
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address,
                details=details,
                severity=severity,
                timestamp=datetime.utcnow()
            )
            
            self.session.add(security_event)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to log security event: {str(e)}")
    
    async def get_security_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get security events."""
        try:
            # Build query
            query = select(SecurityEvent)
            
            if event_type:
                query = query.where(SecurityEvent.event_type == event_type)
            
            if user_id:
                query = query.where(SecurityEvent.user_id == user_id)
            
            if severity:
                query = query.where(SecurityEvent.severity == severity)
            
            # Get total count
            count_query = select(func.count(SecurityEvent.id))
            if event_type:
                count_query = count_query.where(SecurityEvent.event_type == event_type)
            if user_id:
                count_query = count_query.where(SecurityEvent.user_id == user_id)
            if severity:
                count_query = count_query.where(SecurityEvent.severity == severity)
            
            total_result = await self.session.execute(count_query)
            total = total_result.scalar()
            
            # Get events
            query = query.order_by(desc(SecurityEvent.timestamp)).offset(offset).limit(limit)
            events_result = await self.session.execute(query)
            events = events_result.scalars().all()
            
            # Format results
            event_list = []
            for event in events:
                event_list.append({
                    "id": event.id,
                    "event_type": event.event_type,
                    "user_id": event.user_id,
                    "ip_address": event.ip_address,
                    "details": event.details,
                    "severity": event.severity,
                    "timestamp": event.timestamp
                })
            
            return {
                "events": event_list,
                "total": total,
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get security events: {str(e)}")
    
    async def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        try:
            # Get total security events
            total_events_query = select(func.count(SecurityEvent.id))
            total_events_result = await self.session.execute(total_events_query)
            total_events = total_events_result.scalar()
            
            # Get events by type
            events_by_type_query = select(
                SecurityEvent.event_type,
                func.count(SecurityEvent.id).label('count')
            ).group_by(SecurityEvent.event_type)
            
            events_by_type_result = await self.session.execute(events_by_type_query)
            events_by_type = dict(events_by_type_result.all())
            
            # Get events by severity
            events_by_severity_query = select(
                SecurityEvent.severity,
                func.count(SecurityEvent.id).label('count')
            ).group_by(SecurityEvent.severity)
            
            events_by_severity_result = await self.session.execute(events_by_severity_query)
            events_by_severity = dict(events_by_severity_result.all())
            
            # Get failed login attempts
            failed_logins_query = select(func.count(FailedLoginAttempt.id))
            failed_logins_result = await self.session.execute(failed_logins_query)
            failed_logins = failed_logins_result.scalar()
            
            return {
                "total_security_events": total_events,
                "events_by_type": events_by_type,
                "events_by_severity": events_by_severity,
                "total_failed_logins": failed_logins,
                "security_policies": self.security_policies
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get security stats: {str(e)}")
    
    async def update_security_policy(self, policy_name: str, policy_value: Any) -> Dict[str, Any]:
        """Update a security policy."""
        try:
            if policy_name not in self.security_policies:
                raise ValidationError(f"Unknown security policy: {policy_name}")
            
            old_value = self.security_policies[policy_name]
            self.security_policies[policy_name] = policy_value
            
            # Log the policy change
            await self.log_security_event(
                event_type="policy_change",
                user_id=None,
                ip_address="127.0.0.1",
                details={
                    "policy_name": policy_name,
                    "old_value": old_value,
                    "new_value": policy_value
                },
                severity="low"
            )
            
            return {
                "policy_name": policy_name,
                "old_value": old_value,
                "new_value": policy_value,
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            raise SecurityError(f"Failed to update security policy: {str(e)}")
    
    async def generate_2fa_secret(self, user_id: str) -> str:
        """Generate a 2FA secret for a user."""
        try:
            # Generate a random secret
            secret = secrets.token_urlsafe(32)
            
            # Store the secret (encrypted)
            encrypted_secret = await self.encrypt_sensitive_data(secret)
            
            # Update user with 2FA secret
            user_query = select(User).where(User.id == user_id)
            user_result = await self.session.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            if user:
                user.two_factor_secret = encrypted_secret
                await self.session.commit()
            
            return secret
            
        except Exception as e:
            raise SecurityError(f"Failed to generate 2FA secret: {str(e)}")
    
    async def verify_2fa_token(self, user_id: str, token: str) -> bool:
        """Verify a 2FA token."""
        try:
            # Get user's 2FA secret
            user_query = select(User).where(User.id == user_id)
            user_result = await self.session.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            if not user or not user.two_factor_secret:
                return False
            
            # Decrypt the secret
            secret = await self.decrypt_sensitive_data(user.two_factor_secret)
            
            # Verify the token (simplified implementation)
            # In production, you would use a proper TOTP library
            import time
            current_time = int(time.time() // 30)
            
            # Generate expected token
            expected_token = hashlib.sha256(f"{secret}{current_time}".encode()).hexdigest()[:6]
            
            return token == expected_token
            
        except Exception as e:
            raise SecurityError(f"Failed to verify 2FA token: {str(e)}")
    
    async def perform_security_audit(self) -> Dict[str, Any]:
        """Perform a comprehensive security audit."""
        try:
            audit_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_score": 0,
                "checks": []
            }
            
            # Check password policies
            password_check = {
                "name": "Password Policies",
                "status": "pass",
                "score": 100,
                "details": self.security_policies
            }
            audit_results["checks"].append(password_check)
            audit_results["overall_score"] += 100
            
            # Check failed login attempts
            failed_logins_query = select(func.count(FailedLoginAttempt.id)).where(
                FailedLoginAttempt.attempted_at >= datetime.utcnow() - timedelta(hours=24)
            )
            failed_logins_result = await self.session.execute(failed_logins_query)
            recent_failed_logins = failed_logins_result.scalar()
            
            login_check = {
                "name": "Recent Failed Logins",
                "status": "pass" if recent_failed_logins < 100 else "warning",
                "score": max(0, 100 - recent_failed_logins),
                "details": {"recent_failed_logins": recent_failed_logins}
            }
            audit_results["checks"].append(login_check)
            audit_results["overall_score"] += login_check["score"]
            
            # Check security events
            security_events_query = select(func.count(SecurityEvent.id)).where(
                SecurityEvent.timestamp >= datetime.utcnow() - timedelta(hours=24)
            )
            security_events_result = await self.session.execute(security_events_query)
            recent_security_events = security_events_result.scalar()
            
            events_check = {
                "name": "Recent Security Events",
                "status": "pass" if recent_security_events < 50 else "warning",
                "score": max(0, 100 - recent_security_events * 2),
                "details": {"recent_security_events": recent_security_events}
            }
            audit_results["checks"].append(events_check)
            audit_results["overall_score"] += events_check["score"]
            
            # Calculate overall score
            audit_results["overall_score"] = audit_results["overall_score"] / len(audit_results["checks"])
            
            # Determine overall status
            if audit_results["overall_score"] >= 90:
                audit_results["overall_status"] = "excellent"
            elif audit_results["overall_score"] >= 70:
                audit_results["overall_status"] = "good"
            elif audit_results["overall_score"] >= 50:
                audit_results["overall_status"] = "fair"
            else:
                audit_results["overall_status"] = "poor"
            
            return audit_results
            
        except Exception as e:
            raise SecurityError(f"Security audit failed: {str(e)}")
    
    async def detect_anomalous_activity(self, user_id: str, ip_address: str) -> Dict[str, Any]:
        """Detect anomalous user activity."""
        try:
            anomalies = []
            
            # Check for unusual login times
            current_hour = datetime.utcnow().hour
            if current_hour < 6 or current_hour > 22:
                anomalies.append({
                    "type": "unusual_login_time",
                    "severity": "medium",
                    "description": f"Login at unusual hour: {current_hour}:00"
                })
            
            # Check for multiple IP addresses
            recent_logins_query = select(func.count(func.distinct(SecurityEvent.ip_address))).where(
                and_(
                    SecurityEvent.user_id == user_id,
                    SecurityEvent.event_type == "login",
                    SecurityEvent.timestamp >= datetime.utcnow() - timedelta(hours=24)
                )
            )
            recent_logins_result = await self.session.execute(recent_logins_query)
            unique_ips = recent_logins_result.scalar()
            
            if unique_ips > 3:
                anomalies.append({
                    "type": "multiple_ip_addresses",
                    "severity": "high",
                    "description": f"Login from {unique_ips} different IP addresses in 24 hours"
                })
            
            # Check for rapid successive logins
            rapid_logins_query = select(func.count(SecurityEvent.id)).where(
                and_(
                    SecurityEvent.user_id == user_id,
                    SecurityEvent.event_type == "login",
                    SecurityEvent.timestamp >= datetime.utcnow() - timedelta(minutes=5)
                )
            )
            rapid_logins_result = await self.session.execute(rapid_logins_query)
            rapid_logins = rapid_logins_result.scalar()
            
            if rapid_logins > 5:
                anomalies.append({
                    "type": "rapid_successive_logins",
                    "severity": "high",
                    "description": f"{rapid_logins} logins in 5 minutes"
                })
            
            return {
                "user_id": user_id,
                "ip_address": ip_address,
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies,
                "risk_level": "high" if any(a["severity"] == "high" for a in anomalies) else "medium" if anomalies else "low"
            }
            
        except Exception as e:
            raise SecurityError(f"Anomaly detection failed: {str(e)}")

























