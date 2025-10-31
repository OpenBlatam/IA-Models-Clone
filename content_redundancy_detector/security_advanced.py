"""
Advanced security system for enterprise-grade protection
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import secrets
import base64
import json
from datetime import datetime, timedelta
import ipaddress
import re

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Threat types"""
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALICIOUS_CONTENT = "malicious_content"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    DDoS = "ddos"
    INJECTION = "injection"


class AuthMethod(Enum):
    """Authentication methods"""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"


@dataclass
class SecurityEvent:
    """Security event"""
    id: str
    event_type: ThreatType
    severity: SecurityLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class SecurityPolicy:
    """Security policy"""
    id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class UserSession:
    """User session"""
    id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIKey:
    """API Key"""
    id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[str]
    rate_limit: int
    expires_at: Optional[float] = None
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None


class AdvancedSecurityManager:
    """Advanced security manager"""
    
    def __init__(self):
        self._security_events: Dict[str, SecurityEvent] = {}
        self._security_policies: Dict[str, SecurityPolicy] = {}
        self._user_sessions: Dict[str, UserSession] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._blocked_ips: Set[str] = set()
        self._suspicious_ips: Dict[str, List[float]] = {}
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._encryption_key = secrets.token_bytes(32)
    
    def create_api_key(self, user_id: str, name: str, permissions: List[str], 
                      rate_limit: int = 1000, expires_days: Optional[int] = None) -> str:
        """Create a new API key"""
        # Generate API key
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = time.time() + (expires_days * 24 * 3600)
        
        # Create API key record
        api_key_record = APIKey(
            id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions,
            rate_limit=rate_limit,
            expires_at=expires_at
        )
        
        self._api_keys[api_key_record.id] = api_key_record
        
        # Log creation
        self._log_audit_event("api_key_created", {
            "user_id": user_id,
            "key_id": api_key_record.id,
            "key_name": name,
            "permissions": permissions
        })
        
        logger.info(f"API key created for user {user_id}: {name}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for key_record in self._api_keys.values():
            if key_record.key_hash == key_hash and key_record.is_active:
                # Check expiration
                if key_record.expires_at and time.time() > key_record.expires_at:
                    self._log_audit_event("api_key_expired", {
                        "key_id": key_record.id,
                        "user_id": key_record.user_id
                    })
                    return None
                
                # Update last used
                key_record.last_used = time.time()
                
                # Log usage
                self._log_audit_event("api_key_used", {
                    "key_id": key_record.id,
                    "user_id": key_record.user_id
                })
                
                return key_record
        
        return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke API key"""
        if key_id not in self._api_keys:
            return False
        
        key_record = self._api_keys[key_id]
        key_record.is_active = False
        
        # Log revocation
        self._log_audit_event("api_key_revoked", {
            "key_id": key_id,
            "user_id": key_record.user_id
        })
        
        logger.info(f"API key revoked: {key_id}")
        return True
    
    def create_user_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self._user_sessions[session_id] = session
        
        # Log session creation
        self._log_audit_event("session_created", {
            "session_id": session_id,
            "user_id": user_id,
            "ip_address": ip_address
        })
        
        logger.info(f"User session created: {user_id} from {ip_address}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate user session"""
        if session_id not in self._user_sessions:
            return None
        
        session = self._user_sessions[session_id]
        
        # Check if session is active
        if not session.is_active:
            return None
        
        # Check session timeout (24 hours)
        if time.time() - session.last_activity > 24 * 3600:
            session.is_active = False
            self._log_audit_event("session_expired", {
                "session_id": session_id,
                "user_id": session.user_id
            })
            return None
        
        # Update last activity
        session.last_activity = time.time()
        
        return session
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate user session"""
        if session_id not in self._user_sessions:
            return False
        
        session = self._user_sessions[session_id]
        session.is_active = False
        
        # Log session termination
        self._log_audit_event("session_terminated", {
            "session_id": session_id,
            "user_id": session.user_id
        })
        
        logger.info(f"User session terminated: {session_id}")
        return True
    
    def check_rate_limit(self, identifier: str, limit: int, window: int = 3600) -> bool:
        """Check rate limit"""
        current_time = time.time()
        
        if identifier not in self._rate_limits:
            self._rate_limits[identifier] = {
                "requests": [],
                "limit": limit,
                "window": window
            }
        
        rate_limit_data = self._rate_limits[identifier]
        
        # Remove old requests outside the window
        cutoff_time = current_time - window
        rate_limit_data["requests"] = [
            req_time for req_time in rate_limit_data["requests"] 
            if req_time > cutoff_time
        ]
        
        # Check if limit exceeded
        if len(rate_limit_data["requests"]) >= limit:
            # Log rate limit violation
            self._log_security_event(ThreatType.RATE_LIMIT_EXCEEDED, SecurityLevel.MEDIUM, 
                                   identifier, None, f"Rate limit exceeded: {limit} requests per {window}s")
            return False
        
        # Add current request
        rate_limit_data["requests"].append(current_time)
        return True
    
    def check_ip_reputation(self, ip_address: str) -> SecurityLevel:
        """Check IP reputation"""
        # Check if IP is blocked
        if ip_address in self._blocked_ips:
            return SecurityLevel.CRITICAL
        
        # Check if IP is suspicious
        if ip_address in self._suspicious_ips:
            recent_events = [
                event_time for event_time in self._suspicious_ips[ip_address]
                if time.time() - event_time < 3600  # Last hour
            ]
            
            if len(recent_events) >= 5:
                return SecurityLevel.HIGH
            elif len(recent_events) >= 3:
                return SecurityLevel.MEDIUM
        
        # Check if IP is in private range (internal)
        try:
            ip = ipaddress.ip_address(ip_address)
            if ip.is_private:
                return SecurityLevel.LOW
        except ValueError:
            return SecurityLevel.HIGH
        
        return SecurityLevel.LOW
    
    def block_ip(self, ip_address: str, reason: str) -> None:
        """Block IP address"""
        self._blocked_ips.add(ip_address)
        
        # Log blocking
        self._log_security_event(ThreatType.UNAUTHORIZED_ACCESS, SecurityLevel.HIGH,
                               ip_address, None, f"IP blocked: {reason}")
        
        logger.warning(f"IP address blocked: {ip_address} - {reason}")
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock IP address"""
        if ip_address in self._blocked_ips:
            self._blocked_ips.remove(ip_address)
            
            # Log unblocking
            self._log_audit_event("ip_unblocked", {
                "ip_address": ip_address
            })
            
            logger.info(f"IP address unblocked: {ip_address}")
            return True
        
        return False
    
    def mark_suspicious_ip(self, ip_address: str) -> None:
        """Mark IP as suspicious"""
        if ip_address not in self._suspicious_ips:
            self._suspicious_ips[ip_address] = []
        
        self._suspicious_ips[ip_address].append(time.time())
        
        # Keep only recent events (last 24 hours)
        cutoff_time = time.time() - 24 * 3600
        self._suspicious_ips[ip_address] = [
            event_time for event_time in self._suspicious_ips[ip_address]
            if event_time > cutoff_time
        ]
    
    def validate_input(self, input_data: str, input_type: str = "general") -> bool:
        """Validate input for security threats"""
        # Check for SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(--|\#|\/\*|\*\/)",
            r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT)\b)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                self._log_security_event(ThreatType.INJECTION, SecurityLevel.HIGH,
                                       "unknown", None, f"SQL injection attempt detected in {input_type}")
                return False
        
        # Check for XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                self._log_security_event(ThreatType.INJECTION, SecurityLevel.HIGH,
                                       "unknown", None, f"XSS attempt detected in {input_type}")
                return False
        
        # Check for path traversal
        if ".." in input_data or "/" in input_data or "\\" in input_data:
            if input_type in ["file_path", "filename"]:
                self._log_security_event(ThreatType.UNAUTHORIZED_ACCESS, SecurityLevel.MEDIUM,
                                       "unknown", None, f"Path traversal attempt detected in {input_type}")
                return False
        
        return True
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        # Simple encryption using base64 and HMAC
        # In production, use proper encryption like AES
        encoded_data = base64.b64encode(data.encode()).decode()
        signature = hmac.new(self._encryption_key, encoded_data.encode(), hashlib.sha256).hexdigest()
        return f"{encoded_data}:{signature}"
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt sensitive data"""
        try:
            parts = encrypted_data.split(":")
            if len(parts) != 2:
                return None
            
            encoded_data, signature = parts
            
            # Verify signature
            expected_signature = hmac.new(self._encryption_key, encoded_data.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Decode data
            return base64.b64decode(encoded_data).decode()
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return None
    
    def create_security_policy(self, policy_id: str, name: str, description: str, 
                              rules: List[Dict[str, Any]]) -> SecurityPolicy:
        """Create security policy"""
        policy = SecurityPolicy(
            id=policy_id,
            name=name,
            description=description,
            rules=rules
        )
        
        self._security_policies[policy_id] = policy
        
        # Log policy creation
        self._log_audit_event("security_policy_created", {
            "policy_id": policy_id,
            "policy_name": name,
            "rules_count": len(rules)
        })
        
        logger.info(f"Security policy created: {name}")
        return policy
    
    def apply_security_policies(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Apply security policies to request"""
        triggered_events = []
        
        for policy in self._security_policies.values():
            if not policy.is_active:
                continue
            
            for rule in policy.rules:
                if self._evaluate_rule(rule, request_data):
                    event = SecurityEvent(
                        id=secrets.token_urlsafe(16),
                        event_type=ThreatType(rule.get("threat_type", "suspicious_activity")),
                        severity=SecurityLevel(rule.get("severity", "medium")),
                        source_ip=request_data.get("ip_address", "unknown"),
                        user_id=request_data.get("user_id"),
                        description=f"Policy violation: {policy.name} - {rule.get('description', 'Rule triggered')}",
                        metadata={
                            "policy_id": policy.id,
                            "rule": rule
                        }
                    )
                    
                    self._security_events[event.id] = event
                    triggered_events.append(event)
        
        return triggered_events
    
    def _evaluate_rule(self, rule: Dict[str, Any], request_data: Dict[str, Any]) -> bool:
        """Evaluate security rule"""
        rule_type = rule.get("type")
        
        if rule_type == "rate_limit":
            identifier = request_data.get("ip_address", "unknown")
            limit = rule.get("limit", 100)
            window = rule.get("window", 3600)
            return not self.check_rate_limit(identifier, limit, window)
        
        elif rule_type == "ip_reputation":
            ip_address = request_data.get("ip_address", "unknown")
            min_level = SecurityLevel(rule.get("min_level", "medium"))
            return self.check_ip_reputation(ip_address).value >= min_level.value
        
        elif rule_type == "input_validation":
            input_data = request_data.get("input_data", "")
            return not self.validate_input(input_data, rule.get("input_type", "general"))
        
        elif rule_type == "user_agent":
            user_agent = request_data.get("user_agent", "")
            blocked_patterns = rule.get("blocked_patterns", [])
            for pattern in blocked_patterns:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    return True
        
        elif rule_type == "request_size":
            request_size = request_data.get("request_size", 0)
            max_size = rule.get("max_size", 1024 * 1024)  # 1MB default
            return request_size > max_size
        
        return False
    
    def _log_security_event(self, event_type: ThreatType, severity: SecurityLevel,
                           source_ip: str, user_id: Optional[str], description: str,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log security event"""
        event = SecurityEvent(
            id=secrets.token_urlsafe(16),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            metadata=metadata or {}
        )
        
        self._security_events[event.id] = event
        
        # Mark suspicious IP if high severity
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self.mark_suspicious_ip(source_ip)
        
        logger.warning(f"Security event: {event_type.value} - {description}")
    
    def _log_audit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log audit event"""
        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        
        self._audit_log.append(audit_entry)
        
        # Keep only last 10000 audit entries
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]
    
    def get_security_events(self, limit: int = 100) -> List[SecurityEvent]:
        """Get recent security events"""
        events = list(self._security_events.values())
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self._audit_log[-limit:] if limit > 0 else self._audit_log
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        current_time = time.time()
        last_24h = current_time - 24 * 3600
        
        recent_events = [
            event for event in self._security_events.values()
            if event.timestamp > last_24h
        ]
        
        events_by_type = {}
        for event in recent_events:
            event_type = event.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
        
        return {
            "total_events": len(self._security_events),
            "recent_events_24h": len(recent_events),
            "events_by_type": events_by_type,
            "blocked_ips": len(self._blocked_ips),
            "suspicious_ips": len(self._suspicious_ips),
            "active_sessions": len([s for s in self._user_sessions.values() if s.is_active]),
            "active_api_keys": len([k for k in self._api_keys.values() if k.is_active]),
            "security_policies": len(self._security_policies),
            "audit_entries": len(self._audit_log)
        }


# Global security manager
security_manager = AdvancedSecurityManager()


# Helper functions
def create_api_key(user_id: str, name: str, permissions: List[str], 
                  rate_limit: int = 1000, expires_days: Optional[int] = None) -> str:
    """Create API key"""
    return security_manager.create_api_key(user_id, name, permissions, rate_limit, expires_days)


def validate_api_key(api_key: str) -> Optional[APIKey]:
    """Validate API key"""
    return security_manager.validate_api_key(api_key)


def check_rate_limit(identifier: str, limit: int, window: int = 3600) -> bool:
    """Check rate limit"""
    return security_manager.check_rate_limit(identifier, limit, window)


def validate_input(input_data: str, input_type: str = "general") -> bool:
    """Validate input"""
    return security_manager.validate_input(input_data, input_type)


def encrypt_data(data: str) -> str:
    """Encrypt data"""
    return security_manager.encrypt_sensitive_data(data)


def decrypt_data(encrypted_data: str) -> Optional[str]:
    """Decrypt data"""
    return security_manager.decrypt_sensitive_data(encrypted_data)


