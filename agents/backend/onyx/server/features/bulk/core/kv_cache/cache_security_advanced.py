"""
Advanced security system for KV cache.

This module provides comprehensive security features including encryption,
access control, audit logging, and security policies.
"""

import time
import threading
import hashlib
import hmac
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class SecurityLevel(Enum):
    """Security levels."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class AccessPermission(Enum):
    """Access permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    NONE = "none"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    security_level: SecurityLevel
    enable_encryption: bool = True
    enable_access_control: bool = True
    enable_audit_logging: bool = True
    encryption_key: Optional[bytes] = None
    require_authentication: bool = False
    max_failed_attempts: int = 5
    lockout_duration: float = 300.0  # 5 minutes


@dataclass
class AccessControlEntry:
    """Access control entry."""
    principal: str  # User, role, or identifier
    permissions: Set[AccessPermission]
    key_pattern: Optional[str] = None  # Pattern for key-based access
    expires_at: Optional[float] = None


@dataclass
class AuditLogEntry:
    """Audit log entry."""
    timestamp: float
    principal: str
    action: str
    key: str
    result: str  # "success" or "failure"
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheSecurityManager:
    """Security manager for cache."""
    
    def __init__(self, cache: Any, policy: SecurityPolicy):
        self.cache = cache
        self.policy = policy
        self._access_control: Dict[str, AccessControlEntry] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._failed_attempts: Dict[str, int] = {}
        self._locked_accounts: Dict[str, float] = {}
        self._encryption_key = policy.encryption_key or self._generate_key()
        self._lock = threading.Lock()
        
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return hashlib.sha256(str(time.time()).encode()).digest()
        
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data (simplified - use proper encryption in production)."""
        if not self.policy.enable_encryption:
            return data
            
        # Simplified XOR encryption (use proper encryption in production)
        key = self._encryption_key
        encrypted = bytearray(data)
        for i in range(len(encrypted)):
            encrypted[i] ^= key[i % len(key)]
        return bytes(encrypted)
        
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not self.policy.enable_encryption:
            return encrypted_data
            
        # XOR is symmetric
        return self.encrypt(encrypted_data)
        
    def check_permission(
        self,
        principal: str,
        key: str,
        permission: AccessPermission
    ) -> bool:
        """Check if principal has permission for key."""
        if not self.policy.enable_access_control:
            return True
            
        # Check if account is locked
        if principal in self._locked_accounts:
            lockout_end = self._locked_accounts[principal]
            if time.time() < lockout_end:
                return False
            else:
                del self._locked_accounts[principal]
                
        # Check access control
        if principal in self._access_control:
            entry = self._access_control[principal]
            
            # Check expiration
            if entry.expires_at and time.time() > entry.expires_at:
                return False
                
            # Check key pattern
            if entry.key_pattern:
                import re
                if not re.match(entry.key_pattern, key):
                    return False
                    
            # Check permission
            return permission in entry.permissions or AccessPermission.ADMIN in entry.permissions
            
        return False
        
    def grant_permission(
        self,
        principal: str,
        permissions: Set[AccessPermission],
        key_pattern: Optional[str] = None,
        expires_in: Optional[float] = None
    ) -> None:
        """Grant permissions to principal."""
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in
            
        entry = AccessControlEntry(
            principal=principal,
            permissions=permissions,
            key_pattern=key_pattern,
            expires_at=expires_at
        )
        
        with self._lock:
            self._access_control[principal] = entry
            
    def revoke_permission(self, principal: str) -> None:
        """Revoke all permissions for principal."""
        with self._lock:
            self._access_control.pop(principal, None)
            
    def record_audit(
        self,
        principal: str,
        action: str,
        key: str,
        result: str,
        reason: Optional[str] = None
    ) -> None:
        """Record audit log entry."""
        if not self.policy.enable_audit_logging:
            return
            
        entry = AuditLogEntry(
            timestamp=time.time(),
            principal=principal,
            action=action,
            key=key,
            result=result,
            reason=reason
        )
        
        with self._lock:
            self._audit_log.append(entry)
            # Keep only last 10000 entries
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-10000:]
                
    def handle_failed_attempt(self, principal: str) -> bool:
        """Handle failed authentication attempt."""
        with self._lock:
            self._failed_attempts[principal] = self._failed_attempts.get(principal, 0) + 1
            
            if self._failed_attempts[principal] >= self.policy.max_failed_attempts:
                self._locked_accounts[principal] = time.time() + self.policy.lockout_duration
                return True  # Account locked
            return False
            
    def get_audit_log(
        self,
        principal: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[AuditLogEntry]:
        """Get audit log entries."""
        with self._lock:
            entries = list(self._audit_log)
            
        if principal:
            entries = [e for e in entries if e.principal == principal]
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
            
        return entries
        
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        with self._lock:
            return {
                'total_access_control_entries': len(self._access_control),
                'total_audit_entries': len(self._audit_log),
                'locked_accounts': len(self._locked_accounts),
                'failed_attempts': sum(self._failed_attempts.values())
            }


class SecureCache:
    """Cache wrapper with security features."""
    
    def __init__(self, cache: Any, policy: SecurityPolicy):
        self.cache = cache
        self.security_manager = CacheSecurityManager(cache, policy)
        
    def get(self, key: str, principal: str = "anonymous") -> Any:
        """Get value with security checks."""
        # Check permission
        if not self.security_manager.check_permission(principal, key, AccessPermission.READ):
            self.security_manager.record_audit(principal, "get", key, "failure", "Permission denied")
            raise PermissionError(f"Read permission denied for key: {key}")
            
        # Get value
        value = self.cache.get(key)
        
        if value is not None:
            # Decrypt if needed
            if isinstance(value, bytes):
                value = self.security_manager.decrypt(value)
            self.security_manager.record_audit(principal, "get", key, "success")
        else:
            self.security_manager.record_audit(principal, "get", key, "failure", "Key not found")
            
        return value
        
    def put(self, key: str, value: Any, principal: str = "anonymous") -> bool:
        """Put value with security checks."""
        # Check permission
        if not self.security_manager.check_permission(principal, key, AccessPermission.WRITE):
            self.security_manager.record_audit(principal, "put", key, "failure", "Permission denied")
            raise PermissionError(f"Write permission denied for key: {key}")
            
        # Encrypt if needed
        if isinstance(value, bytes):
            encrypted_value = self.security_manager.encrypt(value)
        elif isinstance(value, str):
            encrypted_value = self.security_manager.encrypt(value.encode('utf-8'))
        else:
            encrypted_value = self.security_manager.encrypt(str(value).encode('utf-8'))
            
        result = self.cache.put(key, encrypted_value)
        self.security_manager.record_audit(principal, "put", key, "success" if result else "failure")
        return result
        
    def delete(self, key: str, principal: str = "anonymous") -> bool:
        """Delete value with security checks."""
        # Check permission
        if not self.security_manager.check_permission(principal, key, AccessPermission.DELETE):
            self.security_manager.record_audit(principal, "delete", key, "failure", "Permission denied")
            raise PermissionError(f"Delete permission denied for key: {key}")
            
        result = self.cache.delete(key)
        self.security_manager.record_audit(principal, "delete", key, "success" if result else "failure")
        return result














