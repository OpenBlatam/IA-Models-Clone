"""
PDF Variantes Security
=====================

Security features including encryption, access control, and auditing.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


class PermissionType(str, Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


@dataclass
class AccessToken:
    """Access token for API authentication."""
    token: str
    user_id: str
    permissions: List[str]
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token": self.token,
            "user_id": self.user_id,
            "permissions": self.permissions,
            "expires_at": self.expires_at.isoformat(),
            "created_at": self.created_at.isoformat()
        }


@dataclass
class AuditLog:
    """Audit log entry."""
    log_id: str
    action: str
    user_id: str
    resource_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_id": self.log_id,
            "action": self.action,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


class SecurityManager:
    """Security and access control manager."""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("./uploads/pdf_variantes")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.security_dir = self.upload_dir / "security"
        self.security_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokens: Dict[str, AccessToken] = {}
        self.audit_logs: List[AuditLog] = []
        
        logger.info("Initialized Security Manager")
    
    def generate_token(
        self,
        user_id: str,
        permissions: List[str],
        expires_in_hours: int = 24
    ) -> AccessToken:
        """Generate access token."""
        import secrets
        
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        access_token = AccessToken(
            token=token,
            user_id=user_id,
            permissions=permissions,
            expires_at=expires_at
        )
        
        self.tokens[token] = access_token
        logger.info(f"Generated token for user {user_id}")
        
        return access_token
    
    def validate_token(self, token: str) -> Optional[AccessToken]:
        """Validate access token."""
        access_token = self.tokens.get(token)
        
        if not access_token:
            return None
        
        if access_token.is_expired():
            del self.tokens[token]
            logger.warning(f"Token expired: {token}")
            return None
        
        return access_token
    
    def check_permission(
        self,
        token: str,
        permission: PermissionType
    ) -> bool:
        """Check if token has permission."""
        access_token = self.validate_token(token)
        
        if not access_token:
            return False
        
        # Admin has all permissions
        if PermissionType.ADMIN.value in access_token.permissions:
            return True
        
        return permission.value in access_token.permissions
    
    def log_action(
        self,
        action: str,
        user_id: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security action."""
        import uuid
        
        log = AuditLog(
            log_id=str(uuid.uuid4()),
            action=action,
            user_id=user_id,
            resource_id=resource_id,
            details=details or {}
        )
        
        self.audit_logs.append(log)
        
        # Keep only last 10000 logs
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-10000:]
        
        logger.info(f"Audit log: {action} by {user_id} on {resource_id}")
    
    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditLog]:
        """Get audit logs with filters."""
        logs = self.audit_logs
        
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        
        if resource_id:
            logs = [log for log in logs if log.resource_id == resource_id]
        
        if action:
            logs = [log for log in logs if log.action == action]
        
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        return logs
    
    async def encrypt_pdf(
        self,
        file_id: str,
        password: str,
        algorithm: str = "AES-256"
    ) -> bool:
        """Encrypt PDF file."""
        logger.info(f"Encrypting {file_id} with {algorithm}")
        
        # Placeholder for encryption logic
        self.log_action("encrypt_pdf", "system", file_id, {
            "algorithm": algorithm,
            "encrypted": True
        })
        
        return True
    
    async def decrypt_pdf(
        self,
        file_id: str,
        password: str
    ) -> bool:
        """Decrypt PDF file."""
        logger.info(f"Decrypting {file_id}")
        
        self.log_action("decrypt_pdf", "system", file_id, {
            "decrypted": True
        })
        
        return True







