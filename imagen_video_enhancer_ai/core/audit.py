"""
Audit System
============

System for auditing actions and events.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Audit level."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditAction(Enum):
    """Audit action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS = "access"
    CONFIG_CHANGE = "config_change"
    SECURITY_EVENT = "security_event"


@dataclass
class AuditLog:
    """Audit log entry."""
    timestamp: datetime
    user_id: Optional[str]
    action: AuditAction
    resource: str
    level: AuditLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action.value,
            "resource": self.resource,
            "level": self.level.value,
            "message": self.message,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success
        }


class AuditLogger:
    """Audit logger for recording audit events."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize audit logger.
        
        Args:
            log_file: Optional log file path
        """
        self.log_file = log_file
        self.logs: List[AuditLog] = []
        self.max_logs = 10000
    
    def log(
        self,
        action: AuditAction,
        resource: str,
        message: str,
        user_id: Optional[str] = None,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True
    ):
        """
        Log an audit event.
        
        Args:
            action: Audit action
            resource: Resource affected
            message: Log message
            user_id: Optional user ID
            level: Audit level
            details: Optional details
            ip_address: Optional IP address
            user_agent: Optional user agent
            success: Whether action was successful
        """
        log_entry = AuditLog(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            level=level,
            message=message,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
        
        self.logs.append(log_entry)
        
        # Limit logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(log_entry)
        
        # Also log to standard logger
        log_func = getattr(logger, level.value, logger.info)
        log_func(f"Audit: {action.value} on {resource} by {user_id}: {message}")
    
    def _write_to_file(self, log_entry: AuditLog):
        """Write log entry to file."""
        try:
            if self.log_file:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
    
    def get_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get audit logs with filters.
        
        Args:
            user_id: Optional user ID filter
            action: Optional action filter
            resource: Optional resource filter
            limit: Maximum number of logs
            
        Returns:
            List of audit logs
        """
        logs = self.logs
        
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        if action:
            logs = [log for log in logs if log.action == action]
        if resource:
            logs = [log for log in logs if log.resource == resource]
        
        return logs[-limit:]
    
    def get_recent_logs(self, limit: int = 100) -> List[AuditLog]:
        """
        Get recent audit logs.
        
        Args:
            limit: Maximum number of logs
            
        Returns:
            List of recent audit logs
        """
        return self.logs[-limit:]




