"""Audit logging system"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Audit action types"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    CONVERT = "convert"
    EXPORT = "export"
    SIGN = "sign"
    REVIEW = "review"
    LOGIN = "login"
    LOGOUT = "logout"


@dataclass
class AuditLog:
    """Audit log entry"""
    timestamp: datetime
    user_id: Optional[str]
    action: AuditAction
    resource: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class AuditLogger:
    """Audit logging system"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize audit logger
        
        Args:
            log_file: Optional log file path
        """
        self.log_file = log_file
        self.logs: List[AuditLog] = []
        self.max_memory_logs = 1000
    
    def log(
        self,
        action: AuditAction,
        resource: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Log audit event
        
        Args:
            action: Action type
            resource: Resource name
            user_id: Optional user ID
            resource_id: Optional resource ID
            details: Optional details
            ip_address: Optional IP address
            user_agent: Optional user agent
            success: Success status
            error_message: Optional error message
        """
        audit_log = AuditLog(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
        
        # Store in memory
        self.logs.append(audit_log)
        if len(self.logs) > self.max_memory_logs:
            self.logs.pop(0)
        
        # Write to file if configured
        if self.log_file:
            self._write_to_file(audit_log)
        
        # Log to standard logger
        logger.info(
            f"Audit: {action.value} {resource} by {user_id or 'anonymous'}"
        )
    
    def _write_to_file(self, audit_log: AuditLog):
        """Write audit log to file"""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            log_entry = {
                "timestamp": audit_log.timestamp.isoformat(),
                "user_id": audit_log.user_id,
                "action": audit_log.action.value,
                "resource": audit_log.resource,
                "resource_id": audit_log.resource_id,
                "details": audit_log.details,
                "ip_address": audit_log.ip_address,
                "user_agent": audit_log.user_agent,
                "success": audit_log.success,
                "error_message": audit_log.error_message
            }
            
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
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
        Get audit logs
        
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
        
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]


# Global audit logger
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        from config import settings
        log_file = getattr(settings, 'audit_log_file', None)
        _audit_logger = AuditLogger(log_file)
    return _audit_logger

