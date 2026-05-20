"""
Advanced Audit System
=====================

Advanced audit system with detailed tracking and compliance features.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Audit actions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_CHANGE = "permission_change"
    CONFIG_CHANGE = "config_change"


class AuditLevel(Enum):
    """Audit levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """Audit entry."""
    timestamp: datetime
    user_id: Optional[str]
    action: AuditAction
    resource: str
    level: AuditLevel = AuditLevel.INFO
    success: bool = True
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action.value,
            "resource": self.resource,
            "level": self.level.value,
            "success": self.success,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
            "metadata": self.metadata
        }


class AuditLogger:
    """Advanced audit logger with compliance features."""
    
    def __init__(self, max_entries: int = 100000):
        """
        Initialize advanced audit logger.
        
        Args:
            max_entries: Maximum number of entries to keep
        """
        self.max_entries = max_entries
        self.entries: List[AuditEntry] = []
        self.handlers: List[Callable] = []
        self.enabled = True
    
    def log(
        self,
        action: AuditAction,
        resource: str,
        user_id: Optional[str] = None,
        level: AuditLevel = AuditLevel.INFO,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log audit entry.
        
        Args:
            action: Audit action
            resource: Resource name
            user_id: Optional user ID
            level: Audit level
            success: Whether action was successful
            ip_address: Optional IP address
            user_agent: Optional user agent
            details: Optional details
            metadata: Optional metadata
        """
        if not self.enabled:
            return
        
        entry = AuditEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            level=level,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        
        # Trim if exceeds max entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Call handlers
        for handler in self.handlers:
            try:
                handler(entry)
            except Exception as e:
                logger.error(f"Audit handler failed: {e}")
    
    def add_handler(self, handler: Callable):
        """
        Add audit handler.
        
        Args:
            handler: Handler function
        """
        self.handlers.append(handler)
    
    def get_entries(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[AuditEntry]:
        """
        Get audit entries with filters.
        
        Args:
            user_id: Optional user ID filter
            action: Optional action filter
            resource: Optional resource filter
            since: Optional timestamp filter
            limit: Optional limit
            
        Returns:
            List of audit entries
        """
        results = self.entries.copy()
        
        # Apply filters
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        
        if action:
            results = [e for e in results if e.action == action]
        
        if resource:
            results = [e for e in results if e.resource == resource]
        
        if since:
            results = [e for e in results if e.timestamp >= since]
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Limit results
        if limit:
            results = results[:limit]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit statistics.
        
        Returns:
            Statistics dictionary
        """
        total = len(self.entries)
        by_action = {}
        by_level = {}
        by_user = {}
        success_count = 0
        
        for entry in self.entries:
            # By action
            action_name = entry.action.value
            by_action[action_name] = by_action.get(action_name, 0) + 1
            
            # By level
            level_name = entry.level.value
            by_level[level_name] = by_level.get(level_name, 0) + 1
            
            # By user
            if entry.user_id:
                by_user[entry.user_id] = by_user.get(entry.user_id, 0) + 1
            
            # Success count
            if entry.success:
                success_count += 1
        
        return {
            "total": total,
            "by_action": by_action,
            "by_level": by_level,
            "by_user": by_user,
            "success_count": success_count,
            "failure_count": total - success_count,
            "success_rate": (success_count / total * 100) if total > 0 else 0
        }

