"""
Audit Logger for Color Grading AI
===================================

Comprehensive audit logging for compliance and security.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Audit event types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"


class AuditLevel(Enum):
    """Audit levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event."""
    event_id: str
    event_type: AuditEventType
    level: AuditLevel
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    action: str = ""
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class AuditLogger:
    """
    Comprehensive audit logger.
    
    Features:
    - Multiple event types
    - Compliance logging
    - Security events
    - Search and filtering
    - Retention policies
    """
    
    def __init__(self, audit_dir: str = "audit_logs"):
        """
        Initialize audit logger.
        
        Args:
            audit_dir: Directory for audit logs
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._events: List[AuditEvent] = []
        self._max_memory_events = 1000
        self._retention_days = 90
    
    def log_event(
        self,
        event_type: AuditEventType,
        level: AuditLevel,
        action: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log audit event.
        
        Args:
            event_type: Event type
            level: Audit level
            action: Action description
            user_id: User ID
            session_id: Session ID
            ip_address: IP address
            resource: Resource accessed
            details: Additional details
            success: Whether action succeeded
            error_message: Error message if failed
            
        Returns:
            Event ID
        """
        import uuid
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            level=level,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            action=action,
            resource=resource,
            details=details or {},
            success=success,
            error_message=error_message
        )
        
        # Store in memory
        self._events.append(event)
        if len(self._events) > self._max_memory_events:
            self._events = self._events[-self._max_memory_events:]
        
        # Write to file
        self._write_event(event)
        
        logger.info(f"Audit event logged: {event_type.value} - {action}")
        
        return event_id
    
    def _write_event(self, event: AuditEvent):
        """Write event to file."""
        try:
            # Daily log files
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.audit_dir / f"audit_{date_str}.jsonl"
            
            event_dict = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "level": event.level.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "session_id": event.session_id,
                "ip_address": event.ip_address,
                "action": event.action,
                "resource": event.resource,
                "details": event.details,
                "success": event.success,
                "error_message": event.error_message,
            }
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_dict, ensure_ascii=False) + '\n')
        
        except Exception as e:
            logger.error(f"Error writing audit event: {e}")
    
    def search_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        level: Optional[AuditLevel] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Search audit events.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_date: Start date
            end_date: End date
            level: Filter by level
            limit: Maximum results
            
        Returns:
            List of matching events
        """
        results = []
        
        # Search in memory
        for event in self._events:
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue
            if level and event.level != level:
                continue
            
            results.append(event)
        
        # Search in files if needed
        if len(results) < limit:
            file_results = self._search_files(
                event_type, user_id, start_date, end_date, level, limit - len(results)
            )
            results.extend(file_results)
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda e: e.timestamp, reverse=True)
        
        return results[:limit]
    
    def _search_files(
        self,
        event_type: Optional[AuditEventType],
        user_id: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        level: Optional[AuditLevel],
        limit: int
    ) -> List[AuditEvent]:
        """Search events in log files."""
        results = []
        
        # Get date range for files
        if start_date:
            start_file = start_date.strftime("%Y-%m-%d")
        else:
            start_file = None
        
        if end_date:
            end_file = end_date.strftime("%Y-%m-%d")
        else:
            end_file = datetime.now().strftime("%Y-%m-%d")
        
        # Search files
        for log_file in sorted(self.audit_dir.glob("audit_*.jsonl"), reverse=True):
            file_date = log_file.stem.replace("audit_", "")
            if start_file and file_date < start_file:
                continue
            if end_file and file_date > end_file:
                continue
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                        
                        data = json.loads(line)
                        
                        # Apply filters
                        if event_type and data.get("event_type") != event_type.value:
                            continue
                        if user_id and data.get("user_id") != user_id:
                            continue
                        if level and data.get("level") != level.value:
                            continue
                        
                        event = AuditEvent(
                            event_id=data["event_id"],
                            event_type=AuditEventType(data["event_type"]),
                            level=AuditLevel(data["level"]),
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            user_id=data.get("user_id"),
                            session_id=data.get("session_id"),
                            ip_address=data.get("ip_address"),
                            action=data["action"],
                            resource=data.get("resource"),
                            details=data.get("details", {}),
                            success=data.get("success", True),
                            error_message=data.get("error_message"),
                        )
                        
                        # Date filter
                        if start_date and event.timestamp < start_date:
                            continue
                        if end_date and event.timestamp > end_date:
                            continue
                        
                        results.append(event)
            
            except Exception as e:
                logger.error(f"Error reading audit file {log_file}: {e}")
        
        return results
    
    def cleanup_old_logs(self):
        """Clean up old log files based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self._retention_days)
        
        for log_file in self.audit_dir.glob("audit_*.jsonl"):
            file_date_str = log_file.stem.replace("audit_", "")
            try:
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info(f"Deleted old audit log: {log_file}")
            except ValueError:
                continue
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        from collections import Counter
        
        event_types = Counter(e.event_type.value for e in self._events)
        levels = Counter(e.level.value for e in self._events)
        success_count = sum(1 for e in self._events if e.success)
        failure_count = len(self._events) - success_count
        
        return {
            "total_events": len(self._events),
            "event_types": dict(event_types),
            "levels": dict(levels),
            "success_count": success_count,
            "failure_count": failure_count,
            "retention_days": self._retention_days,
        }

