"""
Audit Logger
============

Advanced audit logging for compliance.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Audit event types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"


@dataclass
class AuditEvent:
    """Audit event."""
    event_type: AuditEventType
    user_id: Optional[str]
    action: str
    resource: str
    result: str  # success, failure, denied
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """Advanced audit logger."""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        self._storage_backend = storage_backend
        self._events: List[AuditEvent] = []
        self._retention_days = 90
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        resource: str,
        result: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log audit event."""
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        
        self._events.append(event)
        
        # Log to standard logger
        logger.info(f"Audit: {event_type.value} - {action} - {resource} - {result}")
        
        # Store in backend if available
        if self._storage_backend:
            try:
                self._storage_backend.store(event)
            except Exception as e:
                logger.error(f"Failed to store audit event: {e}")
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Get audit events."""
        events = self._events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if resource:
            events = [e for e in events if e.resource == resource]
        
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return events[:limit]
    
    def get_audit_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit statistics."""
        events = self.get_events(start_date=start_date, end_date=end_date, limit=10000)
        
        return {
            "total_events": len(events),
            "by_type": {
                event_type.value: sum(1 for e in events if e.event_type == event_type)
                for event_type in AuditEventType
            },
            "by_result": {
                result: sum(1 for e in events if e.result == result)
                for result in ["success", "failure", "denied"]
            },
            "unique_users": len(set(e.user_id for e in events if e.user_id)),
            "unique_resources": len(set(e.resource for e in events))
        }















