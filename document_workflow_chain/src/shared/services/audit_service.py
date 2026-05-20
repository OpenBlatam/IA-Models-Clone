"""
Audit Service
=============

Comprehensive audit service for tracking all system activities and changes.
"""

from __future__ import annotations
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution


logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Audit event type enumeration"""
    # Authentication Events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    
    # Authorization Events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    
    # Data Events
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    IMPORT = "import"
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIGURATION_CHANGE = "configuration_change"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Workflow Events
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_UPDATED = "workflow_updated"
    WORKFLOW_DELETED = "workflow_deleted"
    WORKFLOW_EXECUTED = "workflow_executed"
    NODE_ADDED = "node_added"
    NODE_UPDATED = "node_updated"
    NODE_DELETED = "node_deleted"
    
    # Security Events
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class AuditSeverity(str, Enum):
    """Audit severity enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditStatus(str, Enum):
    """Audit status enumeration"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"


@dataclass
class AuditEvent:
    """Audit event representation"""
    id: str
    event_type: AuditEventType
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: str
    user_agent: str
    resource_type: str
    resource_id: Optional[str]
    action: str
    status: AuditStatus
    severity: AuditSeverity
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditQuery:
    """Audit query parameters"""
    user_id: Optional[str] = None
    event_type: Optional[AuditEventType] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    status: Optional[AuditStatus] = None
    severity: Optional[AuditSeverity] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    ip_address: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class AuditReport:
    """Audit report representation"""
    id: str
    name: str
    description: str
    query: AuditQuery
    generated_at: datetime
    generated_by: str
    total_events: int
    events: List[AuditEvent]
    summary: Dict[str, Any] = field(default_factory=dict)


class AuditService:
    """Comprehensive audit service"""
    
    def __init__(self, max_events: int = 100000, retention_days: int = 365):
        self.max_events = max_events
        self.retention_days = retention_days
        self.events: List[AuditEvent] = []
        self.reports: List[AuditReport] = []
        self.event_index: Dict[str, List[AuditEvent]] = defaultdict(list)
        self.is_running = False
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the audit service"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        logger.info("Audit service started")
    
    async def stop(self):
        """Stop the audit service"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Audit service stopped")
    
    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        session_id: Optional[str],
        ip_address: str,
        user_agent: str,
        resource_type: str,
        resource_id: Optional[str],
        action: str,
        status: AuditStatus,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log audit event"""
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            id=event_id,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            status=status,
            severity=severity,
            timestamp=DateTimeHelpers.now_utc(),
            details=details or {},
            old_values=old_values,
            new_values=new_values,
            metadata=metadata or {}
        )
        
        # Add to events list
        self.events.append(event)
        
        # Add to index
        self.event_index[event_type.value].append(event)
        if user_id:
            self.event_index[f"user:{user_id}"].append(event)
        if resource_type:
            self.event_index[f"resource:{resource_type}"].append(event)
        if resource_id:
            self.event_index[f"resource_id:{resource_id}"].append(event)
        
        # Keep only max_events
        if len(self.events) > self.max_events:
            oldest_event = self.events.pop(0)
            self._remove_from_index(oldest_event)
        
        # Log to logger
        log_level = logging.INFO if status == AuditStatus.SUCCESS else logging.WARNING
        logger.log(
            log_level,
            f"Audit event: {event_type.value} - User: {user_id} - Resource: {resource_type}:{resource_id} - Status: {status.value}"
        )
        
        return event_id
    
    def _remove_from_index(self, event: AuditEvent):
        """Remove event from index"""
        # Remove from event type index
        if event.event_type.value in self.event_index:
            try:
                self.event_index[event.event_type.value].remove(event)
            except ValueError:
                pass
        
        # Remove from user index
        if event.user_id:
            user_key = f"user:{event.user_id}"
            if user_key in self.event_index:
                try:
                    self.event_index[user_key].remove(event)
                except ValueError:
                    pass
        
        # Remove from resource index
        if event.resource_type:
            resource_key = f"resource:{event.resource_type}"
            if resource_key in self.event_index:
                try:
                    self.event_index[resource_key].remove(event)
                except ValueError:
                    pass
        
        # Remove from resource ID index
        if event.resource_id:
            resource_id_key = f"resource_id:{event.resource_id}"
            if resource_id_key in self.event_index:
                try:
                    self.event_index[resource_id_key].remove(event)
                except ValueError:
                    pass
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events"""
        events = self.events.copy()
        
        # Apply filters
        if query.user_id:
            events = [e for e in events if e.user_id == query.user_id]
        
        if query.event_type:
            events = [e for e in events if e.event_type == query.event_type]
        
        if query.resource_type:
            events = [e for e in events if e.resource_type == query.resource_type]
        
        if query.resource_id:
            events = [e for e in events if e.resource_id == query.resource_id]
        
        if query.status:
            events = [e for e in events if e.status == query.status]
        
        if query.severity:
            events = [e for e in events if e.severity == query.severity]
        
        if query.start_date:
            events = [e for e in events if e.timestamp >= query.start_date]
        
        if query.end_date:
            events = [e for e in events if e.timestamp <= query.end_date]
        
        if query.ip_address:
            events = [e for e in events if e.ip_address == query.ip_address]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply pagination
        start = query.offset
        end = start + query.limit
        return events[start:end]
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[AuditEvent]:
        """Get events by user"""
        query = AuditQuery(user_id=user_id, limit=limit)
        return self.query_events(query)
    
    def get_events_by_resource(self, resource_type: str, resource_id: Optional[str] = None, limit: int = 100) -> List[AuditEvent]:
        """Get events by resource"""
        query = AuditQuery(resource_type=resource_type, resource_id=resource_id, limit=limit)
        return self.query_events(query)
    
    def get_events_by_type(self, event_type: AuditEventType, limit: int = 100) -> List[AuditEvent]:
        """Get events by type"""
        query = AuditQuery(event_type=event_type, limit=limit)
        return self.query_events(query)
    
    def get_events_by_severity(self, severity: AuditSeverity, limit: int = 100) -> List[AuditEvent]:
        """Get events by severity"""
        query = AuditQuery(severity=severity, limit=limit)
        return self.query_events(query)
    
    def get_failed_events(self, limit: int = 100) -> List[AuditEvent]:
        """Get failed events"""
        query = AuditQuery(status=AuditStatus.FAILURE, limit=limit)
        return self.query_events(query)
    
    def get_critical_events(self, limit: int = 100) -> List[AuditEvent]:
        """Get critical events"""
        query = AuditQuery(severity=AuditSeverity.CRITICAL, limit=limit)
        return self.query_events(query)
    
    def generate_report(
        self,
        name: str,
        description: str,
        query: AuditQuery,
        generated_by: str
    ) -> AuditReport:
        """Generate audit report"""
        events = self.query_events(query)
        
        # Generate summary
        summary = self._generate_summary(events)
        
        report = AuditReport(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            query=query,
            generated_at=DateTimeHelpers.now_utc(),
            generated_by=generated_by,
            total_events=len(events),
            events=events,
            summary=summary
        )
        
        self.reports.append(report)
        
        # Keep only last 100 reports
        if len(self.reports) > 100:
            self.reports = self.reports[-100:]
        
        logger.info(f"Audit report generated: {name} - {len(events)} events")
        
        return report
    
    def _generate_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate summary for events"""
        if not events:
            return {}
        
        # Event type distribution
        event_types = defaultdict(int)
        for event in events:
            event_types[event.event_type.value] += 1
        
        # Status distribution
        statuses = defaultdict(int)
        for event in events:
            statuses[event.status.value] += 1
        
        # Severity distribution
        severities = defaultdict(int)
        for event in events:
            severities[event.severity.value] += 1
        
        # User distribution
        users = defaultdict(int)
        for event in events:
            if event.user_id:
                users[event.user_id] += 1
        
        # Resource type distribution
        resource_types = defaultdict(int)
        for event in events:
            resource_types[event.resource_type] += 1
        
        # Time range
        timestamps = [event.timestamp for event in events]
        time_range = {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat()
        }
        
        return {
            "total_events": len(events),
            "event_types": dict(event_types),
            "statuses": dict(statuses),
            "severities": dict(severities),
            "users": dict(users),
            "resource_types": dict(resource_types),
            "time_range": time_range
        }
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit statistics"""
        total_events = len(self.events)
        
        # Event type distribution
        event_types = defaultdict(int)
        for event in self.events:
            event_types[event.event_type.value] += 1
        
        # Status distribution
        statuses = defaultdict(int)
        for event in self.events:
            statuses[event.status.value] += 1
        
        # Severity distribution
        severities = defaultdict(int)
        for event in self.events:
            severities[event.severity.value] += 1
        
        # Recent activity (last 24 hours)
        recent_cutoff = DateTimeHelpers.now_utc() - timedelta(hours=24)
        recent_events = [e for e in self.events if e.timestamp > recent_cutoff]
        
        # Failed events
        failed_events = [e for e in self.events if e.status == AuditStatus.FAILURE]
        
        # Critical events
        critical_events = [e for e in self.events if e.severity == AuditSeverity.CRITICAL]
        
        return {
            "total_events": total_events,
            "recent_events_24h": len(recent_events),
            "failed_events": len(failed_events),
            "critical_events": len(critical_events),
            "event_types": dict(event_types),
            "statuses": dict(statuses),
            "severities": dict(severities),
            "reports_generated": len(self.reports),
            "timestamp": DateTimeHelpers.now_utc().isoformat()
        }
    
    def export_events(self, query: AuditQuery, format: str = "json") -> str:
        """Export events in specified format"""
        events = self.query_events(query)
        
        if format == "json":
            return json.dumps({
                "query": {
                    "user_id": query.user_id,
                    "event_type": query.event_type.value if query.event_type else None,
                    "resource_type": query.resource_type,
                    "resource_id": query.resource_id,
                    "status": query.status.value if query.status else None,
                    "severity": query.severity.value if query.severity else None,
                    "start_date": query.start_date.isoformat() if query.start_date else None,
                    "end_date": query.end_date.isoformat() if query.end_date else None,
                    "ip_address": query.ip_address
                },
                "total_events": len(events),
                "events": [
                    {
                        "id": event.id,
                        "event_type": event.event_type.value,
                        "user_id": event.user_id,
                        "session_id": event.session_id,
                        "ip_address": event.ip_address,
                        "user_agent": event.user_agent,
                        "resource_type": event.resource_type,
                        "resource_id": event.resource_id,
                        "action": event.action,
                        "status": event.status.value,
                        "severity": event.severity.value,
                        "timestamp": event.timestamp.isoformat(),
                        "details": event.details,
                        "old_values": event.old_values,
                        "new_values": event.new_values,
                        "metadata": event.metadata
                    }
                    for event in events
                ]
            }, indent=2)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "ID", "Event Type", "User ID", "Session ID", "IP Address", "User Agent",
                "Resource Type", "Resource ID", "Action", "Status", "Severity", "Timestamp",
                "Details", "Old Values", "New Values", "Metadata"
            ])
            
            # Write data
            for event in events:
                writer.writerow([
                    event.id,
                    event.event_type.value,
                    event.user_id or "",
                    event.session_id or "",
                    event.ip_address,
                    event.user_agent,
                    event.resource_type,
                    event.resource_id or "",
                    event.action,
                    event.status.value,
                    event.severity.value,
                    event.timestamp.isoformat(),
                    json.dumps(event.details),
                    json.dumps(event.old_values) if event.old_values else "",
                    json.dumps(event.new_values) if event.new_values else "",
                    json.dumps(event.metadata)
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _cleanup_worker(self):
        """Cleanup old events periodically"""
        while self.is_running:
            try:
                await self._cleanup_old_events()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Audit cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_events(self):
        """Cleanup old events"""
        cutoff_time = DateTimeHelpers.now_utc() - timedelta(days=self.retention_days)
        
        # Remove old events
        old_events = [e for e in self.events if e.timestamp < cutoff_time]
        for event in old_events:
            self.events.remove(event)
            self._remove_from_index(event)
        
        # Remove old reports
        old_reports = [r for r in self.reports if r.generated_at < cutoff_time]
        for report in old_reports:
            self.reports.remove(report)
        
        if old_events or old_reports:
            logger.info(f"Cleaned up {len(old_events)} old events and {len(old_reports)} old reports")


# Global audit service
audit_service = AuditService()


# Utility functions
async def start_audit_service():
    """Start the audit service"""
    await audit_service.start()


async def stop_audit_service():
    """Stop the audit service"""
    await audit_service.stop()


def log_audit_event(
    event_type: AuditEventType,
    user_id: Optional[str],
    session_id: Optional[str],
    ip_address: str,
    user_agent: str,
    resource_type: str,
    resource_id: Optional[str],
    action: str,
    status: AuditStatus,
    severity: AuditSeverity = AuditSeverity.MEDIUM,
    details: Optional[Dict[str, Any]] = None,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Log audit event"""
    return audit_service.log_event(
        event_type, user_id, session_id, ip_address, user_agent,
        resource_type, resource_id, action, status, severity,
        details, old_values, new_values, metadata
    )


def query_audit_events(query: AuditQuery) -> List[AuditEvent]:
    """Query audit events"""
    return audit_service.query_events(query)


def get_audit_statistics() -> Dict[str, Any]:
    """Get audit statistics"""
    return audit_service.get_audit_statistics()


def export_audit_events(query: AuditQuery, format: str = "json") -> str:
    """Export audit events"""
    return audit_service.export_events(query, format)


# Common audit logging functions
def log_authentication_event(
    event_type: AuditEventType,
    user_id: Optional[str],
    ip_address: str,
    user_agent: str,
    status: AuditStatus,
    details: Optional[Dict[str, Any]] = None
) -> str:
    """Log authentication event"""
    return log_audit_event(
        event_type=event_type,
        user_id=user_id,
        session_id=None,
        ip_address=ip_address,
        user_agent=user_agent,
        resource_type="authentication",
        resource_id=user_id,
        action=event_type.value,
        status=status,
        severity=AuditSeverity.HIGH if status == AuditStatus.FAILURE else AuditSeverity.MEDIUM,
        details=details
    )


def log_data_event(
    event_type: AuditEventType,
    user_id: Optional[str],
    session_id: Optional[str],
    ip_address: str,
    user_agent: str,
    resource_type: str,
    resource_id: Optional[str],
    status: AuditStatus,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None
) -> str:
    """Log data event"""
    return log_audit_event(
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        resource_type=resource_type,
        resource_id=resource_id,
        action=event_type.value,
        status=status,
        severity=AuditSeverity.MEDIUM,
        details=details,
        old_values=old_values,
        new_values=new_values
    )


def log_security_event(
    event_type: AuditEventType,
    user_id: Optional[str],
    ip_address: str,
    user_agent: str,
    status: AuditStatus,
    details: Optional[Dict[str, Any]] = None
) -> str:
    """Log security event"""
    return log_audit_event(
        event_type=event_type,
        user_id=user_id,
        session_id=None,
        ip_address=ip_address,
        user_agent=user_agent,
        resource_type="security",
        resource_id=None,
        action=event_type.value,
        status=status,
        severity=AuditSeverity.CRITICAL,
        details=details
    )




