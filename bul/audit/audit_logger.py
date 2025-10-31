"""
BUL Audit Logging System
========================

Comprehensive audit logging for security, compliance, and monitoring.
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import Request, Response
from pydantic import BaseModel, Field
import uuid
import os

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..security import get_encryption

logger = get_logger(__name__)

class AuditEventType(str, Enum):
    """Audit event types"""
    # Authentication & Authorization
    LOGIN_SUCCESS = "auth.login_success"
    LOGIN_FAILED = "auth.login_failed"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token_refresh"
    PERMISSION_DENIED = "auth.permission_denied"
    
    # API Access
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"
    RATE_LIMIT_EXCEEDED = "api.rate_limit_exceeded"
    
    # Document Operations
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_ACCESSED = "document.accessed"
    DOCUMENT_EXPORTED = "document.exported"
    
    # System Operations
    CONFIG_CHANGED = "system.config_changed"
    CACHE_CLEARED = "system.cache_cleared"
    MAINTENANCE_MODE = "system.maintenance_mode"
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    
    # Security Events
    SECURITY_VIOLATION = "security.violation"
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    DATA_BREACH_ATTEMPT = "security.data_breach_attempt"
    UNAUTHORIZED_ACCESS = "security.unauthorized_access"
    
    # Admin Operations
    ADMIN_ACTION = "admin.action"
    USER_CREATED = "admin.user_created"
    USER_UPDATED = "admin.user_updated"
    USER_DELETED = "admin.user_deleted"
    ROLE_CHANGED = "admin.role_changed"
    
    # Webhook Events
    WEBHOOK_CREATED = "webhook.created"
    WEBHOOK_UPDATED = "webhook.updated"
    WEBHOOK_DELETED = "webhook.deleted"
    WEBHOOK_DELIVERED = "webhook.delivered"
    WEBHOOK_FAILED = "webhook.failed"

class AuditSeverity(str, Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditStatus(str, Enum):
    """Audit event status"""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    INFO = "info"

@dataclass
class AuditEvent:
    """Audit event data structure"""
    id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    request_id: Optional[str]
    severity: AuditSeverity
    status: AuditStatus
    message: str
    details: Dict[str, Any]
    resource: Optional[str]
    action: Optional[str]
    outcome: Optional[str]
    risk_score: float
    tags: List[str]
    source: str
    version: str = "1.0"

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        self.encryption = get_encryption()
        
        # Audit storage (in production, use secure database)
        self.audit_events: List[AuditEvent] = []
        self.audit_index: Dict[str, List[str]] = {}  # Index for fast lookups
        
        # Configuration
        self.retention_days = 90
        self.max_events = 100000
        self.compression_enabled = True
        self.encryption_enabled = True
        
        # Risk scoring weights
        self.risk_weights = {
            AuditEventType.LOGIN_FAILED: 0.3,
            AuditEventType.PERMISSION_DENIED: 0.4,
            AuditEventType.RATE_LIMIT_EXCEEDED: 0.2,
            AuditEventType.SECURITY_VIOLATION: 0.8,
            AuditEventType.SUSPICIOUS_ACTIVITY: 0.6,
            AuditEventType.UNAUTHORIZED_ACCESS: 0.9,
            AuditEventType.DATA_BREACH_ATTEMPT: 1.0
        }
    
    async def log_event(
        self,
        event_type: AuditEventType,
        message: str,
        details: Dict[str, Any] = None,
        user_id: str = None,
        session_id: str = None,
        ip_address: str = None,
        user_agent: str = None,
        request_id: str = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        status: AuditStatus = AuditStatus.INFO,
        resource: str = None,
        action: str = None,
        outcome: str = None,
        tags: List[str] = None,
        source: str = "system"
    ) -> str:
        """Log an audit event"""
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(event_type, severity, details)
            
            # Create audit event
            event = AuditEvent(
                id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                severity=severity,
                status=status,
                message=message,
                details=details or {},
                resource=resource,
                action=action,
                outcome=outcome,
                risk_score=risk_score,
                tags=tags or [],
                source=source
            )
            
            # Store event
            await self._store_event(event)
            
            # Index event
            self._index_event(event)
            
            # Check for suspicious patterns
            await self._check_suspicious_patterns(event)
            
            # Log to standard logger if high severity
            if severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                self.logger.warning(f"AUDIT: {event_type.value} - {message} (Risk: {risk_score})")
            
            return event_id
        
        except Exception as e:
            self.logger.error(f"Error logging audit event: {e}")
            return None
    
    async def log_api_request(
        self,
        request: Request,
        response: Response,
        processing_time: float,
        user_id: str = None,
        session_id: str = None
    ):
        """Log API request/response"""
        try:
            # Extract request information
            request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("User-Agent")
            
            # Determine event type and status
            if response.status_code >= 400:
                event_type = AuditEventType.API_ERROR
                status = AuditStatus.FAILURE
                severity = AuditSeverity.MEDIUM
            else:
                event_type = AuditEventType.API_REQUEST
                status = AuditStatus.SUCCESS
                severity = AuditSeverity.LOW
            
            # Prepare details
            details = {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "status_code": response.status_code,
                "processing_time": processing_time,
                "content_length": response.headers.get("Content-Length"),
                "content_type": response.headers.get("Content-Type")
            }
            
            # Add request body for certain endpoints (be careful with sensitive data)
            if request.method in ["POST", "PUT", "PATCH"] and request.url.path in ["/generate", "/admin/config/reload"]:
                try:
                    body = await request.body()
                    if body and len(body) < 1000:  # Only log small bodies
                        details["request_body"] = body.decode("utf-8", errors="ignore")
                except Exception:
                    pass
            
            # Log the event
            await self.log_event(
                event_type=event_type,
                message=f"API {request.method} {request.url.path} - {response.status_code}",
                details=details,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                severity=severity,
                status=status,
                resource=request.url.path,
                action=request.method,
                outcome=f"HTTP_{response.status_code}",
                tags=["api", "http"],
                source="api"
            )
        
        except Exception as e:
            self.logger.error(f"Error logging API request: {e}")
    
    async def log_document_operation(
        self,
        operation: str,
        document_id: str,
        user_id: str = None,
        details: Dict[str, Any] = None
    ):
        """Log document-related operations"""
        try:
            event_type_map = {
                "create": AuditEventType.DOCUMENT_CREATED,
                "update": AuditEventType.DOCUMENT_UPDATED,
                "delete": AuditEventType.DOCUMENT_DELETED,
                "access": AuditEventType.DOCUMENT_ACCESSED,
                "export": AuditEventType.DOCUMENT_EXPORTED
            }
            
            event_type = event_type_map.get(operation, AuditEventType.DOCUMENT_ACCESSED)
            
            await self.log_event(
                event_type=event_type,
                message=f"Document {operation}: {document_id}",
                details=details or {},
                user_id=user_id,
                resource=document_id,
                action=operation,
                tags=["document", operation],
                source="document"
            )
        
        except Exception as e:
            self.logger.error(f"Error logging document operation: {e}")
    
    async def log_security_event(
        self,
        event_type: AuditEventType,
        message: str,
        ip_address: str = None,
        user_id: str = None,
        details: Dict[str, Any] = None
    ):
        """Log security-related events"""
        try:
            await self.log_event(
                event_type=event_type,
                message=message,
                details=details or {},
                user_id=user_id,
                ip_address=ip_address,
                severity=AuditSeverity.HIGH,
                status=AuditStatus.WARNING,
                tags=["security", "threat"],
                source="security"
            )
        
        except Exception as e:
            self.logger.error(f"Error logging security event: {e}")
    
    async def log_admin_action(
        self,
        action: str,
        admin_user_id: str,
        target_resource: str = None,
        details: Dict[str, Any] = None
    ):
        """Log administrative actions"""
        try:
            await self.log_event(
                event_type=AuditEventType.ADMIN_ACTION,
                message=f"Admin action: {action}",
                details=details or {},
                user_id=admin_user_id,
                resource=target_resource,
                action=action,
                severity=AuditSeverity.MEDIUM,
                status=AuditStatus.INFO,
                tags=["admin", "management"],
                source="admin"
            )
        
        except Exception as e:
            self.logger.error(f"Error logging admin action: {e}")
    
    async def _store_event(self, event: AuditEvent):
        """Store audit event"""
        try:
            # Add to memory storage
            self.audit_events.append(event)
            
            # Maintain size limit
            if len(self.audit_events) > self.max_events:
                # Remove oldest events
                self.audit_events = self.audit_events[-self.max_events:]
            
            # In production, also store in secure database
            # await self._store_in_database(event)
        
        except Exception as e:
            self.logger.error(f"Error storing audit event: {e}")
    
    def _index_event(self, event: AuditEvent):
        """Index audit event for fast lookups"""
        try:
            # Index by event type
            if event.event_type.value not in self.audit_index:
                self.audit_index[event.event_type.value] = []
            self.audit_index[event.event_type.value].append(event.id)
            
            # Index by user ID
            if event.user_id:
                user_key = f"user:{event.user_id}"
                if user_key not in self.audit_index:
                    self.audit_index[user_key] = []
                self.audit_index[user_key].append(event.id)
            
            # Index by IP address
            if event.ip_address:
                ip_key = f"ip:{event.ip_address}"
                if ip_key not in self.audit_index:
                    self.audit_index[ip_key] = []
                self.audit_index[ip_key].append(event.id)
            
            # Index by severity
            severity_key = f"severity:{event.severity.value}"
            if severity_key not in self.audit_index:
                self.audit_index[severity_key] = []
            self.audit_index[severity_key].append(event.id)
        
        except Exception as e:
            self.logger.error(f"Error indexing audit event: {e}")
    
    async def _check_suspicious_patterns(self, event: AuditEvent):
        """Check for suspicious patterns in audit events"""
        try:
            # Check for multiple failed logins from same IP
            if event.event_type == AuditEventType.LOGIN_FAILED and event.ip_address:
                recent_failures = await self._get_recent_events(
                    event_type=AuditEventType.LOGIN_FAILED,
                    ip_address=event.ip_address,
                    minutes=15
                )
                
                if len(recent_failures) >= 5:
                    await self.log_security_event(
                        AuditEventType.SUSPICIOUS_ACTIVITY,
                        f"Multiple failed login attempts from IP {event.ip_address}",
                        ip_address=event.ip_address,
                        details={"failed_attempts": len(recent_failures)}
                    )
            
            # Check for high-risk events
            if event.risk_score >= 0.8:
                await self.log_security_event(
                    AuditEventType.SECURITY_VIOLATION,
                    f"High-risk event detected: {event.event_type.value}",
                    ip_address=event.ip_address,
                    user_id=event.user_id,
                    details={"risk_score": event.risk_score, "event_id": event.id}
                )
        
        except Exception as e:
            self.logger.error(f"Error checking suspicious patterns: {e}")
    
    def _calculate_risk_score(self, event_type: AuditEventType, severity: AuditSeverity, details: Dict[str, Any]) -> float:
        """Calculate risk score for audit event"""
        try:
            base_score = self.risk_weights.get(event_type, 0.1)
            
            # Adjust based on severity
            severity_multiplier = {
                AuditSeverity.LOW: 0.5,
                AuditSeverity.MEDIUM: 1.0,
                AuditSeverity.HIGH: 1.5,
                AuditSeverity.CRITICAL: 2.0
            }.get(severity, 1.0)
            
            # Adjust based on details
            detail_multiplier = 1.0
            if details:
                if "admin" in str(details).lower():
                    detail_multiplier *= 1.5
                if "delete" in str(details).lower():
                    detail_multiplier *= 1.3
                if "export" in str(details).lower():
                    detail_multiplier *= 1.2
            
            risk_score = base_score * severity_multiplier * detail_multiplier
            return min(risk_score, 1.0)  # Cap at 1.0
        
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    async def _get_recent_events(
        self,
        event_type: AuditEventType = None,
        user_id: str = None,
        ip_address: str = None,
        minutes: int = 60
    ) -> List[AuditEvent]:
        """Get recent audit events matching criteria"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            filtered_events = []
            for event in self.audit_events:
                if event.timestamp < cutoff_time:
                    continue
                
                if event_type and event.event_type != event_type:
                    continue
                
                if user_id and event.user_id != user_id:
                    continue
                
                if ip_address and event.ip_address != ip_address:
                    continue
                
                filtered_events.append(event)
            
            return filtered_events
        
        except Exception as e:
            self.logger.error(f"Error getting recent events: {e}")
            return []
    
    async def get_audit_events(
        self,
        event_type: AuditEventType = None,
        user_id: str = None,
        ip_address: str = None,
        severity: AuditSeverity = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Get audit events with filtering"""
        try:
            filtered_events = []
            
            for event in self.audit_events:
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                
                if user_id and event.user_id != user_id:
                    continue
                
                if ip_address and event.ip_address != ip_address:
                    continue
                
                if severity and event.severity != severity:
                    continue
                
                if start_date and event.timestamp < start_date:
                    continue
                
                if end_date and event.timestamp > end_date:
                    continue
                
                filtered_events.append(event)
            
            # Sort by timestamp (newest first)
            filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply limit
            return filtered_events[:limit]
        
        except Exception as e:
            self.logger.error(f"Error getting audit events: {e}")
            return []
    
    async def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit statistics"""
        try:
            total_events = len(self.audit_events)
            
            # Count by event type
            event_type_counts = {}
            for event in self.audit_events:
                event_type_counts[event.event_type.value] = event_type_counts.get(event.event_type.value, 0) + 1
            
            # Count by severity
            severity_counts = {}
            for event in self.audit_events:
                severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
            
            # Count by status
            status_counts = {}
            for event in self.audit_events:
                status_counts[event.status.value] = status_counts.get(event.status.value, 0) + 1
            
            # Calculate average risk score
            if self.audit_events:
                avg_risk_score = sum(event.risk_score for event in self.audit_events) / len(self.audit_events)
            else:
                avg_risk_score = 0.0
            
            # Get recent high-risk events
            recent_high_risk = await self._get_recent_events(minutes=60)
            high_risk_count = len([e for e in recent_high_risk if e.risk_score >= 0.7])
            
            return {
                "total_events": total_events,
                "event_type_counts": event_type_counts,
                "severity_counts": severity_counts,
                "status_counts": status_counts,
                "average_risk_score": round(avg_risk_score, 3),
                "recent_high_risk_events": high_risk_count,
                "retention_days": self.retention_days,
                "max_events": self.max_events
            }
        
        except Exception as e:
            self.logger.error(f"Error getting audit stats: {e}")
            return {}
    
    async def export_audit_logs(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        format: str = "json"
    ) -> Union[str, bytes]:
        """Export audit logs"""
        try:
            events = await self.get_audit_events(
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            if format.lower() == "json":
                export_data = {
                    "exported_at": datetime.now().isoformat(),
                    "total_events": len(events),
                    "date_range": {
                        "start": start_date.isoformat() if start_date else None,
                        "end": end_date.isoformat() if end_date else None
                    },
                    "events": [asdict(event) for event in events]
                }
                return json.dumps(export_data, indent=2, default=str)
            
            elif format.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    "ID", "Event Type", "Timestamp", "User ID", "IP Address",
                    "Severity", "Status", "Message", "Risk Score", "Resource", "Action"
                ])
                
                # Write events
                for event in events:
                    writer.writerow([
                        event.id, event.event_type.value, event.timestamp.isoformat(),
                        event.user_id, event.ip_address, event.severity.value,
                        event.status.value, event.message, event.risk_score,
                        event.resource, event.action
                    ])
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        except Exception as e:
            self.logger.error(f"Error exporting audit logs: {e}")
            raise

# Global audit logger
_audit_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

# Audit middleware
async def audit_middleware(request: Request, call_next):
    """Audit middleware for FastAPI"""
    try:
        audit_logger = get_audit_logger()
        start_time = datetime.now()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log API request
        await audit_logger.log_api_request(
            request=request,
            response=response,
            processing_time=processing_time
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in audit middleware: {e}")
        return await call_next(request)


