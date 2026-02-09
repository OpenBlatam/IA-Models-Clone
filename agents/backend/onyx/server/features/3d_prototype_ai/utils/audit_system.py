"""
Audit System - Sistema de compliance y auditoría
=================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Tipos de eventos de auditoría"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_CHANGE = "permission_change"
    CONFIG_CHANGE = "config_change"
    EXPORT = "export"
    SHARE = "share"


@dataclass
class AuditLog:
    """Log de auditoría"""
    id: str
    event_type: AuditEventType
    user_id: str
    resource_type: str
    resource_id: Optional[str]
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None


class AuditSystem:
    """Sistema de auditoría y compliance"""
    
    def __init__(self):
        self.audit_logs: List[AuditLog] = []
        self.max_logs = 100000
        self.retention_days = 365
    
    def log_event(self, event_type: AuditEventType, user_id: str,
                 resource_type: str, action: str,
                 resource_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None,
                 success: bool = True,
                 error_message: Optional[str] = None):
        """Registra un evento de auditoría"""
        from uuid import uuid4
        
        log = AuditLog(
            id=str(uuid4()),
            event_type=event_type,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        self.audit_logs.append(log)
        
        # Limpiar logs antiguos
        if len(self.audit_logs) > self.max_logs:
            self.audit_logs = self.audit_logs[-self.max_logs:]
        
        # Limpiar logs fuera de retención
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        self.audit_logs = [l for l in self.audit_logs if l.timestamp > cutoff]
        
        logger.info(f"Evento de auditoría registrado: {event_type.value} - {action}")
    
    def get_audit_logs(self, user_id: Optional[str] = None,
                      event_type: Optional[AuditEventType] = None,
                      resource_type: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: int = 1000) -> List[Dict[str, Any]]:
        """Obtiene logs de auditoría"""
        logs = self.audit_logs
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
        
        if resource_type:
            logs = [l for l in logs if l.resource_type == resource_type]
        
        if start_date:
            logs = [l for l in logs if l.timestamp >= start_date]
        
        if end_date:
            logs = [l for l in logs if l.timestamp <= end_date]
        
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "id": l.id,
                "event_type": l.event_type.value,
                "user_id": l.user_id,
                "resource_type": l.resource_type,
                "resource_id": l.resource_id,
                "action": l.action,
                "details": l.details,
                "ip_address": l.ip_address,
                "user_agent": l.user_agent,
                "timestamp": l.timestamp.isoformat(),
                "success": l.success,
                "error_message": l.error_message
            }
            for l in logs[:limit]
        ]
    
    def get_user_activity(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Obtiene actividad de un usuario"""
        cutoff = datetime.now() - timedelta(days=days)
        user_logs = [
            l for l in self.audit_logs
            if l.user_id == user_id and l.timestamp > cutoff
        ]
        
        activity_by_type = {}
        for log in user_logs:
            event_type = log.event_type.value
            activity_by_type[event_type] = activity_by_type.get(event_type, 0) + 1
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_events": len(user_logs),
            "events_by_type": activity_by_type,
            "last_activity": user_logs[0].timestamp.isoformat() if user_logs else None
        }
    
    def generate_compliance_report(self, start_date: datetime,
                                   end_date: datetime) -> Dict[str, Any]:
        """Genera reporte de compliance"""
        logs = [
            l for l in self.audit_logs
            if start_date <= l.timestamp <= end_date
        ]
        
        # Estadísticas
        total_events = len(logs)
        successful_events = sum(1 for l in logs if l.success)
        failed_events = total_events - successful_events
        
        events_by_type = {}
        for log in logs:
            event_type = log.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
        
        unique_users = len(set(l.user_id for l in logs))
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_events": total_events,
                "successful_events": successful_events,
                "failed_events": failed_events,
                "unique_users": unique_users
            },
            "events_by_type": events_by_type,
            "compliance_score": (successful_events / total_events * 100) if total_events > 0 else 0
        }




