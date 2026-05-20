"""
Audit System - Sistema de Auditoría
====================================

Sistema completo de auditoría con tracking de acciones, cambios y eventos críticos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Tipo de evento de auditoría."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ACCESS = "access"
    LOGIN = "login"
    LOGOUT = "logout"
    CONFIG_CHANGE = "config_change"
    SECURITY_EVENT = "security_event"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"


class AuditSeverity(Enum):
    """Severidad de auditoría."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditLog:
    """Log de auditoría."""
    audit_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    resource_type: str
    resource_id: str
    action: str
    severity: AuditSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    changes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditSystem:
    """Sistema de auditoría."""
    
    def __init__(self, history_size: int = 1000000):
        self.history_size = history_size
        self.audit_logs: deque = deque(maxlen=history_size)
        self._lock = asyncio.Lock()
    
    def log_event(
        self,
        event_type: AuditEventType,
        resource_type: str,
        resource_id: str,
        action: str,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar evento de auditoría."""
        audit_id = f"audit_{resource_type}_{datetime.now().timestamp()}"
        
        log = AuditLog(
            audit_id=audit_id,
            event_type=event_type,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
            changes=changes,
            metadata=metadata or {},
        )
        
        async def save_log():
            async with self._lock:
                self.audit_logs.append(log)
        
        asyncio.create_task(save_log())
        
        logger.info(f"Audit log: {event_type.value} - {action} on {resource_type}/{resource_id}")
        return audit_id
    
    def get_audit_logs(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Obtener logs de auditoría."""
        logs = list(self.audit_logs)
        
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if resource_type:
            logs = [l for l in logs if l.resource_type == resource_type]
        
        if resource_id:
            logs = [l for l in logs if l.resource_id == resource_id]
        
        if severity:
            logs = [l for l in logs if l.severity == severity]
        
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]
        
        logs.sort(key=lambda l: l.timestamp, reverse=True)
        
        return [
            {
                "audit_id": l.audit_id,
                "event_type": l.event_type.value,
                "user_id": l.user_id,
                "resource_type": l.resource_type,
                "resource_id": l.resource_id,
                "action": l.action,
                "severity": l.severity.value,
                "timestamp": l.timestamp.isoformat(),
                "ip_address": l.ip_address,
                "user_agent": l.user_agent,
                "changes": l.changes,
                "metadata": l.metadata,
            }
            for l in logs[:limit]
        ]
    
    def get_user_activity(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener actividad de usuario."""
        return self.get_audit_logs(user_id=user_id, limit=limit)
    
    def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de recurso."""
        return self.get_audit_logs(
            resource_type=resource_type,
            resource_id=resource_id,
            limit=limit,
        )
    
    def get_audit_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Obtener estadísticas de auditoría."""
        logs = list(self.audit_logs)
        
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]
        
        by_type: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        by_resource: Dict[str, int] = defaultdict(int)
        by_user: Dict[str, int] = defaultdict(int)
        
        for log in logs:
            by_type[log.event_type.value] += 1
            by_severity[log.severity.value] += 1
            by_resource[log.resource_type] += 1
            if log.user_id:
                by_user[log.user_id] += 1
        
        return {
            "total_events": len(logs),
            "events_by_type": dict(by_type),
            "events_by_severity": dict(by_severity),
            "events_by_resource": dict(by_resource),
            "top_users": dict(sorted(by_user.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
    
    def get_audit_system_summary(self) -> Dict[str, Any]:
        """Obtener resumen del sistema."""
        return {
            "total_logs": len(self.audit_logs),
            "max_history_size": self.history_size,
        }



