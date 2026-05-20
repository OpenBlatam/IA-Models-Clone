"""
Audit Log Service - Sistema de auditoría
==========================================

Sistema de logging y auditoría de acciones del sistema.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Acciones auditadas"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VIEW = "view"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    IMPORT = "import"
    ACCESS = "access"


class AuditSeverity(str, Enum):
    """Severidad de auditoría"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditLog:
    """Log de auditoría"""
    id: str
    user_id: Optional[str]
    action: AuditAction
    resource_type: str
    resource_id: Optional[str]
    severity: AuditSeverity
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


class AuditLogService:
    """Servicio de auditoría"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.logs: List[AuditLog] = []
        logger.info("AuditLogService initialized")
    
    def log_action(
        self,
        user_id: Optional[str],
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """Registrar acción en log de auditoría"""
        log_id = f"audit_{int(datetime.now().timestamp())}_{len(self.logs)}"
        
        log = AuditLog(
            id=log_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            severity=severity,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
        )
        
        self.logs.append(log)
        
        # Log también al logger estándar
        log_level = {
            AuditSeverity.LOW: logging.INFO,
            AuditSeverity.MEDIUM: logging.INFO,
            AuditSeverity.HIGH: logging.WARNING,
            AuditSeverity.CRITICAL: logging.ERROR,
        }.get(severity, logging.INFO)
        
        logger.log(
            log_level,
            f"Audit: {action.value} on {resource_type} by {user_id or 'system'}"
        )
        
        return log
    
    def get_user_audit_logs(
        self,
        user_id: str,
        limit: int = 100,
        action: Optional[AuditAction] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Obtener logs de auditoría del usuario"""
        user_logs = [
            l for l in self.logs
            if l.user_id == user_id
            and (action is None or l.action == action)
            and (start_date is None or l.timestamp >= start_date)
            and (end_date is None or l.timestamp <= end_date)
        ]
        
        # Ordenar por timestamp (más recientes primero)
        user_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "id": l.id,
                "action": l.action.value,
                "resource_type": l.resource_type,
                "resource_id": l.resource_id,
                "severity": l.severity.value,
                "timestamp": l.timestamp.isoformat(),
                "success": l.success,
                "details": l.details,
            }
            for l in user_logs[:limit]
        ]
    
    def get_resource_audit_logs(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener logs de auditoría de un recurso"""
        resource_logs = [
            l for l in self.logs
            if l.resource_type == resource_type
            and (resource_id is None or l.resource_id == resource_id)
        ]
        
        resource_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "id": l.id,
                "user_id": l.user_id,
                "action": l.action.value,
                "timestamp": l.timestamp.isoformat(),
                "success": l.success,
                "details": l.details,
            }
            for l in resource_logs[:limit]
        ]
    
    def get_audit_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Obtener estadísticas de auditoría"""
        filtered_logs = [
            l for l in self.logs
            if (start_date is None or l.timestamp >= start_date)
            and (end_date is None or l.timestamp <= end_date)
        ]
        
        if not filtered_logs:
            return {
                "total_logs": 0,
                "period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
            }
        
        # Estadísticas por acción
        action_counts = {}
        for log in filtered_logs:
            action = log.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Estadísticas por severidad
        severity_counts = {}
        for log in filtered_logs:
            severity = log.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Tasa de éxito
        success_count = sum(1 for l in filtered_logs if l.success)
        success_rate = (success_count / len(filtered_logs)) * 100 if filtered_logs else 0
        
        return {
            "total_logs": len(filtered_logs),
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
            "action_breakdown": action_counts,
            "severity_breakdown": severity_counts,
            "success_rate": round(success_rate, 2),
            "failed_actions": len(filtered_logs) - success_count,
        }
    
    def search_audit_logs(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Buscar en logs de auditoría"""
        query_lower = query.lower()
        
        matching_logs = [
            l for l in self.logs
            if query_lower in l.resource_type.lower()
            or query_lower in (l.resource_id or "").lower()
            or query_lower in (l.user_id or "").lower()
            or query_lower in l.action.value.lower()
        ]
        
        matching_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "id": l.id,
                "user_id": l.user_id,
                "action": l.action.value,
                "resource_type": l.resource_type,
                "resource_id": l.resource_id,
                "timestamp": l.timestamp.isoformat(),
            }
            for l in matching_logs[:limit]
        ]




