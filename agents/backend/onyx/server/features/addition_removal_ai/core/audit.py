"""
Audit - Sistema de auditoría avanzado
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Tipos de eventos de auditoría"""
    CONTENT_CREATED = "content_created"
    CONTENT_MODIFIED = "content_modified"
    CONTENT_DELETED = "content_deleted"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    CONFIGURATION_CHANGED = "configuration_changed"
    SECURITY_EVENT = "security_event"


@dataclass
class AuditLog:
    """Log de auditoría"""
    id: str
    event_type: AuditEventType
    user_id: Optional[str]
    resource_id: Optional[str]
    resource_type: Optional[str]
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True


class AuditManager:
    """Gestor de auditoría"""

    def __init__(self):
        """Inicializar gestor de auditoría"""
        self.audit_logs: List[AuditLog] = []
        self.max_logs = 10000

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        action: str = "",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True
    ) -> str:
        """
        Registrar evento de auditoría.

        Args:
            event_type: Tipo de evento
            user_id: ID del usuario
            resource_id: ID del recurso
            resource_type: Tipo de recurso
            action: Acción realizada
            details: Detalles adicionales
            ip_address: Dirección IP
            user_agent: User agent
            success: Si fue exitoso

        Returns:
            ID del log
        """
        log_id = str(uuid.uuid4())
        
        log = AuditLog(
            id=log_id,
            event_type=event_type,
            user_id=user_id,
            resource_id=resource_id,
            resource_type=resource_type,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
        
        self.audit_logs.append(log)
        
        # Limitar tamaño
        if len(self.audit_logs) > self.max_logs:
            self.audit_logs = self.audit_logs[-self.max_logs:]
        
        logger.info(f"Evento de auditoría registrado: {event_type.value} - {action}")
        return log_id

    def query_logs(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Consultar logs de auditoría.

        Args:
            event_type: Filtrar por tipo
            user_id: Filtrar por usuario
            resource_id: Filtrar por recurso
            start_date: Fecha de inicio
            end_date: Fecha de fin
            limit: Límite de resultados

        Returns:
            Lista de logs
        """
        logs = self.audit_logs
        
        # Aplicar filtros
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if resource_id:
            logs = [l for l in logs if l.resource_id == resource_id]
        
        if start_date:
            logs = [l for l in logs if l.timestamp >= start_date]
        
        if end_date:
            logs = [l for l in logs if l.timestamp <= end_date]
        
        # Ordenar y limitar
        logs = sorted(logs, key=lambda l: l.timestamp, reverse=True)[:limit]
        
        return [
            {
                "id": log.id,
                "event_type": log.event_type.value,
                "user_id": log.user_id,
                "resource_id": log.resource_id,
                "resource_type": log.resource_type,
                "action": log.action,
                "details": log.details,
                "ip_address": log.ip_address,
                "timestamp": log.timestamp.isoformat(),
                "success": log.success
            }
            for log in logs
        ]

    def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generar reporte de auditoría.

        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin

        Returns:
            Reporte de auditoría
        """
        logs = self.query_logs(start_date=start_date, end_date=end_date, limit=10000)
        
        # Estadísticas
        by_type = {}
        by_user = {}
        by_resource = {}
        
        for log in logs:
            # Por tipo
            event_type = log["event_type"]
            by_type[event_type] = by_type.get(event_type, 0) + 1
            
            # Por usuario
            user_id = log.get("user_id", "unknown")
            by_user[user_id] = by_user.get(user_id, 0) + 1
            
            # Por recurso
            resource_id = log.get("resource_id")
            if resource_id:
                by_resource[resource_id] = by_resource.get(resource_id, 0) + 1
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(logs),
            "by_type": by_type,
            "by_user": by_user,
            "by_resource": by_resource,
            "success_rate": (
                len([l for l in logs if l["success"]]) / len(logs)
                if logs else 0.0
            )
        }






