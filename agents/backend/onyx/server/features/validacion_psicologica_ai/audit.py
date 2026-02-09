"""
Sistema de Auditoría Avanzado
==============================
Auditoría detallada de acciones y eventos
"""

from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum
import structlog
from collections import defaultdict

logger = structlog.get_logger()


class AuditAction(str, Enum):
    """Acciones auditables"""
    VALIDATION_CREATED = "validation_created"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_DELETED = "validation_deleted"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_REMOVED = "connection_removed"
    PROFILE_VIEWED = "profile_viewed"
    REPORT_EXPORTED = "report_exported"
    SETTINGS_CHANGED = "settings_changed"
    FEEDBACK_SUBMITTED = "feedback_submitted"


class AuditLog:
    """Entrada de auditoría"""
    
    def __init__(
        self,
        user_id: UUID,
        action: AuditAction,
        resource_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        self.id = uuid4()
        self.user_id = user_id
        self.action = action
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.details = details or {}
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "action": self.action.value,
            "resource_id": str(self.resource_id) if self.resource_id else None,
            "resource_type": self.resource_type,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp.isoformat()
        }


class AuditLogger:
    """Logger de auditoría"""
    
    def __init__(self):
        """Inicializar logger"""
        self._logs: List[AuditLog] = []
        logger.info("AuditLogger initialized")
    
    def log(
        self,
        user_id: UUID,
        action: AuditAction,
        resource_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditLog:
        """
        Registrar acción
        
        Args:
            user_id: ID del usuario
            action: Acción realizada
            resource_id: ID del recurso
            resource_type: Tipo de recurso
            details: Detalles adicionales
            ip_address: Dirección IP
            user_agent: User agent
            
        Returns:
            Log de auditoría creado
        """
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_id=resource_id,
            resource_type=resource_type,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self._logs.append(audit_log)
        
        logger.info(
            "Audit log created",
            action=action.value,
            user_id=str(user_id),
            resource_id=str(resource_id) if resource_id else None
        )
        
        return audit_log
    
    def get_logs(
        self,
        user_id: Optional[UUID] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """
        Obtener logs de auditoría
        
        Args:
            user_id: Filtrar por usuario
            action: Filtrar por acción
            resource_type: Filtrar por tipo de recurso
            start_date: Fecha de inicio
            end_date: Fecha de fin
            limit: Límite de resultados
            
        Returns:
            Lista de logs
        """
        logs = self._logs
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if action:
            logs = [l for l in logs if l.action == action]
        
        if resource_type:
            logs = [l for l in logs if l.resource_type == resource_type]
        
        if start_date:
            logs = [l for l in logs if l.timestamp >= start_date]
        
        if end_date:
            logs = [l for l in logs if l.timestamp <= end_date]
        
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]
    
    def get_audit_summary(
        self,
        user_id: Optional[UUID] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtener resumen de auditoría
        
        Args:
            user_id: Filtrar por usuario
            days: Días a analizar
            
        Returns:
            Resumen de auditoría
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        logs = self.get_logs(
            user_id=user_id,
            start_date=cutoff_date
        )
        
        # Agrupar por acción
        action_counts = defaultdict(int)
        for log in logs:
            action_counts[log.action.value] += 1
        
        # Agrupar por tipo de recurso
        resource_counts = defaultdict(int)
        for log in logs:
            if log.resource_type:
                resource_counts[log.resource_type] += 1
        
        return {
            "total_actions": len(logs),
            "period_days": days,
            "action_distribution": dict(action_counts),
            "resource_distribution": dict(resource_counts),
            "most_active_day": self._get_most_active_day(logs) if logs else None
        }
    
    def _get_most_active_day(self, logs: List[AuditLog]) -> Optional[str]:
        """Obtener día más activo"""
        if not logs:
            return None
        
        day_counts = defaultdict(int)
        for log in logs:
            day = log.timestamp.date().isoformat()
            day_counts[day] += 1
        
        return max(day_counts, key=day_counts.get)


# Instancia global del logger de auditoría
audit_logger = AuditLogger()

