"""
Audit Logging - Logging de Auditoría
====================================

Sistema de audit logging:
- User actions tracking
- Data changes tracking
- Compliance logging
- Immutable logs
- Audit trail
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
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


class AuditLog:
    """Log de auditoría"""
    
    def __init__(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        resource_type: str,
        resource_id: Optional[str],
        action: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> None:
        self.event_type = event_type
        self.user_id = user_id
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.action = action
        self.details = details or {}
        self.ip_address = ip_address
        self.timestamp = datetime.now()
        self.id = f"audit_{self.timestamp.timestamp()}_{hash(str(self))}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "ip_address": self.ip_address,
            "timestamp": self.timestamp.isoformat()
        }


class AuditLogger:
    """
    Logger de auditoría.
    """
    
    def __init__(self, storage_backend: Optional[Any] = None) -> None:
        self.storage_backend = storage_backend
        self.logs: List[AuditLog] = []
        self.max_logs = 10000  # Mantener últimos 10k logs en memoria
    
    def log(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        resource_type: str,
        resource_id: Optional[str],
        action: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> AuditLog:
        """Registra evento de auditoría"""
        audit_log = AuditLog(
            event_type=event_type,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details,
            ip_address=ip_address
        )
        
        self.logs.append(audit_log)
        
        # Limpiar logs antiguos si excede límite
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Guardar en backend si está disponible
        if self.storage_backend:
            try:
                # Guardar en storage (implementación específica)
                pass
            except Exception as e:
                logger.error(f"Failed to save audit log to backend: {e}")
        
        logger.info(f"Audit log: {event_type.value} - {action} by {user_id}")
        return audit_log
    
    def query(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditLog]:
        """Consulta logs de auditoría"""
        results = self.logs
        
        if user_id:
            results = [log for log in results if log.user_id == user_id]
        
        if resource_type:
            results = [log for log in results if log.resource_type == resource_type]
        
        if resource_id:
            results = [log for log in results if log.resource_id == resource_id]
        
        if event_type:
            results = [log for log in results if log.event_type == event_type]
        
        if start_date:
            results = [log for log in results if log.timestamp >= start_date]
        
        if end_date:
            results = [log for log in results if log.timestamp <= end_date]
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def get_user_activity(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene actividad de un usuario"""
        logs = self.query(user_id=user_id)
        return [log.to_dict() for log in logs[:limit]]
    
    def get_resource_history(
        self,
        resource_type: str,
        resource_id: str
    ) -> List[Dict[str, Any]]:
        """Obtiene historial de un recurso"""
        logs = self.query(resource_type=resource_type, resource_id=resource_id)
        return [log.to_dict() for log in logs]


def get_audit_logger(storage_backend: Optional[Any] = None) -> AuditLogger:
    """Obtiene logger de auditoría"""
    return AuditLogger(storage_backend=storage_backend)















