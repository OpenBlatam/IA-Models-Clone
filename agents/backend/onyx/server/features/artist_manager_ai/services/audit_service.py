"""
Audit Service
=============

Servicio de auditoría de acciones.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Acciones auditables."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VIEW = "view"
    EXPORT = "export"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class AuditLog:
    """Log de auditoría."""
    id: str
    user_id: str
    artist_id: Optional[str]
    action: AuditAction
    resource_type: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = None
    success: bool = True
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "artist_id": self.artist_id,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success
        }


class AuditService:
    """Servicio de auditoría."""
    
    def __init__(self):
        """Inicializar servicio de auditoría."""
        self.logs: List[AuditLog] = []
        self._logger = logger
    
    def log(
        self,
        user_id: str,
        action: AuditAction,
        resource_type: str,
        artist_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True
    ) -> AuditLog:
        """
        Registrar acción de auditoría.
        
        Args:
            user_id: ID del usuario
            action: Acción realizada
            resource_type: Tipo de recurso
            artist_id: ID del artista (opcional)
            resource_id: ID del recurso (opcional)
            details: Detalles adicionales
            ip_address: IP del usuario
            user_agent: User agent
            success: Si fue exitosa
        
        Returns:
            Log de auditoría creado
        """
        import uuid
        
        log = AuditLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            artist_id=artist_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
        
        self.logs.append(log)
        
        # Mantener solo últimos 10000 logs
        if len(self.logs) > 10000:
            self.logs = self.logs[-10000:]
        
        self._logger.info(
            f"Audit log: {action.value} on {resource_type} by {user_id}",
            extra={"audit_log": log.to_dict()}
        )
        
        return log
    
    def get_logs(
        self,
        user_id: Optional[str] = None,
        artist_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Obtener logs de auditoría.
        
        Args:
            user_id: Filtrar por usuario
            artist_id: Filtrar por artista
            action: Filtrar por acción
            resource_type: Filtrar por tipo de recurso
            limit: Límite de resultados
        
        Returns:
            Lista de logs
        """
        logs = self.logs
        
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        
        if artist_id:
            logs = [log for log in logs if log.artist_id == artist_id]
        
        if action:
            logs = [log for log in logs if log.action == action]
        
        if resource_type:
            logs = [log for log in logs if log.resource_type == resource_type]
        
        # Ordenar por timestamp descendente
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return logs[:limit]
    
    def get_user_activity(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Obtener actividad de usuario.
        
        Args:
            user_id: ID del usuario
            days: Días hacia atrás
        
        Returns:
            Resumen de actividad
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        user_logs = [
            log for log in self.logs
            if log.user_id == user_id and log.timestamp >= cutoff
        ]
        
        # Agrupar por acción
        by_action = {}
        for log in user_logs:
            action = log.action.value
            if action not in by_action:
                by_action[action] = 0
            by_action[action] += 1
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_actions": len(user_logs),
            "actions_by_type": by_action,
            "success_rate": sum(1 for log in user_logs if log.success) / len(user_logs) if user_logs else 0
        }




