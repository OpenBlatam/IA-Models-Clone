"""
Security Audit - Sistema de Auditoría de Seguridad
==================================================

Sistema para registrar y auditar eventos de seguridad.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Tipos de eventos de auditoría"""
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    TOKEN_CREATED = "token_created"
    TOKEN_REVOKED = "token_revoked"
    PERMISSION_DENIED = "permission_denied"
    SECURITY_VIOLATION = "security_violation"
    COMMAND_EXECUTED = "command_executed"
    CONFIG_CHANGED = "config_changed"
    DATA_ACCESSED = "data_accessed"
    DATA_MODIFIED = "data_modified"


@dataclass
class AuditEvent:
    """Evento de auditoría"""
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "username": self.username,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource": self.resource,
            "action": self.action,
            "success": self.success,
            "details": self.details,
            "severity": self.severity
        }


class SecurityAuditor:
    """
    Auditor de seguridad.
    
    Registra eventos de seguridad para auditoría y análisis.
    """
    
    def __init__(self, max_events: int = 100000):
        self.events: List[AuditEvent] = []
        self.max_events = max_events
        self.enabled = True
    
    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ) -> None:
        """
        Registrar evento de auditoría.
        
        Args:
            event_type: Tipo de evento
            user_id: ID de usuario
            username: Nombre de usuario
            ip_address: Dirección IP
            user_agent: User agent
            resource: Recurso afectado
            action: Acción realizada
            success: Si fue exitoso
            details: Detalles adicionales
            severity: Severidad (info, warning, error, critical)
        """
        if not self.enabled:
            return
        
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            success=success,
            details=details or {},
            severity=severity
        )
        
        self.events.append(event)
        
        # Limpiar eventos antiguos si excede el límite
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Log según severidad
        log_method = {
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
            "critical": logger.critical
        }.get(severity, logger.info)
        
        log_method(
            f"🔒 Audit: {event_type.value} - "
            f"User: {username or 'unknown'} - "
            f"Resource: {resource or 'N/A'} - "
            f"Success: {success}",
            extra=event.to_dict()
        )
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Obtener eventos de auditoría con filtros.
        
        Args:
            event_type: Filtrar por tipo
            user_id: Filtrar por usuario
            since: Filtrar desde fecha
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        filtered = self.events
        
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        
        if since:
            filtered = [e for e in filtered if e.timestamp >= since]
        
        # Ordenar por timestamp (más recientes primero)
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered[:limit]
    
    def get_security_violations(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Obtener violaciones de seguridad.
        
        Args:
            since: Filtrar desde fecha
            limit: Límite de resultados
            
        Returns:
            Lista de eventos de seguridad
        """
        return self.get_events(
            event_type=AuditEventType.SECURITY_VIOLATION,
            since=since,
            limit=limit
        )
    
    def get_failed_auth_attempts(
        self,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Obtener intentos de autenticación fallidos.
        
        Args:
            user_id: Filtrar por usuario
            since: Filtrar desde fecha
            limit: Límite de resultados
            
        Returns:
            Lista de eventos
        """
        return [
            e for e in self.get_events(
                event_type=AuditEventType.AUTH_FAILURE,
                user_id=user_id,
                since=since,
                limit=limit
            )
            if not e.success
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de auditoría.
        
        Returns:
            Diccionario con estadísticas
        """
        total = len(self.events)
        
        if total == 0:
            return {
                "total_events": 0,
                "by_type": {},
                "by_severity": {},
                "success_rate": 0.0
            }
        
        by_type = {}
        by_severity = {}
        successful = 0
        
        for event in self.events:
            # Por tipo
            event_type = event.event_type.value
            by_type[event_type] = by_type.get(event_type, 0) + 1
            
            # Por severidad
            by_severity[event.severity] = by_severity.get(event.severity, 0) + 1
            
            # Exitosos
            if event.success:
                successful += 1
        
        return {
            "total_events": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "success_rate": successful / total if total > 0 else 0.0,
            "oldest_event": self.events[0].timestamp.isoformat() if self.events else None,
            "newest_event": self.events[-1].timestamp.isoformat() if self.events else None
        }
    
    def clear_events(self) -> None:
        """Limpiar todos los eventos"""
        self.events.clear()
        logger.info("🧹 Audit events cleared")




