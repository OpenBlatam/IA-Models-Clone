"""
Servicio de auditoría para registro estructurado de eventos importantes.
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings
from config.di_setup import get_service

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    """Tipos de eventos de auditoría."""
    TASK_CREATED = "task_created"
    TASK_UPDATED = "task_updated"
    TASK_DELETED = "task_deleted"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_PAUSED = "agent_paused"
    AGENT_RESUMED = "agent_resumed"
    USER_ACTION = "user_action"
    CONFIG_CHANGED = "config_changed"
    SECURITY_EVENT = "security_event"
    ERROR = "error"
    WARNING = "warning"


class AuditService:
    """
    Servicio para registro de auditoría estructurado con mejoras.
    
    Attributes:
        audit_log_path: Ruta al archivo de log de auditoría
        events: Lista de eventos en memoria
        max_memory_events: Número máximo de eventos a mantener en memoria
    """
    
    def __init__(
        self,
        audit_log_path: Optional[str] = None,
        max_memory_events: int = 1000
    ):
        """
        Inicializar servicio de auditoría con validaciones.
        
        Args:
            audit_log_path: Ruta al archivo de log de auditoría (opcional)
            max_memory_events: Número máximo de eventos en memoria (debe ser entero positivo)
            
        Raises:
            ValueError: Si max_memory_events es inválido
        """
        # Validación
        if not isinstance(max_memory_events, int) or max_memory_events < 1:
            raise ValueError(
                f"max_memory_events debe ser un entero positivo, recibido: {max_memory_events}"
            )
        
        self.audit_log_path = Path(audit_log_path or settings.LOGS_STORAGE_PATH) / "audit.log"
        
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error al crear directorio de auditoría: {e}", exc_info=True)
            raise ValueError(f"No se puede crear directorio de auditoría: {e}") from e
        
        self.events: List[Dict[str, Any]] = []
        self.max_memory_events = max_memory_events
        
        logger.info(
            f"✅ AuditService inicializado: log_path={self.audit_log_path}, "
            f"max_memory_events={max_memory_events}"
        )
    
    def log_event(
        self,
        event_type: AuditEventType,
        details: Dict[str, Any],
        user: Optional[str] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> None:
        """
        Registrar evento de auditoría con validaciones.
        
        Args:
            event_type: Tipo de evento (debe ser AuditEventType)
            details: Detalles del evento (debe ser diccionario)
            user: Usuario que realizó la acción (opcional, debe ser string si se proporciona)
            ip_address: Dirección IP (opcional, debe ser string si se proporciona)
            request_id: ID de la request (opcional, debe ser string si se proporciona)
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(event_type, AuditEventType):
            raise ValueError(f"event_type debe ser un AuditEventType, recibido: {type(event_type)}")
        
        if not isinstance(details, dict):
            raise ValueError(f"details debe ser un diccionario, recibido: {type(details)}")
        
        if user is not None:
            if not isinstance(user, str) or not user.strip():
                raise ValueError(f"user debe ser un string no vacío si se proporciona, recibido: {user}")
            user = user.strip()
        
        if ip_address is not None:
            if not isinstance(ip_address, str) or not ip_address.strip():
                raise ValueError(f"ip_address debe ser un string no vacío si se proporciona, recibido: {ip_address}")
            ip_address = ip_address.strip()
        
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError(f"request_id debe ser un string no vacío si se proporciona, recibido: {request_id}")
            request_id = request_id.strip()
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type.value,
            "user": user,
            "ip_address": ip_address,
            "request_id": request_id,
            "details": details
        }
        
        # Agregar a memoria
        self.events.append(event)
        if len(self.events) > self.max_memory_events:
            removed = self.events.pop(0)
            logger.debug(f"Evento antiguo removido de memoria: {removed.get('event_type')}")
        
        # Escribir a archivo (JSON lines format)
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            logger.debug(f"Evento de auditoría escrito a archivo: {event_type.value}")
        except Exception as e:
            logger.error(f"❌ Error escribiendo evento de auditoría: {e}", exc_info=True)
            # No re-raise para no interrumpir el flujo, pero loguear el error
        
        # Log estructurado
        logger.info(
            f"📋 Audit: {event_type.value}",
            extra={
                "audit_event": True,
                "event_type": event_type.value,
                "user": user,
                "ip_address": ip_address,
                "request_id": request_id,
                **details
            }
        )
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener eventos de auditoría con validaciones.
        
        Args:
            event_type: Filtrar por tipo de evento (opcional, debe ser AuditEventType)
            user: Filtrar por usuario (opcional, debe ser string si se proporciona)
            limit: Número máximo de eventos a retornar (debe ser entero positivo)
        
        Returns:
            Lista de eventos de auditoría
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if event_type is not None and not isinstance(event_type, AuditEventType):
            raise ValueError(f"event_type debe ser un AuditEventType, recibido: {type(event_type)}")
        
        if user is not None:
            if not isinstance(user, str) or not user.strip():
                raise ValueError(f"user debe ser un string no vacío si se proporciona, recibido: {user}")
            user = user.strip()
        
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit debe ser un entero positivo, recibido: {limit}")
        
        # Limitar a máximo razonable
        if limit > 10000:
            logger.warning(f"limit muy alto ({limit}), limitando a 10000")
            limit = 10000
        
        filtered = self.events
        
        if event_type:
            filtered = [e for e in filtered if e["event_type"] == event_type.value]
        
        if user:
            filtered = [e for e in filtered if e.get("user") == user]
        
        # Ordenar por timestamp (más recientes primero)
        filtered.sort(key=lambda x: x["timestamp"], reverse=True)
        
        result = filtered[:limit]
        logger.debug(
            f"Obtenidos {len(result)} eventos de auditoría "
            f"(filtros: event_type={event_type.value if event_type else None}, "
            f"user={user}, limit={limit})"
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de auditoría.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "total_events": len(self.events),
            "events_by_type": {},
            "latest_event": self.events[-1] if self.events else None,
            "log_file": str(self.audit_log_path),
            "log_file_exists": self.audit_log_path.exists()
        }
        
        # Contar por tipo
        for event in self.events:
            event_type = event["event_type"]
            stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
        
        return stats

