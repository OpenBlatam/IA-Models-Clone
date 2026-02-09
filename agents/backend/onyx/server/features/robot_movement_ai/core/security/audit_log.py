"""
Audit Log System
================

Sistema de registro de auditoría.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Nivel de auditoría."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditAction(Enum):
    """Acción de auditoría."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_CHANGE = "permission_change"


@dataclass
class AuditEntry:
    """Entrada de auditoría."""
    entry_id: str
    timestamp: str
    user_id: Optional[str]
    action: AuditAction
    resource_type: str
    resource_id: Optional[str]
    level: AuditLevel
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuditLogger:
    """
    Logger de auditoría.
    
    Registra todas las acciones importantes del sistema.
    """
    
    def __init__(self, log_file: str = "data/audit.log"):
        """
        Inicializar logger de auditoría.
        
        Args:
            log_file: Archivo de log
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[AuditEntry] = []
        self.max_entries = 10000
    
    def log(
        self,
        action: AuditAction,
        resource_type: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        level: AuditLevel = AuditLevel.INFO,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditEntry:
        """
        Registrar entrada de auditoría.
        
        Args:
            action: Acción realizada
            resource_type: Tipo de recurso
            user_id: ID del usuario
            resource_id: ID del recurso
            level: Nivel de auditoría
            message: Mensaje
            metadata: Metadata adicional
            ip_address: Dirección IP
            user_agent: User agent
            
        Returns:
            Entrada de auditoría creada
        """
        entry_id = f"audit_{len(self.entries)}"
        timestamp = datetime.now().isoformat()
        
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            level=level,
            message=message or f"{action.value} {resource_type}",
            metadata=metadata or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.entries.append(entry)
        
        # Limitar tamaño
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Escribir a archivo
        self._write_to_file(entry)
        
        logger.info(f"Audit log: {action.value} {resource_type} by {user_id}")
        
        return entry
    
    def _write_to_file(self, entry: AuditEntry) -> None:
        """Escribir entrada a archivo."""
        try:
            log_line = json.dumps({
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "user_id": entry.user_id,
                "action": entry.action.value,
                "resource_type": entry.resource_type,
                "resource_id": entry.resource_id,
                "level": entry.level.value,
                "message": entry.message,
                "metadata": entry.metadata,
                "ip_address": entry.ip_address,
                "user_agent": entry.user_agent
            }, ensure_ascii=False)
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line + '\n')
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
    
    def query(
        self,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        level: Optional[AuditLevel] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """
        Consultar entradas de auditoría.
        
        Args:
            user_id: Filtrar por usuario
            action: Filtrar por acción
            resource_type: Filtrar por tipo de recurso
            level: Filtrar por nivel
            start_date: Fecha de inicio
            end_date: Fecha de fin
            limit: Límite de resultados
            
        Returns:
            Lista de entradas
        """
        results = self.entries
        
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if action:
            results = [e for e in results if e.action == action]
        if resource_type:
            results = [e for e in results if e.resource_type == resource_type]
        if level:
            results = [e for e in results if e.level == level]
        if start_date:
            results = [e for e in results if e.timestamp >= start_date]
        if end_date:
            results = [e for e in results if e.timestamp <= end_date]
        
        # Ordenar por timestamp (más reciente primero)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de auditoría."""
        if not self.entries:
            return {
                "total_entries": 0,
                "by_action": {},
                "by_level": {},
                "by_resource_type": {}
            }
        
        by_action = {}
        by_level = {}
        by_resource_type = {}
        
        for entry in self.entries:
            by_action[entry.action.value] = by_action.get(entry.action.value, 0) + 1
            by_level[entry.level.value] = by_level.get(entry.level.value, 0) + 1
            by_resource_type[entry.resource_type] = by_resource_type.get(entry.resource_type, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "by_action": by_action,
            "by_level": by_level,
            "by_resource_type": by_resource_type
        }


# Instancia global
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_file: str = "data/audit.log") -> AuditLogger:
    """Obtener instancia global del logger de auditoría."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_file=log_file)
    return _audit_logger






