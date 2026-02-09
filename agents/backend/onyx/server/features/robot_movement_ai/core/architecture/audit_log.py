"""
Sistema de auditoría para Robot Movement AI v2.0
Registro de todas las acciones importantes del sistema
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class AuditAction(str, Enum):
    """Tipos de acciones auditables"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    EXECUTE = "execute"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_DENIED = "permission_denied"
    ERROR = "error"


@dataclass
class AuditLog:
    """Entrada de log de auditoría"""
    id: str
    action: AuditAction
    resource_type: str
    resource_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


class AuditLogger:
    """Logger de auditoría"""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        """
        Inicializar logger de auditoría
        
        Args:
            storage_backend: Backend de almacenamiento (opcional)
        """
        self.storage_backend = storage_backend
        self.logs: List[AuditLog] = []
        self.max_memory_logs: int = 10000
    
    def log(
        self,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Registrar acción de auditoría
        
        Args:
            action: Tipo de acción
            resource_type: Tipo de recurso
            resource_id: ID del recurso
            user_id: ID del usuario
            ip_address: Dirección IP
            user_agent: User agent
            details: Detalles adicionales
            success: Si la acción fue exitosa
            error_message: Mensaje de error si falló
            
        Returns:
            Log de auditoría creado
        """
        import uuid
        
        log = AuditLog(
            id=str(uuid.uuid4()),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            success=success,
            error_message=error_message
        )
        
        # Agregar a memoria
        self.logs.append(log)
        if len(self.logs) > self.max_memory_logs:
            self.logs.pop(0)
        
        # Guardar en backend si está disponible
        if self.storage_backend:
            self._save_to_backend(log)
        
        return log
    
    def _save_to_backend(self, log: AuditLog):
        """Guardar log en backend de almacenamiento"""
        # Implementar según backend específico
        pass
    
    def query(
        self,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Consultar logs de auditoría
        
        Args:
            action: Filtrar por acción
            resource_type: Filtrar por tipo de recurso
            user_id: Filtrar por usuario
            start_date: Fecha de inicio
            end_date: Fecha de fin
            limit: Límite de resultados
            
        Returns:
            Lista de logs que coinciden
        """
        results = self.logs
        
        if action:
            results = [log for log in results if log.action == action]
        
        if resource_type:
            results = [log for log in results if log.resource_type == resource_type]
        
        if user_id:
            results = [log for log in results if log.user_id == user_id]
        
        if start_date:
            results = [log for log in results if log.timestamp >= start_date]
        
        if end_date:
            results = [log for log in results if log.timestamp <= end_date]
        
        # Ordenar por timestamp descendente
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results[:limit]
    
    def get_user_activity(self, user_id: str, limit: int = 100) -> List[AuditLog]:
        """Obtener actividad de un usuario"""
        return self.query(user_id=user_id, limit=limit)
    
    def get_resource_history(self, resource_type: str, resource_id: str, limit: int = 100) -> List[AuditLog]:
        """Obtener historial de un recurso"""
        results = [
            log for log in self.logs
            if log.resource_type == resource_type and log.resource_id == resource_id
        ]
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]


# Instancia global
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Obtener instancia global del logger de auditoría"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    action: AuditAction,
    resource_type: str,
    resource_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
):
    """
    Decorator para auditar funciones
    
    Usage:
        @audit_log(AuditAction.EXECUTE, "robot")
        async def move_robot(robot_id: str):
            ...
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **func_kwargs):
            logger = get_audit_logger()
            
            # Extraer información del contexto
            resource_id_value = resource_id or (args[0] if args else None)
            
            try:
                result = await func(*args, **func_kwargs)
                logger.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id_value) if resource_id_value else None,
                    user_id=user_id,
                    success=True
                )
                return result
            except Exception as e:
                logger.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id_value) if resource_id_value else None,
                    user_id=user_id,
                    success=False,
                    error_message=str(e)
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **func_kwargs):
            logger = get_audit_logger()
            
            resource_id_value = resource_id or (args[0] if args else None)
            
            try:
                result = func(*args, **func_kwargs)
                logger.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id_value) if resource_id_value else None,
                    user_id=user_id,
                    success=True
                )
                return result
            except Exception as e:
                logger.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id_value) if resource_id_value else None,
                    user_id=user_id,
                    success=False,
                    error_message=str(e)
                )
                raise
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator




