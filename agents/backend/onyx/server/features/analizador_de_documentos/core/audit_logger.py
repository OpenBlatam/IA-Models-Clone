"""
Sistema de Auditoría
====================

Sistema para registrar y auditar todas las acciones del sistema.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Tipos de acciones"""
    ANALYSIS = "analysis"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    WORKFLOW_EXECUTE = "workflow_execute"
    MODEL_TRAIN = "model_train"
    MODEL_LOAD = "model_load"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"


@dataclass
class AuditLog:
    """Entrada de auditoría"""
    action_type: ActionType
    user_id: Optional[str]
    document_id: Optional[str]
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: str = None
    success: bool = True
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AuditLogger:
    """
    Logger de auditoría
    
    Registra todas las acciones importantes del sistema:
    - Análisis de documentos
    - Operaciones de usuarios
    - Eventos del sistema
    - Errores
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Inicializar logger de auditoría
        
        Args:
            log_dir: Directorio para logs de auditoría
        """
        if log_dir is None:
            log_dir = os.path.join(
                Path(__file__).parent.parent.parent,
                "logs",
                "audit"
            )
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs: List[AuditLog] = []
        logger.info(f"AuditLogger inicializado: {log_dir}")
    
    def log(
        self,
        action_type: ActionType,
        action: str,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True
    ):
        """
        Registrar acción
        
        Args:
            action_type: Tipo de acción
            action: Descripción de la acción
            user_id: ID del usuario
            document_id: ID del documento
            details: Detalles adicionales
            ip_address: Dirección IP
            user_agent: User agent
            success: Si la acción fue exitosa
        """
        log_entry = AuditLog(
            action_type=action_type,
            user_id=user_id,
            document_id=document_id,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
        
        self.logs.append(log_entry)
        
        # Guardar en archivo
        self._save_log(log_entry)
        
        logger.debug(f"Audit log: {action_type.value} - {action}")
    
    def _save_log(self, log_entry: AuditLog):
        """Guardar log en archivo"""
        try:
            # Archivo por fecha
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"audit_{date_str}.jsonl"
            
            # Convertir a dict
            log_dict = {
                "action_type": log_entry.action_type.value,
                "user_id": log_entry.user_id,
                "document_id": log_entry.document_id,
                "action": log_entry.action,
                "details": log_entry.details,
                "ip_address": log_entry.ip_address,
                "user_agent": log_entry.user_agent,
                "timestamp": log_entry.timestamp,
                "success": log_entry.success
            }
            
            # Agregar a archivo JSONL
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error guardando log de auditoría: {e}")
    
    def get_logs(
        self,
        action_type: Optional[ActionType] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtener logs de auditoría
        
        Args:
            action_type: Filtrar por tipo
            user_id: Filtrar por usuario
            document_id: Filtrar por documento
            limit: Límite de resultados
        
        Returns:
            Lista de logs
        """
        logs = self.logs
        
        if action_type:
            logs = [l for l in logs if l.action_type == action_type]
        
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        
        if document_id:
            logs = [l for l in logs if l.document_id == document_id]
        
        return [
            {
                "action_type": l.action_type.value,
                "user_id": l.user_id,
                "document_id": l.document_id,
                "action": l.action,
                "timestamp": l.timestamp,
                "success": l.success
            }
            for l in logs[-limit:]
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de auditoría"""
        total_logs = len(self.logs)
        
        by_type = {}
        for log in self.logs:
            action_type = log.action_type.value
            by_type[action_type] = by_type.get(action_type, 0) + 1
        
        success_count = sum(1 for l in self.logs if l.success)
        failure_count = total_logs - success_count
        
        return {
            "total_logs": total_logs,
            "by_type": by_type,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / total_logs if total_logs > 0 else 0.0
        }


# Importar os
import os

# Instancia global
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Obtener instancia global del logger de auditoría"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
















