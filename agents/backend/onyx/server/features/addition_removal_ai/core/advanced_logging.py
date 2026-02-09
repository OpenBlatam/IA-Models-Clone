"""
Advanced Logging - Sistema de logging avanzado
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Logger estructurado con formato JSON"""

    def __init__(self, name: str, log_dir: Optional[Path] = None):
        """
        Inicializar logger estructurado.

        Args:
            name: Nombre del logger
            log_dir: Directorio de logs
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_handlers()

    def _setup_handlers(self):
        """Configurar handlers de logging"""
        # Handler para archivo con rotación
        file_handler = RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # Formato estructurado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_operation(
        self,
        operation: str,
        details: Dict[str, Any],
        level: str = "info"
    ):
        """
        Registrar una operación.

        Args:
            operation: Nombre de la operación
            details: Detalles
            level: Nivel de log
        """
        log_data = {
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            **details
        }
        
        log_message = json.dumps(log_data)
        
        if level == "error":
            self.logger.error(log_message)
        elif level == "warning":
            self.logger.warning(log_message)
        elif level == "debug":
            self.logger.debug(log_message)
        else:
            self.logger.info(log_message)

    def log_performance(
        self,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar métricas de rendimiento.

        Args:
            operation: Operación
            duration: Duración en segundos
            metadata: Metadatos adicionales
        """
        self.log_operation(
            "performance",
            {
                "operation": operation,
                "duration": duration,
                "metadata": metadata or {}
            },
            "info"
        )

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar un error.

        Args:
            error: Excepción
            context: Contexto adicional
        """
        self.log_operation(
            "error",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {}
            },
            "error"
        )


class AuditLogger:
    """Logger de auditoría"""

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Inicializar logger de auditoría.

        Args:
            log_dir: Directorio de logs
        """
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Handler específico para auditoría
        audit_file = TimedRotatingFileHandler(
            self.log_dir / "audit.log",
            when='midnight',
            interval=1,
            backupCount=30
        )
        audit_file.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        audit_file.setFormatter(formatter)
        
        self.audit_logger.addHandler(audit_file)

    def log_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar una acción de auditoría.

        Args:
            user_id: ID del usuario
            action: Acción realizada
            resource: Recurso afectado
            details: Detalles adicionales
        """
        audit_entry = {
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        self.audit_logger.info(json.dumps(audit_entry))






