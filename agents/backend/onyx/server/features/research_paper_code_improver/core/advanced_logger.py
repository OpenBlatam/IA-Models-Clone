"""
Advanced Logger - Sistema de logging avanzado
==============================================
"""

import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

logger = logging.getLogger(__name__)


class AdvancedLogger:
    """
    Sistema de logging avanzado con múltiples handlers y formateo.
    """
    
    def __init__(
        self,
        name: str = "research_paper_code_improver",
        log_dir: str = "logs",
        level: int = logging.INFO
    ):
        """
        Inicializar logger avanzado.
        
        Args:
            name: Nombre del logger
            log_dir: Directorio de logs
            level: Nivel de logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Evitar duplicados
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Configura handlers de logging"""
        # Formato estructurado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Handler para archivo (rotación por tamaño)
        file_handler = RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler para errores (rotación diaria)
        error_handler = TimedRotatingFileHandler(
            self.log_dir / "errors.log",
            when='midnight',
            interval=1,
            backupCount=30
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        
        # Handler para auditoría
        audit_handler = TimedRotatingFileHandler(
            self.log_dir / "audit.log",
            when='midnight',
            interval=1,
            backupCount=90
        )
        audit_handler.setLevel(logging.INFO)
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        audit_handler.setFormatter(audit_formatter)
        self.logger.addHandler(audit_handler)
    
    def log_operation(
        self,
        operation: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Registra una operación para auditoría.
        
        Args:
            operation: Nombre de la operación
            user_id: ID del usuario (opcional)
            details: Detalles adicionales (opcional)
        """
        log_data = {
            "operation": operation,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.logger.info(f"AUDIT: {json.dumps(log_data)}")
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registra métricas de performance.
        
        Args:
            operation: Nombre de la operación
            duration_ms: Duración en milisegundos
            metadata: Metadata adicional
        """
        self.logger.debug(
            f"PERF: {operation} - {duration_ms:.2f}ms - {json.dumps(metadata or {})}"
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Registra error con contexto.
        
        Args:
            error: Excepción
            context: Contexto adicional
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.error(
            f"ERROR: {json.dumps(error_data)}",
            exc_info=True
        )
    
    def get_logger(self) -> logging.Logger:
        """Obtiene el logger configurado"""
        return self.logger




