"""
Advanced Logging Service
========================

Sistema de logging avanzado con rotación y niveles.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime


class LoggingService:
    """Servicio de logging avanzado"""
    
    @staticmethod
    def setup_logging(
        log_dir: str = "logs",
        log_level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """Configurar sistema de logging"""
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Formato de logs
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        )
        
        # Handler para archivo con rotación
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(log_format)
        
        # Handler para errores
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "errors.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(log_format)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        
        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
        
        logging.info("Logging system configured")




