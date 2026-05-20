"""
Sistema de logging avanzado
"""

import logging
import sys
from typing import Optional, Dict
from datetime import datetime
from pathlib import Path
import json
from enum import Enum


class LogLevel(str, Enum):
    """Niveles de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AdvancedLogger:
    """Logger avanzado con múltiples handlers"""
    
    def __init__(self, name: str, log_dir: str = "logs",
                 level: LogLevel = LogLevel.INFO,
                 enable_file: bool = True,
                 enable_console: bool = True,
                 enable_json: bool = False):
        """
        Inicializa el logger avanzado
        
        Args:
            name: Nombre del logger
            log_dir: Directorio de logs
            level: Nivel de log
            enable_file: Habilitar logs en archivo
            enable_console: Habilitar logs en consola
            enable_json: Habilitar logs en formato JSON
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Limpiar handlers existentes
        self.logger.handlers.clear()
        
        # Formato estándar
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler de consola
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Handler de archivo
        if enable_file:
            log_file = self.log_dir / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Handler JSON
        if enable_json:
            json_file = self.log_dir / f"{name}.json.log"
            json_handler = logging.FileHandler(json_file)
            json_handler.setFormatter(self._json_formatter)
            self.logger.addHandler(json_handler)
    
    def _json_formatter(self, record: logging.LogRecord) -> str:
        """Formatea log como JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.logger.formatException(record.exc_info)
        
        return json.dumps(log_data)
    
    def debug(self, message: str, **kwargs):
        """Log debug"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical"""
        self.logger.critical(message, **kwargs)
    
    def log_api_request(self, endpoint: str, method: str, status_code: int,
                       duration: float, user_id: Optional[str] = None):
        """Log de request de API"""
        self.info(
            f"API Request: {method} {endpoint} - Status: {status_code} - "
            f"Duration: {duration:.3f}s",
            extra={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration": duration,
                "user_id": user_id
            }
        )
    
    def log_analysis(self, analysis_id: str, user_id: str, duration: float):
        """Log de análisis"""
        self.info(
            f"Analysis completed: {analysis_id} - User: {user_id} - "
            f"Duration: {duration:.3f}s",
            extra={
                "analysis_id": analysis_id,
                "user_id": user_id,
                "duration": duration
            }
        )






