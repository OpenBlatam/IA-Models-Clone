"""
Configuración avanzada de logging para Robot Movement AI v2.0
Soporte para múltiples handlers, rotación, y formateo estructurado
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class LogLevel(str, Enum):
    """Niveles de log disponibles"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredFormatter(logging.Formatter):
    """Formatter estructurado para logs en formato JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear log record como JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Agregar excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Agregar campos extra si existen
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Formatter con colores para consola"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',        # Green
        'WARNING': '\033[33m',     # Yellow
        'ERROR': '\033[31m',       # Red
        'CRITICAL': '\033[35m',    # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear con colores"""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class LoggingConfig:
    """Configurador centralizado de logging"""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_dir: str = "logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 10,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = False,
        enable_colors: bool = True
    ):
        """
        Inicializar configuración de logging
        
        Args:
            log_level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Archivo de log (opcional, se genera automáticamente si no se especifica)
            log_dir: Directorio para logs
            max_bytes: Tamaño máximo de archivo antes de rotar
            backup_count: Número de archivos de backup a mantener
            enable_console: Habilitar logging a consola
            enable_file: Habilitar logging a archivo
            enable_json: Usar formato JSON estructurado
            enable_colors: Habilitar colores en consola
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file or f"{log_dir}/robot_movement_ai.log"
        self.log_dir = Path(log_dir)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.enable_colors = enable_colors
        
        # Crear directorio de logs si no existe
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self, logger_name: str = "robot_movement_ai") -> logging.Logger:
        """
        Configurar logging completo
        
        Args:
            logger_name: Nombre del logger principal
            
        Returns:
            Logger configurado
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)
        
        # Limpiar handlers existentes
        logger.handlers.clear()
        
        # Handler para consola
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            if self.enable_json:
                console_formatter = StructuredFormatter()
            elif self.enable_colors:
                console_formatter = ColoredFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # Handler para archivo con rotación
        if self.enable_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            
            if self.enable_json:
                file_formatter = StructuredFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Handler para errores en archivo separado
        error_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            str(error_file),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        # Configurar loggers de terceros
        self._configure_third_party_loggers()
        
        return logger
    
    def _configure_third_party_loggers(self):
        """Configurar niveles de loggers de librerías externas"""
        # Reducir verbosidad de librerías externas
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Obtener logger con nombre específico"""
        return logging.getLogger(name)


# Instancia global de configuración
_logging_config: Optional[LoggingConfig] = None


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    enable_json: bool = False,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Función helper para configurar logging
    
    Args:
        log_level: Nivel de log
        log_file: Archivo de log
        log_dir: Directorio de logs
        enable_json: Habilitar formato JSON
        enable_colors: Habilitar colores
        
    Returns:
        Logger configurado
    """
    global _logging_config
    
    _logging_config = LoggingConfig(
        log_level=log_level,
        log_file=log_file,
        log_dir=log_dir,
        enable_json=enable_json,
        enable_colors=enable_colors
    )
    
    return _logging_config.setup_logging()


def get_logger(name: str = "robot_movement_ai") -> logging.Logger:
    """Obtener logger"""
    if _logging_config is None:
        setup_logging()
    return LoggingConfig.get_logger(name)


# Context manager para logging con contexto adicional
class LoggingContext:
    """Context manager para agregar contexto a logs"""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Inicializar contexto de logging
        
        Args:
            logger: Logger a usar
            **kwargs: Campos adicionales para agregar a logs
        """
        self.logger = logger
        self.context = kwargs
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        """Entrar al contexto"""
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            record.extra_fields = self.context
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Salir del contexto"""
        logging.setLogRecordFactory(self.old_factory)
        return False


def log_with_context(logger: logging.Logger, **context):
    """
    Decorator para agregar contexto a logs en una función
    
    Usage:
        @log_with_context(logger, robot_id="robot-1")
        def move_robot():
            logger.info("Moving robot")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LoggingContext(logger, **context):
                return func(*args, **kwargs)
        return wrapper
    return decorator




