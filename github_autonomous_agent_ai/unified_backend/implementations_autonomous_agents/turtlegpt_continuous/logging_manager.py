"""
Logging Manager Module
======================

Gestión centralizada de logging del agente.
Proporciona logging estructurado, formateo, niveles configurables y handlers.
"""

import logging
import sys
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Niveles de logging."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """Configuración de logging."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[Path] = None
    file_mode: str = "a"
    console_enabled: bool = True
    file_enabled: bool = False
    structured: bool = False
    include_context: bool = True
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class LogEntry:
    """Entrada de log estructurada."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "logger": self.logger_name,
            "message": self.message,
            "context": self.context,
            "exception": self.exception
        }


class StructuredFormatter(logging.Formatter):
    """Formatter para logs estructurados."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear registro de log."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Agregar contexto si existe
        if hasattr(record, "context") and record.context:
            log_data["context"] = record.context
        
        # Agregar excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Agregar campos adicionales
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", 
                          "funcName", "levelname", "levelno", "lineno", 
                          "module", "msecs", "message", "pathname", "process",
                          "processName", "relativeCreated", "thread", "threadName",
                          "exc_info", "exc_text", "stack_info", "context"]:
                log_data[key] = value
        
        import json
        return json.dumps(log_data, default=str)


class LoggingManager:
    """
    Gestor centralizado de logging.
    
    Proporciona funcionalidades para:
    - Configuración centralizada de logging
    - Formateo estructurado
    - Múltiples handlers (console, file, etc.)
    - Rotación de logs
    - Filtrado y niveles configurables
    - Contexto estructurado
    """
    
    def __init__(self, config: Optional[LogConfig] = None):
        """
        Inicializar gestor de logging.
        
        Args:
            config: Configuración de logging
        """
        self.config = config or LogConfig()
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: List[logging.Handler] = []
        self.log_history: List[LogEntry] = []
        self.max_history_size = 1000
        
        # Configurar root logger
        self._setup_root_logger()
    
    def _setup_root_logger(self) -> None:
        """Configurar root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.level.value)
        
        # Limpiar handlers existentes
        root_logger.handlers.clear()
        
        # Crear formatter
        if self.config.structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                self.config.format,
                datefmt=self.config.date_format
            )
        
        # Handler de consola
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.config.level.value)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            self.handlers.append(console_handler)
        
        # Handler de archivo
        if self.config.file_enabled and self.config.file_path:
            self._setup_file_handler(formatter)
    
    def _setup_file_handler(self, formatter: logging.Formatter) -> None:
        """Configurar handler de archivo con rotación."""
        try:
            from logging.handlers import RotatingFileHandler
            
            # Crear directorio si no existe
            self.config.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                filename=str(self.config.file_path),
                mode=self.config.file_mode,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.config.level.value)
            file_handler.setFormatter(formatter)
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            self.handlers.append(file_handler)
            
        except ImportError:
            logger.warning("RotatingFileHandler not available, using basic FileHandler")
            file_handler = logging.FileHandler(
                str(self.config.file_path),
                mode=self.config.file_mode,
                encoding='utf-8'
            )
            file_handler.setLevel(self.config.level.value)
            file_handler.setFormatter(formatter)
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            self.handlers.append(file_handler)
    
    def get_logger(self, name: str, level: Optional[LogLevel] = None) -> logging.Logger:
        """
        Obtener logger con nombre específico.
        
        Args:
            name: Nombre del logger
            level: Nivel de logging (opcional)
            
        Returns:
            Logger configurado
        """
        if name not in self.loggers:
            logger_instance = logging.getLogger(name)
            if level:
                logger_instance.setLevel(level.value)
            self.loggers[name] = logger_instance
        
        return self.loggers[name]
    
    def set_level(self, level: LogLevel, logger_name: Optional[str] = None) -> None:
        """
        Establecer nivel de logging.
        
        Args:
            level: Nivel de logging
            logger_name: Nombre del logger (None para root)
        """
        target_logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        target_logger.setLevel(level.value)
        
        # Actualizar handlers
        for handler in target_logger.handlers:
            handler.setLevel(level.value)
    
    def log_with_context(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        logger_name: str = __name__
    ) -> None:
        """
        Loggear con contexto estructurado.
        
        Args:
            level: Nivel de logging
            message: Mensaje
            context: Contexto adicional
            logger_name: Nombre del logger
        """
        logger_instance = self.get_logger(logger_name)
        
        # Crear registro con contexto
        extra = {}
        if context and self.config.include_context:
            extra["context"] = context
        
        logger_instance.log(level.value, message, extra=extra)
        
        # Agregar al historial
        self._add_to_history(level, message, context, logger_name)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, logger_name: str = __name__) -> None:
        """Loggear mensaje de debug."""
        self.log_with_context(LogLevel.DEBUG, message, context, logger_name)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None, logger_name: str = __name__) -> None:
        """Loggear mensaje de info."""
        self.log_with_context(LogLevel.INFO, message, context, logger_name)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, logger_name: str = __name__) -> None:
        """Loggear mensaje de warning."""
        self.log_with_context(LogLevel.WARNING, message, context, logger_name)
    
    def error(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        logger_name: str = __name__
    ) -> None:
        """Loggear mensaje de error."""
        logger_instance = self.get_logger(logger_name)
        
        extra = {}
        if context and self.config.include_context:
            extra["context"] = context
        
        if exception:
            logger_instance.error(message, exc_info=exception, extra=extra)
            self._add_to_history(LogLevel.ERROR, message, context, logger_name, str(exception))
        else:
            logger_instance.error(message, extra=extra)
            self._add_to_history(LogLevel.ERROR, message, context, logger_name)
    
    def critical(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        logger_name: str = __name__
    ) -> None:
        """Loggear mensaje crítico."""
        logger_instance = self.get_logger(logger_name)
        
        extra = {}
        if context and self.config.include_context:
            extra["context"] = context
        
        if exception:
            logger_instance.critical(message, exc_info=exception, extra=extra)
            self._add_to_history(LogLevel.CRITICAL, message, context, logger_name, str(exception))
        else:
            logger_instance.critical(message, extra=extra)
            self._add_to_history(LogLevel.CRITICAL, message, context, logger_name)
    
    def add_filter(self, filter_func: Callable[[logging.LogRecord], bool], logger_name: Optional[str] = None) -> None:
        """
        Agregar filtro personalizado.
        
        Args:
            filter_func: Función de filtrado
            logger_name: Nombre del logger (None para root)
        """
        class CustomFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return filter_func(record)
        
        target_logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        custom_filter = CustomFilter()
        target_logger.addFilter(custom_filter)
    
    def get_history(
        self,
        level: Optional[LogLevel] = None,
        limit: Optional[int] = None,
        logger_name: Optional[str] = None
    ) -> List[LogEntry]:
        """
        Obtener historial de logs.
        
        Args:
            level: Filtrar por nivel
            limit: Límite de entradas
            logger_name: Filtrar por logger
            
        Returns:
            Lista de entradas de log
        """
        filtered = self.log_history
        
        if level:
            filtered = [entry for entry in filtered if entry.level == level.name]
        
        if logger_name:
            filtered = [entry for entry in filtered if entry.logger_name == logger_name]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def clear_history(self) -> None:
        """Limpiar historial de logs."""
        self.log_history.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de logging.
        
        Returns:
            Dict con estadísticas
        """
        stats = {
            "total_logs": len(self.log_history),
            "by_level": {},
            "by_logger": {},
            "handlers_count": len(self.handlers),
            "loggers_count": len(self.loggers)
        }
        
        # Contar por nivel
        for entry in self.log_history:
            stats["by_level"][entry.level] = stats["by_level"].get(entry.level, 0) + 1
            stats["by_logger"][entry.logger_name] = stats["by_logger"].get(entry.logger_name, 0) + 1
        
        return stats
    
    def export_history(self, file_path: Path, format: str = "json") -> bool:
        """
        Exportar historial de logs a archivo.
        
        Args:
            file_path: Ruta del archivo
            format: Formato (json, csv, txt)
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            if format == "json":
                import json
                data = [entry.to_dict() for entry in self.log_history]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            
            elif format == "csv":
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    if self.log_history:
                        writer = csv.DictWriter(f, fieldnames=self.log_history[0].to_dict().keys())
                        writer.writeheader()
                        for entry in self.log_history:
                            writer.writerow(entry.to_dict())
            
            elif format == "txt":
                with open(file_path, 'w', encoding='utf-8') as f:
                    for entry in self.log_history:
                        f.write(f"[{entry.timestamp}] {entry.level} - {entry.logger_name}: {entry.message}\n")
                        if entry.context:
                            f.write(f"  Context: {entry.context}\n")
                        if entry.exception:
                            f.write(f"  Exception: {entry.exception}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting log history: {e}", exc_info=True)
            return False
    
    def _add_to_history(
        self,
        level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]],
        logger_name: str,
        exception: Optional[str] = None
    ) -> None:
        """Agregar entrada al historial."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level.name,
            logger_name=logger_name,
            message=message,
            context=context or {},
            exception=exception
        )
        
        self.log_history.append(entry)
        
        # Limitar tamaño del historial
        if len(self.log_history) > self.max_history_size:
            self.log_history = self.log_history[-self.max_history_size:]
    
    def update_config(self, config: LogConfig) -> None:
        """
        Actualizar configuración de logging.
        
        Args:
            config: Nueva configuración
        """
        self.config = config
        self._setup_root_logger()


def create_logging_manager(
    level: LogLevel = LogLevel.INFO,
    file_path: Optional[Path] = None,
    structured: bool = False,
    console_enabled: bool = True,
    file_enabled: bool = False
) -> LoggingManager:
    """
    Factory function para crear LoggingManager.
    
    Args:
        level: Nivel de logging
        file_path: Ruta del archivo de log
        structured: Usar formato estructurado
        console_enabled: Habilitar logging a consola
        file_enabled: Habilitar logging a archivo
        
    Returns:
        Instancia de LoggingManager
    """
    config = LogConfig(
        level=level,
        structured=structured,
        console_enabled=console_enabled,
        file_enabled=file_enabled,
        file_path=file_path
    )
    
    return LoggingManager(config)
