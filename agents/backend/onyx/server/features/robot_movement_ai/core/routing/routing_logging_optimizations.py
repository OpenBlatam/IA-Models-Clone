"""
Routing Logging Optimizations
==============================

Optimizaciones avanzadas de logging.
Incluye: Structured logging, Log rotation, Performance logging, etc.
"""

import logging
import logging.handlers
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class StructuredLogger:
    """Logger estructurado con formato JSON."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        Inicializar logger estructurado.
        
        Args:
            name: Nombre del logger
            log_file: Archivo de log (opcional)
        """
        self.logger = logging.getLogger(name)
        self.log_file = log_file
        self.setup_handler()
    
    def setup_handler(self):
        """Configurar handler de logging."""
        if self.log_file:
            handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
        else:
            handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_structured(self, level: str, message: str, **kwargs):
        """
        Log estructurado.
        
        Args:
            level: Nivel de log
            message: Mensaje
            **kwargs: Campos adicionales
        """
        log_data = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        log_message = json.dumps(log_data)
        
        if level == 'DEBUG':
            self.logger.debug(log_message)
        elif level == 'INFO':
            self.logger.info(log_message)
        elif level == 'WARNING':
            self.logger.warning(log_message)
        elif level == 'ERROR':
            self.logger.error(log_message)
        elif level == 'CRITICAL':
            self.logger.critical(log_message)


class PerformanceLogger:
    """Logger de rendimiento."""
    
    def __init__(self, logger: logging.Logger):
        """
        Inicializar logger de rendimiento.
        
        Args:
            logger: Logger base
        """
        self.logger = logger
        self.performance_logs: list = []
        self.lock = threading.Lock()
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        Log de rendimiento.
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
            **kwargs: Métricas adicionales
        """
        log_entry = {
            'timestamp': time.time(),
            'operation': operation,
            'duration': duration,
            **kwargs
        }
        
        with self.lock:
            self.performance_logs.append(log_entry)
            # Mantener solo últimas 1000 entradas
            if len(self.performance_logs) > 1000:
                self.performance_logs = self.performance_logs[-1000:]
        
        self.logger.info(f"Performance: {operation} took {duration:.4f}s", extra=log_entry)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento."""
        with self.lock:
            if not self.performance_logs:
                return {}
            
            operations = {}
            for entry in self.performance_logs:
                op = entry['operation']
                if op not in operations:
                    operations[op] = []
                operations[op].append(entry['duration'])
            
            stats = {}
            for op, durations in operations.items():
                stats[op] = {
                    'count': len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'mean': sum(durations) / len(durations),
                    'total': sum(durations)
                }
            
            return stats


class LoggingOptimizer:
    """Optimizador completo de logging."""
    
    def __init__(self, enable_structured: bool = True, log_file: Optional[str] = None):
        """
        Inicializar optimizador de logging.
        
        Args:
            enable_structured: Habilitar logging estructurado
            log_file: Archivo de log
        """
        self.enable_structured = enable_structured
        self.structured_logger = None
        self.performance_logger = None
        
        if enable_structured:
            self.structured_logger = StructuredLogger('routing', log_file)
        
        base_logger = logging.getLogger('routing.performance')
        self.performance_logger = PerformanceLogger(base_logger)
    
    def log(self, level: str, message: str, **kwargs):
        """Log con formato estructurado."""
        if self.structured_logger:
            self.structured_logger.log_structured(level, message, **kwargs)
        else:
            getattr(logger, level.lower())(message, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log de rendimiento."""
        if self.performance_logger:
            self.performance_logger.log_performance(operation, duration, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        stats = {
            'structured_logging_enabled': self.enable_structured
        }
        
        if self.performance_logger:
            stats['performance_stats'] = self.performance_logger.get_performance_stats()
        
        return stats

