"""
Structured Logger
=================

Logger estructurado para mejor observabilidad.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


class StructuredLogger:
    """
    Logger estructurado con soporte para JSON y contexto.
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        use_json: bool = False,
        log_file: Optional[str] = None
    ):
        """
        Inicializar logger.
        
        Args:
            name: Nombre del logger
            level: Nivel de logging
            use_json: Usar formato JSON
            log_file: Archivo de log (opcional)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Configurar handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            
            if use_json:
                from .formatters import JSONFormatter
                formatter = JSONFormatter()
            else:
                from .formatters import ColoredFormatter
                formatter = ColoredFormatter()
            
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (si se especifica)
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(JSONFormatter())
                self.logger.addHandler(file_handler)
        
        # Structlog si está disponible
        if STRUCTLOG_AVAILABLE:
            self.struct_logger = structlog.get_logger(name)
        else:
            self.struct_logger = None
    
    def _log(
        self,
        level: int,
        message: str,
        **kwargs
    ):
        """Log interno."""
        extra = {
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            **kwargs
        }
        
        if self.struct_logger:
            log_func = getattr(self.struct_logger, logging.getLevelName(level).lower())
            log_func(message, **kwargs)
        else:
            self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info."""
        self._log(logging.INFO, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def with_context(self, **context):
        """
        Crear logger con contexto adicional.
        
        Args:
            **context: Contexto adicional
            
        Returns:
            Logger con contexto
        """
        class ContextLogger:
            def __init__(self, base_logger, ctx):
                self.base_logger = base_logger
                self.ctx = ctx
            
            def _log(self, level, message, **kwargs):
                self.base_logger._log(level, message, **{**self.ctx, **kwargs})
            
            def info(self, message, **kwargs):
                self._log(logging.INFO, message, **kwargs)
            
            def debug(self, message, **kwargs):
                self._log(logging.DEBUG, message, **kwargs)
            
            def warning(self, message, **kwargs):
                self._log(logging.WARNING, message, **kwargs)
            
            def error(self, message, **kwargs):
                self._log(logging.ERROR, message, **kwargs)
            
            def critical(self, message, **kwargs):
                self._log(logging.CRITICAL, message, **kwargs)
        
        return ContextLogger(self, context)


def setup_logging(
    level: str = "INFO",
    use_json: bool = False,
    log_file: Optional[str] = None
):
    """
    Configurar logging global.
    
    Args:
        level: Nivel de logging
        use_json: Usar formato JSON
        log_file: Archivo de log (opcional)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Limpiar handlers existentes
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    if use_json:
        from .formatters import JSONFormatter
        formatter = JSONFormatter()
    else:
        from .formatters import ColoredFormatter
        formatter = ColoredFormatter()
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (si se especifica)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Configurar structlog si está disponible
    if STRUCTLOG_AVAILABLE:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def get_logger(name: str, **kwargs) -> StructuredLogger:
    """
    Obtener logger.
    
    Args:
        name: Nombre del logger
        **kwargs: Argumentos adicionales
        
    Returns:
        Logger estructurado
    """
    return StructuredLogger(name, **kwargs)


# Funciones de conveniencia para logging específico
def log_route_request(logger: StructuredLogger, request: Dict[str, Any]):
    """Log de request de ruta."""
    logger.info(
        "Route request",
        start_node=request.get("start_node"),
        end_node=request.get("end_node"),
        strategy=request.get("strategy"),
        **request.get("metadata", {})
    )


def log_route_response(logger: StructuredLogger, response: Dict[str, Any]):
    """Log de response de ruta."""
    logger.info(
        "Route found",
        route_length=len(response.get("route", [])),
        confidence=response.get("confidence"),
        metrics=response.get("metrics", {})
    )


def log_training_step(
    logger: StructuredLogger,
    epoch: int,
    step: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None
):
    """Log de paso de entrenamiento."""
    logger.info(
        "Training step",
        epoch=epoch,
        step=step,
        loss=loss,
        **metrics or {}
    )


def log_inference(
    logger: StructuredLogger,
    model_name: str,
    input_shape: tuple,
    inference_time: float,
    **kwargs
):
    """Log de inferencia."""
    logger.info(
        "Inference",
        model=model_name,
        input_shape=input_shape,
        inference_time_ms=inference_time * 1000,
        **kwargs
    )


def log_error(
    logger: StructuredLogger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
):
    """Log de error."""
    logger.error(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        **context or {}
    )

