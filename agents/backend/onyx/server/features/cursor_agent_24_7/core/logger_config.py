"""
Logger Config - Configuración avanzada de logging
==================================================

Configuración profesional de logging con múltiples handlers,
soporte para JSON logging, colores, y múltiples backends.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Any

# Intentar usar structlog si está disponible
try:
    import structlog
    USE_STRUCTLOG = True
except ImportError:
    USE_STRUCTLOG = False
    structlog = None  # type: ignore

# Intentar usar colorlog si está disponible
try:
    import colorlog
    USE_COLORLOG = True
except ImportError:
    USE_COLORLOG = False
    colorlog = None  # type: ignore


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_json: bool = False,
    use_colors: bool = True
) -> logging.Logger:
    """
    Configurar logging avanzado.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Ruta del archivo de log (opcional).
        use_json: Si usar formato JSON (default: False).
        use_colors: Si usar colores en consola (default: True).
    
    Returns:
        Logger configurado.
    
    Raises:
        ValueError: Si log_level es inválido.
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. "
            f"Must be one of {valid_levels}"
        )
    
    # Crear directorio de logs
    if log_file:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create log directory: {e}") from e
    
    if USE_STRUCTLOG and use_json:
        # Configurar structlog para JSON
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
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    elif USE_STRUCTLOG:
        # Configurar structlog con colores
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
                structlog.dev.ConsoleRenderer(colors=use_colors)
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        # Configuración estándar de logging
        handlers: List[logging.Handler] = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if use_colors and USE_COLORLOG:
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_formatter)
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
        
        handlers.append(console_handler)
        
        # File handler
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                if use_json:
                    try:
                        from python_json_logger import jsonlogger
                        formatter = jsonlogger.JsonFormatter(
                            "%(asctime)s %(name)s %(levelname)s %(message)s"
                        )
                    except ImportError:
                        formatter = logging.Formatter(
                            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                        )
                else:
                    formatter = logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)
            except Exception as e:
                logging.warning(f"Failed to create file handler: {e}")
        
        # Configurar root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            handlers=handlers,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging configured (level: {log_level})")
    
    return logger


def get_logger(name: str) -> Union[logging.Logger, Any]:
    """
    Obtener logger configurado.
    
    Args:
        name: Nombre del logger (típicamente __name__).
    
    Returns:
        Logger configurado (structlog o logging estándar).
    
    Raises:
        ValueError: Si name está vacío.
    """
    if not name or not name.strip():
        raise ValueError("Logger name cannot be empty")
    
    if USE_STRUCTLOG:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)
