"""
Logger Config - Configuración avanzada de logging
==================================================

Configuración profesional de logging con múltiples handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Intentar usar structlog si está disponible
try:
    import structlog
    USE_STRUCTLOG = True
except ImportError:
    USE_STRUCTLOG = False


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_json: bool = False,
    use_colors: bool = True
):
    """Configurar logging avanzado"""
    
    # Crear directorio de logs
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if use_colors:
            try:
                import colorlog
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
            except ImportError:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                console_handler.setFormatter(formatter)
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
        
        handlers.append(console_handler)
        
        # File handler
        if log_file:
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
        
        # Configurar root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            handlers=handlers,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging configured (level: {log_level})")
    
    return logger


def get_logger(name: str):
    """Obtener logger configurado"""
    if USE_STRUCTLOG:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


