"""
Logging Utils

Utilities for logging utils.
"""

import logging
import json
from typing import Optional
from datetime import datetime

def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configurar logging
    
    Args:
        level: Nivel de logging
        format_type: 'json' o 'text'
        log_file: Archivo para logs (opcional)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger("physical_store_designer_ai")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remover handlers existentes
    logger.handlers.clear()
    
    # Formato
    if format_type == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (si se especifica)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class JsonFormatter(logging.Formatter):
    """Formatter para logs en formato JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear log record como JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Agregar campos adicionales si existen
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Agregar exception info si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)

