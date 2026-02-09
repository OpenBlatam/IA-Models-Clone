"""
Log Formatters
==============

Formatters para logging estructurado.
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Formatter JSON para logs estructurados."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear record como JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Agregar campos extra
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Agregar exception info si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Formatter con colores para consola."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear record con colores."""
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        
        # Formato básico
        message = f"{color}[{record.levelname}]{reset} {record.name}: {record.getMessage()}"
        
        # Agregar campos extra si existen
        if hasattr(record, "extra"):
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            if extra_str:
                message += f" | {extra_str}"
        
        # Agregar exception info si existe
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class StructuredFormatter(logging.Formatter):
    """Formatter estructurado con contexto."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear record estructurado."""
        parts = [
            f"[{record.levelname}]",
            f"{record.name}:",
            record.getMessage()
        ]
        
        # Agregar campos extra
        if hasattr(record, "extra"):
            extra_parts = [f"{k}={v}" for k, v in record.extra.items()]
            if extra_parts:
                parts.append("|")
                parts.extend(extra_parts)
        
        message = " ".join(parts)
        
        # Agregar exception info si existe
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message

