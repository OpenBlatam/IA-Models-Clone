"""
Configuración de logging estructurado con rotación y mejor formato.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from config.settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configurar logging con handlers de consola y archivo con rotación.
    
    Args:
        log_level: Nivel de logging (INFO, DEBUG, etc.)
        log_file: Ruta al archivo de log (opcional)
        max_bytes: Tamaño máximo del archivo antes de rotar (default: 10MB)
        backup_count: Número de archivos de backup a mantener (default: 5)
    """
    # Convertir string a nivel de logging si es necesario
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = log_level or (logging.DEBUG if settings.DEBUG else logging.INFO)
    
    # Crear directorio de logs si no existe
    log_dir = Path(settings.LOGS_STORAGE_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar handlers
    handlers = []
    
    # Handler de consola con formato mejorado
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    handlers.append(console_handler)
    
    # Handler de archivo con rotación
    if log_file:
        log_path = log_dir / log_file
    else:
        log_path = log_dir / "github_agent.log"
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    
    # Formato más detallado para archivo
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    handlers.append(file_handler)
    
    # Configurar logging root
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Sobrescribir configuración previa
    )
    
    # Configurar niveles específicos para librerías externas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Log de inicio
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configurado: nivel={logging.getLevelName(level)}, archivo={log_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Obtener un logger con el nombre especificado.
    
    Args:
        name: Nombre del logger (típicamente __name__)
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)


