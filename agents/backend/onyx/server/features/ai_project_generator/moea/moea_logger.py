"""
MOEA Logger - Sistema de logging avanzado
==========================================
Sistema de logging estructurado para herramientas MOEA
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class MOEALogger:
    """Logger personalizado para MOEA"""
    
    def __init__(
        self,
        name: str = "MOEA",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        json_format: bool = False
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Limpiar handlers existentes
        self.logger.handlers.clear()
        
        # Formato estándar
        if json_format:
            formatter = self._json_formatter
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Handler para archivo (si se especifica)
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _json_formatter(self, record):
        """Formateador JSON para logs estructurados"""
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
            log_data["exception"] = self.format_exception(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)
    
    def format_exception(self, exc_info):
        """Formatear excepción"""
        import traceback
        return traceback.format_exception(*exc_info)
    
    def info(self, message: str, **kwargs):
        """Log info"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error"""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning"""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug"""
        self.logger.debug(message, **kwargs)
    
    def success(self, message: str):
        """Log success (info level con emoji)"""
        self.logger.info(f"✅ {message}")
    
    def failure(self, message: str):
        """Log failure (error level con emoji)"""
        self.logger.error(f"❌ {message}")


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    json_format: bool = False
) -> MOEALogger:
    """Configurar logging para MOEA"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Crear directorio de logs si no existe
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    return MOEALogger(
        name="MOEA",
        level=log_level,
        log_file=log_file,
        json_format=json_format
    )


def main():
    """Ejemplo de uso"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Logger")
    parser.add_argument(
        '--log-file',
        help='Archivo de log'
    )
    parser.add_argument(
        '--level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Nivel de logging'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Formato JSON'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(
        log_file=args.log_file,
        level=args.level,
        json_format=args.json
    )
    
    # Ejemplos
    logger.info("Mensaje informativo")
    logger.success("Operación exitosa")
    logger.warning("Advertencia")
    logger.error("Error de ejemplo")
    logger.debug("Mensaje de debug")


if __name__ == "__main__":
    main()

