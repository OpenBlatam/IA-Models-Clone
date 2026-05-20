"""
Logging Config - Configuración Avanzada de Logging
===================================================

Configuración avanzada de logging estructurado.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedLoggingConfig:
    """Configuración avanzada de logging"""

    @staticmethod
    def setup_logging(
        log_level: str = "INFO",
        log_file: Optional[Path] = None,
        json_logging: bool = False,
    ):
        """
        Configura logging avanzado.

        Args:
            log_level: Nivel de logging
            log_file: Archivo de log (opcional)
            json_logging: Si usar formato JSON
        """
        # Crear directorio de logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configurar formato
        if json_logging:
            formatter = AdvancedLoggingConfig._create_json_formatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level))

        # Handler para archivo
        if log_file is None:
            log_file = log_dir / f"ai_project_generator_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        # Configurar loggers específicos
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)

    @staticmethod
    def _create_json_formatter():
        """Crea un formatter JSON"""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }

                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)

                return json.dumps(log_data, ensure_ascii=False)

        return JSONFormatter()

    @staticmethod
    def get_log_stats(log_file: Path) -> Dict[str, Any]:
        """
        Obtiene estadísticas de logs.

        Args:
            log_file: Archivo de log

        Returns:
            Estadísticas de logs
        """
        if not log_file.exists():
            return {"error": "Archivo de log no existe"}

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            level_counts = {
                "DEBUG": 0,
                "INFO": 0,
                "WARNING": 0,
                "ERROR": 0,
                "CRITICAL": 0,
            }

            for line in lines:
                for level in level_counts.keys():
                    if level in line:
                        level_counts[level] += 1
                        break

            return {
                "total_lines": len(lines),
                "level_counts": level_counts,
                "file_size_bytes": log_file.stat().st_size,
            }
        except Exception as e:
            return {"error": str(e)}


