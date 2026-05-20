"""
Structured Logging - Logging estructurado
==========================================

Sistema de logging estructurado para deep learning.
Sigue mejores prácticas de logging.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Entrada de log estructurada"""
    timestamp: datetime
    level: str
    message: str
    module: str
    function: str
    line: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convertir a JSON"""
        return json.dumps(self.to_dict(), indent=2)


class StructuredLogger:
    """Logger estructurado para deep learning"""
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        format_string: Optional[str] = None
    ):
        """
        Inicializar logger estructurado.
        
        Args:
            name: Nombre del logger
            log_file: Archivo de log (opcional)
            level: Nivel de logging
            format_string: Formato personalizado (opcional)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if format_string:
            formatter = logging.Formatter(format_string)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.log_entries: List[LogEntry] = []
        self.max_entries = 10000  # Keep last 10k entries in memory
    
    def _create_log_entry(
        self,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEntry:
        """Crear entrada de log"""
        import inspect
        frame = inspect.currentframe().f_back.f_back
        module = frame.f_globals.get('__name__', 'unknown')
        function = frame.f_code.co_name
        line = frame.f_lineno
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            module=module,
            function=function,
            line=line,
            metadata=metadata or {},
        )
        
        # Store entry
        self.log_entries.append(entry)
        if len(self.log_entries) > self.max_entries:
            self.log_entries.pop(0)
        
        return entry
    
    def info(self, message: str, **metadata) -> None:
        """Log info"""
        entry = self._create_log_entry("INFO", message, metadata)
        self.logger.info(f"{message} | {json.dumps(metadata)}" if metadata else message)
    
    def warning(self, message: str, **metadata) -> None:
        """Log warning"""
        entry = self._create_log_entry("WARNING", message, metadata)
        self.logger.warning(f"{message} | {json.dumps(metadata)}" if metadata else message)
    
    def error(self, message: str, **metadata) -> None:
        """Log error"""
        entry = self._create_log_entry("ERROR", message, metadata)
        self.logger.error(f"{message} | {json.dumps(metadata)}" if metadata else message)
    
    def debug(self, message: str, **metadata) -> None:
        """Log debug"""
        entry = self._create_log_entry("DEBUG", message, metadata)
        self.logger.debug(f"{message} | {json.dumps(metadata)}" if metadata else message)
    
    def log_training_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        **metrics
    ) -> None:
        """Log training step"""
        metadata = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            **metrics
        }
        self.info(f"Training step {step} (epoch {epoch})", **metadata)
    
    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log validation results"""
        metadata = {
            "epoch": epoch,
            **metrics
        }
        self.info(f"Validation epoch {epoch}", **metadata)
    
    def get_recent_logs(
        self,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """
        Obtener logs recientes.
        
        Args:
            level: Filtrar por nivel (opcional)
            limit: Número máximo de entradas
        
        Returns:
            Lista de entradas de log
        """
        logs = self.log_entries[-limit:]
        
        if level:
            logs = [log for log in logs if log.level == level.upper()]
        
        return logs
    
    def export_logs(self, filepath: str, format: str = "json") -> bool:
        """
        Exportar logs a archivo.
        
        Args:
            filepath: Ruta del archivo
            format: Formato ('json' o 'txt')
        
        Returns:
            True si se exportó exitosamente
        """
        try:
            if format == "json":
                logs_data = [entry.to_dict() for entry in self.log_entries]
                with open(filepath, 'w') as f:
                    json.dump(logs_data, f, indent=2)
            else:
                with open(filepath, 'w') as f:
                    for entry in self.log_entries:
                        f.write(f"{entry.timestamp} [{entry.level}] {entry.message}\n")
                        if entry.metadata:
                            f.write(f"  Metadata: {json.dumps(entry.metadata)}\n")
            
            return True
        except Exception as e:
            self.error(f"Error exporting logs: {e}")
            return False




