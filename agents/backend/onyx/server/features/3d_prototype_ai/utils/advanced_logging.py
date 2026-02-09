"""
Advanced Logging - Sistema de logging avanzado completo
========================================================
"""

import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Niveles de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AdvancedLogging:
    """Sistema de logging avanzado completo"""
    
    def __init__(self, log_dir: str = "logs", max_bytes: int = 10 * 1024 * 1024,
                 backup_count: int = 5):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs: List[Dict[str, Any]] = []
        self.max_logs = 10000
        self.setup_logging(max_bytes, backup_count)
    
    def setup_logging(self, max_bytes: int, backup_count: int):
        """Configura sistema de logging"""
        # Formato detallado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler rotativo por tamaño
        file_handler = RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Handler rotativo por tiempo
        timed_handler = TimedRotatingFileHandler(
            self.log_dir / "app_daily.log",
            when='midnight',
            interval=1,
            backupCount=30
        )
        timed_handler.setFormatter(formatter)
        timed_handler.setLevel(logging.INFO)
        
        # Handler para errores
        error_handler = RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(timed_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
    
    def log_structured(self, level: LogLevel, message: str,
                      context: Optional[Dict[str, Any]] = None,
                      user_id: Optional[str] = None,
                      request_id: Optional[str] = None):
        """Log estructurado"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "message": message,
            "context": context or {},
            "user_id": user_id,
            "request_id": request_id
        }
        
        self.logs.append(log_entry)
        
        # Mantener solo últimos N logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Log usando logger estándar
        log_func = getattr(logging, level.value.lower())
        log_message = f"{message} | Context: {context} | User: {user_id} | Request: {request_id}"
        log_func(log_message)
        
        return log_entry
    
    def get_logs(self, level: Optional[LogLevel] = None,
                user_id: Optional[str] = None,
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                limit: int = 1000) -> List[Dict[str, Any]]:
        """Obtiene logs"""
        logs = self.logs
        
        if level:
            logs = [l for l in logs if l["level"] == level.value]
        
        if user_id:
            logs = [l for l in logs if l.get("user_id") == user_id]
        
        if start_date:
            logs = [l for l in logs if datetime.fromisoformat(l["timestamp"]) >= start_date]
        
        if end_date:
            logs = [l for l in logs if datetime.fromisoformat(l["timestamp"]) <= end_date]
        
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return logs[:limit]
    
    def get_log_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Obtiene estadísticas de logs"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_logs = [
            l for l in self.logs
            if datetime.fromisoformat(l["timestamp"]) > cutoff
        ]
        
        level_counts = {}
        for level in LogLevel:
            level_counts[level.value] = sum(
                1 for l in recent_logs if l["level"] == level.value
            )
        
        return {
            "period_days": days,
            "total_logs": len(recent_logs),
            "by_level": level_counts,
            "error_rate": (level_counts.get("ERROR", 0) + level_counts.get("CRITICAL", 0)) / len(recent_logs) * 100 if recent_logs else 0
        }
    
    def export_logs(self, format: str = "json", 
                   level: Optional[LogLevel] = None,
                   limit: int = 10000) -> str:
        """Exporta logs"""
        logs = self.get_logs(level=level, limit=limit)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            file_path = self.log_dir / f"export_{timestamp}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False, default=str)
        else:
            file_path = self.log_dir / f"export_{timestamp}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                for log in logs:
                    f.write(f"{log['timestamp']} [{log['level']}] {log['message']}\n")
        
        return str(file_path)




