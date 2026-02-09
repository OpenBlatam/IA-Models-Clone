"""
Centralized Logging - Logging centralizado
==========================================

Integración con sistemas de logging centralizados:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- AWS CloudWatch
- Google Cloud Logging
- Azure Monitor
"""

import logging
import json
from typing import Optional, Dict, Any, List, Protocol
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LoggingBackend(str, Enum):
    """Backends de logging"""
    ELK = "elk"
    CLOUDWATCH = "cloudwatch"
    GCP_LOGGING = "gcp_logging"
    AZURE_MONITOR = "azure_monitor"
    FILE = "file"


class CentralizedLogger(Protocol):
    """Protocol para loggers centralizados"""
    
    def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None: ...
    
    def flush(self) -> None: ...


class ELKLogger:
    """Logger para ELK Stack"""
    
    def __init__(
        self,
        elasticsearch_hosts: List[str],
        index_prefix: str = "app-logs",
        **kwargs: Any
    ) -> None:
        self.elasticsearch_hosts = elasticsearch_hosts
        self.index_prefix = index_prefix
        self._client: Optional[Any] = None
    
    def _get_client(self) -> Any:
        """Obtiene cliente de Elasticsearch"""
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
                self._client = Elasticsearch(hosts=self.elasticsearch_hosts)
            except ImportError:
                logger.error("elasticsearch not available")
                raise
        return self._client
    
    def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Envía log a Elasticsearch"""
        try:
            client = self._get_client()
            index_name = f"{self.index_prefix}-{datetime.now().strftime('%Y.%m.%d')}"
            
            log_entry = {
                "@timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                **(extra or {})
            }
            
            client.index(index=index_name, document=log_entry)
        except Exception as e:
            logger.error(f"Failed to send log to ELK: {e}")
    
    def flush(self) -> None:
        """Flush logs"""
        pass


class CloudWatchLogger:
    """Logger para AWS CloudWatch"""
    
    def __init__(
        self,
        log_group: str,
        log_stream: str,
        region: str = "us-east-1",
        **kwargs: Any
    ) -> None:
        self.log_group = log_group
        self.log_stream = log_stream
        self.region = region
        self._client: Optional[Any] = None
        self._buffer: List[Dict[str, Any]] = []
    
    def _get_client(self) -> Any:
        """Obtiene cliente de CloudWatch Logs"""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("logs", region_name=self.region)
            except ImportError:
                logger.error("boto3 not available")
                raise
        return self._client
    
    def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Envía log a CloudWatch"""
        try:
            log_entry = {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "message": json.dumps({
                    "level": level,
                    "message": message,
                    **(extra or {})
                })
            }
            
            self._buffer.append(log_entry)
            
            # Enviar en batch cuando hay suficientes
            if len(self._buffer) >= 10:
                self.flush()
        except Exception as e:
            logger.error(f"Failed to send log to CloudWatch: {e}")
    
    def flush(self) -> None:
        """Flush logs a CloudWatch"""
        if not self._buffer:
            return
        
        try:
            client = self._get_client()
            client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=self._buffer
            )
            self._buffer = []
        except Exception as e:
            logger.error(f"Failed to flush logs to CloudWatch: {e}")


class StructuredLogger:
    """Logger estructurado con múltiples backends"""
    
    def __init__(
        self,
        backends: List[CentralizedLogger],
        default_level: str = "INFO"
    ) -> None:
        self.backends = backends
        self.default_level = default_level
    
    def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Envía log a todos los backends"""
        for backend in self.backends:
            try:
                backend.log(level, message, extra)
            except Exception as e:
                logger.error(f"Backend logging failed: {e}")
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug"""
        self.log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info"""
        self.log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning"""
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error"""
        self.log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical"""
        self.log("CRITICAL", message, **kwargs)
    
    def flush(self) -> None:
        """Flush todos los backends"""
        for backend in self.backends:
            try:
                backend.flush()
            except Exception as e:
                logger.error(f"Backend flush failed: {e}")


def get_centralized_logger(
    backend: LoggingBackend = LoggingBackend.FILE,
    **kwargs: Any
) -> CentralizedLogger:
    """
    Obtiene logger centralizado.
    
    Args:
        backend: Backend de logging
        **kwargs: Configuración específica
    
    Returns:
        Logger centralizado
    """
    if backend == LoggingBackend.ELK:
        hosts = kwargs.get("elasticsearch_hosts", ["localhost:9200"])
        if isinstance(hosts, str):
            hosts = [hosts]
        return ELKLogger(elasticsearch_hosts=hosts, **kwargs)
    elif backend == LoggingBackend.CLOUDWATCH:
        return CloudWatchLogger(**kwargs)
    else:
        # Fallback a logging estándar
        return logging.getLogger(__name__)















