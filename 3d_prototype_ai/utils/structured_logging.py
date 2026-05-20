"""
Structured Logging - Sistema de logging estructurado
======================================================

Soporta:
- JSON logging
- Centralized logging (ELK Stack, CloudWatch)
- Log levels y filtros
- Context propagation
"""

import logging
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pythonjsonlogger import jsonlogger


class StructuredLogger:
    """Logger estructurado con formato JSON"""
    
    def __init__(self, name: str, level: int = logging.INFO,
                 output_file: Optional[str] = None,
                 enable_console: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Formato JSON
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            timestamp=True
        )
        
        # Handler para consola
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Handler para archivo
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _log(self, level: int, message: str, **kwargs):
        """Log con contexto adicional"""
        extra = {
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log de información"""
        self._log(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log de error"""
        self._log(logging.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log de advertencia"""
        self._log(logging.WARNING, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log de debug"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log crítico"""
        self._log(logging.CRITICAL, message, **kwargs)


class CloudWatchLogger:
    """Logger para AWS CloudWatch"""
    
    def __init__(self, log_group: str, log_stream: str = "default"):
        self.log_group = log_group
        self.log_stream = log_stream
        self.client = None
        self._setup()
    
    def _setup(self):
        """Configura CloudWatch"""
        try:
            import boto3
            self.client = boto3.client('logs')
            logger.info("CloudWatch logger configured")
        except ImportError:
            logger.warning("boto3 not available. Install with: pip install boto3")
        except Exception as e:
            logger.error(f"Failed to setup CloudWatch: {e}")
    
    def log(self, message: str, level: str = "INFO", **kwargs):
        """Envía log a CloudWatch"""
        if not self.client:
            return
        
        try:
            log_event = {
                'timestamp': int(datetime.utcnow().timestamp() * 1000),
                'message': json.dumps({
                    "level": level,
                    "message": message,
                    **kwargs
                })
            }
            
            self.client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=[log_event]
            )
        except Exception as e:
            logger.error(f"Failed to send log to CloudWatch: {e}")


class ELKLogger:
    """Logger para ELK Stack (Elasticsearch, Logstash, Kibana)"""
    
    def __init__(self, elasticsearch_url: str = "http://localhost:9200",
                 index_name: str = "3d-prototype-ai-logs"):
        self.elasticsearch_url = elasticsearch_url
        self.index_name = index_name
        self.client = None
        self._setup()
    
    def _setup(self):
        """Configura Elasticsearch"""
        try:
            from elasticsearch import Elasticsearch
            self.client = Elasticsearch([self.elasticsearch_url])
            logger.info("ELK logger configured")
        except ImportError:
            logger.warning("elasticsearch not available. Install with: pip install elasticsearch")
        except Exception as e:
            logger.error(f"Failed to setup ELK: {e}")
    
    def log(self, message: str, level: str = "INFO", **kwargs):
        """Envía log a Elasticsearch"""
        if not self.client:
            return
        
        try:
            doc = {
                "@timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
                **kwargs
            }
            
            self.client.index(index=self.index_name, body=doc)
        except Exception as e:
            logger.error(f"Failed to send log to ELK: {e}")


class CentralizedLogging:
    """Sistema de logging centralizado que soporta múltiples backends"""
    
    def __init__(self, 
                 enable_cloudwatch: bool = False,
                 enable_elk: bool = False,
                 cloudwatch_config: Optional[Dict] = None,
                 elk_config: Optional[Dict] = None):
        self.structured_logger = StructuredLogger("3d_prototype_ai")
        self.cloudwatch_logger = None
        self.elk_logger = None
        
        if enable_cloudwatch:
            config = cloudwatch_config or {}
            self.cloudwatch_logger = CloudWatchLogger(
                log_group=config.get("log_group", "3d-prototype-ai"),
                log_stream=config.get("log_stream", "default")
            )
        
        if enable_elk:
            config = elk_config or {}
            self.elk_logger = ELKLogger(
                elasticsearch_url=config.get("url", "http://localhost:9200"),
                index_name=config.get("index", "3d-prototype-ai-logs")
            )
    
    def log(self, level: str, message: str, **kwargs):
        """Log a todos los backends configurados"""
        # Structured logger
        getattr(self.structured_logger, level.lower())(message, **kwargs)
        
        # CloudWatch
        if self.cloudwatch_logger:
            self.cloudwatch_logger.log(message, level=level, **kwargs)
        
        # ELK
        if self.elk_logger:
            self.elk_logger.log(message, level=level, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log de información"""
        self.log("INFO", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log de error"""
        self.log("ERROR", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log de advertencia"""
        self.log("WARNING", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log de debug"""
        self.log("DEBUG", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log crítico"""
        self.log("CRITICAL", message, **kwargs)


# Logger global
logger = logging.getLogger(__name__)




