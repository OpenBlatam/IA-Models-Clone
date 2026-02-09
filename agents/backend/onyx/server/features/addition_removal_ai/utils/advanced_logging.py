"""
Advanced Logging with Structured Logging
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import sys


class StructuredLogger:
    """Structured logger for training and inference"""
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        format_json: bool = False
    ):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            log_file: Log file path
            level: Logging level
            format_json: Use JSON format
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.format_json = format_json
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if format_json:
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            if format_json:
                file_handler.setFormatter(JsonFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            self.logger.addHandler(file_handler)
    
    def log_metric(self, name: str, value: float, step: int = 0, **kwargs):
        """Log metric"""
        data = {
            "metric": name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        if self.format_json:
            self.logger.info(json.dumps(data))
        else:
            self.logger.info(f"Metric: {name}={value} (step={step})")
    
    def log_event(self, event: str, **kwargs):
        """Log event"""
        data = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        if self.format_json:
            self.logger.info(json.dumps(data))
        else:
            self.logger.info(f"Event: {event} - {kwargs}")
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log error with context"""
        data = {
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        if self.format_json:
            self.logger.error(json.dumps(data))
        else:
            self.logger.error(f"Error: {error} - Context: {context}")


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, 'metric'):
            log_data['metric'] = record.metric
        if hasattr(record, 'value'):
            log_data['value'] = record.value
        if hasattr(record, 'step'):
            log_data['step'] = record.step
        
        return json.dumps(log_data)


def create_logger(
    name: str,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> StructuredLogger:
    """Factory function to create structured logger"""
    return StructuredLogger(name, log_file, format_json=json_format)

