"""
Structured Logging
Enhanced logging with structured data
"""

import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime
import sys


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record
            
        Returns:
            JSON string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


class ModelLogger:
    """
    Logger for model operations
    """
    
    def __init__(self, name: str = "model", level: int = logging.INFO):
        """
        Initialize model logger
        
        Args:
            name: Logger name
            level: Log level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Add handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(handler)
    
    def log_inference(
        self,
        model_name: str,
        input_shape: tuple,
        output_shape: tuple,
        inference_time_ms: float,
        success: bool = True,
        **kwargs
    ):
        """
        Log inference
        
        Args:
            model_name: Model name
            input_shape: Input shape
            output_shape: Output shape
            inference_time_ms: Inference time in ms
            success: Whether inference was successful
            **kwargs: Additional fields
        """
        self.logger.info(
            "Model inference",
            extra={
                "event": "inference",
                "model": model_name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "inference_time_ms": inference_time_ms,
                "success": success,
                **kwargs
            }
        )
    
    def log_training(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        **kwargs
    ):
        """
        Log training step
        
        Args:
            epoch: Epoch number
            loss: Loss value
            metrics: Additional metrics
            **kwargs: Additional fields
        """
        self.logger.info(
            "Training step",
            extra={
                "event": "training",
                "epoch": epoch,
                "loss": loss,
                "metrics": metrics,
                **kwargs
            }
        )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log error
        
        Args:
            error_type: Error type
            error_message: Error message
            context: Additional context
        """
        self.logger.error(
            "Error occurred",
            extra={
                "event": "error",
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {}
            },
            exc_info=True
        )


def create_model_logger(name: str = "model") -> ModelLogger:
    """Factory for model logger"""
    return ModelLogger(name)













