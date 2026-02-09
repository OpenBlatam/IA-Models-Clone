"""
Advanced Logging System
=======================
Structured logging with multiple handlers
"""

from typing import Dict, Any, Optional
import structlog
import logging
from pathlib import Path
from datetime import datetime
import json

logger = structlog.get_logger()


class AdvancedLogger:
    """
    Advanced logging system with structured logging
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        log_level: str = "INFO",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True
    ):
        """
        Initialize advanced logger
        
        Args:
            log_dir: Log directory
            log_level: Log level
            enable_file_logging: Enable file logging
            enable_console_logging: Enable console logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_level = getattr(logging, log_level.upper())
        
        self._setup_logging(enable_file_logging, enable_console_logging)
        
        logger.info("AdvancedLogger initialized", log_dir=str(self.log_dir))
    
    def _setup_logging(
        self,
        enable_file: bool,
        enable_console: bool
    ) -> None:
        """Setup logging handlers"""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if enable_file else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup file handler
        if enable_file:
            log_file = self.log_dir / f"training_{datetime.utcnow().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.setLevel(self.log_level)
        
        # Setup console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            
            root_logger = logging.getLogger()
            root_logger.addHandler(console_handler)
            root_logger.setLevel(self.log_level)
    
    def log_training_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        **kwargs
    ) -> None:
        """
        Log training step
        
        Args:
            step: Training step
            epoch: Current epoch
            loss: Loss value
            learning_rate: Learning rate
            **kwargs: Additional metrics
        """
        logger.info(
            "Training step",
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def log_validation(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log validation results
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
        """
        logger.info(
            "Validation results",
            epoch=epoch,
            **metrics
        )
    
    def log_model_info(
        self,
        model_name: str,
        num_parameters: int,
        model_size_mb: float
    ) -> None:
        """
        Log model information
        
        Args:
            model_name: Model name
            num_parameters: Number of parameters
            model_size_mb: Model size in MB
        """
        logger.info(
            "Model information",
            model_name=model_name,
            num_parameters=num_parameters,
            model_size_mb=model_size_mb
        )


# Global advanced logger
advanced_logger = AdvancedLogger()




