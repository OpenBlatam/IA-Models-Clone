"""
Advanced Logging Utilities
Advanced logging and monitoring utilities
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AdvancedLogger:
    """
    Advanced logging utilities
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Path = Path("logs"),
        log_level: int = logging.INFO,
    ):
        """
        Initialize advanced logger
        
        Args:
            name: Logger name
            log_dir: Log directory
            log_level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        step_str = f" (step {step})" if step is not None else ""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Metrics{step_str}: {metrics_str}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration
        
        Args:
            config: Configuration dictionary
        """
        config_str = json.dumps(config, indent=2)
        self.logger.info(f"Configuration:\n{config_str}")
    
    def log_exception(self, exception: Exception, context: Optional[str] = None) -> None:
        """
        Log exception with context
        
        Args:
            exception: Exception to log
            context: Additional context (optional)
        """
        context_str = f" in {context}" if context else ""
        self.logger.error(f"Exception{context_str}: {str(exception)}", exc_info=True)



