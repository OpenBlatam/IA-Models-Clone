"""
Advanced Logging Utilities
===========================

Structured logging and error tracking.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import traceback

logger = logging.getLogger(__name__)


def setup_logging(
    log_dir: Path,
    log_level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Setup advanced logging configuration.
    
    Args:
        log_dir: Log directory
        log_level: Logging level
        log_to_file: Log to file
        log_to_console: Log to console
        log_format: Custom log format
        
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('deep_learning')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """
    Structured logger for training.
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize training logger.
        
        Args:
            log_dir: Log directory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.jsonl"
        self.logs = []
    
    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log epoch information.
        
        Args:
            epoch: Epoch number
            metrics: Training metrics
            config: Configuration
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics,
            'config': config or {}
        }
        
        self.logs.append(log_entry)
        
        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs."""
        return self.logs


class PerformanceLogger:
    """
    Logger for performance metrics.
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize performance logger.
        
        Args:
            log_dir: Log directory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "performance_log.jsonl"
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance metric.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'metadata': metadata or {}
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class ErrorTracker:
    """
    Track and log errors.
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize error tracker.
        
        Args:
            log_dir: Log directory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.error_file = self.log_dir / "errors.jsonl"
        self.errors = []
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log error.
        
        Args:
            error: Exception object
            context: Additional context
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.errors.append(error_entry)
        
        # Append to file
        with open(self.error_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')
        
        logger.error(f"Error logged: {error}")
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all errors."""
        return self.errors
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get error summary by type."""
        summary = {}
        for error in self.errors:
            error_type = error['error_type']
            summary[error_type] = summary.get(error_type, 0) + 1
        return summary



