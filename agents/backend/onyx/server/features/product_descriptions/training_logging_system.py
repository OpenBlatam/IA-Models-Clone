from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from contextlib import contextmanager
import functools
from datetime import datetime, timedelta
import hashlib
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import structlog
from structlog.stdlib import LoggerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.panel import Panel
from error_handling_debugging import ErrorHandlingDebuggingSystem, ErrorSeverity, ErrorCategory
from robust_operations import RobustOperations, OperationResult
        import psutil
from typing import Any, List, Dict, Optional
"""
Comprehensive Training Logging System

This module provides robust logging for ML training progress and errors with:
- Structured logging for training metrics and progress
- Error categorization and tracking
- Performance monitoring and resource usage
- Cybersecurity-specific event logging
- Integration with existing robust operations
- Real-time progress visualization
- Log persistence and analysis tools
"""




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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class TrainingEventType(Enum):
    """Types of training events to log."""
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    LOSS_UPDATE = "loss_update"
    METRIC_UPDATE = "metric_update"
    VALIDATION_START = "validation_start"
    VALIDATION_END = "validation_end"
    MODEL_SAVE = "model_save"
    MODEL_LOAD = "model_load"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_ALERT = "performance_alert"
    RESOURCE_USAGE = "resource_usage"


class LogLevel(Enum):
    """Log levels for training events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    epoch: int
    batch: int
    loss: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_time: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: str
    severity: str
    description: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    port: Optional[int] = None
    protocol: Optional[str] = None
    threat_level: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    disk_io: Optional[float] = None
    network_io: Optional[float] = None
    batch_time: Optional[float] = None
    epoch_time: Optional[float] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class TrainingLogger:
    """Comprehensive training logger with structured logging."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: LogLevel = LogLevel.INFO,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_rich: bool = True,
                 max_log_files: int = 10,
                 log_rotation_size: int = 100 * 1024 * 1024):  # 100MB
        """
        Initialize the training logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Minimum log level to record
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_rich: Enable rich console output
            max_log_files: Maximum number of log files to keep
            log_rotation_size: Size at which to rotate log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = log_level
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_rich = enable_rich
        
        self.max_log_files = max_log_files
        self.log_rotation_size = log_rotation_size
        
        # Initialize components
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # Metrics storage
        self.training_metrics: List[TrainingMetrics] = []
        self.security_events: List[SecurityEvent] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Progress tracking
        self.current_epoch = 0
        self.current_batch = 0
        self.total_epochs = 0
        self.total_batches = 0
        
        # Rich console for beautiful output
        if self.enable_rich:
            self.console = Console()
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
        
        # Setup loggers
        self._setup_loggers()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("TrainingLogger initialized", 
                   session_id=self.session_id,
                   log_dir=str(self.log_dir),
                   log_level=self.log_level.value)
    
    def _setup_loggers(self) -> Any:
        """Setup different loggers for different purposes."""
        # Main training logger
        self.training_logger = structlog.get_logger("training")
        
        # Security events logger
        self.security_logger = structlog.get_logger("security")
        
        # Performance logger
        self.performance_logger = structlog.get_logger("performance")
        
        # Error logger
        self.error_logger = structlog.get_logger("errors")
        
        # File handlers
        if self.enable_file:
            self._setup_file_handlers()
    
    def _setup_file_handlers(self) -> Any:
        """Setup file handlers for different log types."""
        # Training logs
        training_log_file = self.log_dir / f"training_{self.session_id}.log"
        training_handler = logging.FileHandler(training_log_file)
        training_handler.setLevel(getattr(logging, self.log_level.value.upper()))
        
        # Security logs
        security_log_file = self.log_dir / f"security_{self.session_id}.log"
        security_handler = logging.FileHandler(security_log_file)
        security_handler.setLevel(logging.INFO)
        
        # Performance logs
        performance_log_file = self.log_dir / f"performance_{self.session_id}.log"
        performance_handler = logging.FileHandler(performance_log_file)
        performance_handler.setLevel(logging.INFO)
        
        # Error logs
        error_log_file = self.log_dir / f"errors_{self.session_id}.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        
        # Add handlers to loggers
        logging.getLogger("training").addHandler(training_handler)
        logging.getLogger("security").addHandler(security_handler)
        logging.getLogger("performance").addHandler(performance_handler)
        logging.getLogger("errors").addHandler(error_handler)
    
    def log_training_event(self, 
                          event_type: TrainingEventType,
                          message: str,
                          metrics: Optional[TrainingMetrics] = None,
                          level: LogLevel = LogLevel.INFO,
                          **kwargs):
        """Log a training event with structured data."""
        log_data = {
            "event_type": event_type.value,
            "message": message,
            "level": level.value,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "epoch": self.current_epoch,
            "batch": self.current_batch,
            "total_epochs": self.total_epochs,
            "total_batches": self.total_batches,
            **kwargs
        }
        
        if metrics:
            log_data["metrics"] = metrics.to_dict()
        
        # Log based on level
        log_method = getattr(self.training_logger, level.value)
        log_method(message, **log_data)
        
        # Store metrics if provided
        if metrics:
            self.training_metrics.append(metrics)
    
    def log_security_event(self, 
                          event: SecurityEvent,
                          level: LogLevel = LogLevel.INFO):
        """Log a security event."""
        log_data = {
            "event_type": event.event_type,
            "severity": event.severity,
            "description": event.description,
            "source_ip": event.source_ip,
            "destination_ip": event.destination_ip,
            "port": event.port,
            "protocol": event.protocol,
            "threat_level": event.threat_level,
            "confidence": event.confidence,
            "timestamp": event.timestamp or datetime.now().isoformat(),
            "session_id": self.session_id,
            "metadata": event.metadata or {}
        }
        
        log_method = getattr(self.security_logger, level.value)
        log_method(event.description, **log_data)
        
        # Store security event
        self.security_events.append(event)
    
    def log_performance_metrics(self, 
                               metrics: PerformanceMetrics,
                               level: LogLevel = LogLevel.INFO):
        """Log performance metrics."""
        log_data = {
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "gpu_usage": metrics.gpu_usage,
            "gpu_memory": metrics.gpu_memory,
            "disk_io": metrics.disk_io,
            "network_io": metrics.network_io,
            "batch_time": metrics.batch_time,
            "epoch_time": metrics.epoch_time,
            "timestamp": metrics.timestamp or datetime.now().isoformat(),
            "session_id": self.session_id
        }
        
        log_method = getattr(self.performance_logger, level.value)
        log_method("Performance metrics", **log_data)
        
        # Store performance metrics
        self.performance_metrics.append(metrics)
    
    def log_error(self, 
                  error: Exception,
                  context: Optional[Dict[str, Any]] = None,
                  level: LogLevel = LogLevel.ERROR):
        """Log an error with context."""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "epoch": self.current_epoch,
            "batch": self.current_batch,
            "context": context or {}
        }
        
        log_method = getattr(self.error_logger, level.value)
        log_method(f"Training error: {str(error)}", **error_data)
    
    def start_training(self, total_epochs: int, total_batches: int):
        """Log training start."""
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        
        self.log_training_event(
            TrainingEventType.INFO,
            "Training started",
            level=LogLevel.INFO,
            total_epochs=total_epochs,
            total_batches=total_batches
        )
        
        if self.enable_rich:
            self.progress.start()
    
    def end_training(self, final_metrics: Optional[TrainingMetrics] = None):
        """Log training end."""
        training_duration = datetime.now() - self.start_time
        
        self.log_training_event(
            TrainingEventType.INFO,
            "Training completed",
            metrics=final_metrics,
            level=LogLevel.INFO,
            training_duration=str(training_duration),
            total_metrics_logged=len(self.training_metrics)
        )
        
        if self.enable_rich:
            self.progress.stop()
    
    def start_epoch(self, epoch: int):
        """Log epoch start."""
        self.current_epoch = epoch
        
        self.log_training_event(
            TrainingEventType.EPOCH_START,
            f"Starting epoch {epoch}",
            level=LogLevel.INFO,
            epoch=epoch
        )
        
        if self.enable_rich:
            self.progress.add_task(f"Epoch {epoch}", total=self.total_batches)
    
    def end_epoch(self, epoch: int, metrics: Optional[TrainingMetrics] = None):
        """Log epoch end."""
        self.log_training_event(
            TrainingEventType.EPOCH_END,
            f"Completed epoch {epoch}",
            metrics=metrics,
            level=LogLevel.INFO,
            epoch=epoch
        )
    
    def start_batch(self, batch: int, epoch: int):
        """Log batch start."""
        self.current_batch = batch
        
        self.log_training_event(
            TrainingEventType.BATCH_START,
            f"Starting batch {batch} in epoch {epoch}",
            level=LogLevel.DEBUG,
            batch=batch,
            epoch=epoch
        )
    
    def end_batch(self, 
                  batch: int, 
                  epoch: int, 
                  metrics: TrainingMetrics,
                  update_progress: bool = True):
        """Log batch end with metrics."""
        self.log_training_event(
            TrainingEventType.BATCH_END,
            f"Completed batch {batch} in epoch {epoch}",
            metrics=metrics,
            level=LogLevel.DEBUG,
            batch=batch,
            epoch=epoch
        )
        
        if update_progress and self.enable_rich:
            self.progress.update(self.progress.task_ids[-1], advance=1)
    
    def log_loss(self, 
                 epoch: int, 
                 batch: int, 
                 loss: float, 
                 learning_rate: Optional[float] = None):
        """Log loss update."""
        metrics = TrainingMetrics(
            epoch=epoch,
            batch=batch,
            loss=loss,
            learning_rate=learning_rate,
            timestamp=datetime.now().isoformat()
        )
        
        self.log_training_event(
            TrainingEventType.LOSS_UPDATE,
            f"Loss: {loss:.4f} (Epoch {epoch}, Batch {batch})",
            metrics=metrics,
            level=LogLevel.INFO
        )
    
    def log_validation(self, 
                      epoch: int, 
                      validation_loss: float, 
                      validation_accuracy: float,
                      **kwargs):
        """Log validation results."""
        metrics = TrainingMetrics(
            epoch=epoch,
            batch=0,  # Validation doesn't have batches
            loss=validation_loss,
            accuracy=validation_accuracy,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )
        
        self.log_training_event(
            TrainingEventType.VALIDATION_END,
            f"Validation - Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.4f}",
            metrics=metrics,
            level=LogLevel.INFO
        )
    
    def log_model_save(self, 
                      file_path: str, 
                      epoch: int, 
                      metrics: Optional[TrainingMetrics] = None):
        """Log model saving."""
        self.log_training_event(
            TrainingEventType.MODEL_SAVE,
            f"Model saved to {file_path}",
            metrics=metrics,
            level=LogLevel.INFO,
            file_path=file_path,
            epoch=epoch
        )
    
    def log_model_load(self, 
                      file_path: str, 
                      success: bool, 
                      error_message: Optional[str] = None):
        """Log model loading."""
        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"Model loaded from {file_path}" if success else f"Model loading failed: {error_message}"
        
        self.log_training_event(
            TrainingEventType.MODEL_LOAD,
            message,
            level=level,
            file_path=file_path,
            success=success,
            error_message=error_message
        )
    
    def log_security_anomaly(self, 
                            source_ip: str,
                            destination_ip: str,
                            threat_level: str,
                            confidence: float,
                            description: str):
        """Log security anomaly detection."""
        event = SecurityEvent(
            event_type="anomaly_detection",
            severity="high" if threat_level == "high" else "medium",
            description=description,
            source_ip=source_ip,
            destination_ip=destination_ip,
            threat_level=threat_level,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        self.log_security_event(event, level=LogLevel.WARNING)
    
    def log_performance_alert(self, 
                             alert_type: str,
                             message: str,
                             metrics: Optional[PerformanceMetrics] = None):
        """Log performance alert."""
        self.log_training_event(
            TrainingEventType.PERFORMANCE_ALERT,
            f"Performance alert: {message}",
            level=LogLevel.WARNING,
            alert_type=alert_type,
            message=message
        )
        
        if metrics:
            self.log_performance_metrics(metrics, level=LogLevel.WARNING)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.training_metrics:
            return {"error": "No training metrics available"}
        
        losses = [m.loss for m in self.training_metrics if m.loss is not None]
        accuracies = [m.accuracy for m in self.training_metrics if m.accuracy is not None]
        
        summary = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_epochs": self.total_epochs,
            "total_batches": self.total_batches,
            "metrics_count": len(self.training_metrics),
            "security_events_count": len(self.security_events),
            "performance_metrics_count": len(self.performance_metrics)
        }
        
        if losses:
            summary.update({
                "final_loss": losses[-1],
                "min_loss": min(losses),
                "max_loss": max(losses),
                "avg_loss": np.mean(losses)
            })
        
        if accuracies:
            summary.update({
                "final_accuracy": accuracies[-1],
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "avg_accuracy": np.mean(accuracies)
            })
        
        return summary
    
    def save_metrics_to_csv(self, file_path: Optional[str] = None):
        """Save training metrics to CSV file."""
        if not self.training_metrics:
            logger.warning("No training metrics to save")
            return
        
        if file_path is None:
            file_path = self.log_dir / f"training_metrics_{self.session_id}.csv"
        
        df = pd.DataFrame([m.to_dict() for m in self.training_metrics])
        df.to_csv(file_path, index=False)
        
        logger.info(f"Training metrics saved to {file_path}")
    
    def plot_training_curves(self, 
                           save_path: Optional[str] = None,
                           show_plot: bool = False):
        """Plot training curves."""
        if not self.training_metrics:
            logger.warning("No training metrics to plot")
            return
        
        df = pd.DataFrame([m.to_dict() for m in self.training_metrics])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        if 'loss' in df.columns and df['loss'].notna().any():
            axes[0, 0].plot(df['loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Loss')
        
        # Accuracy curve
        if 'accuracy' in df.columns and df['accuracy'].notna().any():
            axes[0, 1].plot(df['accuracy'])
            axes[0, 1].set_title('Training Accuracy')
            axes[0, 1].set_xlabel('Batch')
            axes[0, 1].set_ylabel('Accuracy')
        
        # Learning rate curve
        if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
            axes[1, 0].plot(df['learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Learning Rate')
        
        # Memory usage
        if 'memory_usage' in df.columns and df['memory_usage'].notna().any():
            axes[1, 1].plot(df['memory_usage'])
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.log_dir / f"training_curves_{self.session_id}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        logger.info(f"Training curves saved to {save_path}")
    
    def cleanup(self) -> Any:
        """Cleanup resources."""
        if self.enable_rich:
            self.progress.stop()
        
        # Rotate log files if needed
        self._rotate_log_files()
        
        logger.info("TrainingLogger cleanup completed")
    
    def _rotate_log_files(self) -> Any:
        """Rotate log files if they exceed size limit."""
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_size > self.log_rotation_size:
                # Create backup
                backup_file = log_file.with_suffix(f".{int(time.time())}.log")
                log_file.rename(backup_file)
                
                # Create new log file
                log_file.touch()


class PerformanceMonitor:
    """Monitor system performance during training."""
    
    def __init__(self) -> Any:
        self.start_time = time.time()
        self.metrics_history: List[PerformanceMetrics] = []
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        
        gpu_usage = None
        gpu_memory = None
        
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            except Exception:
                pass
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            timestamp=datetime.now().isoformat()
        )
    
    def check_performance_alerts(self, metrics: PerformanceMetrics) -> List[str]:
        """Check for performance alerts."""
        alerts = []
        
        if metrics.cpu_usage > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > 90:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.gpu_memory and metrics.gpu_memory > 8000:  # 8GB
            alerts.append(f"High GPU memory usage: {metrics.gpu_memory:.1f}MB")
        
        return alerts


class TrainingLoggerDecorator:
    """Decorator for automatic training logging."""
    
    def __init__(self, logger: TrainingLogger):
        
    """__init__ function."""
self.logger = logger
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                # Log function start
                self.logger.log_training_event(
                    TrainingEventType.INFO,
                    f"Starting {func.__name__}",
                    level=LogLevel.DEBUG
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful completion
                execution_time = time.time() - start_time
                self.logger.log_training_event(
                    TrainingEventType.INFO,
                    f"Completed {func.__name__} in {execution_time:.2f}s",
                    level=LogLevel.DEBUG,
                    execution_time=execution_time
                )
                
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                self.logger.log_error(
                    e,
                    context={
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "execution_time": execution_time
                    }
                )
                raise
        
        return wrapper


# Utility functions
def create_training_logger(config: Optional[Dict[str, Any]] = None) -> TrainingLogger:
    """Create a training logger with default configuration."""
    default_config = {
        "log_dir": "logs",
        "log_level": LogLevel.INFO,
        "enable_console": True,
        "enable_file": True,
        "enable_rich": True,
        "max_log_files": 10,
        "log_rotation_size": 100 * 1024 * 1024
    }
    
    if config:
        default_config.update(config)
    
    return TrainingLogger(**default_config)


def log_training_progress(logger: TrainingLogger):
    """Decorator for logging training progress."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract training parameters
            model = kwargs.get('model') or (args[0] if args else None)
            dataloader = kwargs.get('dataloader') or (args[1] if len(args) > 1 else None)
            epochs = kwargs.get('epochs') or (args[2] if len(args) > 2 else 1)
            
            if dataloader:
                total_batches = len(dataloader) * epochs
                logger.start_training(epochs, total_batches)
            
            try:
                return func(*args, **kwargs)
            finally:
                logger.end_training()
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = create_training_logger({
        "log_dir": "training_logs",
        "enable_rich": True
    })
    
    # Example training loop with logging
    @log_training_progress(logger)
    def train_model(model, dataloader, epochs=10) -> Any:
        for epoch in range(epochs):
            logger.start_epoch(epoch)
            
            for batch_idx, (data, target) in enumerate(dataloader):
                logger.start_batch(batch_idx, epoch)
                
                # Simulate training
                loss = 1.0 / (batch_idx + 1)  # Decreasing loss
                accuracy = 0.5 + (batch_idx * 0.01)  # Increasing accuracy
                
                metrics = TrainingMetrics(
                    epoch=epoch,
                    batch=batch_idx,
                    loss=loss,
                    accuracy=accuracy,
                    learning_rate=0.001,
                    timestamp=datetime.now().isoformat()
                )
                
                logger.end_batch(batch_idx, epoch, metrics)
                
                # Log performance metrics
                perf_metrics = logger.performance_monitor.get_current_metrics()
                logger.log_performance_metrics(perf_metrics)
                
                # Check for alerts
                alerts = logger.performance_monitor.check_performance_alerts(perf_metrics)
                for alert in alerts:
                    logger.log_performance_alert("system", alert, perf_metrics)
            
            logger.end_epoch(epoch)
    
    # Run example
    model = nn.Linear(10, 2)
    dataloader = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(100)]
    
    train_model(model, dataloader, epochs=2)
    
    # Generate reports
    summary = logger.get_training_summary()
    print("Training Summary:", json.dumps(summary, indent=2))
    
    logger.save_metrics_to_csv()
    logger.plot_training_curves()
    logger.cleanup() 