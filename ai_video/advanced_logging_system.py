from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import logging
import logging.handlers
import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np
from contextlib import contextmanager
import traceback
import threading
from collections import defaultdict, deque
            import psutil
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Logging System for AI Training

Comprehensive logging system for training progress, errors, and metrics tracking.
"""


@dataclass
class TrainingMetrics:
    """Structured training metrics."""
    epoch: int
    batch: int
    total_batches: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    memory_usage: Optional[Dict[str, float]] = None
    batch_time: Optional[float] = None
    data_time: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class ErrorLog:
    """Structured error logging."""
    error_type: str
    error_message: str
    operation: str
    timestamp: datetime
    traceback: str
    context: Dict[str, Any]
    severity: str = "ERROR"

class AdvancedLogger:
    """Advanced logging system for AI training."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 experiment_name: str = "training_experiment",
                 log_level: int = logging.INFO,
                 max_log_files: int = 10,
                 max_log_size: int = 10 * 1024 * 1024):  # 10MB
        """
        Initialize advanced logging system.
        
        Args:
            log_dir: Directory to store log files
            experiment_name: Name of the training experiment
            log_level: Logging level
            max_log_files: Maximum number of log files to keep
            max_log_size: Maximum size of each log file in bytes
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_level = log_level
        self.max_log_files = max_log_files
        self.max_log_size = max_log_size
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k metrics
        self.error_history = deque(maxlen=1000)     # Keep last 1k errors
        self.training_start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # Performance tracking
        self.batch_times = deque(maxlen=100)
        self.data_times = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info(f"Advanced logging system initialized for experiment: {experiment_name}")
    
    def _setup_loggers(self) -> Any:
        """Setup different loggers for different purposes."""
        # Main logger
        self.logger = logging.getLogger(f"{self.experiment_name}_main")
        self.logger.setLevel(self.log_level)
        self.logger.handlers.clear()
        
        # Training progress logger
        self.training_logger = logging.getLogger(f"{self.experiment_name}_training")
        self.training_logger.setLevel(self.log_level)
        self.training_logger.handlers.clear()
        
        # Error logger
        self.error_logger = logging.getLogger(f"{self.experiment_name}_errors")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()
        
        # Metrics logger
        self.metrics_logger = logging.getLogger(f"{self.experiment_name}_metrics")
        self.metrics_logger.setLevel(self.log_level)
        self.metrics_logger.handlers.clear()
        
        # Setup handlers for each logger
        self._setup_main_logger()
        self._setup_training_logger()
        self._setup_error_logger()
        self._setup_metrics_logger()
    
    def _setup_main_logger(self) -> Any:
        """Setup main logger with console and file handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        main_log_file = self.log_dir / f"{self.experiment_name}_main.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=self.max_log_size,
            backupCount=self.max_log_files
        )
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_training_logger(self) -> Any:
        """Setup training progress logger."""
        training_log_file = self.log_dir / f"{self.experiment_name}_training.log"
        training_handler = logging.handlers.RotatingFileHandler(
            training_log_file,
            maxBytes=self.max_log_size,
            backupCount=self.max_log_files
        )
        training_handler.setLevel(self.log_level)
        training_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        training_handler.setFormatter(training_formatter)
        self.training_logger.addHandler(training_handler)
    
    def _setup_error_logger(self) -> Any:
        """Setup error logger."""
        error_log_file = self.log_dir / f"{self.experiment_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_log_size,
            backupCount=self.max_log_files
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
            'Traceback: %(traceback)s\n'
            'Context: %(context)s\n'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
    
    def _setup_metrics_logger(self) -> Any:
        """Setup metrics logger for structured data."""
        metrics_log_file = self.log_dir / f"{self.experiment_name}_metrics.jsonl"f"
        metrics_handler = logging.handlers.RotatingFileHandler(
            metrics_log_file,
            maxBytes=self.max_log_size,
            backupCount=self.max_log_files
        )
        metrics_handler.setLevel(self.log_level)
        
        # Custom formatter for JSON lines
        class JSONFormatter(logging.Formatter):
            def format(self, record) -> Any:
                if hasattr(record, 'metrics_data'):
                    return json.dumps(record.metrics_data, default=str)
                return super()"
        
        metrics_formatter = JSONFormatter()
        metrics_handler.setFormatter(metrics_formatter)
        self.metrics_logger.addHandler(metrics_handler)
    
    def start_training(self, config: Dict[str, Any]):
        """Log training start with configuration."""
        self.training_start_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Start time: {self.training_start_time}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
        
        # Log system information
        self._log_system_info()
        
        self.training_logger.info(f"Training started at {self.training_start_time}")
    
    def end_training(self, final_metrics: Dict[str, Any]):
        """Log training end with final metrics."""
        if self.training_start_time:
            training_duration = datetime.now() - self.training_start_time
            
            self.logger.info("=" * 80)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info("=" * 80)
            self.logger.info(f"Duration: {training_duration}")
            self.logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2, default=str)}")
            
            self.training_logger.info(f"Training completed in {training_duration}")
            
            # Save final summary
            self._save_training_summary(final_metrics, training_duration)
    
    def start_epoch(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
        
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")
        self.training_logger.info(f"Epoch {epoch}/{total_epochs} started at {self.epoch_start_time}")
    
    def end_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch end with metrics."""
        if self.epoch_start_time:
            epoch_duration = datetime.now() - self.epoch_start_time
            
            self.logger.info(f"Epoch {epoch} completed in {epoch_duration}")
            self.logger.info(f"Epoch metrics: {json.dumps(metrics, indent=2)}")
            
            self.training_logger.info(f"Epoch {epoch} completed in {epoch_duration}")
            self.training_logger.info(f"Epoch metrics: {metrics}")
    
    def log_batch_progress(self, 
                          epoch: int, 
                          batch: int, 
                          total_batches: int,
                          loss: float,
                          accuracy: Optional[float] = None,
                          learning_rate: Optional[float] = None,
                          gradient_norm: Optional[float] = None):
        """Log batch progress with detailed metrics."""
        self.batch_start_time = time.time()
        
        # Calculate progress percentage
        progress = (batch / total_batches) * 100
        
        # Create metrics object
        metrics = TrainingMetrics(
            epoch=epoch,
            batch=batch,
            total_batches=total_batches,
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            memory_usage=self._get_memory_usage(),
            timestamp=datetime.now()
        )
        
        # Store metrics
        with self._lock:
            self.metrics_history.append(metrics)
        
        # Log progress
        progress_msg = (f"Epoch {epoch}, Batch {batch}/{total_batches} "
                       f"({progress:.1f}%) - Loss: {loss:.4f}")
        
        if accuracy is not None:
            progress_msg += f", Accuracy: {accuracy:.4f}"
        
        if learning_rate is not None:
            progress_msg += f", LR: {learning_rate:.6f}"
        
        if gradient_norm is not None:
            progress_msg += f", Grad Norm: {gradient_norm:.4f}"
        
        # Log to training logger
        self.training_logger.info(progress_msg)
        
        # Log metrics as JSON
        self._log_metrics_json(metrics)
        
        # Log to console every 10 batches or at specific progress points
        if (batch % 10 == 0 or 
            batch == 1 or 
            batch == total_batches or 
            progress % 25 == 0):
            self.logger.info(progress_msg)
    
    def log_error(self, 
                  error: Exception, 
                  operation: str, 
                  context: Dict[str, Any] = None,
                  severity: str = "ERROR"):
        """Log error with detailed context."""
        error_log = ErrorLog(
            error_type=type(error).__name__,
            error_message=str(error),
            operation=operation,
            timestamp=datetime.now(),
            traceback=traceback.format_exc(),
            context=context or {},
            severity=severity
        )
        
        # Store error
        with self._lock:
            self.error_history.append(error_log)
        
        # Log to error logger
        error_msg = (f"Error in {operation}: {error}\n"
                    f"Traceback: {error_log.traceback}\n"
                    f"Context: {error_log.context}")
        
        self.error_logger.error(error_msg, extra={
            'traceback': error_log.traceback,
            'context': error_log.context
        })
        
        # Log to main logger
        self.logger.error(f"Error in {operation}: {error}")
    
    def log_validation(self, 
                      epoch: int, 
                      metrics: Dict[str, float],
                      is_best: bool = False):
        """Log validation results."""
        validation_msg = f"Validation (Epoch {epoch}): {json.dumps(metrics, indent=2)}"
        
        if is_best:
            validation_msg += " [BEST MODEL]"
            self.logger.info("🎉 New best model achieved!")
        
        self.logger.info(validation_msg)
        self.training_logger.info(validation_msg)
    
    def log_memory_usage(self) -> Any:
        """Log current memory usage."""
        memory_info = self._get_memory_usage()
        
        memory_msg = (f"Memory Usage - "
                     f"CPU: {memory_info.get('cpu_memory_gb', 0):.2f}GB, "
                     f"GPU: {memory_info.get('gpu_memory_gb', {}).get('allocated', 0):.2f}GB")
        
        self.training_logger.info(memory_msg)
        
        # Store memory usage
        with self._lock:
            self.memory_usage_history.append(memory_info)
    
    def log_performance_metrics(self, batch_time: float, data_time: float):
        """Log performance metrics."""
        with self._lock:
            self.batch_times.append(batch_time)
            self.data_times.append(data_time)
        
        avg_batch_time = np.mean(self.batch_times)
        avg_data_time = np.mean(self.data_times)
        
        performance_msg = (f"Performance - "
                          f"Batch time: {batch_time:.3f}s (avg: {avg_batch_time:.3f}s), "
                          f"Data time: {data_time:.3f}s (avg: {avg_data_time:.3f}s)")
        
        self.training_logger.info(performance_msg)
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "architecture": str(type(model).__name__)
        }
        
        self.logger.info(f"Model Information: {json.dumps(model_info, indent=2)}")
        self.training_logger.info(f"Model: {model_info['architecture']} "
                                f"({model_info['total_parameters']:,} parameters)")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.logger.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
        self.training_logger.info(f"Hyperparameters: {hyperparams}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            # Calculate statistics
            losses = [m.loss for m in self.metrics_history if m.loss is not None]
            accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
            
            summary = {
                "total_metrics": len(self.metrics_history),
                "total_errors": len(self.error_history),
                "training_duration": None,
                "loss_stats": {
                    "min": min(losses) if losses else None,
                    "max": max(losses) if losses else None,
                    "mean": np.mean(losses) if losses else None,
                    "std": np.std(losses) if losses else None
                },
                "accuracy_stats": {
                    "min": min(accuracies) if accuracies else None,
                    "max": max(accuracies) if accuracies else None,
                    "mean": np.mean(accuracies) if accuracies else None,
                    "std": np.std(accuracies) if accuracies else None
                }
            }
            
            if self.training_start_time:
                summary["training_duration"] = str(datetime.now() - self.training_start_time)
            
            return summary
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}
        
        try:
            process = psutil.Process()
            memory_info['cpu_memory_gb'] = process.memory_info().rss / (1024**3)
        except ImportError:
            memory_info['cpu_memory_gb'] = 0.0
        
        if torch.cuda.is_available():
            try:
                memory_info['gpu_memory_gb'] = {
                    'allocated': torch.cuda.memory_allocated() / (1024**3),
                    'reserved': torch.cuda.memory_reserved() / (1024**3),
                    'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)
                }
            except Exception:
                memory_info['gpu_memory_gb'] = {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}
        else:
            memory_info['gpu_memory_gb'] = {'allocated': 0.0, 'reserved': 0.0, 'max_allocated': 0.0}
        
        return memory_info
    
    def _log_system_info(self) -> Any:
        """Log system information."""
        system_info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "platform": sys.platform
        }
        
        self.logger.info(f"System Information: {json.dumps(system_info, indent=2)}")
    
    def _log_metrics_json(self, metrics: TrainingMetrics):
        """Log metrics as JSON line."""
        metrics_data = asdict(metrics)
        metrics_data['timestamp'] = metrics_data['timestamp'].isoformat()
        
        # Create a custom log record
        record = logging.LogRecord(
            name=self.metrics_logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None
        )
        record.metrics_data = metrics_data
        
        self.metrics_logger.handle(record)
    
    def _save_training_summary(self, final_metrics: Dict[str, Any], duration: timedelta):
        """Save training summary to file."""
        summary = {
            "experiment_name": self.experiment_name,
            "training_duration": str(duration),
            "final_metrics": final_metrics,
            "summary_stats": self.get_training_summary(),
            "error_count": len(self.error_history),
            "completion_time": datetime.now().isoformat()
        }
        
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved to {summary_file}")
    
    @contextmanager
    def training_context(self, operation: str):
        """Context manager for training operations with automatic error logging."""
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.training_logger.info(f"{operation} completed in {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start_time
            self.log_error(e, operation, {"duration": duration})
            raise

class TrainingProgressTracker:
    """Track and display training progress."""
    
    def __init__(self, logger: AdvancedLogger):
        
    """__init__ function."""
self.logger = logger
        self.epoch_progress = {}
        self.overall_progress = 0
    
    def update_progress(self, epoch: int, batch: int, total_batches: int, total_epochs: int):
        """Update and log progress."""
        # Calculate epoch progress
        epoch_progress = (batch / total_batches) * 100
        
        # Calculate overall progress
        epoch_weight = 1 / total_epochs
        overall_progress = ((epoch - 1) * epoch_weight + (batch / total_batches) * epoch_weight) * 100
        
        # Update progress
        self.epoch_progress[epoch] = epoch_progress
        self.overall_progress = overall_progress
        
        # Log progress
        progress_msg = (f"Progress - Epoch {epoch}: {epoch_progress:.1f}%, "
                       f"Overall: {overall_progress:.1f}%")
        
        self.logger.training_logger.info(progress_msg)
        
        # Log milestone achievements
        if overall_progress >= 25 and self.overall_progress < 25:
            self.logger.logger.info("🎯 25% of training completed!")
        elif overall_progress >= 50 and self.overall_progress < 50:
            self.logger.logger.info("🎯 50% of training completed!")
        elif overall_progress >= 75 and self.overall_progress < 75:
            self.logger.logger.info("🎯 75% of training completed!")
        elif overall_progress >= 90 and self.overall_progress < 90:
            self.logger.logger.info("🎯 90% of training completed!")

# Example usage
def example_usage():
    """Example of using the advanced logging system."""
    
    # Initialize logger
    logger = AdvancedLogger(
        log_dir="logs",
        experiment_name="example_training",
        log_level=logging.INFO
    )
    
    # Start training
    config = {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 10,
        "model": "ResNet18"
    }
    logger.start_training(config)
    
    # Log model info
    model = torch.nn.Linear(784, 10)
    logger.log_model_info(model)
    
    # Simulate training loop
    for epoch in range(3):
        logger.start_epoch(epoch + 1, 3)
        
        for batch in range(10):
            # Simulate training step
            loss = 1.0 - (epoch * 10 + batch) * 0.01
            accuracy = 0.5 + (epoch * 10 + batch) * 0.05
            
            logger.log_batch_progress(
                epoch=epoch + 1,
                batch=batch + 1,
                total_batches=10,
                loss=loss,
                accuracy=accuracy,
                learning_rate=1e-3
            )
        
        # Log validation
        val_metrics = {"val_loss": loss * 0.8, "val_accuracy": accuracy * 1.1}
        logger.log_validation(epoch + 1, val_metrics, is_best=(epoch == 2))
        
        logger.end_epoch(epoch + 1, val_metrics)
    
    # End training
    final_metrics = {"final_loss": 0.7, "final_accuracy": 0.85}
    logger.end_training(final_metrics)
    
    # Print summary
    summary = logger.get_training_summary()
    print(f"Training Summary: {json.dumps(summary, indent=2)}")

match __name__:
    case "__main__":
    example_usage() 