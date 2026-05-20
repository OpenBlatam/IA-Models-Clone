from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import logging
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
import asyncio
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
from contextlib import contextmanager
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
from collections import defaultdict, deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from torch.autograd import detect_anomaly
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_config import settings
            import psutil
from typing import Any, List, Dict, Optional
"""
Comprehensive training logger service for ads generation features.
Provides detailed logging for training progress, errors, metrics, and performance monitoring.
"""


logger = setup_logger()

class LogLevel(Enum):
    """Log levels for training."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class TrainingPhase(Enum):
    """Training phases for detailed logging."""
    DATA_PREPARATION = "data_preparation"
    MODEL_LOADING = "model_loading"
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    CLEANUP = "cleanup"

class MetricType(Enum):
    """Types of metrics to log."""
    LOSS = "loss"
    ACCURACY = "accuracy"
    LEARNING_RATE = "learning_rate"
    GRADIENT_NORM = "gradient_norm"
    MEMORY_USAGE = "memory_usage"
    TIME = "time"
    CUSTOM = "custom"

@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    accuracy: Optional[float] = None
    gradient_norm: Optional[float] = None
    memory_usage: Optional[float] = None
    time_elapsed: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None
    phase: TrainingPhase = TrainingPhase.TRAINING
    timestamp: datetime = None
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ErrorLog:
    """Error logging data structure."""
    error_type: str
    error_message: str
    traceback: str
    phase: TrainingPhase
    user_id: Optional[int] = None
    model_name: Optional[str] = None
    timestamp: datetime = None
    context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> Any:
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TrainingProgress:
    """Training progress tracking."""
    total_epochs: int
    current_epoch: int
    total_steps: int
    current_step: int
    phase: TrainingPhase
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    status: str = "running"
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100
    
    @property
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed time."""
        return datetime.now() - self.start_time

class TrainingLogger:
    """Comprehensive training logger with multiple output formats."""
    
    def __init__(
        self,
        log_dir: str = "logs/training",
        user_id: Optional[int] = None,
        model_name: Optional[str] = None,
        enable_tensorboard: bool = True,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        enable_redis_logging: bool = True,
        max_log_files: int = 10,
        log_interval: int = 10,
        enable_autograd_debug: bool = False
    ):
        """Initialize the training logger."""
        self.log_dir = Path(log_dir)
        self.user_id = user_id
        self.model_name = model_name
        self.enable_tensorboard = enable_tensorboard
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.enable_redis_logging = enable_redis_logging
        self.max_log_files = max_log_files
        self.log_interval = log_interval
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_logging()
        self._setup_tensorboard()
        self._setup_redis()
        
        # Training state
        self.training_progress = None
        self.metrics_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=100)
        self.start_time = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.memory_tracker = MemoryTracker()
        
        # PyTorch debugging
        self.enable_autograd_debug = enable_autograd_debug
        self.autograd_context = None
        
        logger.info(f"Training logger initialized for user {user_id}, model {model_name}")
        if self.enable_autograd_debug:
            logger.info("PyTorch autograd anomaly detection enabled")
    
    def _setup_logging(self) -> Any:
        """Setup file and console logging."""
        if self.enable_file_logging:
            self._setup_file_logging()
        
        if self.enable_console_logging:
            self._setup_console_logging()
    
    def _setup_file_logging(self) -> Any:
        """Setup file logging with rotation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{self.user_id}_{self.model_name}_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        self.file_handler = file_handler
    
    def _setup_console_logging(self) -> Any:
        """Setup console logging."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        self.console_handler = console_handler
    
    def _setup_tensorboard(self) -> Any:
        """Setup TensorBoard logging."""
        if self.enable_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(tb_dir))
        else:
            self.tensorboard_writer = None
    
    def _setup_redis(self) -> Any:
        """Setup Redis for distributed logging."""
        self._redis_client = None
        if self.enable_redis_logging:
            self._redis_client = None  # Will be initialized lazily
    
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._redis_client is None and self.enable_redis_logging:
            self._redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    def start_training(
        self,
        total_epochs: int,
        total_steps: int,
        phase: TrainingPhase = TrainingPhase.TRAINING
    ):
        """Start training session."""
        self.start_time = datetime.now()
        self.training_progress = TrainingProgress(
            total_epochs=total_epochs,
            current_epoch=0,
            total_steps=total_steps,
            current_step=0,
            phase=phase,
            start_time=self.start_time
        )
        
        # Enable autograd debugging if requested
        if self.enable_autograd_debug:
            self.autograd_context = detect_anomaly()
            self.autograd_context.__enter__()
            self.log_info("PyTorch autograd anomaly detection started")
        
        self._log_training_start()
    
    def _log_training_start(self) -> Any:
        """Log training start information."""
        message = f"Training started - Epochs: {self.training_progress.total_epochs}, Steps: {self.training_progress.total_steps}"
        self.log_info(message, phase=self.training_progress.phase)
        
        # Log system information
        self._log_system_info()
    
    def _log_system_info(self) -> Any:
        """Log system information."""
        system_info = {
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "memory_usage": self.memory_tracker.get_memory_usage()
        }
        
        self.log_info(f"System info: {json.dumps(system_info, indent=2)}")
    
    def update_progress(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Update training progress and log metrics."""
        with self._lock:
            if self.training_progress:
                self.training_progress.current_epoch = epoch
                self.training_progress.current_step = step
            
            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                step=step,
                loss=loss,
                learning_rate=learning_rate,
                accuracy=accuracy,
                gradient_norm=gradient_norm,
                memory_usage=self.memory_tracker.get_memory_usage(),
                time_elapsed=(datetime.now() - self.start_time).total_seconds() if self.start_time else None,
                custom_metrics=custom_metrics,
                phase=self.training_progress.phase if self.training_progress else TrainingPhase.TRAINING
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Log at intervals
            if step % self.log_interval == 0:
                self._log_metrics(metrics)
                self._log_to_tensorboard(metrics)
                self._log_progress_update()
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        message = (
            f"Epoch {metrics.epoch}/{self.training_progress.total_epochs}, "
            f"Step {metrics.step}/{self.training_progress.total_steps} - "
            f"Loss: {metrics.loss:.4f}, LR: {metrics.learning_rate:.6f}"
        )
        
        if metrics.accuracy is not None:
            message += f", Accuracy: {metrics.accuracy:.4f}"
        
        if metrics.gradient_norm is not None:
            message += f", Grad Norm: {metrics.gradient_norm:.4f}"
        
        if metrics.memory_usage is not None:
            message += f", Memory: {metrics.memory_usage:.1f}MB"
        
        self.log_info(message, phase=metrics.phase)
    
    def _log_to_tensorboard(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard."""
        if self.tensorboard_writer:
            global_step = metrics.epoch * self.training_progress.total_steps + metrics.step
            
            self.tensorboard_writer.add_scalar('Loss/Train', metrics.loss, global_step)
            self.tensorboard_writer.add_scalar('Learning_Rate', metrics.learning_rate, global_step)
            
            if metrics.accuracy is not None:
                self.tensorboard_writer.add_scalar('Accuracy/Train', metrics.accuracy, global_step)
            
            if metrics.gradient_norm is not None:
                self.tensorboard_writer.add_scalar('Gradient_Norm', metrics.gradient_norm, global_step)
            
            if metrics.memory_usage is not None:
                self.tensorboard_writer.add_scalar('Memory_Usage', metrics.memory_usage, global_step)
            
            if metrics.custom_metrics:
                for name, value in metrics.custom_metrics.items():
                    self.tensorboard_writer.add_scalar(f'Custom/{name}', value, global_step)
    
    def _log_progress_update(self) -> Any:
        """Log progress update."""
        if self.training_progress:
            progress = self.training_progress.progress_percentage
            elapsed = self.training_progress.elapsed_time
            
            # Estimate completion time
            if progress > 0:
                estimated_total = elapsed / (progress / 100)
                estimated_remaining = estimated_total - elapsed
                estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining.total_seconds())
                
                message = (
                    f"Progress: {progress:.1f}% - "
                    f"Elapsed: {str(elapsed).split('.')[0]} - "
                    f"ETA: {estimated_completion.strftime('%H:%M:%S')}"
                )
                self.log_info(message)
    
    def log_error(
        self,
        error: Exception,
        phase: TrainingPhase,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log training error with full context."""
        error_log = ErrorLog(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            phase=phase,
            user_id=self.user_id,
            model_name=self.model_name,
            context=context or {}
        )
        
        # Store error
        self.error_history.append(error_log)
        
        # Log error
        message = f"Error in {phase.value}: {error_log.error_type} - {error_log.error_message}"
        self.log_error(message, phase=phase)
        
        # Log to file with full traceback
        if self.enable_file_logging:
            logger.error(f"Full error context: {json.dumps(asdict(error_log), indent=2)}")
    
    def log_info(self, message: str, phase: Optional[TrainingPhase] = None):
        """Log info message."""
        if phase:
            message = f"[{phase.value.upper()}] {message}"
        logger.info(message)
    
    def log_warning(self, message: str, phase: Optional[TrainingPhase] = None):
        """Log warning message."""
        if phase:
            message = f"[{phase.value.upper()}] {message}"
        logger.warning(message)
    
    def log_debug(self, message: str, phase: Optional[TrainingPhase] = None):
        """Log debug message."""
        if phase:
            message = f"[{phase.value.upper()}] {message}"
        logger.debug(message)
    
    def end_training(self, status: str = "completed"):
        """End training session."""
        if self.training_progress:
            self.training_progress.status = status
            self.training_progress.estimated_completion = datetime.now()
        
        # Disable autograd debugging
        if self.autograd_context:
            self.autograd_context.__exit__(None, None, None)
            self.autograd_context = None
            self.log_info("PyTorch autograd anomaly detection stopped")
        
        # Log final statistics
        self._log_training_summary()
        
        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
    
    def _log_training_summary(self) -> Any:
        """Log training summary."""
        if not self.metrics_history:
            return
        
        # Calculate statistics
        losses = [m.loss for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
        
        summary = {
            "total_steps": len(self.metrics_history),
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "max_loss": max(losses) if losses else None,
            "avg_loss": np.mean(losses) if losses else None,
            "final_accuracy": accuracies[-1] if accuracies else None,
            "max_accuracy": max(accuracies) if accuracies else None,
            "avg_accuracy": np.mean(accuracies) if accuracies else None,
            "total_time": (datetime.now() - self.start_time).total_seconds() if self.start_time else None,
            "errors_count": len(self.error_history)
        }
        
        self.log_info(f"Training summary: {json.dumps(summary, indent=2)}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
        
        losses = [m.loss for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics if m.accuracy is not None]
        
        return {
            "current_epoch": self.training_progress.current_epoch if self.training_progress else 0,
            "current_step": self.training_progress.current_step if self.training_progress else 0,
            "progress_percentage": self.training_progress.progress_percentage if self.training_progress else 0,
            "recent_loss_avg": np.mean(losses) if losses else None,
            "recent_accuracy_avg": np.mean(accuracies) if accuracies else None,
            "total_errors": len(self.error_history),
            "memory_usage": self.memory_tracker.get_memory_usage()
        }
    
    def create_training_plots(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """Create training visualization plots."""
        if not self.metrics_history:
            return {}
        
        plots = {}
        
        # Loss plot
        epochs = [m.epoch for m in self.metrics_history]
        losses = [m.loss for m in self.metrics_history]
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, losses, 'b-', alpha=0.7)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
        if accuracies:
            plt.subplot(2, 2, 2)
            plt.plot(accuracies, 'g-', alpha=0.7)
            plt.title('Training Accuracy')
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        lrs = [m.learning_rate for m in self.metrics_history]
        plt.subplot(2, 2, 3)
        plt.plot(lrs, 'r-', alpha=0.7)
        plt.title('Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        
        # Memory usage plot
        memory_usage = [m.memory_usage for m in self.metrics_history if m.memory_usage is not None]
        if memory_usage:
            plt.subplot(2, 2, 4)
            plt.plot(memory_usage, 'm-', alpha=0.7)
            plt.title('Memory Usage')
            plt.xlabel('Step')
            plt.ylabel('Memory (MB)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plots['training_plots'] = save_path
        
        plt.close()
        
        return plots
    
    async def save_logs_to_redis(self, key_prefix: str = "training_logs"):
        """Save logs to Redis for distributed access."""
        if not self.enable_redis_logging:
            return
        
        redis = await self.redis_client
        if not redis:
            return
        
        # Save metrics
        metrics_data = [asdict(m) for m in self.metrics_history]
        await redis.set(
            f"{key_prefix}:metrics:{self.user_id}:{self.model_name}",
            json.dumps(metrics_data),
            ex=3600  # 1 hour expiration
        )
        
        # Save errors
        errors_data = [asdict(e) for e in self.error_history]
        await redis.set(
            f"{key_prefix}:errors:{self.user_id}:{self.model_name}",
            json.dumps(errors_data),
            ex=3600
        )
        
        # Save training stats
        stats = self.get_training_stats()
        await redis.set(
            f"{key_prefix}:stats:{self.user_id}:{self.model_name}",
            json.dumps(stats),
            ex=3600
        )
    
    def cleanup_old_logs(self, days: int = 7):
        """Clean up old log files."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete log file {log_file}: {e}")
    
    def close(self) -> Any:
        """Close the logger and cleanup resources."""
        # Remove handlers
        if hasattr(self, 'file_handler'):
            logger.removeHandler(self.file_handler)
        
        if hasattr(self, 'console_handler'):
            logger.removeHandler(self.console_handler)
        
        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Disable autograd debugging if still active
        if self.autograd_context:
            self.autograd_context.__exit__(None, None, None)
            self.autograd_context = None
        
        logger.info("Training logger closed")
    
    def check_tensor_anomalies(self, tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Check for anomalies in a tensor (NaN, Inf, etc.)."""
        if not isinstance(tensor, torch.Tensor):
            return {"status": "not_tensor", "name": name}
        
        anomalies = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item(),
            "has_neg_inf": torch.isneginf(tensor).any().item(),
            "min_value": tensor.min().item() if tensor.numel() > 0 else None,
            "max_value": tensor.max().item() if tensor.numel() > 0 else None,
            "mean_value": tensor.mean().item() if tensor.numel() > 0 else None,
            "std_value": tensor.std().item() if tensor.numel() > 0 else None,
            "num_elements": tensor.numel()
        }
        
        # Log anomalies if found
        if anomalies["has_nan"] or anomalies["has_inf"] or anomalies["has_neg_inf"]:
            self.log_warning(f"Tensor anomalies detected in {name}: {anomalies}")
        
        return anomalies
    
    def check_gradient_anomalies(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check for gradient anomalies in model parameters."""
        gradient_info = {
            "total_params": 0,
            "params_with_grad": 0,
            "params_with_nan_grad": 0,
            "params_with_inf_grad": 0,
            "gradient_norms": [],
            "anomalous_params": []
        }
        
        for name, param in model.named_parameters():
            gradient_info["total_params"] += 1
            
            if param.grad is not None:
                gradient_info["params_with_grad"] += 1
                
                # Check for gradient anomalies
                grad = param.grad
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                
                if has_nan:
                    gradient_info["params_with_nan_grad"] += 1
                    gradient_info["anomalous_params"].append({
                        "name": name,
                        "issue": "nan_gradient",
                        "shape": list(grad.shape)
                    })
                
                if has_inf:
                    gradient_info["params_with_inf_grad"] += 1
                    gradient_info["anomalous_params"].append({
                        "name": name,
                        "issue": "inf_gradient",
                        "shape": list(grad.shape)
                    })
                
                # Calculate gradient norm
                grad_norm = grad.norm().item()
                gradient_info["gradient_norms"].append(grad_norm)
        
        # Log gradient statistics
        if gradient_info["gradient_norms"]:
            gradient_info["avg_grad_norm"] = np.mean(gradient_info["gradient_norms"])
            gradient_info["max_grad_norm"] = np.max(gradient_info["gradient_norms"])
            gradient_info["min_grad_norm"] = np.min(gradient_info["gradient_norms"])
        
        if gradient_info["params_with_nan_grad"] > 0 or gradient_info["params_with_inf_grad"] > 0:
            self.log_warning(f"Gradient anomalies detected: {gradient_info}")
        
        return gradient_info
    
    def enable_autograd_debugging(self) -> Any:
        """Enable PyTorch autograd anomaly detection."""
        if not self.enable_autograd_debug:
            self.enable_autograd_debug = True
            if self.autograd_context is None:
                self.autograd_context = detect_anomaly()
                self.autograd_context.__enter__()
                self.log_info("PyTorch autograd anomaly detection enabled")
    
    def disable_autograd_debugging(self) -> Any:
        """Disable PyTorch autograd anomaly detection."""
        if self.autograd_context:
            self.autograd_context.__exit__(None, None, None)
            self.autograd_context = None
            self.enable_autograd_debug = False
            self.log_info("PyTorch autograd anomaly detection disabled")
    
    def log_tensor_debug_info(self, tensor: torch.Tensor, name: str = "tensor", step: int = None):
        """Log detailed tensor debugging information."""
        anomalies = self.check_tensor_anomalies(tensor, name)
        
        if step is not None:
            anomalies["step"] = step
        
        # Log to TensorBoard if available
        if self.tensorboard_writer and step is not None:
            if anomalies["has_nan"]:
                self.tensorboard_writer.add_scalar(f'Debug/{name}_has_nan', 1, step)
            if anomalies["has_inf"]:
                self.tensorboard_writer.add_scalar(f'Debug/{name}_has_inf', 1, step)
            if anomalies["mean_value"] is not None:
                self.tensorboard_writer.add_scalar(f'Debug/{name}_mean', anomalies["mean_value"], step)
            if anomalies["std_value"] is not None:
                self.tensorboard_writer.add_scalar(f'Debug/{name}_std', anomalies["std_value"], step)
        
        return anomalies

class MemoryTracker:
    """Track memory usage during training."""
    
    def __init__(self) -> Any:
        """Initialize memory tracker."""
        self.start_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback to torch memory if available
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage."""
        return self._get_memory_usage()
    
    def get_memory_increase(self) -> float:
        """Get memory increase since start."""
        current = self._get_memory_usage()
        return current - self.start_memory

@contextmanager
def training_logger_context(
    user_id: int,
    model_name: str,
    log_dir: str = "logs/training",
    **kwargs
):
    """Context manager for training logger."""
    logger_instance = TrainingLogger(
        log_dir=log_dir,
        user_id=user_id,
        model_name=model_name,
        **kwargs
    )
    
    try:
        yield logger_instance
    finally:
        logger_instance.close()

class AsyncTrainingLogger(TrainingLogger):
    """Async version of training logger for async training loops."""
    
    async def log_async(self, message: str, level: LogLevel = LogLevel.INFO, phase: Optional[TrainingPhase] = None):
        """Async logging method."""
        if level == LogLevel.INFO:
            self.log_info(message, phase)
        elif level == LogLevel.WARNING:
            self.log_warning(message, phase)
        elif level == LogLevel.ERROR:
            self.log_error(Exception(message), phase or TrainingPhase.TRAINING)
        elif level == LogLevel.DEBUG:
            self.log_debug(message, phase)
    
    async def update_progress_async(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Async version of update_progress."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.update_progress,
            epoch, step, loss, learning_rate, accuracy, gradient_norm, custom_metrics
        )
    
    async def check_tensor_anomalies_async(self, tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Async version of check_tensor_anomalies."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check_tensor_anomalies, tensor, name)
    
    async def check_gradient_anomalies_async(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Async version of check_gradient_anomalies."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check_gradient_anomalies, model)
    
    async def log_tensor_debug_info_async(self, tensor: torch.Tensor, name: str = "tensor", step: int = None) -> Dict[str, Any]:
        """Async version of log_tensor_debug_info."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.log_tensor_debug_info, tensor, name, step)
    
    async def enable_autograd_debugging_async(self) -> Any:
        """Async version of enable_autograd_debugging."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.enable_autograd_debugging)
    
    async def disable_autograd_debugging_async(self) -> Any:
        """Async version of disable_autograd_debugging."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.disable_autograd_debugging)
    
    async def save_logs_periodically(self, interval: int = 60):
        """Periodically save logs to Redis."""
        while True:
            try:
                await self.save_logs_to_redis()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error saving logs to Redis: {e}")
                await asyncio.sleep(interval) 