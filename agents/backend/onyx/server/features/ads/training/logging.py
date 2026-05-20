"""
Unified Training Logging System for the ads feature.

This module consolidates all training logging functionality from the scattered implementations:
- training_logger.py (comprehensive training logging)

The new structure follows Clean Architecture principles with clear separation of concerns.
"""

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
# Heavy plotting libraries are imported lazily inside plotting functions to reduce import overhead
from io import StringIO
from torch.autograd import detect_anomaly
import psutil

from ...config import get_settings


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
    """Comprehensive training logger service for ads generation features."""
    
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
        self.enable_autograd_debug = enable_autograd_debug
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_logging()
        self._setup_file_logging()
        self._setup_console_logging()
        if self.enable_tensorboard:
            self._setup_tensorboard()
        if self.enable_redis_logging:
            self._setup_redis()
        
        # Training state
        self.training_progress: Optional[TrainingProgress] = None
        self.metrics_history: List[TrainingMetrics] = []
        self.error_logs: List[ErrorLog] = []
        self.memory_tracker = MemoryTracker()
        
        # Logging state
        self.last_log_time = time.time()
        self.logger = logging.getLogger(__name__)
        
        # Settings
        self.settings = get_settings()
    
    def _setup_logging(self) -> None:
        """Setup basic logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_file_logging(self) -> None:
        """Setup file logging."""
        if not self.enable_file_logging:
            return
        
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)
    
    def _setup_console_logging(self) -> None:
        """Setup console logging."""
        if not self.enable_console_logging:
            return
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(console_handler)
    
    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        try:
            tensorboard_dir = self.log_dir / "tensorboard"
            tensorboard_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(tensorboard_dir))
            self.logger.info(f"TensorBoard logging enabled at {tensorboard_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to setup TensorBoard: {e}")
            self.enable_tensorboard = False
    
    def _setup_redis(self) -> None:
        """Setup Redis logging."""
        try:
            self.redis_pool = None
            self.logger.info("Redis logging enabled")
        except Exception as e:
            self.logger.warning(f"Failed to setup Redis: {e}")
            self.enable_redis_logging = False
    
    @property
    async def redis_client(self):
        """Get Redis client."""
        if not self.enable_redis_logging or not self.redis_pool:
            return None
        
        try:
            return await self.redis_pool.acquire()
        except Exception as e:
            self.logger.warning(f"Failed to get Redis client: {e}")
            return None
    
    def start_training(
        self,
        total_epochs: int,
        total_steps: int,
        phase: TrainingPhase = TrainingPhase.TRAINING
    ):
        """Start training logging."""
        self.training_progress = TrainingProgress(
            total_epochs=total_epochs,
            current_epoch=0,
            total_steps=total_steps,
            current_step=0,
            phase=phase,
            start_time=datetime.now()
        )
        
        self._log_training_start()
        self._log_system_info()
        
        self.logger.info(f"Training started: {total_epochs} epochs, {total_steps} steps")
    
    def _log_training_start(self) -> None:
        """Log training start information."""
        start_info = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id,
            "model_name": self.model_name,
            "log_dir": str(self.log_dir),
            "settings": {
                "enable_tensorboard": self.enable_tensorboard,
                "enable_file_logging": self.enable_file_logging,
                "enable_console_logging": self.enable_console_logging,
                "enable_redis_logging": self.enable_redis_logging
            }
        }
        
        start_log_file = self.log_dir / "training_start.json"
        with open(start_log_file, 'w') as f:
            json.dump(start_info, f, indent=2, default=str)
    
    def _log_system_info(self) -> None:
        """Log system information."""
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        }
        
        system_log_file = self.log_dir / "system_info.json"
        with open(system_log_file, 'w') as f:
            json.dump(system_info, f, indent=2, default=str)
    
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
        if not self.training_progress:
            return
        
        # Update progress
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
            time_elapsed=time.time() - self.last_log_time,
            custom_metrics=custom_metrics,
            phase=self.training_progress.phase
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Log metrics
        self._log_metrics(metrics)
        
        # Update last log time
        self.last_log_time = time.time()
        
        # Log progress periodically
        if step % self.log_interval == 0:
            self._log_progress_update()
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        # Console logging
        log_message = (
            f"Epoch {metrics.epoch}/{self.training_progress.total_epochs}, "
            f"Step {metrics.step}/{self.training_progress.total_steps}, "
            f"Loss: {metrics.loss:.4f}, "
            f"LR: {metrics.learning_rate:.6f}"
        )
        
        if metrics.accuracy is not None:
            log_message += f", Accuracy: {metrics.accuracy:.4f}"
        
        self.logger.info(log_message)
        
        # TensorBoard logging
        if self.enable_tensorboard:
            self._log_to_tensorboard(metrics)
        
        # File logging
        try:
            metrics_file = self.log_dir / "metrics.jsonl"
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(asdict(metrics), default=str) + '\n')
        except Exception as e:
            self.logger.warning(f"Failed to write metrics file: {e}")
    
    def _log_to_tensorboard(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard."""
        try:
            global_step = metrics.epoch * self.training_progress.total_steps + metrics.step
            
            self.tensorboard_writer.add_scalar('Loss', metrics.loss, global_step)
            self.tensorboard_writer.add_scalar('Learning_Rate', metrics.learning_rate, global_step)
            
            if metrics.accuracy is not None:
                self.tensorboard_writer.add_scalar('Accuracy', metrics.accuracy, global_step)
            
            if metrics.gradient_norm is not None:
                self.tensorboard_writer.add_scalar('Gradient_Norm', metrics.gradient_norm, global_step)
            
            if metrics.memory_usage is not None:
                self.tensorboard_writer.add_scalar('Memory_Usage', metrics.memory_usage, global_step)
            
            if metrics.custom_metrics:
                for name, value in metrics.custom_metrics.items():
                    self.tensorboard_writer.add_scalar(f'Custom/{name}', value, global_step)
            
            self.tensorboard_writer.flush()
        except Exception as e:
            self.logger.warning(f"Failed to log to TensorBoard: {e}")
    
    def _log_progress_update(self) -> None:
        """Log progress update."""
        if not self.training_progress:
            return
        
        progress = self.training_progress.progress_percentage
        elapsed = self.training_progress.elapsed_time
        
        self.logger.info(
            f"Progress: {progress:.1f}% ({elapsed.total_seconds():.0f}s elapsed)"
        )
    
    def log_error(
        self,
        error: Exception,
        phase: TrainingPhase,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log training error."""
        error_log = ErrorLog(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            phase=phase,
            user_id=self.user_id,
            model_name=self.model_name,
            context=context
        )
        
        self.error_logs.append(error_log)
        
        # Log to console
        self.logger.error(f"Training error in {phase.value}: {error}")
        
        # Log to file
        try:
            error_file = self.log_dir / "errors.jsonl"
            with open(error_file, 'a') as f:
                f.write(json.dumps(asdict(error_log), default=str) + '\n')
        except Exception as e:
            self.logger.warning(f"Failed to write error file: {e}")
        
        # Log to TensorBoard
        if self.enable_tensorboard:
            try:
                global_step = (
                    self.training_progress.current_epoch * self.training_progress.total_steps +
                    self.training_progress.current_step
                )
                self.tensorboard_writer.add_text('Error', str(error), global_step)
            except Exception as e:
                self.logger.warning(f"Failed to log error to TensorBoard: {e}")
    
    def log_info(self, message: str, phase: Optional[TrainingPhase] = None):
        """Log info message."""
        phase_str = f" [{phase.value}]" if phase else ""
        self.logger.info(f"{message}{phase_str}")
    
    def log_warning(self, message: str, phase: Optional[TrainingPhase] = None):
        """Log warning message."""
        phase_str = f" [{phase.value}]" if phase else ""
        self.logger.warning(f"{message}{phase_str}")
    
    def log_debug(self, message: str, phase: Optional[TrainingPhase] = None):
        """Log debug message."""
        phase_str = f" [{phase.value}]" if phase else ""
        self.logger.debug(f"{message}{phase_str}")
    
    def end_training(self, status: str = "completed"):
        """End training logging."""
        if not self.training_progress:
            return
        
        self.training_progress.status = status
        end_time = datetime.now()
        duration = end_time - self.training_progress.start_time
        
        # Log training summary
        self._log_training_summary()
        
        # Close TensorBoard
        if self.enable_tensorboard:
            self.tensorboard_writer.close()
        
        self.logger.info(f"Training {status} in {duration}")
    
    def _log_training_summary(self) -> None:
        """Log training summary."""
        if not self.training_progress:
            return
        
        summary = {
            "status": self.training_progress.status,
            "start_time": self.training_progress.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.training_progress.start_time).total_seconds(),
            "total_epochs": self.training_progress.total_epochs,
            "total_steps": self.training_progress.total_steps,
            "metrics_count": len(self.metrics_history),
            "errors_count": len(self.error_logs),
            "final_loss": self.metrics_history[-1].loss if self.metrics_history else None,
            "final_accuracy": self.metrics_history[-1].accuracy if self.metrics_history else None
        }
        
        try:
            summary_file = self.log_dir / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to write training summary: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_progress:
            return {}
        
        if not self.metrics_history:
            return {"status": "no_metrics"}
        
        losses = [m.loss for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
        learning_rates = [m.learning_rate for m in self.metrics_history]
        
        stats = {
            "total_metrics": len(self.metrics_history),
            "total_errors": len(self.error_logs),
            "loss_stats": {
                "min": min(losses),
                "max": max(losses),
                "mean": np.mean(losses),
                "std": np.std(losses)
            },
            "learning_rate_stats": {
                "min": min(learning_rates),
                "max": max(learning_rates),
                "mean": np.mean(learning_rates)
            }
        }
        
        if accuracies:
            stats["accuracy_stats"] = {
                "min": min(accuracies),
                "max": max(accuracies),
                "mean": np.mean(accuracies),
                "std": np.std(accuracies)
            }
        
        return stats
    
    def create_training_plots(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """Create training plots."""
        if not self.metrics_history:
            return {}
        
        plots = {}
        
        try:
            # Lazy import heavy plotting libraries
            import matplotlib.pyplot as plt  # type: ignore
            try:
                import seaborn as sns  # type: ignore
                sns.set_theme(style="whitegrid")
            except Exception:
                pass
            # Loss plot
            plt.figure(figsize=(10, 6))
            epochs = [m.epoch for m in self.metrics_history]
            losses = [m.loss for m in self.metrics_history]
            plt.plot(epochs, losses, 'b-', label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.grid(True)
            
            loss_plot_path = save_path or str(self.log_dir / "loss_plot.png")
            plt.savefig(loss_plot_path)
            plt.close()
            plots['loss'] = loss_plot_path
            
            # Accuracy plot (if available)
            accuracies = [m.accuracy for m in self.metrics_history if m.accuracy is not None]
            if accuracies:
                plt.figure(figsize=(10, 6))
                plt.plot(epochs[:len(accuracies)], accuracies, 'g-', label='Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Training Accuracy Over Time')
                plt.legend()
                plt.grid(True)
                
                acc_plot_path = save_path or str(self.log_dir / "accuracy_plot.png")
                plt.savefig(acc_plot_path)
                plt.close()
                plots['accuracy'] = acc_plot_path
            
            # Learning rate plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, learning_rates, 'r-', label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Over Time')
            plt.legend()
            plt.grid(True)
            
            lr_plot_path = save_path or str(self.log_dir / "learning_rate_plot.png")
            plt.savefig(lr_plot_path)
            plt.close()
            plots['learning_rate'] = lr_plot_path
            
        except Exception as e:
            self.logger.error(f"Failed to create training plots: {e}")
        
        return plots
    
    async def save_logs_to_redis(self, key_prefix: str = "training_logs"):
        """Save logs to Redis."""
        if not self.enable_redis_logging:
            return
        
        try:
            redis_client = await self.redis_client
            if not redis_client:
                return
            
            # Save metrics
            if self.metrics_history:
                metrics_key = f"{key_prefix}:metrics:{self.user_id}:{self.model_name}"
                metrics_data = [asdict(m) for m in self.metrics_history[-100:]]  # Last 100 metrics
                await redis_client.set(metrics_key, json.dumps(metrics_data))
            
            # Save errors
            if self.error_logs:
                errors_key = f"{key_prefix}:errors:{self.user_id}:{self.model_name}"
                errors_data = [asdict(e) for e in self.error_logs]
                await redis_client.set(errors_key, json.dumps(errors_data))
            
            # Save summary
            summary_key = f"{key_prefix}:summary:{self.user_id}:{self.model_name}"
            summary_data = self.get_training_stats()
            await redis_client.set(summary_key, json.dumps(summary_data))
            
            self.logger.info("Logs saved to Redis successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save logs to Redis: {e}")
    
    def cleanup_old_logs(self, days: int = 7):
        """Clean up old log files."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    log_file.unlink()
                    self.logger.info(f"Removed old log file: {log_file}")
            
            for plot_file in self.log_dir.glob("*.png"):
                if plot_file.stat().st_mtime < cutoff_time.timestamp():
                    plot_file.unlink()
                    self.logger.info(f"Removed old plot file: {plot_file}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
    
    def close(self) -> None:
        """Close the logger and cleanup resources."""
        try:
            if self.enable_tensorboard:
                self.tensorboard_writer.close()
            
            if self.enable_redis_logging and self.redis_pool:
                asyncio.create_task(self.redis_pool.close())
            
            self.logger.info("Training logger closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close training logger: {e}")
    
    def check_tensor_anomalies(self, tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Check for tensor anomalies."""
        try:
            anomalies = {
                "has_nan": torch.isnan(tensor).any().item(),
                "has_inf": torch.isinf(tensor).any().item(),
                "min_value": tensor.min().item(),
                "max_value": tensor.max().item(),
                "mean_value": tensor.mean().item(),
                "std_value": tensor.std().item(),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype)
            }
            
            if anomalies["has_nan"] or anomalies["has_inf"]:
                self.logger.warning(f"Tensor {name} has anomalies: {anomalies}")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to check tensor anomalies: {e}")
            return {"error": str(e)}
    
    def check_gradient_anomalies(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check for gradient anomalies."""
        try:
            anomalies = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_anomalies = self.check_tensor_anomalies(param.grad, f"grad_{name}")
                    anomalies[name] = grad_anomalies
                    
                    # Check for gradient explosion/vanishing
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 10.0:
                        self.logger.warning(f"Gradient explosion detected in {name}: {grad_norm}")
                    elif grad_norm < 1e-6:
                        self.logger.warning(f"Gradient vanishing detected in {name}: {grad_norm}")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to check gradient anomalies: {e}")
            return {"error": str(e)}
    
    def enable_autograd_debugging(self) -> None:
        """Enable PyTorch autograd debugging."""
        try:
            detect_anomaly()
            self.enable_autograd_debug = True
            self.logger.info("Autograd debugging enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable autograd debugging: {e}")
    
    def disable_autograd_debugging(self) -> None:
        """Disable PyTorch autograd debugging."""
        try:
            self.enable_autograd_debug = False
            self.logger.info("Autograd debugging disabled")
        except Exception as e:
            self.logger.error(f"Failed to disable autograd debugging: {e}")
    
    def log_tensor_debug_info(self, tensor: torch.Tensor, name: str = "tensor", step: int = None):
        """Log detailed tensor debug information."""
        try:
            debug_info = self.check_tensor_anomalies(tensor, name)
            
            if step is not None and self.enable_tensorboard:
                for key, value in debug_info.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f'Tensor/{name}_{key}', value, step)
            
            self.logger.debug(f"Tensor {name} debug info: {debug_info}")
            
        except Exception as e:
            self.logger.error(f"Failed to log tensor debug info: {e}")


class MemoryTracker:
    """Memory usage tracker."""
    
    def __init__(self) -> None:
        """Initialize memory tracker."""
        self.initial_memory = self._get_memory_usage()
        self.last_memory = self.initial_memory
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage."""
        current_memory = self._get_memory_usage()
        self.last_memory = current_memory
        return current_memory
    
    def get_memory_increase(self) -> float:
        """Get memory increase since initialization."""
        current_memory = self._get_memory_usage()
        return current_memory - self.initial_memory


@contextmanager
def training_logger_context(
    user_id: int,
    model_name: str,
    log_dir: str = "logs/training",
    **kwargs
):
    """Context manager for training logger."""
    logger = TrainingLogger(
        log_dir=log_dir,
        user_id=user_id,
        model_name=model_name,
        **kwargs
    )
    try:
        yield logger
    finally:
        logger.close()


class AsyncTrainingLogger(TrainingLogger):
    """Asynchronous training logger."""
    
    async def log_async(self, message: str, level: LogLevel = LogLevel.INFO, phase: Optional[TrainingPhase] = None):
        """Log message asynchronously."""
        phase_str = f" [{phase.value}]" if phase else ""
        
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
        """Update progress asynchronously."""
        self.update_progress(
            epoch, step, loss, learning_rate, accuracy, gradient_norm, custom_metrics
        )
    
    async def check_tensor_anomalies_async(self, tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Check tensor anomalies asynchronously."""
        return self.check_tensor_anomalies(tensor, name)
    
    async def check_gradient_anomalies_async(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Check gradient anomalies asynchronously."""
        return self.check_gradient_anomalies(model)
    
    async def log_tensor_debug_info_async(self, tensor: torch.Tensor, name: str = "tensor", step: int = None) -> Dict[str, Any]:
        """Log tensor debug info asynchronously."""
        self.log_tensor_debug_info(tensor, name, step)
        return {"status": "logged"}
    
    async def enable_autograd_debugging_async(self) -> None:
        """Enable autograd debugging asynchronously."""
        self.enable_autograd_debugging()
    
    async def disable_autograd_debugging_async(self) -> None:
        """Disable autograd debugging asynchronously."""
        self.disable_autograd_debugging()
    
    async def save_logs_periodically(self, interval: int = 60):
        """Save logs to Redis periodically."""
        while True:
            await asyncio.sleep(interval)
            await self.save_logs_to_redis()


# Global utility functions
def get_training_logger(
    log_dir: str = "logs/training",
    user_id: Optional[int] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> TrainingLogger:
    """Get a global training logger instance."""
    return TrainingLogger(log_dir, user_id, model_name, **kwargs)


def get_async_training_logger(
    log_dir: str = "logs/training",
    user_id: Optional[int] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> AsyncTrainingLogger:
    """Get a global async training logger instance."""
    return AsyncTrainingLogger(log_dir, user_id, model_name, **kwargs)
