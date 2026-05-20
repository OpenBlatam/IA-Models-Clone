from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
import queue
import traceback
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from typing import Any, List, Dict, Optional
import asyncio
"""
Training Logger System

Comprehensive logging system for tracking training progress, metrics,
errors, and performance insights in the email sequence AI system.
"""



class LogLevel(Enum):
    """Log levels for training events"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TrainingEventType(Enum):
    """Types of training events"""
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    BATCH_START = "batch_start"
    BATCH_END = "batch_end"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    WARNING = "warning"
    METRIC_UPDATE = "metric_update"
    CONFIG_CHANGE = "config_change"
    RESOURCE_USAGE = "resource_usage"
    MODEL_SAVE = "model_save"
    EARLY_STOPPING = "early_stopping"
    LEARNING_RATE_CHANGE = "learning_rate_change"
    GRADIENT_UPDATE = "gradient_update"


@dataclass
class TrainingEvent:
    """Training event data structure"""
    timestamp: datetime
    event_type: TrainingEventType
    level: LogLevel
    message: str
    data: Dict[str, Any]
    epoch: Optional[int] = None
    batch: Optional[int] = None
    step: Optional[int] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Training metrics data structure"""
    epoch: int
    batch: int
    step: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_time: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None


class TrainingLogger:
    """Comprehensive training logger with progress tracking and error handling"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "email_sequence_training",
        log_level: LogLevel = LogLevel.INFO,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        enable_metrics_logging: bool = True,
        enable_visualization: bool = True,
        max_log_files: int = 10,
        flush_interval: int = 100
    ):
        """Initialize the training logger"""
        
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_level = log_level
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.enable_metrics_logging = enable_metrics_logging
        self.enable_visualization = enable_visualization
        self.max_log_files = max_log_files
        self.flush_interval = flush_interval
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging components
        self._setup_logging()
        self._setup_metrics_tracking()
        self._setup_event_queue()
        
        # Training state
        self.training_start_time = None
        self.current_epoch = 0
        self.current_batch = 0
        self.current_step = 0
        self.total_epochs = 0
        self.total_batches = 0
        self.total_steps = 0
        
        # Performance tracking
        self.epoch_times = []
        self.batch_times = []
        self.loss_history = []
        self.accuracy_history = []
        self.validation_loss_history = []
        self.validation_accuracy_history = []
        self.learning_rate_history = []
        self.gradient_norm_history = []
        
        # Error tracking
        self.error_count = 0
        self.warning_count = 0
        self.critical_errors = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.log_info(f"Training logger initialized for experiment: {experiment_name}")
    
    def _setup_logging(self) -> Any:
        """Setup logging configuration"""
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        self.logger = logging.getLogger(f"training.{self.experiment_name}")
        self.logger.setLevel(getattr(logging, self.log_level.value))
        self.logger.handlers.clear()
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.log_level.value))
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            log_file = self.log_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.log_level.value))
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
            
            # Error file handler
            error_log_file = self.log_dir / f"{self.experiment_name}_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            error_handler = logging.FileHandler(error_log_file)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(error_handler)
    
    def _setup_metrics_tracking(self) -> Any:
        """Setup metrics tracking"""
        
        if self.enable_metrics_logging:
            self.metrics_file = self.log_dir / f"{self.experiment_name}_metrics.jsonl"
            self.events_file = self.log_dir / f"{self.experiment_name}_events.jsonl"
            
            # Initialize metrics file
            if not self.metrics_file.exists():
                self.metrics_file.touch()
            
            # Initialize events file
            if not self.events_file.exists():
                self.events_file.touch()
    
    def _setup_event_queue(self) -> Any:
        """Setup event queue for async logging"""
        
        self.event_queue = queue.Queue()
        self.event_thread = threading.Thread(target=self._process_events, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self.event_thread.start()
    
    def _process_events(self) -> Any:
        """Process events from the queue"""
        
        while True:
            try:
                event = self.event_queue.get(timeout=1)
                if event is None:  # Shutdown signal
                    break
                
                self._write_event(event)
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
    
    def _write_event(self, event: TrainingEvent):
        """Write event to file"""
        
        try:
            with open(self.events_file, 'a', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                event_dict = asdict(event)
                event_dict['timestamp'] = event.timestamp.isoformat()
                event_dict['event_type'] = event.event_type.value
                event_dict['level'] = event.level.value
                json.dump(event_dict, f)
                f.write('\n')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception as e:
            print(f"Error writing event: {e}")
    
    def _write_metrics(self, metrics: TrainingMetrics):
        """Write metrics to file"""
        
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                metrics_dict = asdict(metrics)
                json.dump(metrics_dict, f)
                f.write('\n')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception as e:
            print(f"Error writing metrics: {e}")
    
    def _log_event(self, event_type: TrainingEventType, level: LogLevel, message: str, data: Dict[str, Any] = None):
        """Log a training event"""
        
        event = TrainingEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            level=level,
            message=message,
            data=data or {},
            epoch=self.current_epoch,
            batch=self.current_batch,
            step=self.current_step
        )
        
        # Add to queue for async processing
        self.event_queue.put(event)
        
        # Log to standard logger
        log_method = getattr(self.logger, level.value.lower())
        log_method(f"[{event_type.value.upper()}] {message}")
    
    def start_training(self, total_epochs: int, total_batches: int = None, config: Dict[str, Any] = None):
        """Start training session"""
        
        with self._lock:
            self.training_start_time = datetime.now()
            self.total_epochs = total_epochs
            self.total_batches = total_batches
            self.current_epoch = 0
            self.current_batch = 0
            self.current_step = 0
            
            # Reset tracking
            self.epoch_times = []
            self.batch_times = []
            self.loss_history = []
            self.accuracy_history = []
            self.validation_loss_history = []
            self.validation_accuracy_history = []
            self.learning_rate_history = []
            self.gradient_norm_history = []
            self.error_count = 0
            self.warning_count = 0
            self.critical_errors = []
        
        self._log_event(
            TrainingEventType.EPOCH_START,
            LogLevel.INFO,
            f"Starting training session - Total epochs: {total_epochs}",
            {"total_epochs": total_epochs, "total_batches": total_batches, "config": config}
        )
        
        self.log_info(f"Training started at {self.training_start_time}")
        if config:
            self.log_info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    def start_epoch(self, epoch: int, total_batches: int = None):
        """Start a new epoch"""
        
        with self._lock:
            self.current_epoch = epoch
            self.current_batch = 0
            epoch_start_time = time.time()
        
        self._log_event(
            TrainingEventType.EPOCH_START,
            LogLevel.INFO,
            f"Starting epoch {epoch}/{self.total_epochs}",
            {"epoch": epoch, "total_epochs": self.total_epochs, "total_batches": total_batches}
        )
        
        self.log_info(f"Epoch {epoch}/{self.total_epochs} started")
    
    def end_epoch(self, epoch_metrics: Dict[str, Any]):
        """End current epoch"""
        
        with self._lock:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - time.time()
            self.epoch_times.append(epoch_duration)
        
        # Update metrics
        if 'loss' in epoch_metrics:
            self.loss_history.append(epoch_metrics['loss'])
        if 'accuracy' in epoch_metrics:
            self.accuracy_history.append(epoch_metrics['accuracy'])
        if 'validation_loss' in epoch_metrics:
            self.validation_loss_history.append(epoch_metrics['validation_loss'])
        if 'validation_accuracy' in epoch_metrics:
            self.validation_accuracy_history.append(epoch_metrics['validation_accuracy'])
        if 'learning_rate' in epoch_metrics:
            self.learning_rate_history.append(epoch_metrics['learning_rate'])
        
        # Calculate progress
        progress = (self.current_epoch / self.total_epochs) * 100
        eta = self._calculate_eta()
        
        self._log_event(
            TrainingEventType.EPOCH_END,
            LogLevel.INFO,
            f"Epoch {self.current_epoch} completed - Progress: {progress:.1f}% - ETA: {eta}",
            {
                "epoch": self.current_epoch,
                "duration": epoch_duration,
                "metrics": epoch_metrics,
                "progress": progress,
                "eta": eta
            }
        )
        
        self.log_info(f"Epoch {self.current_epoch} completed in {epoch_duration:.2f}s")
        self.log_info(f"Metrics: {epoch_metrics}")
    
    def start_batch(self, batch: int, total_batches: int = None):
        """Start a new batch"""
        
        with self._lock:
            self.current_batch = batch
            self.current_step += 1
            batch_start_time = time.time()
        
        if batch % self.flush_interval == 0:
            self._log_event(
                TrainingEventType.BATCH_START,
                LogLevel.DEBUG,
                f"Batch {batch} started",
                {"batch": batch, "total_batches": total_batches, "step": self.current_step}
            )
    
    def end_batch(self, batch_metrics: Dict[str, Any]):
        """End current batch"""
        
        with self._lock:
            batch_end_time = time.time()
            batch_duration = batch_end_time - time.time()
            self.batch_times.append(batch_duration)
        
        # Update metrics
        if 'loss' in batch_metrics:
            self.loss_history.append(batch_metrics['loss'])
        if 'accuracy' in batch_metrics:
            self.accuracy_history.append(batch_metrics['accuracy'])
        if 'learning_rate' in batch_metrics:
            self.learning_rate_history.append(batch_metrics['learning_rate'])
        if 'gradient_norm' in batch_metrics:
            self.gradient_norm_history.append(batch_metrics['gradient_norm'])
        
        # Log metrics
        if self.enable_metrics_logging:
            metrics = TrainingMetrics(
                epoch=self.current_epoch,
                batch=self.current_batch,
                step=self.current_step,
                loss=batch_metrics.get('loss', 0.0),
                accuracy=batch_metrics.get('accuracy'),
                learning_rate=batch_metrics.get('learning_rate'),
                gradient_norm=batch_metrics.get('gradient_norm'),
                training_time=batch_duration,
                memory_usage=batch_metrics.get('memory_usage'),
                gpu_usage=batch_metrics.get('gpu_usage')
            )
            self._write_metrics(metrics)
        
        # Log progress periodically
        if self.current_batch % self.flush_interval == 0:
            progress = (self.current_batch / (self.total_batches or 1)) * 100
            self._log_event(
                TrainingEventType.BATCH_END,
                LogLevel.INFO,
                f"Batch {self.current_batch} completed - Progress: {progress:.1f}%",
                {
                    "batch": self.current_batch,
                    "duration": batch_duration,
                    "metrics": batch_metrics,
                    "progress": progress
                }
            )
    
    def log_validation(self, validation_metrics: Dict[str, Any]):
        """Log validation results"""
        
        self._log_event(
            TrainingEventType.VALIDATION,
            LogLevel.INFO,
            f"Validation completed - Loss: {validation_metrics.get('loss', 'N/A')}, Accuracy: {validation_metrics.get('accuracy', 'N/A')}",
            validation_metrics
        )
        
        self.log_info(f"Validation metrics: {validation_metrics}")
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, Any] = None):
        """Log model checkpoint"""
        
        self._log_event(
            TrainingEventType.CHECKPOINT,
            LogLevel.INFO,
            f"Model checkpoint saved to {checkpoint_path}",
            {"checkpoint_path": checkpoint_path, "metrics": metrics}
        )
        
        self.log_info(f"Checkpoint saved: {checkpoint_path}")
    
    def log_error(self, error: Exception, context: str = "", operation: str = ""):
        """Log training error"""
        
        with self._lock:
            self.error_count += 1
        
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Check if it's a critical error
        is_critical = any(keyword in error_message.lower() for keyword in [
            'out of memory', 'cuda error', 'critical', 'fatal', 'corruption'
        ])
        
        if is_critical:
            with self._lock:
                self.critical_errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error_message,
                    'context': context,
                    'operation': operation,
                    'stack_trace': stack_trace
                })
        
        self._log_event(
            TrainingEventType.ERROR,
            LogLevel.ERROR if not is_critical else LogLevel.CRITICAL,
            f"Training error in {operation}: {error_message}",
            {
                "error": error_message,
                "context": context,
                "operation": operation,
                "stack_trace": stack_trace,
                "is_critical": is_critical
            }
        )
        
        self.logger.error(f"Error in {operation}: {error_message}")
        if self.log_level == LogLevel.DEBUG:
            self.logger.error(f"Stack trace: {stack_trace}")
    
    def log_warning(self, message: str, context: str = ""):
        """Log training warning"""
        
        with self._lock:
            self.warning_count += 1
        
        self._log_event(
            TrainingEventType.WARNING,
            LogLevel.WARNING,
            f"Training warning: {message}",
            {"message": message, "context": context}
        )
        
        self.logger.warning(f"Warning: {message}")
    
    def log_metric_update(self, metric_name: str, value: float, step: int = None):
        """Log metric update"""
        
        self._log_event(
            TrainingEventType.METRIC_UPDATE,
            LogLevel.DEBUG,
            f"Metric update: {metric_name} = {value}",
            {"metric_name": metric_name, "value": value, "step": step or self.current_step}
        )
    
    def log_learning_rate_change(self, old_lr: float, new_lr: float, reason: str = ""):
        """Log learning rate change"""
        
        self._log_event(
            TrainingEventType.LEARNING_RATE_CHANGE,
            LogLevel.INFO,
            f"Learning rate changed: {old_lr} -> {new_lr}",
            {"old_lr": old_lr, "new_lr": new_lr, "reason": reason}
        )
        
        self.log_info(f"Learning rate changed from {old_lr} to {new_lr} ({reason})")
    
    def log_gradient_update(self, gradient_norm: float, clip_threshold: float = None):
        """Log gradient update"""
        
        data = {"gradient_norm": gradient_norm}
        if clip_threshold:
            data["clip_threshold"] = clip_threshold
            data["was_clipped"] = gradient_norm > clip_threshold
        
        self._log_event(
            TrainingEventType.GRADIENT_UPDATE,
            LogLevel.DEBUG,
            f"Gradient norm: {gradient_norm}",
            data
        )
    
    def log_resource_usage(self, memory_usage: float, gpu_usage: float = None, cpu_usage: float = None):
        """Log resource usage"""
        
        self._log_event(
            TrainingEventType.RESOURCE_USAGE,
            LogLevel.DEBUG,
            f"Resource usage - Memory: {memory_usage:.2f}GB, GPU: {gpu_usage:.1f}%",
            {
                "memory_usage": memory_usage,
                "gpu_usage": gpu_usage,
                "cpu_usage": cpu_usage
            }
        )
    
    def log_early_stopping(self, reason: str, best_epoch: int, best_metric: float):
        """Log early stopping"""
        
        self._log_event(
            TrainingEventType.EARLY_STOPPING,
            LogLevel.INFO,
            f"Early stopping triggered: {reason}",
            {
                "reason": reason,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "current_epoch": self.current_epoch
            }
        )
        
        self.log_info(f"Early stopping triggered: {reason} (Best epoch: {best_epoch}, Best metric: {best_metric})")
    
    def end_training(self, final_metrics: Dict[str, Any] = None):
        """End training session"""
        
        with self._lock:
            training_end_time = datetime.now()
            total_duration = training_end_time - self.training_start_time
        
        # Calculate statistics
        stats = self._calculate_training_stats()
        
        self._log_event(
            TrainingEventType.EPOCH_END,
            LogLevel.INFO,
            f"Training completed - Duration: {total_duration}",
            {
                "total_duration": str(total_duration),
                "final_metrics": final_metrics,
                "statistics": stats
            }
        )
        
        self.log_info(f"Training completed in {total_duration}")
        self.log_info(f"Final metrics: {final_metrics}")
        self.log_info(f"Training statistics: {stats}")
        
        # Generate training report
        self._generate_training_report(final_metrics, stats)
    
    def _calculate_training_stats(self) -> Dict[str, Any]:
        """Calculate training statistics"""
        
        with self._lock:
            stats = {
                "total_epochs": self.current_epoch,
                "total_steps": self.current_step,
                "total_errors": self.error_count,
                "total_warnings": self.warning_count,
                "critical_errors": len(self.critical_errors)
            }
            
            if self.epoch_times:
                stats["avg_epoch_time"] = np.mean(self.epoch_times)
                stats["total_training_time"] = sum(self.epoch_times)
            
            if self.batch_times:
                stats["avg_batch_time"] = np.mean(self.batch_times)
            
            if self.loss_history:
                stats["final_loss"] = self.loss_history[-1]
                stats["min_loss"] = min(self.loss_history)
                stats["max_loss"] = max(self.loss_history)
                stats["loss_improvement"] = self.loss_history[0] - self.loss_history[-1]
            
            if self.accuracy_history:
                stats["final_accuracy"] = self.accuracy_history[-1]
                stats["max_accuracy"] = max(self.accuracy_history)
            
            if self.validation_loss_history:
                stats["final_validation_loss"] = self.validation_loss_history[-1]
                stats["min_validation_loss"] = min(self.validation_loss_history)
            
            if self.validation_accuracy_history:
                stats["final_validation_accuracy"] = self.validation_accuracy_history[-1]
                stats["max_validation_accuracy"] = max(self.validation_accuracy_history)
        
        return stats
    
    def _calculate_eta(self) -> str:
        """Calculate estimated time to completion"""
        
        if not self.epoch_times or self.current_epoch == 0:
            return "Unknown"
        
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - self.current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        
        return str(timedelta(seconds=int(eta_seconds)))
    
    def _generate_training_report(self, final_metrics: Dict[str, Any], stats: Dict[str, Any]):
        """Generate comprehensive training report"""
        
        report_file = self.log_dir / f"{self.experiment_name}_report.json"
        
        report = {
            "experiment_name": self.experiment_name,
            "training_start": self.training_start_time.isoformat(),
            "training_end": datetime.now().isoformat(),
            "final_metrics": final_metrics,
            "statistics": stats,
            "error_summary": {
                "total_errors": self.error_count,
                "total_warnings": self.warning_count,
                "critical_errors": len(self.critical_errors),
                "critical_error_details": self.critical_errors
            },
            "performance_metrics": {
                "loss_history": self.loss_history,
                "accuracy_history": self.accuracy_history,
                "validation_loss_history": self.validation_loss_history,
                "validation_accuracy_history": self.validation_accuracy_history,
                "learning_rate_history": self.learning_rate_history,
                "gradient_norm_history": self.gradient_norm_history
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        self.log_info(f"Training report saved to {report_file}")
    
    def create_visualizations(self, save_path: str = None):
        """Create training visualizations"""
        
        if not self.enable_visualization:
            return
        
        if save_path is None:
            save_path = self.log_dir / f"{self.experiment_name}_visualizations.png"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Progress - {self.experiment_name}', fontsize=16)
        
        # Loss plot
        if self.loss_history:
            axes[0, 0].plot(self.loss_history, label='Training Loss')
            if self.validation_loss_history:
                axes[0, 0].plot(self.validation_loss_history, label='Validation Loss')
            axes[0, 0].set_title('Loss Over Time')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy plot
        if self.accuracy_history:
            axes[0, 1].plot(self.accuracy_history, label='Training Accuracy')
            if self.validation_accuracy_history:
                axes[0, 1].plot(self.validation_accuracy_history, label='Validation Accuracy')
            axes[0, 1].set_title('Accuracy Over Time')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate plot
        if self.learning_rate_history:
            axes[0, 2].plot(self.learning_rate_history)
            axes[0, 2].set_title('Learning Rate Over Time')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].grid(True)
        
        # Gradient norm plot
        if self.gradient_norm_history:
            axes[1, 0].plot(self.gradient_norm_history)
            axes[1, 0].set_title('Gradient Norm Over Time')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].grid(True)
        
        # Epoch times plot
        if self.epoch_times:
            axes[1, 1].plot(self.epoch_times)
            axes[1, 1].set_title('Epoch Duration Over Time')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Duration (seconds)')
            axes[1, 1].grid(True)
        
        # Error count plot
        if self.error_count > 0 or self.warning_count > 0:
            axes[1, 2].bar(['Errors', 'Warnings'], [self.error_count, self.warning_count])
            axes[1, 2].set_title('Error and Warning Counts')
            axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log_info(f"Training visualizations saved to {save_path}")
    
    @contextmanager
    def training_context(self, total_epochs: int, config: Dict[str, Any] = None):
        """Context manager for training sessions"""
        
        try:
            self.start_training(total_epochs, config=config)
            yield self
        except Exception as e:
            self.log_error(e, "Training session", "training_context")
            raise
        finally:
            self.end_training()
    
    @contextmanager
    def epoch_context(self, epoch: int, total_batches: int = None):
        """Context manager for epochs"""
        
        try:
            self.start_epoch(epoch, total_batches)
            yield self
        except Exception as e:
            self.log_error(e, f"Epoch {epoch}", "epoch_context")
            raise
    
    @contextmanager
    def batch_context(self, batch: int, total_batches: int = None):
        """Context manager for batches"""
        
        try:
            self.start_batch(batch, total_batches)
            yield self
        except Exception as e:
            self.log_error(e, f"Batch {batch}", "batch_context")
            raise
    
    # Convenience logging methods
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def log_critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        
        with self._lock:
            return {
                "experiment_name": self.experiment_name,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "current_step": self.current_step,
                "progress": (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "critical_errors": len(self.critical_errors),
                "training_start_time": self.training_start_time.isoformat() if self.training_start_time else None,
                "current_loss": self.loss_history[-1] if self.loss_history else None,
                "current_accuracy": self.accuracy_history[-1] if self.accuracy_history else None
            }
    
    def cleanup(self) -> Any:
        """Cleanup resources"""
        
        # Signal event thread to stop
        self.event_queue.put(None)
        self.event_thread.join(timeout=5)
        
        # Generate final visualizations
        if self.enable_visualization:
            self.create_visualizations()
        
        self.log_info("Training logger cleanup completed")


# Utility functions
def create_training_logger(
    experiment_name: str,
    log_dir: str = "logs",
    log_level: str = "INFO",
    **kwargs
) -> TrainingLogger:
    """Create a training logger with default settings"""
    
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        log_level=LogLevel(log_level.upper()),
        **kwargs
    )


def load_training_logs(log_file: str) -> List[TrainingEvent]:
    """Load training events from log file"""
    
    events = []
    with open(log_file, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        for line in f:
            try:
                event_data = json.loads(line.strip())
                event = TrainingEvent(
                    timestamp=datetime.fromisoformat(event_data['timestamp']),
                    event_type=TrainingEventType(event_data['event_type']),
                    level=LogLevel(event_data['level']),
                    message=event_data['message'],
                    data=event_data['data'],
                    epoch=event_data.get('epoch'),
                    batch=event_data.get('batch'),
                    step=event_data.get('step'),
                    duration=event_data.get('duration'),
                    error=event_data.get('error'),
                    stack_trace=event_data.get('stack_trace')
                )
                events.append(event)
            except Exception as e:
                print(f"Error loading event: {e}")
    
    return events


def analyze_training_logs(events: List[TrainingEvent]) -> Dict[str, Any]:
    """Analyze training events and extract insights"""
    
    analysis = {
        "total_events": len(events),
        "event_types": {},
        "error_analysis": {},
        "performance_analysis": {},
        "timeline_analysis": {}
    }
    
    # Count event types
    for event in events:
        event_type = event.event_type.value
        analysis["event_types"][event_type] = analysis["event_types"].get(event_type, 0) + 1
    
    # Analyze errors
    error_events = [e for e in events if e.event_type == TrainingEventType.ERROR]
    analysis["error_analysis"]["total_errors"] = len(error_events)
    analysis["error_analysis"]["critical_errors"] = len([e for e in error_events if e.level == LogLevel.CRITICAL])
    
    # Analyze performance
    epoch_events = [e for e in events if e.event_type == TrainingEventType.EPOCH_END]
    if epoch_events:
        analysis["performance_analysis"]["total_epochs"] = len(epoch_events)
        analysis["performance_analysis"]["avg_epoch_duration"] = np.mean([e.duration for e in epoch_events if e.duration])
    
    # Timeline analysis
    if events:
        start_time = min(e.timestamp for e in events)
        end_time = max(e.timestamp for e in events)
        analysis["timeline_analysis"]["start_time"] = start_time.isoformat()
        analysis["timeline_analysis"]["end_time"] = end_time.isoformat()
        analysis["timeline_analysis"]["total_duration"] = str(end_time - start_time)
    
    return analysis


if __name__ == "__main__":
    # Example usage
    logger = create_training_logger("test_experiment", log_level="DEBUG")
    
    with logger.training_context(10, {"learning_rate": 0.001}):
        for epoch in range(10):
            with logger.epoch_context(epoch, 100):
                for batch in range(100):
                    with logger.batch_context(batch, 100):
                        # Simulate training
                        loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.1)
                        accuracy = 0.8 + epoch * 0.02 + np.random.normal(0, 0.05)
                        
                        logger.end_batch({
                            "loss": loss,
                            "accuracy": accuracy,
                            "learning_rate": 0.001
                        })
                
                logger.end_epoch({
                    "loss": loss,
                    "accuracy": accuracy,
                    "validation_loss": loss * 1.1,
                    "validation_accuracy": accuracy * 0.95
                })
    
    logger.cleanup() 