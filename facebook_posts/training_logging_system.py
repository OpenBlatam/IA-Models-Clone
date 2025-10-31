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

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import logging
import traceback
import time
import os
import gc
import json
import csv
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
import threading
import queue
from contextlib import contextmanager
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
    from deep_learning_framework import DeepLearningFramework, FrameworkConfig, TaskType
    from evaluation_metrics import EvaluationManager, MetricConfig, MetricType
    from gradient_clipping_nan_handling import NumericalStabilityManager
    from early_stopping_scheduling import TrainingManager
    from efficient_data_loading import EfficientDataLoader
    from data_splitting_validation import DataSplitter
    from training_evaluation import TrainingManager as TrainingEvalManager
    from diffusion_models import DiffusionModel, DiffusionConfig
    from advanced_transformers import AdvancedTransformerModel
    from llm_training import AdvancedLLMTrainer
    from model_finetuning import ModelFineTuner
    from custom_modules import AdvancedNeuralNetwork
    from weight_initialization import AdvancedWeightInitializer
    from normalization_techniques import AdvancedLayerNorm
    from loss_functions import AdvancedCrossEntropyLoss
    from optimization_algorithms import AdvancedAdamW
    from attention_mechanisms import MultiHeadAttention
    from tokenization_sequence import AdvancedTokenizer
from framework_utils import MetricsTracker, ModelAnalyzer, PerformanceMonitor
from deep_learning_integration import DeepLearningIntegration, IntegrationConfig, IntegrationType, ComponentType
    from robust_error_handling import RobustErrorHandler, RobustDataLoader, RobustModelHandler
import psutil
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Training Logging System
Comprehensive logging for training progress and errors with real-time monitoring.
"""

warnings.filterwarnings('ignore')

# Import our custom modules
try:
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")


class LogLevel(Enum):
    """Log levels for training logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TrainingPhase(Enum):
    """Training phases for detailed logging."""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    MODEL_SETUP = "model_setup"
    TRAINING_START = "training_start"
    EPOCH_START = "epoch_start"
    BATCH_START = "batch_start"
    FORWARD_PASS = "forward_pass"
    LOSS_COMPUTATION = "loss_computation"
    BACKWARD_PASS = "backward_pass"
    OPTIMIZATION_STEP = "optimization_step"
    BATCH_END = "batch_end"
    EPOCH_END = "epoch_end"
    VALIDATION = "validation"
    CHECKPOINT_SAVING = "checkpoint_saving"
    TRAINING_END = "training_end"
    ERROR_RECOVERY = "error_recovery"


class ErrorCategory(Enum):
    """Error categories for detailed error tracking."""
    DATA_LOADING = "data_loading"
    MODEL_INFERENCE = "model_inference"
    LOSS_COMPUTATION = "loss_computation"
    GRADIENT_COMPUTATION = "gradient_computation"
    OPTIMIZATION = "optimization"
    MEMORY = "memory"
    DEVICE = "device"
    CHECKPOINT = "checkpoint"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class TrainingMetrics:
    """Training metrics for logging."""
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    memory_usage: float = 0.0
    gpu_memory: float = 0.0
    training_time: float = 0.0
    validation_loss: float = 0.0
    validation_accuracy: float = 0.0
    best_loss: float = float('inf')
    best_accuracy: float = 0.0
    patience_counter: int = 0
    early_stopping_triggered: bool = False


@dataclass
class ErrorInfo:
    """Error information for detailed logging."""
    category: ErrorCategory
    phase: TrainingPhase
    error_message: str
    traceback: str
    timestamp: datetime
    epoch: int = 0
    batch: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_time: float = 0.0


class TrainingLogger:
    """Comprehensive training logger with progress and error tracking."""
    
    def __init__(self, log_dir: str = "training_logs", experiment_name: str = None):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Training state
        self.metrics = TrainingMetrics()
        self.error_history: List[ErrorInfo] = []
        self.progress_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()
        
        # Real-time monitoring
        self.monitoring_queue = queue.Queue()
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Log files
        self.metrics_file = self.experiment_dir / "training_metrics.csv"
        self.errors_file = self.experiment_dir / "training_errors.json"
        self.progress_file = self.experiment_dir / "training_progress.json"
        self.config_file = self.experiment_dir / "training_config.json"
        
        # Initialize log files
        self._initialize_log_files()
        
        # Start monitoring
        self.start_monitoring()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger(f"training_logger_{self.experiment_name}")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.experiment_dir / "training.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_log_files(self) -> Any:
        """Initialize log files with headers."""
        # Metrics CSV file
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'epoch', 'batch', 'total_batches', 'loss', 'accuracy',
                'learning_rate', 'gradient_norm', 'memory_usage', 'gpu_memory',
                'training_time', 'validation_loss', 'validation_accuracy',
                'best_loss', 'best_accuracy', 'patience_counter'
            ])
        
        # Errors JSON file
        with open(self.errors_file, 'w') as f:
            json.dump([], f, indent=2)
        
        # Progress JSON file
        with open(self.progress_file, 'w') as f:
            json.dump([], f, indent=2)
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start with configuration."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Start time: {datetime.now()}")
        self.logger.info(f"Log directory: {self.experiment_dir}")
        
        # Log configuration
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save configuration
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.start_time = time.time()
    
    def log_phase_start(self, phase: TrainingPhase, **kwargs):
        """Log the start of a training phase."""
        self.logger.debug(f"Phase started: {phase.value}")
        
        if phase == TrainingPhase.EPOCH_START:
            self.epoch_start_time = time.time()
            self.metrics.epoch = kwargs.get('epoch', 0)
            self.logger.info(f"Epoch {self.metrics.epoch} started")
        
        elif phase == TrainingPhase.BATCH_START:
            self.batch_start_time = time.time()
            self.metrics.batch = kwargs.get('batch', 0)
            self.metrics.total_batches = kwargs.get('total_batches', 0)
        
        # Log phase with context
        phase_data = {
            'phase': phase.value,
            'timestamp': datetime.now().isoformat(),
            'epoch': self.metrics.epoch,
            'batch': self.metrics.batch,
            **kwargs
        }
        
        self.progress_history.append(phase_data)
        self._save_progress()
    
    def log_phase_end(self, phase: TrainingPhase, **kwargs):
        """Log the end of a training phase."""
        self.logger.debug(f"Phase ended: {phase.value}")
        
        if phase == TrainingPhase.EPOCH_END:
            epoch_time = time.time() - self.epoch_start_time
            self.logger.info(f"Epoch {self.metrics.epoch} completed in {epoch_time:.2f}s")
        
        elif phase == TrainingPhase.BATCH_END:
            batch_time = time.time() - self.batch_start_time
            if self.metrics.batch % 10 == 0:  # Log every 10 batches
                self.logger.info(f"Batch {self.metrics.batch}/{self.metrics.total_batches} "
                               f"completed in {batch_time:.3f}s")
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics."""
        # Update metrics
        for key, value in metrics.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        
        # Calculate additional metrics
        self.metrics.training_time = time.time() - self.start_time
        self.metrics.memory_usage = self._get_memory_usage()
        self.metrics.gpu_memory = self._get_gpu_memory_usage()
        
        # Log to console
        if self.metrics.batch % 10 == 0:  # Log every 10 batches
            self.logger.info(
                f"Epoch {self.metrics.epoch}, "
                f"Batch {self.metrics.batch}/{self.metrics.total_batches}, "
                f"Loss: {self.metrics.loss:.4f}, "
                f"Accuracy: {self.metrics.accuracy:.4f}, "
                f"LR: {self.metrics.learning_rate:.6f}"
            )
        
        # Save to CSV
        self._save_metrics()
        
        # Update progress
        self._update_progress()
    
    def log_error(self, error: Exception, phase: TrainingPhase, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
        """Log training error with detailed information."""
        error_info = ErrorInfo(
            category=category,
            phase=phase,
            error_message=str(error),
            traceback=traceback.format_exc(),
            timestamp=datetime.now(),
            epoch=self.metrics.epoch,
            batch=self.metrics.batch,
            context_data=kwargs
        )
        
        # Log error
        self.logger.error(f"Error in {phase.value}: {str(error)}")
        self.logger.error(f"Category: {category.value}")
        self.logger.error(f"Epoch: {self.metrics.epoch}, Batch: {self.metrics.batch}")
        self.logger.error(f"Traceback: {error_info.traceback}")
        
        # Add to history
        self.error_history.append(error_info)
        
        # Save errors
        self._save_errors()
        
        # Update error statistics
        self._update_error_statistics()
    
    def log_recovery_attempt(self, error_info: ErrorInfo, recovery_successful: bool, recovery_time: float):
        """Log error recovery attempt."""
        error_info.recovery_attempted = True
        error_info.recovery_successful = recovery_successful
        error_info.recovery_time = recovery_time
        
        if recovery_successful:
            self.logger.info(f"Recovery successful for {error_info.category.value} error")
        else:
            self.logger.warning(f"Recovery failed for {error_info.category.value} error")
    
    def log_validation_results(self, validation_metrics: Dict[str, Any]):
        """Log validation results."""
        self.logger.info("Validation Results:")
        for metric, value in validation_metrics.items():
            self.logger.info(f"  {metric}: {value}")
        
        # Update metrics
        self.metrics.validation_loss = validation_metrics.get('loss', 0.0)
        self.metrics.validation_accuracy = validation_metrics.get('accuracy', 0.0)
        
        # Check for best metrics
        if self.metrics.validation_loss < self.metrics.best_loss:
            self.metrics.best_loss = self.metrics.validation_loss
            self.logger.info(f"New best validation loss: {self.metrics.best_loss:.4f}")
        
        if self.metrics.validation_accuracy > self.metrics.best_accuracy:
            self.metrics.best_accuracy = self.metrics.validation_accuracy
            self.logger.info(f"New best validation accuracy: {self.metrics.best_accuracy:.4f}")
    
    def log_training_end(self, final_metrics: Dict[str, Any]):
        """Log training end with final metrics."""
        total_time = time.time() - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total training time: {total_time:.2f}s")
        self.logger.info(f"Final metrics: {final_metrics}")
        
        # Save final summary
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_time': total_time,
            'final_metrics': final_metrics,
            'error_count': len(self.error_history),
            'total_epochs': self.metrics.epoch
        }
        
        with open(self.experiment_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Stop monitoring
        self.stop_monitoring()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
            return 0.0
        except Exception:
            return 0.0
    
    def _save_metrics(self) -> Any:
        """Save metrics to CSV file."""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                self.metrics.epoch,
                self.metrics.batch,
                self.metrics.total_batches,
                self.metrics.loss,
                self.metrics.accuracy,
                self.metrics.learning_rate,
                self.metrics.gradient_norm,
                self.metrics.memory_usage,
                self.metrics.gpu_memory,
                self.metrics.training_time,
                self.metrics.validation_loss,
                self.metrics.validation_accuracy,
                self.metrics.best_loss,
                self.metrics.best_accuracy,
                self.metrics.patience_counter
            ])
    
    def _save_errors(self) -> Any:
        """Save errors to JSON file."""
        errors_data = []
        for error in self.error_history:
            errors_data.append({
                'category': error.category.value,
                'phase': error.phase.value,
                'error_message': error.error_message,
                'timestamp': error.timestamp.isoformat(),
                'epoch': error.epoch,
                'batch': error.batch,
                'recovery_attempted': error.recovery_attempted,
                'recovery_successful': error.recovery_successful,
                'recovery_time': error.recovery_time
            })
        
        with open(self.errors_file, 'w') as f:
            json.dump(errors_data, f, indent=2)
    
    def _save_progress(self) -> Any:
        """Save progress to JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress_history, f, indent=2)
    
    def _update_progress(self) -> Any:
        """Update progress tracking."""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'epoch': self.metrics.epoch,
            'batch': self.metrics.batch,
            'total_batches': self.metrics.total_batches,
            'loss': self.metrics.loss,
            'accuracy': self.metrics.accuracy,
            'learning_rate': self.metrics.learning_rate,
            'memory_usage': self.metrics.memory_usage,
            'gpu_memory': self.metrics.gpu_memory
        }
        
        self.progress_history.append(progress_data)
        self._save_progress()
    
    def _update_error_statistics(self) -> Any:
        """Update error statistics."""
        error_stats = defaultdict(int)
        for error in self.error_history:
            error_stats[error.category.value] += 1
        
        self.logger.info("Error Statistics:")
        for category, count in error_stats.items():
            self.logger.info(f"  {category}: {count}")
    
    def start_monitoring(self) -> Any:
        """Start real-time monitoring."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self) -> Any:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self) -> Any:
        """Real-time monitoring loop."""
        while self.monitoring_active:
            try:
                # Monitor system resources
                memory_usage = self._get_memory_usage()
                gpu_memory = self._get_gpu_memory_usage()
                
                # Check for potential issues
                if memory_usage > 1000:  # 1GB threshold
                    self.logger.warning(f"High memory usage: {memory_usage:.2f} MB")
                
                if gpu_memory > 8000:  # 8GB threshold
                    self.logger.warning(f"High GPU memory usage: {gpu_memory:.2f} MB")
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                time.sleep(60)  # Wait longer on error


class TrainingProgressTracker:
    """Advanced training progress tracker with visualization."""
    
    def __init__(self, logger: TrainingLogger):
        self.logger = logger
        self.metrics_history: List[Dict[str, Any]] = []
        self.error_summary: Dict[str, int] = defaultdict(int)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics history."""
        self.metrics_history.append(metrics.copy())
        
        # Update performance metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.performance_metrics[key].append(value)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        summary = {
            'current_epoch': latest_metrics.get('epoch', 0),
            'current_batch': latest_metrics.get('batch', 0),
            'total_batches': latest_metrics.get('total_batches', 0),
            'current_loss': latest_metrics.get('loss', 0.0),
            'current_accuracy': latest_metrics.get('accuracy', 0.0),
            'best_loss': latest_metrics.get('best_loss', float('inf')),
            'best_accuracy': latest_metrics.get('best_accuracy', 0.0),
            'learning_rate': latest_metrics.get('learning_rate', 0.0),
            'training_time': latest_metrics.get('training_time', 0.0),
            'error_count': len(self.logger.error_history),
            'memory_usage': latest_metrics.get('memory_usage', 0.0),
            'gpu_memory': latest_metrics.get('gpu_memory', 0.0)
        }
        
        return summary
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics."""
        error_categories = defaultdict(int)
        error_phases = defaultdict(int)
        
        for error in self.logger.error_history:
            error_categories[error.category.value] += 1
            error_phases[error.phase.value] += 1
        
        return {
            'total_errors': len(self.logger.error_history),
            'error_categories': dict(error_categories),
            'error_phases': dict(error_phases),
            'recovery_success_rate': self._calculate_recovery_success_rate()
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate error recovery success rate."""
        recovery_attempts = sum(1 for error in self.logger.error_history if error.recovery_attempted)
        recovery_successes = sum(1 for error in self.logger.error_history if error.recovery_successful)
        
        if recovery_attempts == 0:
            return 0.0
        
        return recovery_successes / recovery_attempts
    
    def create_training_plots(self, save_dir: str = None):
        """Create training progress plots."""
        if not self.metrics_history:
            return
        
        save_dir = Path(save_dir) if save_dir else self.logger.experiment_dir / "plots"
        save_dir.mkdir(exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.logger.experiment_name}', fontsize=16)
        
        # Extract data
        epochs = [m.get('epoch', 0) for m in self.metrics_history]
        losses = [m.get('loss', 0.0) for m in self.metrics_history]
        accuracies = [m.get('accuracy', 0.0) for m in self.metrics_history]
        learning_rates = [m.get('learning_rate', 0.0) for m in self.metrics_history]
        
        # Loss plot
        axes[0, 0].plot(epochs, losses, 'b-', label='Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, accuracies, 'g-', label='Training Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(epochs, learning_rates, 'r-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Error summary plot
        error_summary = self.get_error_summary()
        if error_summary['error_categories']:
            categories = list(error_summary['error_categories'].keys())
            counts = list(error_summary['error_categories'].values())
            
            axes[1, 1].bar(categories, counts, color='orange')
            axes[1, 1].set_title('Error Categories')
            axes[1, 1].set_xlabel('Error Category')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.logger.info(f"Training plots saved to {save_dir}")


class TrainingLoggingManager:
    """Manager for comprehensive training logging."""
    
    def __init__(self, experiment_name: str = None, log_dir: str = "training_logs"):
        self.logger = TrainingLogger(log_dir, experiment_name)
        self.progress_tracker = TrainingProgressTracker(self.logger)
        self.error_handler = RobustErrorHandler(self.logger.logger)
        
        # Training state
        self.training_active = False
        self.current_epoch = 0
        self.current_batch = 0
    
    def start_training(self, config: Dict[str, Any]):
        """Start training with comprehensive logging."""
        self.training_active = True
        self.logger.log_training_start(config)
        
        # Log initial configuration
        self.logger.logger.info("Training configuration:")
        for key, value in config.items():
            self.logger.logger.info(f"  {key}: {value}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.current_epoch = epoch
        self.logger.log_phase_start(TrainingPhase.EPOCH_START, epoch=epoch, total_epochs=total_epochs)
        
        # Update progress tracker
        self.progress_tracker.update_metrics({
            'epoch': epoch,
            'total_epochs': total_epochs
        })
    
    def log_batch_start(self, batch: int, total_batches: int):
        """Log batch start."""
        self.current_batch = batch
        self.logger.log_phase_start(TrainingPhase.BATCH_START, batch=batch, total_batches=total_batches)
    
    def log_batch_metrics(self, metrics: Dict[str, Any]):
        """Log batch metrics."""
        # Add context information
        metrics.update({
            'epoch': self.current_epoch,
            'batch': self.current_batch
        })
        
        self.logger.log_metrics(metrics)
        self.progress_tracker.update_metrics(metrics)
    
    def log_batch_end(self) -> Any:
        """Log batch end."""
        self.logger.log_phase_end(TrainingPhase.BATCH_END)
    
    def log_epoch_end(self, epoch_metrics: Dict[str, Any]):
        """Log epoch end."""
        self.logger.log_phase_end(TrainingPhase.EPOCH_END, **epoch_metrics)
        
        # Update progress tracker
        self.progress_tracker.update_metrics(epoch_metrics)
    
    def log_validation(self, validation_metrics: Dict[str, Any]):
        """Log validation results."""
        self.logger.log_validation_results(validation_metrics)
        
        # Update progress tracker
        self.progress_tracker.update_metrics(validation_metrics)
    
    def log_error(self, error: Exception, phase: TrainingPhase, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs):
        """Log error with recovery attempt."""
        # Log error
        self.logger.log_error(error, phase, category, **kwargs)
        
        # Attempt recovery
        recovery_start = time.time()
        recovery_successful = self._attempt_error_recovery(error, category, **kwargs)
        recovery_time = time.time() - recovery_start
        
        # Log recovery attempt
        if self.logger.error_history:
            latest_error = self.logger.error_history[-1]
            self.logger.log_recovery_attempt(latest_error, recovery_successful, recovery_time)
    
    def _attempt_error_recovery(self, error: Exception, category: ErrorCategory, **kwargs) -> bool:
        """Attempt error recovery based on category."""
        try:
            if category == ErrorCategory.MEMORY:
                # Clear memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            
            elif category == ErrorCategory.GRADIENT_COMPUTATION:
                # Clear gradients
                if 'model' in kwargs:
                    for param in kwargs['model'].parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                return True
            
            elif category == ErrorCategory.OPTIMIZATION:
                # Skip optimization step
                return True
            
            else:
                # Generic recovery
                time.sleep(1)
                return True
                
        except Exception as recovery_error:
            self.logger.logger.error(f"Recovery failed: {str(recovery_error)}")
            return False
    
    def end_training(self, final_metrics: Dict[str, Any]):
        """End training with final summary."""
        self.training_active = False
        self.logger.log_training_end(final_metrics)
        
        # Create training plots
        self.progress_tracker.create_training_plots()
        
        # Print final summary
        self.print_training_summary()
    
    def print_training_summary(self) -> Any:
        """Print comprehensive training summary."""
        summary = self.progress_tracker.get_training_summary()
        error_summary = self.progress_tracker.get_error_summary()
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Experiment: {self.logger.experiment_name}")
        print(f"Total epochs: {summary.get('current_epoch', 0)}")
        print(f"Final loss: {summary.get('current_loss', 0.0):.4f}")
        print(f"Best loss: {summary.get('best_loss', 0.0):.4f}")
        print(f"Final accuracy: {summary.get('current_accuracy', 0.0):.4f}")
        print(f"Best accuracy: {summary.get('best_accuracy', 0.0):.4f}")
        print(f"Total training time: {summary.get('training_time', 0.0):.2f}s")
        print(f"Total errors: {error_summary.get('total_errors', 0)}")
        print(f"Recovery success rate: {error_summary.get('recovery_success_rate', 0.0):.2%}")
        print("=" * 60)


def demonstrate_training_logging():
    """Demonstrate comprehensive training logging."""
    print("Training Logging System Demonstration")
    print("=" * 60)
    
    # Create logging manager
    logging_manager = TrainingLoggingManager("demo_experiment")
    
    # Simulate training configuration
    config = {
        'model_type': 'neural_network',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'optimizer': 'adam',
        'loss_function': 'cross_entropy'
    }
    
    # Start training
    logging_manager.start_training(config)
    
    # Simulate training loop
    for epoch in range(3):  # Simulate 3 epochs
        logging_manager.log_epoch_start(epoch, 10)
        
        for batch in range(5):  # Simulate 5 batches per epoch
            logging_manager.log_batch_start(batch, 5)
            
            # Simulate batch metrics
            batch_metrics = {
                'loss': 1.0 - (epoch * 0.2 + batch * 0.05),
                'accuracy': 0.5 + (epoch * 0.1 + batch * 0.02),
                'learning_rate': 0.001 * (0.9 ** epoch),
                'gradient_norm': 0.5 + np.random.random() * 0.5,
                'total_batches': 5
            }
            
            logging_manager.log_batch_metrics(batch_metrics)
            logging_manager.log_batch_end()
            
            # Simulate occasional errors
            if epoch == 1 and batch == 2:
                try:
                    raise ValueError("Simulated training error")
                except Exception as e:
                    logging_manager.log_error(e, TrainingPhase.LOSS_COMPUTATION, ErrorCategory.LOSS_COMPUTATION)
        
        # Simulate epoch metrics
        epoch_metrics = {
            'epoch_loss': batch_metrics['loss'],
            'epoch_accuracy': batch_metrics['accuracy'],
            'epoch_time': 30.0
        }
        
        logging_manager.log_epoch_end(epoch_metrics)
        
        # Simulate validation
        validation_metrics = {
            'validation_loss': batch_metrics['loss'] + 0.1,
            'validation_accuracy': batch_metrics['accuracy'] - 0.05
        }
        
        logging_manager.log_validation(validation_metrics)
    
    # End training
    final_metrics = {
        'final_loss': 0.3,
        'final_accuracy': 0.85,
        'total_epochs': 3,
        'total_time': 90.0
    }
    
    logging_manager.end_training(final_metrics)
    
    print("\nTraining logging demonstration completed!")
    print(f"Log files saved to: {logging_manager.logger.experiment_dir}")


if __name__ == "__main__":
    # Demonstrate training logging
    demonstrate_training_logging() 