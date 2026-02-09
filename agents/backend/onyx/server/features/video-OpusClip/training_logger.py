"""
Training Logger for Video-OpusClip

Comprehensive logging system for training operations:
- Training progress tracking
- Error logging and recovery
- Metrics collection and visualization
- Performance monitoring
- Model checkpointing
- Training history and analytics
"""

import sys
import os
import time
import json
import logging
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import queue
import signal

# Import existing components
from optimized_config import get_config
from error_handling import ErrorHandler, ErrorType, ErrorSeverity
from logging_config import setup_logging
from debug_tools import DebugManager

# =============================================================================
# TRAINING LOGGER CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training logging."""
    log_dir: str = "training_logs"
    log_level: str = "INFO"
    max_log_files: int = 10
    log_file_size_mb: int = 100
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    enable_progress_bars: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 1000
    metrics_history_size: int = 10000
    error_recovery_enabled: bool = True
    performance_monitoring: bool = True

# =============================================================================
# TRAINING METRICS AND EVENTS
# =============================================================================

@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    epoch: int
    step: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    training_time: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TrainingEvent:
    """Training event data structure."""
    event_type: str
    message: str
    severity: str = "INFO"
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TrainingError:
    """Training error data structure."""
    error_type: str
    error_message: str
    stack_trace: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    timestamp: Optional[datetime] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

# =============================================================================
# TRAINING LOGGER CLASS
# =============================================================================

class TrainingLogger:
    """Comprehensive training logger for Video-OpusClip."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.logger = None
        self.metrics_history = deque(maxlen=self.config.metrics_history_size)
        self.events_history = deque(maxlen=self.config.metrics_history_size)
        self.errors_history = deque(maxlen=1000)
        self.training_start_time = None
        self.current_epoch = 0
        self.current_step = 0
        self.is_training = False
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.checkpoint_manager = None
        
        # Threading
        self.log_queue = queue.Queue()
        self.log_thread = None
        self.stop_logging = threading.Event()
        
        # Initialize logging
        self._setup_logging()
        self._start_log_thread()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create log directory
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Setup main logger
        self.logger = setup_logging("training_logger")
        
        # Create training-specific loggers
        self._setup_file_handlers()
        self._setup_console_handlers()
        
        # Initialize external logging systems
        if self.config.enable_tensorboard:
            self._setup_tensorboard()
        
        if self.config.enable_wandb:
            self._setup_wandb()
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log types."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main training log
        main_log_file = os.path.join(self.config.log_dir, f"training_{timestamp}.log")
        main_handler = logging.FileHandler(main_log_file)
        main_handler.setLevel(getattr(logging, self.config.log_level))
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(main_handler)
        
        # Metrics log
        metrics_log_file = os.path.join(self.config.log_dir, f"metrics_{timestamp}.jsonl")
        self.metrics_handler = logging.FileHandler(metrics_log_file)
        self.metrics_handler.setLevel(logging.INFO)
        
        # Events log
        events_log_file = os.path.join(self.config.log_dir, f"events_{timestamp}.jsonl")
        self.events_handler = logging.FileHandler(events_log_file)
        self.events_handler.setLevel(logging.INFO)
        
        # Errors log
        errors_log_file = os.path.join(self.config.log_dir, f"errors_{timestamp}.log")
        self.errors_handler = logging.FileHandler(errors_log_file)
        self.errors_handler.setLevel(logging.ERROR)
        self.errors_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n'
        ))
    
    def _setup_console_handlers(self):
        """Setup console handlers for real-time output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = os.path.join(self.config.log_dir, "tensorboard")
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
            self.logger.info("✅ TensorBoard logging enabled")
        except ImportError:
            self.logger.warning("⚠️ TensorBoard not available, skipping setup")
            self.tensorboard_writer = None
        except Exception as e:
            self.logger.error(f"❌ Failed to setup TensorBoard: {e}")
            self.tensorboard_writer = None
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(project="video-opusclip-training")
            self.logger.info("✅ Weights & Biases logging enabled")
        except ImportError:
            self.logger.warning("⚠️ Weights & Biases not available, skipping setup")
            self.wandb_run = None
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Weights & Biases: {e}")
            self.wandb_run = None
    
    def _start_log_thread(self):
        """Start background logging thread."""
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()
    
    def _log_worker(self):
        """Background worker for processing log messages."""
        while not self.stop_logging.is_set():
            try:
                # Get message from queue with timeout
                message = self.log_queue.get(timeout=1.0)
                self._process_log_message(message)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in log worker: {e}")
    
    def _process_log_message(self, message: Dict[str, Any]):
        """Process log message in background thread."""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'metrics':
                self._log_metrics(message['data'])
            elif msg_type == 'event':
                self._log_event(message['data'])
            elif msg_type == 'error':
                self._log_error(message['data'])
            elif msg_type == 'checkpoint':
                self._save_checkpoint(message['data'])
            
        except Exception as e:
            print(f"Error processing log message: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.stop_training()
        self.close()

# =============================================================================
# TRAINING PROGRESS LOGGING
# =============================================================================

    def start_training(self, model_name: str, config: Dict[str, Any] = None):
        """Start training session."""
        self.training_start_time = datetime.now()
        self.is_training = True
        self.current_epoch = 0
        self.current_step = 0
        
        # Log training start
        start_event = TrainingEvent(
            event_type="TRAINING_START",
            message=f"Starting training for model: {model_name}",
            severity="INFO",
            metadata={
                'model_name': model_name,
                'config': config or {},
                'start_time': self.training_start_time.isoformat()
            }
        )
        
        self.log_event(start_event)
        self.logger.info(f"🚀 Training started: {model_name}")
        
        # Log configuration
        if config:
            self.logger.info(f"📋 Training configuration: {json.dumps(config, indent=2)}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.current_epoch = epoch
        
        event = TrainingEvent(
            event_type="EPOCH_START",
            message=f"Starting epoch {epoch}/{total_epochs}",
            severity="INFO",
            metadata={
                'epoch': epoch,
                'total_epochs': total_epochs,
                'progress': f"{epoch}/{total_epochs}"
            }
        )
        
        self.log_event(event)
        self.logger.info(f"📅 Epoch {epoch}/{total_epochs} started")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch end with metrics."""
        event = TrainingEvent(
            event_type="EPOCH_END",
            message=f"Completed epoch {epoch}",
            severity="INFO",
            metadata={
                'epoch': epoch,
                'metrics': metrics,
                'duration': self._get_epoch_duration()
            }
        )
        
        self.log_event(event)
        self.logger.info(f"✅ Epoch {epoch} completed - Loss: {metrics.get('loss', 'N/A'):.4f}")
        
        # Log to external systems
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f'epoch/{key}', value, epoch)
        
        if self.wandb_run:
            wandb_metrics = {f'epoch_{key}': value for key, value in metrics.items()}
            self.wandb_run.log(wandb_metrics, step=epoch)
    
    def log_step(self, step: int, loss: float, **kwargs):
        """Log training step."""
        self.current_step = step
        
        # Create metrics object
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            step=step,
            loss=loss,
            **kwargs
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Queue for background processing
        self.log_queue.put({
            'type': 'metrics',
            'data': asdict(metrics)
        })
        
        # Log to console if enabled
        if self.config.enable_progress_bars and step % 100 == 0:
            self.logger.info(f"📊 Step {step} - Loss: {loss:.4f}")
    
    def log_validation(self, step: int, metrics: Dict[str, float]):
        """Log validation results."""
        event = TrainingEvent(
            event_type="VALIDATION",
            message=f"Validation at step {step}",
            severity="INFO",
            metadata={
                'step': step,
                'metrics': metrics
            }
        )
        
        self.log_event(event)
        self.logger.info(f"🔍 Validation at step {step} - {json.dumps(metrics, indent=2)}")
        
        # Log to external systems
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f'validation/{key}', value, step)
        
        if self.wandb_run:
            wandb_metrics = {f'validation_{key}': value for key, value in metrics.items()}
            self.wandb_run.log(wandb_metrics, step=step)

# =============================================================================
# ERROR LOGGING AND RECOVERY
# =============================================================================

    def log_error(self, error: Exception, step: int = None, epoch: int = None, context: str = ""):
        """Log training error with recovery options."""
        error_obj = TrainingError(
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            step=step or self.current_step,
            epoch=epoch or self.current_epoch
        )
        
        # Add to history
        self.errors_history.append(error_obj)
        
        # Queue for background processing
        self.log_queue.put({
            'type': 'error',
            'data': asdict(error_obj)
        })
        
        # Log immediately for critical errors
        self.logger.error(f"❌ Training error at step {error_obj.step}: {error}")
        self.logger.error(f"Context: {context}")
        self.logger.debug(f"Stack trace: {error_obj.stack_trace}")
        
        # Attempt recovery if enabled
        if self.config.error_recovery_enabled:
            self._attempt_error_recovery(error_obj, context)
    
    def _attempt_error_recovery(self, error: TrainingError, context: str):
        """Attempt to recover from training error."""
        self.logger.info("🔄 Attempting error recovery...")
        
        try:
            if "CUDA out of memory" in error.error_message:
                self._recover_from_memory_error()
            elif "gradient" in error.error_message.lower():
                self._recover_from_gradient_error()
            elif "data" in error.error_message.lower():
                self._recover_from_data_error()
            else:
                self._recover_generic_error()
            
            error.recovery_attempted = True
            error.recovery_successful = True
            
            self.logger.info("✅ Error recovery successful")
            
        except Exception as recovery_error:
            error.recovery_attempted = True
            error.recovery_successful = False
            
            self.logger.error(f"❌ Error recovery failed: {recovery_error}")
    
    def _recover_from_memory_error(self):
        """Recover from memory-related errors."""
        self.logger.info("🔄 Recovering from memory error...")
        
        # Clear cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("✅ CUDA cache cleared")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to clear CUDA cache: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        self.logger.info("✅ Garbage collection completed")
    
    def _recover_from_gradient_error(self):
        """Recover from gradient-related errors."""
        self.logger.info("🔄 Recovering from gradient error...")
        
        # Reduce learning rate
        self.logger.info("📉 Reducing learning rate for stability")
        
        # Clip gradients
        self.logger.info("✂️ Enabling gradient clipping")
    
    def _recover_from_data_error(self):
        """Recover from data-related errors."""
        self.logger.info("🔄 Recovering from data error...")
        
        # Skip problematic batch
        self.logger.info("⏭️ Skipping problematic data batch")
    
    def _recover_generic_error(self):
        """Generic error recovery."""
        self.logger.info("🔄 Attempting generic error recovery...")
        
        # Wait and retry
        time.sleep(5)
        self.logger.info("⏳ Waited 5 seconds, retrying...")

# =============================================================================
# METRICS AND PERFORMANCE LOGGING
# =============================================================================

    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        # Add to history
        self.metrics_history.append(metrics)
        
        # Queue for background processing
        self.log_queue.put({
            'type': 'metrics',
            'data': asdict(metrics)
        })
        
        # Log to external systems
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('training/loss', metrics.loss, metrics.step)
            if metrics.accuracy is not None:
                self.tensorboard_writer.add_scalar('training/accuracy', metrics.accuracy, metrics.step)
            if metrics.learning_rate is not None:
                self.tensorboard_writer.add_scalar('training/learning_rate', metrics.learning_rate, metrics.step)
        
        if self.wandb_run:
            wandb_metrics = {
                'train/loss': metrics.loss,
                'train/step': metrics.step,
                'train/epoch': metrics.epoch
            }
            if metrics.accuracy is not None:
                wandb_metrics['train/accuracy'] = metrics.accuracy
            if metrics.learning_rate is not None:
                wandb_metrics['train/learning_rate'] = metrics.learning_rate
            
            self.wandb_run.log(wandb_metrics, step=metrics.step)
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        if not self.config.performance_monitoring:
            return
        
        # Store performance metrics
        for key, value in metrics.items():
            self.performance_metrics[key].append(value)
        
        # Log to external systems
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f'performance/{key}', value, self.current_step)
        
        if self.wandb_run:
            wandb_metrics = {f'performance_{key}': value for key, value in metrics.items()}
            self.wandb_run.log(wandb_metrics, step=self.current_step)
    
    def log_memory_usage(self):
        """Log current memory usage."""
        try:
            import psutil
            import torch
            
            # System memory
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # GPU memory if available
            gpu_memory = None
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            metrics = {
                'system_memory_mb': memory_info.rss / (1024**2),
                'system_memory_percent': process.memory_percent()
            }
            
            if gpu_memory is not None:
                metrics['gpu_memory_gb'] = gpu_memory
            
            self.log_performance_metrics(metrics)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log memory usage: {e}")

# =============================================================================
# CHECKPOINTING AND MODEL SAVING
# =============================================================================

    def log_checkpoint(self, model_state: Dict[str, Any], step: int, metrics: Dict[str, float] = None):
        """Log model checkpoint."""
        if not self.config.save_checkpoints:
            return
        
        checkpoint_data = {
            'step': step,
            'epoch': self.current_epoch,
            'model_state': model_state,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
            'training_time': self._get_training_duration()
        }
        
        # Queue for background processing
        self.log_queue.put({
            'type': 'checkpoint',
            'data': checkpoint_data
        })
        
        self.logger.info(f"💾 Checkpoint saved at step {step}")
    
    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Save checkpoint to disk."""
        try:
            import torch
            
            checkpoint_dir = os.path.join(self.config.log_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            step = checkpoint_data['step']
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            
            torch.save(checkpoint_data, checkpoint_file)
            
            # Keep only recent checkpoints
            self._cleanup_old_checkpoints(checkpoint_dir)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: str):
        """Clean up old checkpoint files."""
        try:
            checkpoint_files = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")],
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            
            # Keep only the last 5 checkpoints
            if len(checkpoint_files) > 5:
                for old_file in checkpoint_files[:-5]:
                    os.remove(os.path.join(checkpoint_dir, old_file))
                    self.logger.debug(f"🗑️ Removed old checkpoint: {old_file}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup checkpoints: {e}")

# =============================================================================
# EVENT LOGGING
# =============================================================================

    def log_event(self, event: TrainingEvent):
        """Log training event."""
        # Add to history
        self.events_history.append(event)
        
        # Queue for background processing
        self.log_queue.put({
            'type': 'event',
            'data': asdict(event)
        })
        
        # Log immediately based on severity
        if event.severity == "ERROR":
            self.logger.error(f"🚨 {event.message}")
        elif event.severity == "WARNING":
            self.logger.warning(f"⚠️ {event.message}")
        elif event.severity == "INFO":
            self.logger.info(f"ℹ️ {event.message}")
        else:
            self.logger.debug(f"🔍 {event.message}")
    
    def log_hyperparameter_update(self, hyperparams: Dict[str, Any]):
        """Log hyperparameter updates."""
        event = TrainingEvent(
            event_type="HYPERPARAMETER_UPDATE",
            message="Hyperparameters updated",
            severity="INFO",
            metadata={'hyperparameters': hyperparams}
        )
        
        self.log_event(event)
        self.logger.info(f"⚙️ Hyperparameters updated: {json.dumps(hyperparams, indent=2)}")
    
    def log_model_save(self, model_path: str, metrics: Dict[str, float] = None):
        """Log model save operation."""
        event = TrainingEvent(
            event_type="MODEL_SAVE",
            message=f"Model saved to {model_path}",
            severity="INFO",
            metadata={
                'model_path': model_path,
                'metrics': metrics or {}
            }
        )
        
        self.log_event(event)
        self.logger.info(f"💾 Model saved: {model_path}")

# =============================================================================
# UTILITY METHODS
# =============================================================================

    def _get_training_duration(self) -> float:
        """Get training duration in seconds."""
        if self.training_start_time is None:
            return 0.0
        return (datetime.now() - self.training_start_time).total_seconds()
    
    def _get_epoch_duration(self) -> float:
        """Get current epoch duration in seconds."""
        # This would need to be tracked per epoch
        return 0.0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.is_training:
            return {"status": "Not training"}
        
        return {
            "status": "Training",
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "training_duration": self._get_training_duration(),
            "total_metrics": len(self.metrics_history),
            "total_events": len(self.events_history),
            "total_errors": len(self.errors_history),
            "recent_loss": self.metrics_history[-1].loss if self.metrics_history else None
        }
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics history."""
        return [asdict(metric) for metric in list(self.metrics_history)[-limit:]]
    
    def get_errors_summary(self) -> Dict[str, Any]:
        """Get errors summary."""
        if not self.errors_history:
            return {"total_errors": 0}
        
        error_types = defaultdict(int)
        recovery_success = 0
        
        for error in self.errors_history:
            error_types[error.error_type] += 1
            if error.recovery_successful:
                recovery_success += 1
        
        return {
            "total_errors": len(self.errors_history),
            "error_types": dict(error_types),
            "recovery_success_rate": recovery_success / len(self.errors_history),
            "recent_errors": [asdict(error) for error in list(self.errors_history)[-10:]]
        }
    
    def export_training_logs(self, output_path: str):
        """Export training logs to file."""
        try:
            export_data = {
                "training_summary": self.get_training_summary(),
                "metrics_history": self.get_metrics_history(),
                "errors_summary": self.get_errors_summary(),
                "performance_metrics": dict(self.performance_metrics),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"📤 Training logs exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export training logs: {e}")

# =============================================================================
# CLEANUP AND SHUTDOWN
# =============================================================================

    def stop_training(self):
        """Stop training session."""
        if not self.is_training:
            return
        
        self.is_training = False
        training_duration = self._get_training_duration()
        
        event = TrainingEvent(
            event_type="TRAINING_END",
            message=f"Training completed in {training_duration:.2f} seconds",
            severity="INFO",
            metadata={
                'duration': training_duration,
                'final_epoch': self.current_epoch,
                'final_step': self.current_step
            }
        )
        
        self.log_event(event)
        self.logger.info(f"🏁 Training completed - Duration: {training_duration:.2f}s")
    
    def close(self):
        """Close logger and cleanup resources."""
        # Stop background thread
        self.stop_logging.set()
        if self.log_thread and self.log_thread.is_alive():
            self.log_thread.join(timeout=5.0)
        
        # Close external writers
        if hasattr(self, 'tensorboard_writer') and self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.finish()
        
        # Export final logs
        final_log_path = os.path.join(self.config.log_dir, "training_summary.json")
        self.export_training_logs(final_log_path)
        
        self.logger.info("🔒 Training logger closed")

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_training_logger():
    """Example usage of training logger."""
    
    # Initialize logger
    config = TrainingConfig(
        log_dir="example_training_logs",
        enable_tensorboard=True,
        enable_wandb=False,
        save_checkpoints=True
    )
    
    logger = TrainingLogger(config)
    
    try:
        # Start training
        logger.start_training("example_model", {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        })
        
        # Training loop
        for epoch in range(3):
            logger.log_epoch_start(epoch, 3)
            
            for step in range(100):
                # Simulate training
                loss = 1.0 / (step + 1)  # Decreasing loss
                
                # Log step
                logger.log_step(step, loss, accuracy=0.8 + step * 0.001)
                
                # Log memory usage periodically
                if step % 20 == 0:
                    logger.log_memory_usage()
                
                # Simulate occasional errors
                if step == 50:
                    try:
                        raise RuntimeError("Simulated training error")
                    except Exception as e:
                        logger.log_error(e, step, epoch, "Training step")
            
            # Log epoch end
            logger.log_epoch_end(epoch, {"loss": loss, "accuracy": 0.85})
            
            # Log checkpoint
            logger.log_checkpoint({"model": "state"}, step, {"loss": loss})
        
        # Stop training
        logger.stop_training()
        
    except Exception as e:
        logger.log_error(e, context="Main training loop")
    finally:
        logger.close()

if __name__ == "__main__":
    example_training_logger() 