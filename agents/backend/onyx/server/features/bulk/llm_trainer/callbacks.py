"""
Training Callbacks Module
==========================

Custom callbacks for training monitoring and logging.

Author: BUL System
Date: 2024
"""

import logging
import time
import torch
from typing import Optional, Dict, Any
from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class TrainingProgressCallback(TrainerCallback):
    """
    Callback to log training progress and metrics.
    
    Logs:
    - Training metrics at each logging step
    - Epoch completion information
    - Loss values
    
    Example:
        >>> callback = TrainingProgressCallback()
        >>> trainer.add_callback(callback)
    """
    
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        """
        Log training metrics.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            logs: Logs dictionary
            **kwargs: Additional arguments
        """
        if logs:
            # Format log message
            log_items = []
            for key, value in logs.items():
                if isinstance(value, float):
                    log_items.append(f"{key}={value:.4f}")
                else:
                    log_items.append(f"{key}={value}")
            
            logger.info(f"Step {state.global_step}: {', '.join(log_items)}")
    
    def on_epoch_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Log epoch end information.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            **kwargs: Additional arguments
        """
        if state.log_history:
            last_log = state.log_history[-1]
            loss = last_log.get('loss', 'N/A')
            logger.info(f"Epoch {state.epoch} completed. Loss: {loss}")
    
    def on_train_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Log training start.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            **kwargs: Additional arguments
        """
        logger.info("=" * 80)
        logger.info("Training started")
        logger.info(f"Total epochs: {args.num_train_epochs}")
        logger.info(f"Batch size: {args.per_device_train_batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info("=" * 80)
    
    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Log training end.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            **kwargs: Additional arguments
        """
        logger.info("=" * 80)
        logger.info("Training completed")
        if state.log_history:
            final_loss = state.log_history[-1].get('loss', 'N/A')
            logger.info(f"Final loss: {final_loss}")
        logger.info("=" * 80)


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback for early stopping based on evaluation loss.
    
    Stops training if evaluation loss doesn't improve for a specified
    number of patience steps/epochs.
    
    Attributes:
        patience: Number of evaluations to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        best_loss: Best loss seen so far
        patience_counter: Current patience counter
        
    Example:
        >>> callback = EarlyStoppingCallback(patience=3, min_delta=0.001)
        >>> trainer.add_callback(callback)
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        """
        Initialize EarlyStoppingCallback.
        
        Args:
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.patience_counter = 0
        self.stopped_epoch = 0
    
    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Check if early stopping should be triggered.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            logs: Evaluation logs
            **kwargs: Additional arguments
        """
        if logs is None or "eval_loss" not in logs:
            return
        
        current_loss = logs["eval_loss"]
        
        if self.best_loss is None:
            self.best_loss = current_loss
            logger.info(f"Early stopping: Initial eval loss = {current_loss:.4f}")
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            logger.info(f"Early stopping: Improved eval loss = {current_loss:.4f}")
        else:
            self.patience_counter += 1
            logger.info(
                f"Early stopping: No improvement ({self.patience_counter}/{self.patience}). "
                f"Best loss = {self.best_loss:.4f}, Current = {current_loss:.4f}"
            )
            
            if self.patience_counter >= self.patience:
                logger.warning(
                    f"Early stopping triggered after {state.epoch} epochs. "
                    f"Best loss was {self.best_loss:.4f}"
                )
                control.should_training_stop = True
                self.stopped_epoch = state.epoch


class MemoryMonitoringCallback(TrainerCallback):
    """
    Callback to monitor GPU memory usage during training.
    
    Logs memory usage at regular intervals and warns if memory is high.
    
    Attributes:
        log_interval: Steps between memory logging
        warning_threshold: Memory usage threshold for warnings (0-1)
        
    Example:
        >>> callback = MemoryMonitoringCallback(log_interval=100)
        >>> trainer.add_callback(callback)
    """
    
    def __init__(self, log_interval: int = 100, warning_threshold: float = 0.9):
        """
        Initialize MemoryMonitoringCallback.
        
        Args:
            log_interval: Steps between memory logging
            warning_threshold: Memory threshold for warnings (0-1)
        """
        self.log_interval = log_interval
        self.warning_threshold = warning_threshold
        self.last_log_step = 0
    
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        """Log memory usage at specified intervals."""
        if state.global_step - self.last_log_step >= self.log_interval:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                
                logger.info(
                    f"GPU Memory - Allocated: {allocated:.2f} GB, "
                    f"Reserved: {reserved:.2f} GB, "
                    f"Peak: {max_allocated:.2f} GB"
                )
                
                # Check if memory usage is high
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    usage_ratio = reserved / total_memory
                    if usage_ratio > self.warning_threshold:
                        logger.warning(
                            f"High GPU memory usage: {usage_ratio:.1%}. "
                            f"Consider reducing batch size or sequence length."
                        )
            
            self.last_log_step = state.global_step


class TrainingTimeCallback(TrainerCallback):
    """
    Callback to track training time and estimate completion.
    
    Tracks elapsed time and estimates remaining time based on progress.
    
    Example:
        >>> callback = TrainingTimeCallback()
        >>> trainer.add_callback(callback)
    """
    
    def __init__(self):
        """Initialize TrainingTimeCallback."""
        self.start_time: Optional[float] = None
        self.last_step_time: Optional[float] = None
    
    def on_train_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Record training start time."""
        self.start_time = time.time()
        logger.info("Training time tracking started")
    
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs
    ):
        """Log training progress and time estimates."""
        if self.start_time is None:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if self.last_step_time:
            step_time = current_time - self.last_step_time
            steps_per_second = 1.0 / step_time if step_time > 0 else 0
        else:
            steps_per_second = state.global_step / elapsed_time if elapsed_time > 0 else 0
        
        # Estimate remaining time
        if state.max_steps and steps_per_second > 0:
            remaining_steps = state.max_steps - state.global_step
            estimated_remaining = remaining_steps / steps_per_second
            logger.info(
                f"Training Progress - Elapsed: {elapsed_time/60:.1f} min, "
                f"Speed: {steps_per_second:.2f} steps/s, "
                f"Est. remaining: {estimated_remaining/60:.1f} min"
            )
        elif args.num_train_epochs and steps_per_second > 0:
            # Estimate based on epochs
            steps_per_epoch = state.global_step / max(state.epoch, 1)
            remaining_epochs = args.num_train_epochs - state.epoch
            remaining_steps = remaining_epochs * steps_per_epoch
            estimated_remaining = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            logger.info(
                f"Training Progress - Elapsed: {elapsed_time/60:.1f} min, "
                f"Speed: {steps_per_second:.2f} steps/s, "
                f"Est. remaining: {estimated_remaining/60:.1f} min"
            )
        
        self.last_step_time = current_time
    
    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log total training time."""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")

