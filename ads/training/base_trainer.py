"""
Base trainer class for the ads training system.

This module provides the foundation for all training implementations,
ensuring consistency and common functionality across different training approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import asyncio
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class TrainingPhase(Enum):
    """Training phases for tracking progress."""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    TRAINING = "training"
    VALIDATION = "validation"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"

class TrainingStatus(Enum):
    """Training status indicators."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    accuracy: Optional[float] = None
    learning_rate: float = 0.0
    gradient_norm: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None
    training_time: float = 0.0
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "memory_usage": self.memory_usage,
            "gpu_utilization": self.gpu_utilization,
            "training_time": self.training_time,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
        }

@dataclass
class TrainingConfig:
    """Configuration for training sessions."""
    # Basic settings
    model_name: str = "default_model"
    dataset_name: str = "default_dataset"
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    
    # Optimization settings
    optimizer_name: str = "adam"
    scheduler_name: str = "cosine"
    weight_decay: float = 0.0001
    gradient_clipping: Optional[float] = 1.0
    
    # Hardware settings
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4
    pin_memory: bool = True
    
    # Advanced settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
    checkpoint_save_frequency: int = 5
    
    # Paths
    model_save_path: str = "./models"
    checkpoint_path: str = "./checkpoints"
    log_path: str = "./logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "optimizer_name": self.optimizer_name,
            "scheduler_name": self.scheduler_name,
            "weight_decay": self.weight_decay,
            "gradient_clipping": self.gradient_clipping,
            "device": self.device,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_save_frequency": self.checkpoint_save_frequency,
            "model_save_path": self.model_save_path,
            "checkpoint_path": self.checkpoint_path,
            "log_path": self.log_path,
        }

@dataclass
class TrainingResult:
    """Result of a training session."""
    success: bool = False
    status: TrainingStatus = TrainingStatus.PENDING
    final_metrics: Optional[TrainingMetrics] = None
    best_metrics: Optional[TrainingMetrics] = None
    training_history: List[TrainingMetrics] = field(default_factory=list)
    error_message: Optional[str] = None
    training_time: float = 0.0
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "status": self.status.value,
            "final_metrics": self.final_metrics.to_dict() if self.final_metrics else None,
            "best_metrics": self.best_metrics.to_dict() if self.best_metrics else None,
            "training_history": [m.to_dict() for m in self.training_history],
            "error_message": self.error_message,
            "training_time": self.training_time,
            "model_path": self.model_path,
            "checkpoint_path": self.checkpoint_path,
        }

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    This class defines the common interface and provides shared functionality
    for all training implementations in the ads system.
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.status = TrainingStatus.PENDING
        self.current_phase = TrainingPhase.INITIALIZATION
        self.metrics = TrainingMetrics()
        self.start_time = None
        self.training_history: List[TrainingMetrics] = []
        self._callbacks: List[Callable] = []
        self._stop_requested = False
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config.model_name}")
    
    @abstractmethod
    async def setup_training(self) -> bool:
        """Set up the training environment and resources."""
        pass
    
    @abstractmethod
    async def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    async def validate(self, epoch: int) -> TrainingMetrics:
        """Validate the model."""
        pass
    
    @abstractmethod
    async def save_checkpoint(self, epoch: int, metrics: TrainingMetrics) -> str:
        """Save a training checkpoint."""
        pass
    
    @abstractmethod
    async def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load a training checkpoint."""
        pass
    
    async def train(self) -> TrainingResult:
        """Execute the complete training process."""
        try:
            self.start_time = datetime.now()
            self.status = TrainingStatus.RUNNING
            self.current_phase = TrainingPhase.INITIALIZATION
            
            logger.info(f"Starting training for {self.config.model_name}")
            
            # Setup training
            if not await self.setup_training():
                raise RuntimeError("Failed to setup training")
            
            self.current_phase = TrainingPhase.TRAINING
            
            # Training loop
            best_metrics = None
            early_stopping_counter = 0
            
            for epoch in range(self.config.num_epochs):
                if self._stop_requested:
                    logger.info("Training stopped by user request")
                    break
                
                # Train epoch
                train_metrics = await self.train_epoch(epoch)
                self.metrics = train_metrics
                self.training_history.append(train_metrics)
                
                # Validate
                self.current_phase = TrainingPhase.VALIDATION
                val_metrics = await self.validate(epoch)
                train_metrics.validation_loss = val_metrics.validation_loss
                train_metrics.validation_accuracy = val_metrics.validation_accuracy
                
                # Update best metrics
                if best_metrics is None or (
                    val_metrics.validation_loss is not None and 
                    val_metrics.validation_loss < best_metrics.validation_loss
                ):
                    best_metrics = train_metrics
                    early_stopping_counter = 0
                    
                    # Save best model
                    await self.save_checkpoint(epoch, train_metrics)
                else:
                    early_stopping_counter += 1
                
                # Check early stopping
                if early_stopping_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Regular checkpointing
                if (epoch + 1) % self.config.checkpoint_save_frequency == 0:
                    await self.save_checkpoint(epoch, train_metrics)
                
                # Log progress
                self._log_progress(epoch, train_metrics)
                
                # Execute callbacks
                await self._execute_callbacks(epoch, train_metrics)
            
            self.current_phase = TrainingPhase.COMPLETED
            self.status = TrainingStatus.COMPLETED
            
            training_time = (datetime.now() - self.start_time).total_seconds()
            
            result = TrainingResult(
                success=True,
                status=self.status,
                final_metrics=self.metrics,
                best_metrics=best_metrics,
                training_history=self.training_history,
                training_time=training_time,
                model_path=await self._get_final_model_path(),
                checkpoint_path=await self._get_final_checkpoint_path()
            )
            
            logger.info(f"Training completed successfully in {training_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.current_phase = TrainingPhase.FAILED
            self.status = TrainingStatus.FAILED
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            result = TrainingResult(
                success=False,
                status=self.status,
                error_message=error_msg,
                training_time=(datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            )
            
            return result
    
    def add_callback(self, callback: Callable):
        """Add a callback function to be executed during training."""
        self._callbacks.append(callback)
    
    def stop_training(self):
        """Request to stop training."""
        self._stop_requested = True
        logger.info("Training stop requested")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "status": self.status.value,
            "phase": self.current_phase.value,
            "current_metrics": self.metrics.to_dict(),
            "training_history_length": len(self.training_history),
            "stop_requested": self._stop_requested
        }
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        os.makedirs(self.config.log_path, exist_ok=True)
    
    def _log_progress(self, epoch: int, metrics: TrainingMetrics):
        """Log training progress."""
        logger.info(
            f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            f"Loss: {metrics.loss:.4f}, "
            f"LR: {metrics.learning_rate:.6f}"
        )
        
        if metrics.validation_loss is not None:
            logger.info(f"Validation Loss: {metrics.validation_loss:.4f}")
    
    async def _execute_callbacks(self, epoch: int, metrics: TrainingMetrics):
        """Execute registered callbacks."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(epoch, metrics)
                else:
                    callback(epoch, metrics)
            except Exception as e:
                logger.warning(f"Callback execution failed: {e}")
    
    async def _get_final_model_path(self) -> Optional[str]:
        """Get the path to the final trained model."""
        # Implementation depends on specific trainer
        return None
    
    async def _get_final_checkpoint_path(self) -> Optional[str]:
        """Get the path to the final checkpoint."""
        # Implementation depends on specific trainer
        return None
