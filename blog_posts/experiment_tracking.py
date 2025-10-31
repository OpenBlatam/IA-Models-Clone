from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import json
import yaml
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import structlog
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import hashlib
import pickle
import zipfile
    import wandb
    import mlflow
            import sys
            from PIL import Image
            import torchvision.transforms as transforms
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Experiment Tracking and Model Checkpointing System
==================================================

This module provides a comprehensive experiment tracking and model checkpointing
system with support for multiple tracking backends (WandB, TensorBoard, MLflow)
and robust checkpoint management for deep learning experiments.

Key Features:
1. Multi-backend experiment tracking
2. Comprehensive model checkpointing
3. Experiment metadata management
4. Performance monitoring and visualization
5. Reproducibility and versioning
6. Integration with modular architecture
"""


# Optional imports for different tracking backends
try:
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlog.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# =============================================================================
# EXPERIMENT METADATA
# =============================================================================

@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking."""
    
    # Basic experiment information
    experiment_name: str
    experiment_id: str
    description: Optional[str] = None
    tags: List[str] = None
    version: str = "1.0.0"
    
    # Timestamps
    start_time: datetime = None
    end_time: Optional[datetime] = None
    
    # System information
    python_version: str = None
    torch_version: str = None
    cuda_version: Optional[str] = None
    gpu_info: Optional[Dict[str, Any]] = None
    
    # Configuration
    config_hash: str = None
    config_path: Optional[str] = None
    
    # Status
    status: str = "running"  # running, completed, failed, stopped
    
    def __post_init__(self) -> Any:
        """Initialize default values."""
        if self.tags is None:
            self.tags = []
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.python_version is None:
            self.python_version = sys.version
        if self.torch_version is None:
            self.torch_version = torch.__version__
        if self.cuda_version is None and torch.cuda.is_available():
            self.cuda_version = torch.version.cuda
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def generate_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(self.experiment_name.encode()).hexdigest()[:8]
        return f"{timestamp}_{name_hash}"
    
    def mark_completed(self) -> Any:
        """Mark experiment as completed."""
        self.status = "completed"
        self.end_time = datetime.now()
    
    def mark_failed(self, error: str = None):
        """Mark experiment as failed."""
        self.status = "failed"
        self.end_time = datetime.now()
        if error:
            logger.error("Experiment failed", experiment_id=self.experiment_id, error=error)


# =============================================================================
# EXPERIMENT TRACKING BACKENDS
# =============================================================================

class BaseTracker:
    """Base class for experiment tracking backends."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        
    """__init__ function."""
self.experiment_name = experiment_name
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to tracking backend."""
        raise NotImplementedError
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters to tracking backend."""
        raise NotImplementedError
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model file to tracking backend."""
        raise NotImplementedError
    
    def log_artifact(self, artifact_path: str, artifact_name: str):
        """Log artifact to tracking backend."""
        raise NotImplementedError
    
    def log_image(self, image_path: str, image_name: str, step: int = None):
        """Log image to tracking backend."""
        raise NotImplementedError
    
    def log_text(self, text: str, text_name: str, step: int = None):
        """Log text to tracking backend."""
        raise NotImplementedError
    
    def finish(self) -> Any:
        """Finish experiment tracking."""
        raise NotImplementedError


class WandBTracker(BaseTracker):
    """Weights & Biases experiment tracker."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        
    """__init__ function."""
super().__init__(experiment_name, config)
        
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        
        # Initialize WandB
        wandb.init(
            project=config.get("wandb_project", "deep-learning-project"),
            name=experiment_name,
            config=config,
            tags=config.get("tags", []),
            notes=config.get("description", "")
        )
        
        self.logger.info("WandB tracker initialized", experiment=experiment_name)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to WandB."""
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters to WandB."""
        wandb.config.update(hyperparameters)
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to WandB."""
        artifact = wandb.Artifact(name=model_name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def log_artifact(self, artifact_path: str, artifact_name: str):
        """Log artifact to WandB."""
        artifact = wandb.Artifact(name=artifact_name, type="artifact")
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)
    
    def log_image(self, image_path: str, image_name: str, step: int = None):
        """Log image to WandB."""
        image = wandb.Image(image_path)
        wandb.log({image_name: image}, step=step)
    
    def log_text(self, text: str, text_name: str, step: int = None):
        """Log text to WandB."""
        wandb.log({text_name: wandb.Html(text)}, step=step)
    
    def finish(self) -> Any:
        """Finish WandB experiment."""
        wandb.finish()
        self.logger.info("WandB experiment finished")


class TensorBoardTracker(BaseTracker):
    """TensorBoard experiment tracker."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        
    """__init__ function."""
super().__init__(experiment_name, config)
        
        # Create log directory
        log_dir = Path(config.get("tensorboard_log_dir", "logs/tensorboard"))
        self.log_dir = log_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Log hyperparameters
        self.log_hyperparameters(config)
        
        self.logger.info("TensorBoard tracker initialized", log_dir=str(self.log_dir))
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            if step is not None:
                self.writer.add_scalar(name, value, step)
            else:
                self.writer.add_scalar(name, value)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        # Convert to format suitable for TensorBoard
        hparams = {}
        for key, value in hyperparameters.items():
            if isinstance(value, (int, float, str, bool)):
                hparams[key] = value
        
        self.writer.add_hparams(hparams, {})
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to TensorBoard."""
        # TensorBoard doesn't directly support model logging
        # We'll log the model path as text
        self.log_text(f"Model saved at: {model_path}", f"{model_name}_path")
    
    def log_artifact(self, artifact_path: str, artifact_name: str):
        """Log artifact to TensorBoard."""
        # Log artifact path as text
        self.log_text(f"Artifact saved at: {artifact_path}", f"{artifact_name}_path")
    
    def log_image(self, image_path: str, image_name: str, step: int = None):
        """Log image to TensorBoard."""
        try:
            
            image = Image.open(image_path)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            transform = transforms.ToTensor()
            image_tensor = transform(image)
            
            if step is not None:
                self.writer.add_image(image_name, image_tensor, step)
            else:
                self.writer.add_image(image_name, image_tensor)
        except Exception as e:
            self.logger.warning("Failed to log image to TensorBoard", error=str(e))
    
    def log_text(self, text: str, text_name: str, step: int = None):
        """Log text to TensorBoard."""
        if step is not None:
            self.writer.add_text(text_name, text, step)
        else:
            self.writer.add_text(text_name, text)
    
    def finish(self) -> Any:
        """Finish TensorBoard experiment."""
        self.writer.close()
        self.logger.info("TensorBoard experiment finished")


class MLflowTracker(BaseTracker):
    """MLflow experiment tracker."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        
    """__init__ function."""
super().__init__(experiment_name, config)
        
        if not MLFLOW_AVAILABLE:
            raise ImportError("mlflow is not installed. Install with: pip install mlflow")
        
        # Set MLflow tracking URI
        tracking_uri = config.get("mlflow_tracking_uri", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        # Start run
        mlflow.start_run()
        
        self.logger.info("MLflow tracker initialized", experiment=experiment_name)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow."""
        for name, value in metrics.items():
            if step is not None:
                mlflow.log_metric(name, value, step=step)
            else:
                mlflow.log_metric(name, value)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters to MLflow."""
        mlflow.log_params(hyperparameters)
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to MLflow."""
        mlflow.log_artifact(model_path, artifact_path=f"models/{model_name}")
    
    def log_artifact(self, artifact_path: str, artifact_name: str):
        """Log artifact to MLflow."""
        mlflow.log_artifact(artifact_path, artifact_path=artifact_name)
    
    def log_image(self, image_path: str, image_name: str, step: int = None):
        """Log image to MLflow."""
        mlflow.log_artifact(image_path, artifact_path=f"images/{image_name}")
    
    def log_text(self, text: str, text_name: str, step: int = None):
        """Log text to MLflow."""
        # Save text to file and log as artifact
        text_path = f"/tmp/{text_name}.txt"
        with open(text_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(text)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        mlflow.log_artifact(text_path, artifact_path=f"text/{text_name}")
    
    def finish(self) -> Any:
        """Finish MLflow experiment."""
        mlflow.end_run()
        self.logger.info("MLflow experiment finished")


# =============================================================================
# MODEL CHECKPOINTING
# =============================================================================

@dataclass
class CheckpointMetadata:
    """Metadata for model checkpoints."""
    
    checkpoint_id: str
    experiment_id: str
    epoch: int
    step: int
    timestamp: datetime
    
    # Model information
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    
    # Training metrics
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    
    # Model configuration
    model_config: Dict[str, Any] = None
    training_config: Dict[str, Any] = None
    
    # File information
    file_size: Optional[int] = None
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def generate_id(self) -> str:
        """Generate unique checkpoint ID."""
        timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_{self.experiment_id}_epoch_{self.epoch}_step_{self.step}_{timestamp}"


class ModelCheckpointer:
    """Comprehensive model checkpointing system."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 10):
        
    """__init__ function."""
self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = structlog.get_logger(__name__)
        
        # Load checkpoint registry
        self.registry_file = self.checkpoint_dir / "checkpoint_registry.json"
        self.checkpoint_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load checkpoint registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return json.load(f)
        return {"checkpoints": {}, "experiments": {}}
    
    def _save_registry(self) -> Any:
        """Save checkpoint registry to file."""
        with open(self.registry_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.checkpoint_registry, f, indent=2, default=str)
    
    def save_checkpoint(
        self,
        experiment_id: str,
        model: nn.Module,
        epoch: int,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint with comprehensive metadata.
        
        Args:
            experiment_id: Unique experiment identifier
            model: PyTorch model
            epoch: Current epoch
            step: Current training step
            optimizer: Optimizer state
            scheduler: Scheduler state
            train_loss: Training loss
            val_loss: Validation loss
            train_accuracy: Training accuracy
            val_accuracy: Validation accuracy
            model_config: Model configuration
            training_config: Training configuration
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint metadata
        checkpoint_metadata = CheckpointMetadata(
            checkpoint_id="",
            experiment_id=experiment_id,
            epoch=epoch,
            step=step,
            timestamp=datetime.now(),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict() if optimizer else None,
            scheduler_state_dict=scheduler.state_dict() if scheduler else None,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            model_config=model_config,
            training_config=training_config
        )
        
        # Generate checkpoint ID and file path
        checkpoint_id = checkpoint_metadata.generate_id()
        checkpoint_metadata.checkpoint_id = checkpoint_id
        
        if is_best:
            checkpoint_filename = f"best_model_{experiment_id}.pt"
        else:
            checkpoint_filename = f"{checkpoint_id}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Save checkpoint
        checkpoint_data = {
            "metadata": checkpoint_metadata.to_dict(),
            "model_state_dict": checkpoint_metadata.model_state_dict,
            "optimizer_state_dict": checkpoint_metadata.optimizer_state_dict,
            "scheduler_state_dict": checkpoint_metadata.scheduler_state_dict,
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update metadata
        checkpoint_metadata.file_size = checkpoint_path.stat().st_size
        checkpoint_metadata.file_path = str(checkpoint_path)
        
        # Update registry
        self._update_registry(checkpoint_metadata)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints(experiment_id)
        
        self.logger.info("Checkpoint saved", 
                        checkpoint_id=checkpoint_id,
                        path=str(checkpoint_path),
                        is_best=is_best)
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Tuple[CheckpointMetadata, int, int]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
            
        Returns:
            Tuple of (metadata, epoch, step)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint_data["model_state_dict"])
        
        # Load optimizer state
        if optimizer and checkpoint_data["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        # Load scheduler state
        if scheduler and checkpoint_data["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        
        # Create metadata object
        metadata = CheckpointMetadata(**checkpoint_data["metadata"])
        
        self.logger.info("Checkpoint loaded", 
                        checkpoint_id=metadata.checkpoint_id,
                        epoch=metadata.epoch,
                        step=metadata.step)
        
        return metadata, metadata.epoch, metadata.step
    
    def get_best_checkpoint(self, experiment_id: str) -> Optional[str]:
        """Get path to best checkpoint for experiment."""
        experiment_checkpoints = self.checkpoint_registry.get("experiments", {}).get(experiment_id, [])
        
        if not experiment_checkpoints:
            return None
        
        # Find checkpoint with best validation loss
        best_checkpoint = None
        best_val_loss = float('inf')
        
        for checkpoint_id in experiment_checkpoints:
            checkpoint_info = self.checkpoint_registry["checkpoints"].get(checkpoint_id, {})
            val_loss = checkpoint_info.get("val_loss")
            
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = checkpoint_info.get("file_path")
        
        return best_checkpoint
    
    def list_checkpoints(self, experiment_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for an experiment."""
        experiment_checkpoints = self.checkpoint_registry.get("experiments", {}).get(experiment_id, [])
        
        checkpoints = []
        for checkpoint_id in experiment_checkpoints:
            checkpoint_info = self.checkpoint_registry["checkpoints"].get(checkpoint_id, {})
            if checkpoint_info:
                checkpoints.append(checkpoint_info)
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return checkpoints
    
    def _update_registry(self, metadata: CheckpointMetadata):
        """Update checkpoint registry."""
        # Add checkpoint to registry
        self.checkpoint_registry["checkpoints"][metadata.checkpoint_id] = {
            "experiment_id": metadata.experiment_id,
            "epoch": metadata.epoch,
            "step": metadata.step,
            "timestamp": metadata.timestamp.isoformat(),
            "file_path": metadata.file_path,
            "file_size": metadata.file_size,
            "train_loss": metadata.train_loss,
            "val_loss": metadata.val_loss,
            "train_accuracy": metadata.train_accuracy,
            "val_accuracy": metadata.val_accuracy
        }
        
        # Add to experiment's checkpoint list
        if metadata.experiment_id not in self.checkpoint_registry["experiments"]:
            self.checkpoint_registry["experiments"][metadata.experiment_id] = []
        
        self.checkpoint_registry["experiments"][metadata.experiment_id].append(metadata.checkpoint_id)
        
        # Save registry
        self._save_registry()
    
    def _cleanup_old_checkpoints(self, experiment_id: str):
        """Remove old checkpoints to maintain storage limits."""
        experiment_checkpoints = self.checkpoint_registry.get("experiments", {}).get(experiment_id, [])
        
        if len(experiment_checkpoints) <= self.max_checkpoints:
            return
        
        # Get checkpoint info and sort by timestamp
        checkpoint_info = []
        for checkpoint_id in experiment_checkpoints:
            info = self.checkpoint_registry["checkpoints"].get(checkpoint_id, {})
            if info:
                checkpoint_info.append((checkpoint_id, info))
        
        checkpoint_info.sort(key=lambda x: x[1].get("timestamp", ""))
        
        # Remove oldest checkpoints (keep best model)
        checkpoints_to_remove = checkpoint_info[:-self.max_checkpoints]
        
        for checkpoint_id, info in checkpoints_to_remove:
            # Remove file
            file_path = info.get("file_path")
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
            
            # Remove from registry
            self.checkpoint_registry["checkpoints"].pop(checkpoint_id, None)
            self.checkpoint_registry["experiments"][experiment_id].remove(checkpoint_id)
            
            self.logger.info("Removed old checkpoint", checkpoint_id=checkpoint_id)
        
        self._save_registry()


# =============================================================================
# EXPERIMENT TRACKER
# =============================================================================

class ExperimentTracker:
    """Main experiment tracking and checkpointing system."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        
    """__init__ function."""
self.experiment_name = experiment_name
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Create experiment metadata
        self.metadata = ExperimentMetadata(
            experiment_name=experiment_name,
            experiment_id="",
            description=config.get("description", ""),
            tags=config.get("tags", [])
        )
        self.metadata.experiment_id = self.metadata.generate_id()
        
        # Initialize tracking backends
        self.trackers = []
        self._initialize_trackers()
        
        # Initialize checkpointer
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        max_checkpoints = config.get("max_checkpoints", 10)
        self.checkpointer = ModelCheckpointer(checkpoint_dir, max_checkpoints)
        
        # Performance monitoring
        self.start_time = time.time()
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }
        
        self.logger.info("Experiment tracker initialized", 
                        experiment_id=self.metadata.experiment_id)
    
    def _initialize_trackers(self) -> Any:
        """Initialize tracking backends based on configuration."""
        tracking_config = self.config.get("tracking", {})
        
        # Initialize WandB if enabled
        if tracking_config.get("use_wandb", False) and WANDB_AVAILABLE:
            try:
                wandb_tracker = WandBTracker(self.experiment_name, self.config)
                self.trackers.append(wandb_tracker)
                self.logger.info("WandB tracker initialized")
            except Exception as e:
                self.logger.warning("Failed to initialize WandB tracker", error=str(e))
        
        # Initialize TensorBoard if enabled
        if tracking_config.get("use_tensorboard", True):
            try:
                tensorboard_tracker = TensorBoardTracker(self.experiment_name, self.config)
                self.trackers.append(tensorboard_tracker)
                self.logger.info("TensorBoard tracker initialized")
            except Exception as e:
                self.logger.warning("Failed to initialize TensorBoard tracker", error=str(e))
        
        # Initialize MLflow if enabled
        if tracking_config.get("use_mlflow", False) and MLFLOW_AVAILABLE:
            try:
                mlflow_tracker = MLflowTracker(self.experiment_name, self.config)
                self.trackers.append(mlflow_tracker)
                self.logger.info("MLflow tracker initialized")
            except Exception as e:
                self.logger.warning("Failed to initialize MLflow tracker", error=str(e))
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to all tracking backends."""
        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Log to all trackers
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to {tracker.__class__.__name__}", error=str(e))
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters to all tracking backends."""
        for tracker in self.trackers:
            try:
                tracker.log_hyperparameters(hyperparameters)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to {tracker.__class__.__name__}", error=str(e))
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to all tracking backends."""
        for tracker in self.trackers:
            try:
                tracker.log_model(model_path, model_name)
            except Exception as e:
                self.logger.warning(f"Failed to log model to {tracker.__class__.__name__}", error=str(e))
    
    def log_artifact(self, artifact_path: str, artifact_name: str):
        """Log artifact to all tracking backends."""
        for tracker in self.trackers:
            try:
                tracker.log_artifact(artifact_path, artifact_name)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact to {tracker.__class__.__name__}", error=str(e))
    
    def log_image(self, image_path: str, image_name: str, step: int = None):
        """Log image to all tracking backends."""
        for tracker in self.trackers:
            try:
                tracker.log_image(image_path, image_name, step)
            except Exception as e:
                self.logger.warning(f"Failed to log image to {tracker.__class__.__name__}", error=str(e))
    
    def save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        is_best: bool = False
    ) -> str:
        """Save model checkpoint."""
        checkpoint_path = self.checkpointer.save_checkpoint(
            experiment_id=self.metadata.experiment_id,
            model=model,
            epoch=epoch,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            model_config=self.config.get("model", {}),
            training_config=self.config.get("training", {}),
            is_best=is_best
        )
        
        # Log checkpoint to trackers
        self.log_artifact(checkpoint_path, "checkpoint")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Tuple[int, int]:
        """Load model checkpoint."""
        metadata, epoch, step = self.checkpointer.load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        
        return epoch, step
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        return self.checkpointer.get_best_checkpoint(self.metadata.experiment_id)
    
    def create_performance_plots(self, output_dir: str = "plots"):
        """Create performance visualization plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create loss plot
        if self.metrics_history["train_loss"] and self.metrics_history["val_loss"]:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics_history["train_loss"], label="Train Loss")
            plt.plot(self.metrics_history["val_loss"], label="Validation Loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(self.metrics_history["train_accuracy"], label="Train Accuracy")
            plt.plot(self.metrics_history["val_accuracy"], label="Validation Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            
            # Learning curves
            plt.subplot(2, 2, 3)
            train_loss = np.array(self.metrics_history["train_loss"])
            val_loss = np.array(self.metrics_history["val_loss"])
            plt.plot(train_loss, label="Train Loss")
            plt.plot(val_loss, label="Validation Loss")
            plt.yscale('log')
            plt.title("Learning Curves (Log Scale)")
            plt.xlabel("Step")
            plt.ylabel("Loss (log)")
            plt.legend()
            plt.grid(True)
            
            # Loss difference
            plt.subplot(2, 2, 4)
            loss_diff = np.array(self.metrics_history["val_loss"]) - np.array(self.metrics_history["train_loss"])
            plt.plot(loss_diff, label="Val Loss - Train Loss")
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.title("Overfitting Indicator")
            plt.xlabel("Step")
            plt.ylabel("Loss Difference")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = output_dir / f"{self.metadata.experiment_id}_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log plot to trackers
            self.log_image(str(plot_path), "performance_plot")
            
            self.logger.info("Performance plots created", plot_path=str(plot_path))
    
    def finish(self, status: str = "completed"):
        """Finish experiment tracking."""
        # Update metadata
        self.metadata.status = status
        self.metadata.end_time = datetime.now()
        
        # Calculate experiment duration
        duration = time.time() - self.start_time
        
        # Log final metrics
        final_metrics = {
            "experiment_duration_seconds": duration,
            "total_steps": len(self.metrics_history["train_loss"]),
            "final_train_loss": self.metrics_history["train_loss"][-1] if self.metrics_history["train_loss"] else None,
            "final_val_loss": self.metrics_history["val_loss"][-1] if self.metrics_history["val_loss"] else None,
            "best_val_loss": min(self.metrics_history["val_loss"]) if self.metrics_history["val_loss"] else None,
            "final_train_accuracy": self.metrics_history["train_accuracy"][-1] if self.metrics_history["train_accuracy"] else None,
            "final_val_accuracy": self.metrics_history["val_accuracy"][-1] if self.metrics_history["val_accuracy"] else None,
            "best_val_accuracy": max(self.metrics_history["val_accuracy"]) if self.metrics_history["val_accuracy"] else None
        }
        
        self.log_metrics(final_metrics)
        
        # Create performance plots
        self.create_performance_plots()
        
        # Finish all trackers
        for tracker in self.trackers:
            try:
                tracker.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish {tracker.__class__.__name__}", error=str(e))
        
        # Save experiment metadata
        metadata_path = Path("experiments") / f"{self.metadata.experiment_id}_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.metadata.to_dict(), f, indent=2, default=str)
        
        self.logger.info("Experiment tracking finished", 
                        experiment_id=self.metadata.experiment_id,
                        status=status,
                        duration=duration)


# =============================================================================
# CONTEXT MANAGER FOR EXPERIMENT TRACKING
# =============================================================================

@contextmanager
def experiment_tracking(experiment_name: str, config: Dict[str, Any]):
    """
    Context manager for experiment tracking.
    
    Usage:
        with experiment_tracking("my_experiment", config) as tracker:
            # Your training loop here
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.save_checkpoint(model, epoch=1, step=100)
    """
    tracker = ExperimentTracker(experiment_name, config)
    
    try:
        yield tracker
        tracker.finish("completed")
    except Exception as e:
        tracker.finish("failed")
        raise


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main():
    """Example usage of the experiment tracking system."""
    
    # Example configuration
    config = {
        "description": "Example experiment for demonstration",
        "tags": ["example", "demo"],
        "tracking": {
            "use_wandb": False,
            "use_tensorboard": True,
            "use_mlflow": False
        },
        "checkpoint_dir": "checkpoints",
        "max_checkpoints": 5,
        "model": {
            "type": "transformer",
            "name": "bert-base-uncased"
        },
        "training": {
            "epochs": 10,
            "learning_rate": 2e-5
        }
    }
    
    # Example usage with context manager
    with experiment_tracking("example_experiment", config) as tracker:
        # Log hyperparameters
        tracker.log_hyperparameters({
            "learning_rate": 2e-5,
            "batch_size": 16,
            "epochs": 10
        })
        
        # Simulate training loop
        for epoch in range(3):
            for step in range(10):
                # Simulate metrics
                train_loss = 1.0 - (epoch * 10 + step) * 0.01
                val_loss = train_loss + 0.1
                train_acc = 0.5 + (epoch * 10 + step) * 0.02
                val_acc = train_acc - 0.05
                
                # Log metrics
                tracker.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc
                }, step=epoch * 10 + step)
                
                # Save checkpoint every 5 steps
                if step % 5 == 0:
                    # Create dummy model for demonstration
                    model = nn.Linear(10, 1)
                    optimizer = torch.optim.Adam(model.parameters())
                    
                    tracker.save_checkpoint(
                        model=model,
                        epoch=epoch,
                        step=epoch * 10 + step,
                        optimizer=optimizer,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_accuracy=train_acc,
                        val_accuracy=val_acc,
                        is_best=(val_loss < 0.5)  # Example best condition
                    )
    
    print("Experiment tracking example completed!")


match __name__:
    case "__main__":
    main() 