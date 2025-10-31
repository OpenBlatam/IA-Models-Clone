from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import os
import json
import yaml
import pickle
import shutil
from pathlib import Path
from datetime import datetime
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
import logging
from contextlib import contextmanager
import threading
from collections import defaultdict
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
    import wandb
    import mlflow
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.config_manager import ExperimentConfig
from typing import Any, List, Dict, Optional
import asyncio
"""
Experiment Tracker and Model Checkpointing System

This module provides comprehensive experiment tracking and model checkpointing including:
- Multi-backend experiment tracking (W&B, MLflow, TensorBoard, Local)
- Automated model checkpointing with versioning
- Experiment comparison and analysis
- Reproducibility management
- Performance monitoring and alerting
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


logger = setup_logger()

class TrackingBackend(Enum):
    """Available tracking backends."""
    WANDB = "wandb"
    MLFLOW = "mlflow"
    TENSORBOARD = "tensorboard"
    LOCAL = "local"
    CUSTOM = "custom"

@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking."""
    experiment_id: str
    experiment_name: str
    project_name: str
    created_at: datetime
    tags: List[str] = field(default_factory=list)
    description: str = ""
    git_commit: str = ""
    python_version: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CheckpointInfo:
    """Information about a model checkpoint."""
    checkpoint_id: str
    experiment_id: str
    epoch: int
    step: int
    metrics: Dict[str, float]
    file_path: str
    file_size: int
    created_at: datetime
    is_best: bool = False
    checkpoint_type: str = "model"  # model, optimizer, scheduler, full
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExperimentTracker:
    """Comprehensive experiment tracking system."""
    
    def __init__(self, config: ExperimentConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logger
        self.experiment_id = config.experiment_id or str(uuid.uuid4())
        self.metadata = None
        self.backend = None
        self.checkpoint_manager = None
        
        # Initialize tracking backend
        self._initialize_backend()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=config.max_checkpoints,
            save_optimizer=config.save_optimizer,
            save_scheduler=config.save_scheduler
        )
        
        # Tracking state
        self.current_epoch = 0
        self.current_step = 0
        self.metrics_history = defaultdict(list)
        self.hyperparameters = {}
        
    def _initialize_backend(self) -> Any:
        """Initialize the tracking backend."""
        backend_name = self.config.tracking_backend.lower()
        
        if backend_name == "wandb" and WANDB_AVAILABLE:
            self.backend = WandbBackend(self.config)
        elif backend_name == "mlflow" and MLFLOW_AVAILABLE:
            self.backend = MLflowBackend(self.config)
        elif backend_name == "tensorboard":
            self.backend = TensorboardBackend(self.config)
        elif backend_name == "local":
            self.backend = LocalBackend(self.config)
        else:
            self.logger.warning(f"Backend {backend_name} not available, using local backend")
            self.backend = LocalBackend(self.config)
    
    def start_experiment(self, metadata: Optional[ExperimentMetadata] = None):
        """Start a new experiment."""
        if metadata is None:
            metadata = ExperimentMetadata(
                experiment_id=self.experiment_id,
                experiment_name=self.config.experiment_name,
                project_name=self.config.project_name,
                created_at=datetime.now(),
                tags=self.config.tags
            )
        
        self.metadata = metadata
        
        # Start backend tracking
        if self.backend:
            self.backend.start_experiment(metadata)
        
        # Create experiment directory
        experiment_dir = Path(self.config.checkpoint_dir) / self.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        self._save_experiment_metadata()
        
        self.logger.info(f"Experiment started: {self.experiment_id}")
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters for the experiment."""
        self.hyperparameters = hyperparameters
        
        if self.backend:
            self.backend.log_hyperparameters(hyperparameters)
        
        # Save to local file
        hp_file = Path(self.config.checkpoint_dir) / self.experiment_id / "hyperparameters.yaml"
        with open(hp_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(hyperparameters, f, default_flow_style=False)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, epoch: Optional[int] = None):
        """Log metrics for the experiment."""
        if step is not None:
            self.current_step = step
        if epoch is not None:
            self.current_epoch = epoch
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['step'] = self.current_step
        metrics['epoch'] = self.current_epoch
        
        # Store in history
        for key, value in metrics.items():
            if key not in ['timestamp', 'step', 'epoch']:
                self.metrics_history[key].append({
                    'value': value,
                    'step': self.current_step,
                    'epoch': self.current_epoch,
                    'timestamp': metrics['timestamp']
                })
        
        # Log to backend
        if self.backend:
            self.backend.log_metrics(metrics, self.current_step)
        
        # Log to console if configured
        if self.config.log_frequency > 0 and self.current_step % self.config.log_frequency == 0:
            self.logger.info(f"Step {self.current_step}, Epoch {self.current_epoch}: {metrics}")
    
    def log_model_architecture(self, model: nn.Module):
        """Log model architecture."""
        if self.backend:
            self.backend.log_model_architecture(model)
        
        # Save model summary locally
        model_summary = self._generate_model_summary(model)
        summary_file = Path(self.config.checkpoint_dir) / self.experiment_id / "model_summary.txt"
        with open(summary_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(model_summary)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       is_best: bool = False) -> str:
        """Save a model checkpoint."""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            experiment_id=self.experiment_id,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=self.current_epoch,
            step=self.current_step,
            metrics=metrics or {},
            is_best=is_best
        )
        
        # Log checkpoint to backend
        if self.backend:
            self.backend.log_checkpoint(checkpoint_path, metrics or {})
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, 
                       model: nn.Module,
                       checkpoint_path: str,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """Load a model checkpoint."""
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # Update current state
        self.current_epoch = checkpoint_info.get('epoch', 0)
        self.current_step = checkpoint_info.get('step', 0)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_info
    
    def log_gradients(self, model: nn.Module, step: int):
        """Log gradient information."""
        if not self.config.log_gradients:
            return
        
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[f"{name}_grad_norm"] = param.grad.norm().item()
                gradients[f"{name}_grad_mean"] = param.grad.mean().item()
                gradients[f"{name}_grad_std"] = param.grad.std().item()
        
        if self.backend:
            self.backend.log_metrics(gradients, step)
    
    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """Log images for visualization."""
        if self.backend:
            self.backend.log_images(images, step)
    
    def log_text(self, text_data: Dict[str, str], step: int):
        """Log text data."""
        if self.backend:
            self.backend.log_text(text_data, step)
    
    def end_experiment(self) -> Any:
        """End the experiment."""
        if self.backend:
            self.backend.end_experiment()
        
        # Save final experiment summary
        self._save_experiment_summary()
        
        self.logger.info(f"Experiment ended: {self.experiment_id}")
    
    def _save_experiment_metadata(self) -> Any:
        """Save experiment metadata to file."""
        metadata_file = Path(self.config.checkpoint_dir) / self.experiment_id / "metadata.yaml"
        metadata_dict = asdict(self.metadata)
        metadata_dict['created_at'] = metadata_dict['created_at'].isoformat()
        
        with open(metadata_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(metadata_dict, f, default_flow_style=False)
    
    def _save_experiment_summary(self) -> Any:
        """Save experiment summary."""
        summary = {
            'experiment_id': self.experiment_id,
            'experiment_name': self.config.experiment_name,
            'project_name': self.config.project_name,
            'start_time': self.metadata.created_at.isoformat() if self.metadata else None,
            'end_time': datetime.now().isoformat(),
            'total_steps': self.current_step,
            'total_epochs': self.current_epoch,
            'hyperparameters': self.hyperparameters,
            'final_metrics': {key: values[-1]['value'] if values else None 
                            for key, values in self.metrics_history.items()},
            'checkpoints': self.checkpoint_manager.get_checkpoint_info(self.experiment_id)
        }
        
        summary_file = Path(self.config.checkpoint_dir) / self.experiment_id / "experiment_summary.yaml"
        with open(summary_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(summary, f, default_flow_style=False)
    
    def _generate_model_summary(self, model: nn.Module) -> str:
        """Generate a summary of the model architecture."""
        summary = []
        summary.append(f"Model Architecture Summary")
        summary.append(f"=" * 50)
        summary.append(f"Model: {type(model).__name__}")
        summary.append(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        summary.append(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        summary.append("")
        summary.append("Layer Details:")
        summary.append("-" * 30)
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                param_count = sum(p.numel() for p in module.parameters())
                summary.append(f"{name}: {type(module).__name__} - {param_count:,} parameters")
        
        return "\n".join(summary)

class CheckpointManager:
    """Manages model checkpoints with versioning and cleanup."""
    
    def __init__(self, 
                 checkpoint_dir: str = "./checkpoints",
                 max_checkpoints: int = 5,
                 save_optimizer: bool = True,
                 save_scheduler: bool = True):
        
    """__init__ function."""
self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.logger = logger
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self,
                       experiment_id: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       epoch: int = 0,
                       step: int = 0,
                       metrics: Optional[Dict[str, float]] = None,
                       is_best: bool = False) -> str:
        """Save a model checkpoint."""
        experiment_dir = self.checkpoint_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}"
        if is_best:
            checkpoint_id += "_best"
        
        checkpoint_path = experiment_dir / f"{checkpoint_id}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
            'is_best': is_best,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_id': checkpoint_id
        }
        
        if self.save_optimizer and optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if self.save_scheduler and scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            experiment_id=experiment_id,
            epoch=epoch,
            step=step,
            metrics=metrics or {},
            file_path=str(checkpoint_path),
            file_size=checkpoint_path.stat().st_size,
            created_at=datetime.now(),
            is_best=is_best
        )
        
        self._save_checkpoint_info(experiment_id, checkpoint_info)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(experiment_id)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self,
                       model: nn.Module,
                       checkpoint_path: str,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """Load a model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state if available
        if optimizer and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        return checkpoint_data
    
    def get_best_checkpoint(self, experiment_id: str) -> Optional[str]:
        """Get the path to the best checkpoint for an experiment."""
        experiment_dir = self.checkpoint_dir / experiment_id
        
        if not experiment_dir.exists():
            return None
        
        # Look for best checkpoint
        for checkpoint_file in experiment_dir.glob("*_best.pt"):
            return str(checkpoint_file)
        
        return None
    
    def get_latest_checkpoint(self, experiment_id: str) -> Optional[str]:
        """Get the path to the latest checkpoint for an experiment."""
        experiment_dir = self.checkpoint_dir / experiment_id
        
        if not experiment_dir.exists():
            return None
        
        # Get all checkpoint files
        checkpoint_files = list(experiment_dir.glob("*.pt"))
        
        if not checkpoint_files:
            return None
        
        # Return the most recent one
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        return str(latest_checkpoint)
    
    def get_checkpoint_info(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get information about all checkpoints for an experiment."""
        info_file = self.checkpoint_dir / experiment_id / "checkpoint_info.yaml"
        
        if not info_file.exists():
            return []
        
        with open(info_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            checkpoint_infos = yaml.safe_load(f)
        
        return checkpoint_infos or []
    
    def _save_checkpoint_info(self, experiment_id: str, checkpoint_info: CheckpointInfo):
        """Save checkpoint information."""
        info_file = self.checkpoint_dir / experiment_id / "checkpoint_info.yaml"
        
        # Load existing info
        checkpoint_infos = self.get_checkpoint_info(experiment_id)
        
        # Add new checkpoint info
        checkpoint_infos.append(asdict(checkpoint_info))
        
        # Save updated info
        with open(info_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(checkpoint_infos, f, default_flow_style=False)
    
    def _cleanup_old_checkpoints(self, experiment_id: str):
        """Remove old checkpoints to stay within limits."""
        checkpoint_infos = self.get_checkpoint_info(experiment_id)
        
        if len(checkpoint_infos) <= self.max_checkpoints:
            return
        
        # Sort by creation time (oldest first)
        checkpoint_infos.sort(key=lambda x: x['created_at'])
        
        # Remove oldest checkpoints (keep best checkpoint if it exists)
        checkpoints_to_remove = []
        best_checkpoint_kept = False
        
        for checkpoint_info in checkpoint_infos[:-self.max_checkpoints]:
            if checkpoint_info['is_best'] and not best_checkpoint_kept:
                best_checkpoint_kept = True
                continue
            
            checkpoints_to_remove.append(checkpoint_info)
        
        # Remove checkpoints
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info['file_path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update checkpoint info file
        remaining_checkpoints = [c for c in checkpoint_infos if c not in checkpoints_to_remove]
        info_file = self.checkpoint_dir / experiment_id / "checkpoint_info.yaml"
        with open(info_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(remaining_checkpoints, f, default_flow_style=False)

# Tracking Backend Implementations
class TrackingBackendBase:
    """Base class for tracking backends."""
    
    def __init__(self, config: ExperimentConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logger
    
    def start_experiment(self, metadata: ExperimentMetadata):
        """Start experiment tracking."""
        raise NotImplementedError
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any):
        """Log hyperparameters."""
        raise NotImplementedError
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics."""
        raise NotImplementedError
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]):
        """Log checkpoint."""
        raise NotImplementedError
    
    def log_model_architecture(self, model: nn.Module):
        """Log model architecture."""
        raise NotImplementedError
    
    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """Log images."""
        raise NotImplementedError
    
    def log_text(self, text_data: Dict[str, str], step: int):
        """Log text data."""
        raise NotImplementedError
    
    def end_experiment(self) -> Any:
        """End experiment tracking."""
        raise NotImplementedError

class WandbBackend(TrackingBackendBase):
    """Weights & Biases tracking backend."""
    
    def __init__(self, config: ExperimentConfig):
        
    """__init__ function."""
super().__init__(config)
        self.run = None
    
    def start_experiment(self, metadata: ExperimentMetadata):
        """Start W&B experiment."""
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not available")
        
        self.run = wandb.init(
            project=self.config.project_name,
            name=metadata.experiment_name,
            id=metadata.experiment_id,
            tags=metadata.tags,
            config=self.config.tracking_params
        )
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters to W&B."""
        if self.run:
            self.run.config.update(hyperparameters)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to W&B."""
        if self.run:
            self.run.log(metrics, step=step)
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]):
        """Log checkpoint to W&B."""
        if self.run:
            self.run.save(checkpoint_path)
    
    def log_model_architecture(self, model: nn.Module):
        """Log model architecture to W&B."""
        if self.run:
            self.run.watch(model)
    
    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """Log images to W&B."""
        if self.run:
            for name, image in images.items():
                self.run.log({name: wandb.Image(image)}, step=step)
    
    def log_text(self, text_data: Dict[str, str], step: int):
        """Log text to W&B."""
        if self.run:
            for name, text in text_data.items():
                self.run.log({name: wandb.Html(text)}, step=step)
    
    def end_experiment(self) -> Any:
        """End W&B experiment."""
        if self.run:
            self.run.finish()

class TensorboardBackend(TrackingBackendBase):
    """TensorBoard tracking backend."""
    
    def __init__(self, config: ExperimentConfig):
        
    """__init__ function."""
super().__init__(config)
        self.writer = None
    
    def start_experiment(self, metadata: ExperimentMetadata):
        """Start TensorBoard experiment."""
        log_dir = Path(self.config.checkpoint_dir) / metadata.experiment_id / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(log_dir))
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        if self.writer:
            self.writer.add_hparams(hyperparameters, {})
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard."""
        if self.writer:
            for name, value in metrics.items():
                if name not in ['timestamp', 'step', 'epoch']:
                    self.writer.add_scalar(name, value, step)
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]):
        """Log checkpoint to TensorBoard."""
        # TensorBoard doesn't have native checkpoint logging
        pass
    
    def log_model_architecture(self, model: nn.Module):
        """Log model architecture to TensorBoard."""
        if self.writer:
            # Create dummy input for model graph
            dummy_input = torch.randn(1, 3, 224, 224)  # Adjust based on your model
            try:
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                self.logger.warning(f"Could not log model architecture: {e}")
    
    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """Log images to TensorBoard."""
        if self.writer:
            for name, image in images.items():
                self.writer.add_image(name, image, step)
    
    def log_text(self, text_data: Dict[str, str], step: int):
        """Log text to TensorBoard."""
        if self.writer:
            for name, text in text_data.items():
                self.writer.add_text(name, text, step)
    
    def end_experiment(self) -> Any:
        """End TensorBoard experiment."""
        if self.writer:
            self.writer.close()

class LocalBackend(TrackingBackendBase):
    """Local file-based tracking backend."""
    
    def __init__(self, config: ExperimentConfig):
        
    """__init__ function."""
super().__init__(config)
        self.metrics_file = None
        self.hyperparameters_file = None
    
    def start_experiment(self, metadata: ExperimentMetadata):
        """Start local experiment tracking."""
        experiment_dir = Path(self.config.checkpoint_dir) / metadata.experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = experiment_dir / "metrics.jsonl"
        self.hyperparameters_file = experiment_dir / "hyperparameters.yaml"
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters to local file."""
        if self.hyperparameters_file:
            with open(self.hyperparameters_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(hyperparameters, f, default_flow_style=False)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to local file."""
        if self.metrics_file:
            metrics['step'] = step
            metrics['timestamp'] = datetime.now().isoformat()
            
            with open(self.metrics_file, 'a') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(json.dumps(metrics) + '\n')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]):
        """Log checkpoint to local file."""
        # Checkpoint is already saved by CheckpointManager
        pass
    
    def log_model_architecture(self, model: nn.Module):
        """Log model architecture to local file."""
        # Model architecture is logged by ExperimentTracker
        pass
    
    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """Log images to local file."""
        experiment_dir = Path(self.config.checkpoint_dir) / self.config.experiment_id
        images_dir = experiment_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for name, image in images.items():
            image_path = images_dir / f"{name}_step_{step}.png"
            # Save image using torchvision or PIL
            pass
    
    def log_text(self, text_data: Dict[str, str], step: int):
        """Log text to local file."""
        experiment_dir = Path(self.config.checkpoint_dir) / self.config.experiment_id
        text_dir = experiment_dir / "text"
        text_dir.mkdir(exist_ok=True)
        
        for name, text in text_data.items():
            text_path = text_dir / f"{name}_step_{step}.txt"
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
    
    def end_experiment(self) -> Any:
        """End local experiment tracking."""
        # Nothing to do for local backend
        pass

# Utility functions
def create_experiment_tracker(config: ExperimentConfig) -> ExperimentTracker:
    """Create an experiment tracker with the given configuration."""
    return ExperimentTracker(config)

@contextmanager
def experiment_context(config: ExperimentConfig, metadata: Optional[ExperimentMetadata] = None):
    """Context manager for experiment tracking."""
    tracker = ExperimentTracker(config)
    try:
        tracker.start_experiment(metadata)
        yield tracker
    finally:
        tracker.end_experiment()

# Example usage
if __name__ == "__main__":
    # Create experiment config
    config = ExperimentConfig(
        experiment_name="test_experiment",
        project_name="test_project",
        track_experiments=True,
        tracking_backend="local",
        save_checkpoints=True,
        checkpoint_dir="./checkpoints"
    )
    
    # Create experiment tracker
    tracker = ExperimentTracker(config)
    
    # Start experiment
    metadata = ExperimentMetadata(
        experiment_id="test_123",
        experiment_name="test_experiment",
        project_name="test_project",
        created_at=datetime.now(),
        tags=["test", "example"]
    )
    
    tracker.start_experiment(metadata)
    
    # Log hyperparameters
    hyperparameters = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100
    }
    tracker.log_hyperparameters(hyperparameters)
    
    # Log metrics
    for step in range(10):
        metrics = {
            "loss": 1.0 - step * 0.1,
            "accuracy": step * 0.1
        }
        tracker.log_metrics(metrics, step=step)
    
    # End experiment
    tracker.end_experiment() 