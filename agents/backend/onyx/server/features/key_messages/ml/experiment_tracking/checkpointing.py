from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import json
import time
import shutil
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import structlog
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Checkpointing System for Key Messages ML Pipeline
Handles model and training state saving/loading with various strategies
"""


logger = structlog.get_logger(__name__)

@dataclass
class ModelCheckpoint:
    """Represents a model checkpoint."""
    model_state: Dict[str, torch.Tensor]
    model_config: Dict[str, Any]
    model_path: str
    timestamp: float
    step: int
    epoch: int
    metrics: Dict[str, float]
    
    def save(self, path: str):
        """Save checkpoint to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save(self.model_state, f"{path}_model.pt")
        
        # Save metadata
        metadata = {
            'model_config': self.model_config,
            'timestamp': self.timestamp,
            'step': self.step,
            'epoch': self.epoch,
            'metrics': self.metrics
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelCheckpoint':
        """Load checkpoint from disk."""
        # Load model state
        model_state = torch.load(f"{path}_model.pt", map_location='cpu')
        
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            metadata = json.load(f)
        
        return cls(
            model_state=model_state,
            model_config=metadata['model_config'],
            model_path=path,
            timestamp=metadata['timestamp'],
            step=metadata['step'],
            epoch=metadata['epoch'],
            metrics=metadata['metrics']
        )

@dataclass
class TrainingCheckpoint:
    """Represents a complete training checkpoint."""
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    checkpoint_path: str
    timestamp: float
    step: int
    epoch: int
    metrics: Dict[str, float]
    best_metric: Optional[float] = None
    best_metric_name: Optional[str] = None
    
    def save(self, path: str):
        """Save training checkpoint to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save(self.model_state, f"{path}_model.pt")
        
        # Save optimizer state
        torch.save(self.optimizer_state, f"{path}_optimizer.pt")
        
        # Save scheduler state if available
        if self.scheduler_state is not None:
            torch.save(self.scheduler_state, f"{path}_scheduler.pt")
        
        # Save metadata
        metadata = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'timestamp': self.timestamp,
            'step': self.step,
            'epoch': self.epoch,
            'metrics': self.metrics,
            'best_metric': self.best_metric,
            'best_metric_name': self.best_metric_name
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingCheckpoint':
        """Load training checkpoint from disk."""
        # Load model state
        model_state = torch.load(f"{path}_model.pt", map_location='cpu')
        
        # Load optimizer state
        optimizer_state = torch.load(f"{path}_optimizer.pt", map_location='cpu')
        
        # Load scheduler state if available
        scheduler_state = None
        scheduler_path = f"{path}_scheduler.pt"
        if os.path.exists(scheduler_path):
            scheduler_state = torch.load(scheduler_path, map_location='cpu')
        
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            metadata = json.load(f)
        
        return cls(
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            model_config=metadata['model_config'],
            training_config=metadata['training_config'],
            checkpoint_path=path,
            timestamp=metadata['timestamp'],
            step=metadata['step'],
            epoch=metadata['epoch'],
            metrics=metadata['metrics'],
            best_metric=metadata.get('best_metric'),
            best_metric_name=metadata.get('best_metric_name')
        )

@dataclass
class CheckpointStrategy:
    """Strategy for checkpoint saving."""
    save_steps: int = 1000
    save_total_limit: int = 3
    save_best_only: bool = False
    monitor: str = "loss"
    mode: str = "min"  # "min" or "max"
    save_on_epoch_end: bool = True
    save_on_training_end: bool = True
    
    def __post_init__(self) -> Any:
        """Validate strategy parameters."""
        if self.mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")
        
        if self.save_steps <= 0:
            raise ValueError("save_steps must be positive")
        
        if self.save_total_limit <= 0:
            raise ValueError("save_total_limit must be positive")

class CheckpointManager:
    """Manages model and training checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", 
                 strategy: Optional[CheckpointStrategy] = None):
        
    """__init__ function."""
self.checkpoint_dir = Path(checkpoint_dir)
        self.strategy = strategy or CheckpointStrategy()
        self.checkpoints: List[str] = []
        self.best_metric = None
        self.best_checkpoint_path = None
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
        
        logger.info("CheckpointManager initialized", 
                   checkpoint_dir=str(self.checkpoint_dir),
                   strategy=self.strategy)
    
    def _load_existing_checkpoints(self) -> Any:
        """Load existing checkpoints from disk."""
        if not self.checkpoint_dir.exists():
            return
        
        # Find all checkpoint directories
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_"):
                self.checkpoints.append(str(item))
        
        # Sort by timestamp
        self.checkpoints.sort(key=lambda x: self._get_checkpoint_timestamp(x))
        
        # Find best checkpoint if save_best_only is enabled
        if self.strategy.save_best_only and self.checkpoints:
            self._find_best_checkpoint()
    
    def _get_checkpoint_timestamp(self, checkpoint_path: str) -> float:
        """Get timestamp from checkpoint metadata."""
        try:
            metadata_path = f"{checkpoint_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    metadata = json.load(f)
                return metadata.get('timestamp', 0)
        except Exception:
            pass
        return 0
    
    def _find_best_checkpoint(self) -> Any:
        """Find the best checkpoint based on monitored metric."""
        best_metric = None
        best_path = None
        
        for checkpoint_path in self.checkpoints:
            try:
                metadata_path = f"{checkpoint_path}_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        metadata = json.load(f)
                    
                    metric_value = metadata.get('metrics', {}).get(self.strategy.monitor)
                    if metric_value is not None:
                        if best_metric is None:
                            best_metric = metric_value
                            best_path = checkpoint_path
                        elif self.strategy.mode == "min" and metric_value < best_metric:
                            best_metric = metric_value
                            best_path = checkpoint_path
                        elif self.strategy.mode == "max" and metric_value > best_metric:
                            best_metric = metric_value
                            best_path = checkpoint_path
            except Exception as e:
                logger.warning(f"Failed to read checkpoint metadata: {e}")
        
        self.best_metric = best_metric
        self.best_checkpoint_path = best_path
    
    def should_save_checkpoint(self, step: int, metric_value: Optional[float] = None) -> bool:
        """Determine if a checkpoint should be saved."""
        # Check if we should save based on steps
        if step % self.strategy.save_steps == 0:
            return True
        
        # Check if we should save based on best metric
        if self.strategy.save_best_only and metric_value is not None:
            if self.best_metric is None:
                return True
            
            if self.strategy.mode == "min" and metric_value < self.best_metric:
                return True
            
            if self.strategy.mode == "max" and metric_value > self.best_metric:
                return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module, optimizer: Optimizer,
                       scheduler: Optional[_LRScheduler] = None,
                       epoch: int = 0, step: int = 0,
                       metrics: Optional[Dict[str, float]] = None,
                       model_config: Optional[Dict[str, Any]] = None,
                       training_config: Optional[Dict[str, Any]] = None) -> str:
        """Save a training checkpoint."""
        metrics = metrics or {}
        model_config = model_config or {}
        training_config = training_config or {}
        
        # Create checkpoint name
        timestamp = int(time.time())
        checkpoint_name = f"checkpoint_step_{step}_epoch_{epoch}_{timestamp}"
        checkpoint_path = str(self.checkpoint_dir / checkpoint_name)
        
        # Get current metric value
        current_metric = metrics.get(self.strategy.monitor)
        
        # Create training checkpoint
        checkpoint = TrainingCheckpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler else None,
            model_config=model_config,
            training_config=training_config,
            checkpoint_path=checkpoint_path,
            timestamp=timestamp,
            step=step,
            epoch=epoch,
            metrics=metrics,
            best_metric=current_metric,
            best_metric_name=self.strategy.monitor
        )
        
        # Save checkpoint
        checkpoint.save(checkpoint_path)
        
        # Update tracking
        self.checkpoints.append(checkpoint_path)
        
        # Update best checkpoint if necessary
        if current_metric is not None:
            if self.best_metric is None:
                self.best_metric = current_metric
                self.best_checkpoint_path = checkpoint_path
            elif self.strategy.mode == "min" and current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_checkpoint_path = checkpoint_path
            elif self.strategy.mode == "max" and current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_checkpoint_path = checkpoint_path
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info("Checkpoint saved", 
                   checkpoint_path=checkpoint_path,
                   step=step,
                   epoch=epoch,
                   metrics=metrics)
        
        return checkpoint_path
    
    def save_model_checkpoint(self, model: nn.Module, model_config: Dict[str, Any],
                             step: int = 0, epoch: int = 0,
                             metrics: Optional[Dict[str, float]] = None) -> str:
        """Save a model-only checkpoint."""
        metrics = metrics or {}
        
        # Create checkpoint name
        timestamp = int(time.time())
        checkpoint_name = f"model_checkpoint_step_{step}_epoch_{epoch}_{timestamp}"
        checkpoint_path = str(self.checkpoint_dir / checkpoint_name)
        
        # Create model checkpoint
        checkpoint = ModelCheckpoint(
            model_state=model.state_dict(),
            model_config=model_config,
            model_path=checkpoint_path,
            timestamp=timestamp,
            step=step,
            epoch=epoch,
            metrics=metrics
        )
        
        # Save checkpoint
        checkpoint.save(checkpoint_path)
        
        logger.info("Model checkpoint saved", 
                   checkpoint_path=checkpoint_path,
                   step=step,
                   epoch=epoch)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> TrainingCheckpoint:
        """Load a training checkpoint."""
        try:
            checkpoint = TrainingCheckpoint.load(checkpoint_path)
            logger.info("Checkpoint loaded", checkpoint_path=checkpoint_path)
            return checkpoint
        except Exception as e:
            logger.error("Failed to load checkpoint", checkpoint_path=checkpoint_path, error=str(e))
            raise
    
    def load_model_checkpoint(self, checkpoint_path: str) -> ModelCheckpoint:
        """Load a model checkpoint."""
        try:
            checkpoint = ModelCheckpoint.load(checkpoint_path)
            logger.info("Model checkpoint loaded", checkpoint_path=checkpoint_path)
            return checkpoint
        except Exception as e:
            logger.error("Failed to load model checkpoint", checkpoint_path=checkpoint_path, error=str(e))
            raise
    
    def load_best_checkpoint(self) -> Optional[TrainingCheckpoint]:
        """Load the best checkpoint based on monitored metric."""
        if self.best_checkpoint_path is None:
            logger.warning("No best checkpoint found")
            return None
        
        return self.load_checkpoint(self.best_checkpoint_path)
    
    def load_latest_checkpoint(self) -> Optional[TrainingCheckpoint]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            logger.warning("No checkpoints found")
            return None
        
        latest_checkpoint = self.checkpoints[-1]
        return self.load_checkpoint(latest_checkpoint)
    
    def restore_training_state(self, model: nn.Module, optimizer: Optimizer,
                              scheduler: Optional[_LRScheduler] = None,
                              checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Restore training state from checkpoint."""
        if checkpoint_path is None:
            checkpoint = self.load_latest_checkpoint()
        else:
            checkpoint = self.load_checkpoint(checkpoint_path)
        
        if checkpoint is None:
            logger.warning("No checkpoint to restore from")
            return {}
        
        # Restore model state
        model.load_state_dict(checkpoint.model_state)
        
        # Restore optimizer state
        optimizer.load_state_dict(checkpoint.optimizer_state)
        
        # Restore scheduler state
        if scheduler is not None and checkpoint.scheduler_state is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state)
        
        logger.info("Training state restored", 
                   checkpoint_path=checkpoint.checkpoint_path,
                   step=checkpoint.step,
                   epoch=checkpoint.epoch)
        
        return {
            'step': checkpoint.step,
            'epoch': checkpoint.epoch,
            'metrics': checkpoint.metrics,
            'best_metric': checkpoint.best_metric
        }
    
    def _cleanup_old_checkpoints(self) -> Any:
        """Remove old checkpoints based on save_total_limit."""
        if len(self.checkpoints) <= self.strategy.save_total_limit:
            return
        
        # Remove oldest checkpoints
        checkpoints_to_remove = self.checkpoints[:-self.strategy.save_total_limit]
        
        for checkpoint_path in checkpoints_to_remove:
            try:
                # Remove checkpoint files
                for ext in ['_model.pt', '_optimizer.pt', '_scheduler.pt', '_metadata.json']:
                    file_path = f"{checkpoint_path}{ext}"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                # Remove from tracking
                self.checkpoints.remove(checkpoint_path)
                
                logger.info("Old checkpoint removed", checkpoint_path=checkpoint_path)
                
            except Exception as e:
                logger.error("Failed to remove old checkpoint", 
                           checkpoint_path=checkpoint_path, error=str(e))
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about all checkpoints."""
        info = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'total_checkpoints': len(self.checkpoints),
            'strategy': {
                'save_steps': self.strategy.save_steps,
                'save_total_limit': self.strategy.save_total_limit,
                'save_best_only': self.strategy.save_best_only,
                'monitor': self.strategy.monitor,
                'mode': self.strategy.mode
            },
            'best_checkpoint': self.best_checkpoint_path,
            'best_metric': self.best_metric,
            'checkpoints': []
        }
        
        for checkpoint_path in self.checkpoints:
            try:
                metadata_path = f"{checkpoint_path}_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        metadata = json.load(f)
                    
                    info['checkpoints'].append({
                        'path': checkpoint_path,
                        'step': metadata.get('step'),
                        'epoch': metadata.get('epoch'),
                        'timestamp': metadata.get('timestamp'),
                        'metrics': metadata.get('metrics', {}),
                        'is_best': checkpoint_path == self.best_checkpoint_path
                    })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint info: {e}")
        
        return info
    
    def clear_checkpoints(self) -> Any:
        """Clear all checkpoints."""
        for checkpoint_path in self.checkpoints[:]:
            try:
                # Remove checkpoint files
                for ext in ['_model.pt', '_optimizer.pt', '_scheduler.pt', '_metadata.json']:
                    file_path = f"{checkpoint_path}{ext}"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                logger.info("Checkpoint removed", checkpoint_path=checkpoint_path)
                
            except Exception as e:
                logger.error("Failed to remove checkpoint", 
                           checkpoint_path=checkpoint_path, error=str(e))
        
        self.checkpoints.clear()
        self.best_metric = None
        self.best_checkpoint_path = None
        
        logger.info("All checkpoints cleared") 