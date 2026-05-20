from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy
import pickle
import shutil
    import wandb
    from torch.utils.tensorboard import SummaryWriter
    import torch
                    import torchvision.utils as vutils
                    import numpy as np
                        import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional
import asyncio
"""
Experiment Tracking System
==========================

This module provides comprehensive experiment tracking and model checkpointing
for AI video generation experiments.

Features:
- WandB integration for experiment tracking
- TensorBoard integration for visualization
- Custom logging and metrics tracking
- Model checkpointing with versioning
- Experiment metadata management
- Performance monitoring
- Artifact management
"""


# Optional imports for external tracking tools
try:
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    # Experiment identification
    experiment_name: str = "default_experiment"
    project_name: str = "ai_video_generation"
    run_id: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Tracking tools
    use_wandb: bool = False
    use_tensorboard: bool = True
    use_custom_logging: bool = True
    
    # Logging settings
    log_frequency: int = 100
    save_frequency: int = 1000
    eval_frequency: int = 500
    
    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    max_checkpoints: int = 5
    checkpoint_metrics: List[str] = field(default_factory=lambda: ["loss", "val_loss"])
    
    # Artifact settings
    save_artifacts: bool = True
    artifact_dir: str = "artifacts"
    save_config: bool = True
    save_samples: bool = True
    save_model: bool = True
    
    def __post_init__(self) -> Any:
        """Generate run ID if not provided."""
        if not self.run_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{self.experiment_name}_{timestamp}"


@dataclass
class CheckpointInfo:
    """Information about a model checkpoint."""
    
    checkpoint_path: str
    epoch: int
    step: int
    metrics: Dict[str, float]
    timestamp: str
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointInfo':
        """Create from dictionary."""
        return cls(**data)


class ExperimentTracker:
    """Main experiment tracking class."""
    
    def __init__(self, config: ExperimentConfig):
        
    """__init__ function."""
self.config = config
        self.start_time = time.time()
        self.current_step = 0
        self.current_epoch = 0
        
        # Initialize tracking tools
        self.wandb_run = None
        self.tensorboard_writer = None
        self.custom_logger = None
        
        # Checkpoint management
        self.checkpoints: List[CheckpointInfo] = []
        self.best_metrics: Dict[str, float] = {}
        
        # Metrics tracking
        self.metrics_history: Dict[str, List[float]] = {}
        self.artifacts: List[str] = []
        
        # Initialize tracking
        self._initialize_tracking()
        
        logger.info(f"Experiment tracker initialized: {config.experiment_name}")
    
    def _initialize_tracking(self) -> Any:
        """Initialize tracking tools."""
        # Create directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.artifact_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize WandB
        if self.config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        
        # Initialize TensorBoard
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self._init_tensorboard()
        
        # Initialize custom logging
        if self.config.use_custom_logging:
            self._init_custom_logging()
    
    def _init_wandb(self) -> Any:
        """Initialize WandB tracking."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                id=self.config.run_id,
                config={
                    "experiment_name": self.config.experiment_name,
                    "description": self.config.description,
                    "tags": self.config.tags,
                    "start_time": datetime.now().isoformat()
                },
                tags=self.config.tags,
                notes=self.config.description
            )
            logger.info(f"WandB initialized: {self.wandb_run.id}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    def _init_tensorboard(self) -> Any:
        """Initialize TensorBoard tracking."""
        try:
            log_dir = Path("runs") / self.config.run_id
            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard initialized: {log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.tensorboard_writer = None
    
    def _init_custom_logging(self) -> Any:
        """Initialize custom logging."""
        log_dir = Path("logs") / self.config.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create custom logger
        self.custom_logger = logging.getLogger(f"experiment_{self.config.run_id}")
        self.custom_logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_dir / "experiment.log")
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        self.custom_logger.addHandler(fh)
        logger.info(f"Custom logging initialized: {log_dir}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all tracking tools."""
        if step is None:
            step = self.current_step
        
        # Update metrics history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    self.tensorboard_writer.add_scalar(key, value, step)
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")
        
        # Log to custom logger
        if self.custom_logger:
            try:
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                self.custom_logger.info(f"Step {step}: {metrics_str}")
            except Exception as e:
                logger.warning(f"Failed to log to custom logger: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to tracking tools."""
        # Log to WandB
        if self.wandb_run:
            try:
                wandb.config.update(config)
            except Exception as e:
                logger.warning(f"Failed to log config to WandB: {e}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            try:
                # Log config as text
                config_str = json.dumps(config, indent=2)
                self.tensorboard_writer.add_text("Configuration", config_str, 0)
            except Exception as e:
                logger.warning(f"Failed to log config to TensorBoard: {e}")
        
        # Save config file
        if self.config.save_config:
            config_path = Path(self.config.artifact_dir) / "config.json"
            try:
                with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(config, f, indent=2)
                self.artifacts.append(str(config_path))
            except Exception as e:
                logger.warning(f"Failed to save config: {e}")
    
    def save_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        metrics: Optional[Dict[str, float]] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        is_best: bool = False
    ) -> Optional[str]:
        """Save model checkpoint."""
        if epoch is None:
            epoch = self.current_epoch
        if step is None:
            step = self.current_step
        if metrics is None:
            metrics = {}
        
        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_path="",
            epoch=epoch,
            step=step,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            model_state=model.state_dict() if TORCH_AVAILABLE else {},
            optimizer_state=optimizer.state_dict() if optimizer else None,
            scheduler_state=scheduler.state_dict() if scheduler else None,
            config=self.config.__dict__
        )
        
        # Determine checkpoint path
        if is_best:
            checkpoint_name = "best_model.pth"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pth"
        
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        
        # Save checkpoint
        try:
            if TORCH_AVAILABLE:
                torch.save(checkpoint_info.to_dict(), checkpoint_path)
            else:
                # Fallback to pickle
                with open(checkpoint_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    pickle.dump(checkpoint_info.to_dict(), f)
            
            checkpoint_info.checkpoint_path = str(checkpoint_path)
            self.checkpoints.append(checkpoint_info)
            
            # Update best metrics
            for metric_name, value in metrics.items():
                if metric_name not in self.best_metrics or value < self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
            
            # Clean up old checkpoints
            if self.config.save_best_only and len(self.checkpoints) > self.config.max_checkpoints:
                self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self) -> Any:
        """Remove old checkpoints to save space."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort by step (keep most recent)
        self.checkpoints.sort(key=lambda x: x.step, reverse=True)
        
        # Remove old checkpoints
        checkpoints_to_remove = self.checkpoints[self.config.max_checkpoints:]
        for checkpoint in checkpoints_to_remove:
            try:
                Path(checkpoint.checkpoint_path).unlink()
                logger.info(f"Removed old checkpoint: {checkpoint.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_path}: {e}")
        
        # Update checkpoints list
        self.checkpoints = self.checkpoints[:self.config.max_checkpoints]
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[CheckpointInfo]:
        """Load checkpoint from file."""
        try:
            if TORCH_AVAILABLE:
                checkpoint_data = torch.load(checkpoint_path)
            else:
                with open(checkpoint_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    checkpoint_data = pickle.load(f)
            
            checkpoint_info = CheckpointInfo.from_dict(checkpoint_data)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_info
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def load_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            return None
        
        latest_checkpoint = max(self.checkpoints, key=lambda x: x.step)
        return self.load_checkpoint(latest_checkpoint.checkpoint_path)
    
    def load_best_checkpoint(self, metric_name: str = "val_loss") -> Optional[CheckpointInfo]:
        """Load the best checkpoint based on metric."""
        if not self.checkpoints:
            return None
        
        best_checkpoint = min(self.checkpoints, key=lambda x: x.metrics.get(metric_name, float('inf')))
        return self.load_checkpoint(best_checkpoint.checkpoint_path)
    
    def save_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "file"):
        """Save artifact to tracking tools."""
        artifact_path = Path(file_path)
        if not artifact_path.exists():
            logger.warning(f"Artifact file not found: {file_path}")
            return
        
        # Save to WandB
        if self.wandb_run and self.config.save_artifacts:
            try:
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type=artifact_type,
                    description=f"Artifact from {self.config.experiment_name}"
                )
                artifact.add_file(file_path)
                self.wandb_run.log_artifact(artifact)
                logger.info(f"Artifact saved to WandB: {artifact_name}")
            except Exception as e:
                logger.warning(f"Failed to save artifact to WandB: {e}")
        
        # Copy to artifact directory
        if self.config.save_artifacts:
            try:
                dest_path = Path(self.config.artifact_dir) / artifact_path.name
                shutil.copy2(file_path, dest_path)
                self.artifacts.append(str(dest_path))
                logger.info(f"Artifact copied: {dest_path}")
            except Exception as e:
                logger.warning(f"Failed to copy artifact: {e}")
    
    def log_samples(self, samples: List[Any], sample_names: List[str], step: int):
        """Log sample outputs (images, videos, etc.)."""
        if not self.config.save_samples:
            return
        
        # Save samples to artifact directory
        samples_dir = Path(self.config.artifact_dir) / "samples" / f"step_{step}"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        for sample, name in zip(samples, sample_names):
            try:
                sample_path = samples_dir / f"{name}.png"
                
                # Handle different sample types
                if hasattr(sample, 'save'):
                    sample.save(sample_path)
                elif TORCH_AVAILABLE and isinstance(sample, torch.Tensor):
                    # Save tensor as image
                    vutils.save_image(sample, sample_path)
                else:
                    # Try to save as numpy array
                    if isinstance(sample, np.ndarray):
                        plt.imsave(sample_path, sample)
                    else:
                        logger.warning(f"Unknown sample type: {type(sample)}")
                        continue
                
                # Log to tracking tools
                self.save_artifact(str(sample_path), f"sample_{name}_step_{step}", "sample")
                
            except Exception as e:
                logger.warning(f"Failed to save sample {name}: {e}")
    
    def log_video(self, video_path: str, video_name: str, step: int):
        """Log video outputs."""
        if not self.config.save_samples:
            return
        
        try:
            # Log to WandB
            if self.wandb_run:
                wandb.log({f"video/{video_name}": wandb.Video(video_path)}, step=step)
            
            # Save to artifact directory
            if self.config.save_artifacts:
                dest_path = Path(self.config.artifact_dir) / f"video_{video_name}_step_{step}.mp4"
                shutil.copy2(video_path, dest_path)
                self.artifacts.append(str(dest_path))
            
            logger.info(f"Video logged: {video_name}")
            
        except Exception as e:
            logger.warning(f"Failed to log video {video_name}: {e}")
    
    def update_step(self, step: int):
        """Update current step."""
        self.current_step = step
    
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        duration = time.time() - self.start_time
        
        return {
            "experiment_name": self.config.experiment_name,
            "run_id": self.config.run_id,
            "duration_seconds": duration,
            "total_steps": self.current_step,
            "total_epochs": self.current_epoch,
            "best_metrics": self.best_metrics,
            "total_checkpoints": len(self.checkpoints),
            "total_artifacts": len(self.artifacts),
            "wandb_enabled": self.wandb_run is not None,
            "tensorboard_enabled": self.tensorboard_writer is not None
        }
    
    def close(self) -> Any:
        """Close experiment tracking."""
        # Close TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Finish WandB run
        if self.wandb_run:
            self.wandb_run.finish()
        
        # Log final summary
        summary = self.get_experiment_summary()
        logger.info(f"Experiment completed: {summary}")
        
        # Save summary to file
        summary_path = Path(self.config.artifact_dir) / "experiment_summary.json"
        try:
            with open(summary_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save experiment summary: {e}")


class CheckpointManager:
    """Dedicated checkpoint management class."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        
    """__init__ function."""
self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: List[CheckpointInfo] = []
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self) -> Any:
        """Load existing checkpoints from directory."""
        for checkpoint_file in self.checkpoint_dir.glob("*.pth"):
            try:
                checkpoint_info = self._load_checkpoint_info(checkpoint_file)
                if checkpoint_info:
                    self.checkpoints.append(checkpoint_info)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
        
        # Sort by step
        self.checkpoints.sort(key=lambda x: x.step)
        logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints")
    
    def _load_checkpoint_info(self, checkpoint_path: Path) -> Optional[CheckpointInfo]:
        """Load checkpoint info from file."""
        try:
            if TORCH_AVAILABLE:
                data = torch.load(checkpoint_path)
            else:
                with open(checkpoint_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = pickle.load(f)
            
            checkpoint_info = CheckpointInfo.from_dict(data)
            checkpoint_info.checkpoint_path = str(checkpoint_path)
            return checkpoint_info
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint info from {checkpoint_path}: {e}")
            return None
    
    def save_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        metrics: Dict[str, float],
        epoch: int,
        step: int,
        is_best: bool = False
    ) -> Optional[str]:
        """Save checkpoint."""
        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_path="",
            epoch=epoch,
            step=step,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            model_state=model.state_dict() if TORCH_AVAILABLE else {},
            optimizer_state=optimizer.state_dict() if optimizer else None,
            scheduler_state=scheduler.state_dict() if scheduler else None
        )
        
        # Determine filename
        if is_best:
            filename = "best_model.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}_step_{step}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        try:
            if TORCH_AVAILABLE:
                torch.save(checkpoint_info.to_dict(), checkpoint_path)
            else:
                with open(checkpoint_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    pickle.dump(checkpoint_info.to_dict(), f)
            
            checkpoint_info.checkpoint_path = str(checkpoint_path)
            self.checkpoints.append(checkpoint_info)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[CheckpointInfo]:
        """Load checkpoint."""
        return self._load_checkpoint_info(Path(checkpoint_path))
    
    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get latest checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x.step)
    
    def get_best_checkpoint(self, metric_name: str = "val_loss") -> Optional[CheckpointInfo]:
        """Get best checkpoint based on metric."""
        if not self.checkpoints:
            return None
        return min(self.checkpoints, key=lambda x: x.metrics.get(metric_name, float('inf')))
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all checkpoints."""
        return sorted(self.checkpoints, key=lambda x: x.step)
    
    def cleanup_old_checkpoints(self, max_checkpoints: int = 5):
        """Remove old checkpoints."""
        if len(self.checkpoints) <= max_checkpoints:
            return
        
        # Sort by step and keep most recent
        self.checkpoints.sort(key=lambda x: x.step, reverse=True)
        
        checkpoints_to_remove = self.checkpoints[max_checkpoints:]
        for checkpoint in checkpoints_to_remove:
            try:
                Path(checkpoint.checkpoint_path).unlink()
                logger.info(f"Removed old checkpoint: {checkpoint.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint.checkpoint_path}: {e}")
        
        self.checkpoints = self.checkpoints[:max_checkpoints]


# Convenience functions
def create_experiment_tracker(
    experiment_name: str,
    project_name: str = "ai_video_generation",
    use_wandb: bool = False,
    use_tensorboard: bool = True,
    **kwargs
) -> ExperimentTracker:
    """Create experiment tracker with default settings."""
    config = ExperimentConfig(
        experiment_name=experiment_name,
        project_name=project_name,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        **kwargs
    )
    return ExperimentTracker(config)


def create_checkpoint_manager(checkpoint_dir: str = "checkpoints") -> CheckpointManager:
    """Create checkpoint manager."""
    return CheckpointManager(checkpoint_dir)


if __name__ == "__main__":
    # Example usage
    print("🔬 Experiment Tracking System")
    print("=" * 40)
    
    # Create experiment tracker
    tracker = create_experiment_tracker(
        "test_experiment",
        use_wandb=False,
        use_tensorboard=True
    )
    
    # Log configuration
    config = {
        "model_type": "diffusion",
        "learning_rate": 1e-4,
        "batch_size": 8
    }
    tracker.log_config(config)
    
    # Simulate training
    for step in range(100):
        tracker.update_step(step)
        
        # Log metrics
        metrics = {
            "loss": 1.0 / (step + 1),
            "val_loss": 1.2 / (step + 1),
            "learning_rate": 1e-4
        }
        tracker.log_metrics(metrics, step)
        
        # Save checkpoint every 20 steps
        if step % 20 == 0:
            # Mock model and optimizer
            class MockModel:
                def state_dict(self) -> Any:
                    return {"weights": [1, 2, 3]}
            
            class MockOptimizer:
                def state_dict(self) -> Any:
                    return {"lr": 1e-4}
            
            model = MockModel()
            optimizer = MockOptimizer()
            
            tracker.save_checkpoint(
                model=model,
                optimizer=optimizer,
                metrics=metrics,
                epoch=step // 20,
                step=step,
                is_best=(step == 80)
            )
    
    # Close tracker
    tracker.close()
    
    print("✅ Experiment tracking example completed!")
    print(f"📊 Summary: {tracker.get_experiment_summary()}") 