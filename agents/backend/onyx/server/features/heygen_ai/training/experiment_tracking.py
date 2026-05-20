"""
Experiment Tracking Module for HeyGen AI
=========================================

Implements experiment tracking following best practices:
- Weights & Biases (wandb) integration
- TensorBoard integration
- Metric logging
- Model checkpointing
- Hyperparameter tracking
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard not available. Install with: pip install tensorboard")

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking.
    
    Attributes:
        project_name: Project name for tracking
        experiment_name: Name of this experiment
        use_wandb: Use Weights & Biases
        use_tensorboard: Use TensorBoard
        log_dir: Directory for TensorBoard logs
        save_checkpoints: Save model checkpoints
        checkpoint_dir: Directory for checkpoints
    """
    project_name: str = "heygen_ai"
    experiment_name: str = "experiment_1"
    use_wandb: bool = True
    use_tensorboard: bool = True
    log_dir: str = "./logs/tensorboard"
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"


class ExperimentTracker:
    """Tracks experiments with wandb and/or TensorBoard.
    
    Features:
    - Metric logging
    - Hyperparameter tracking
    - Model checkpointing
    - Artifact management
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment tracker.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ExperimentTracker")
        
        # Initialize wandb
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=config.project_name,
                    name=config.experiment_name,
                    config=asdict(config),
                )
                self.logger.info("Weights & Biases initialized")
            except Exception as e:
                self.logger.warning(f"wandb initialization failed: {e}")
        
        # Initialize TensorBoard
        self.tensorboard_writer = None
        if config.use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                log_path = Path(config.log_dir) / config.experiment_name
                log_path.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(str(log_path))
                self.logger.info(f"TensorBoard initialized: {log_path}")
            except Exception as e:
                self.logger.warning(f"TensorBoard initialization failed: {e}")
        
        # Create checkpoint directory
        if config.save_checkpoints:
            checkpoint_path = Path(config.checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics to tracking systems.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch
            prefix: Prefix for metric names
        """
        # Add prefix to metric names
        prefixed_metrics = {
            f"{prefix}/{k}" if prefix else k: v
            for k, v in metrics.items()
        }
        
        # Log to wandb
        if self.wandb_run:
            try:
                if step is not None:
                    self.wandb_run.log(prefixed_metrics, step=step)
                else:
                    self.wandb_run.log(prefixed_metrics)
            except Exception as e:
                self.logger.warning(f"wandb logging failed: {e}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            try:
                for key, value in prefixed_metrics.items():
                    self.tensorboard_writer.add_scalar(key, value, step or 0)
            except Exception as e:
                self.logger.warning(f"TensorBoard logging failed: {e}")
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        if self.wandb_run:
            try:
                self.wandb_run.config.update(hyperparameters)
            except Exception as e:
                self.logger.warning(f"wandb hyperparameter logging failed: {e}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Optional metrics to save
            filename: Optional custom filename
        
        Returns:
            Path to saved checkpoint
        """
        if not self.config.save_checkpoints:
            return ""
        
        checkpoint_path = Path(self.config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_file = checkpoint_path / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_file)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        # Log checkpoint as artifact to wandb
        if self.wandb_run:
            try:
                artifact = wandb.Artifact(
                    f"checkpoint-{epoch}",
                    type="model"
                )
                artifact.add_file(str(checkpoint_file))
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                self.logger.warning(f"wandb artifact logging failed: {e}")
        
        return str(checkpoint_file)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = Path(self.config.checkpoint_dir)
            checkpoints = list(checkpoint_path.glob("checkpoint_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint
    
    def log_model_summary(self, model: torch.nn.Module) -> None:
        """Log model architecture summary.
        
        Args:
            model: Model to summarize
        """
        if self.wandb_run:
            try:
                # Log model parameters count
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                
                self.wandb_run.config.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                })
            except Exception as e:
                self.logger.warning(f"wandb model summary failed: {e}")
    
    def finish(self) -> None:
        """Finish experiment tracking."""
        if self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception as e:
                self.logger.warning(f"wandb finish failed: {e}")
        
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
            except Exception as e:
                self.logger.warning(f"TensorBoard close failed: {e}")
        
        self.logger.info("Experiment tracking finished")



