"""
Experiment Tracking - TensorBoard and W&B Integration
=======================================================

Utilities for experiment tracking and logging.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger(__name__)

# Try to import tracking libraries
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class ExperimentTracker:
    """
    Unified interface for experiment tracking (TensorBoard and W&B).
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[Path] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for TensorBoard logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_config: W&B configuration
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) if log_dir else Path("logs") / experiment_name
        
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(self.log_dir))
            logger.info(f"TensorBoard logging to {self.log_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available, install with: pip install tensorboard")
        
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=wandb_project or "deep-learning",
                name=experiment_name,
                config=wandb_config or {}
            )
            logger.info(f"W&B tracking initialized for {experiment_name}")
        elif use_wandb and not WANDB_AVAILABLE:
            logger.warning("W&B not available, install with: pip install wandb")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value.
        
        Args:
            tag: Tag name
            value: Scalar value
            step: Step number
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(tag, value, step)
        
        if self.wandb_run:
            wandb.log({tag: value}, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        """
        Log model graph.
        
        Args:
            model: PyTorch model
            input_sample: Sample input tensor
        """
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, input_sample)
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """
        Log histogram of values.
        
        Args:
            tag: Tag name
            values: Tensor of values
            step: Step number
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(tag, values, step)
        
        if self.wandb_run:
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def close(self) -> None:
        """Close all tracking writers."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        logger.info("Experiment tracking closed")



