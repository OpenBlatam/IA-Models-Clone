"""
Experiment Tracking

Provides utilities for tracking experiments with wandb, tensorboard, etc.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import experiment tracking libraries
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ExperimentTracker:
    """
    Unified experiment tracker supporting multiple backends.
    """
    
    def __init__(
        self,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        tensorboard_log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            use_wandb: Enable Weights & Biases
            use_tensorboard: Enable TensorBoard
            wandb_project: W&B project name
            tensorboard_log_dir: TensorBoard log directory
            experiment_name: Experiment name
        """
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize W&B
        self.wandb_run = None
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "music-generation",
                name=self.experiment_name
            )
            self.wandb_run = wandb.run
            logger.info("Weights & Biases tracking enabled")
        
        # Initialize TensorBoard
        self.tensorboard_writer = None
        if self.use_tensorboard:
            log_dir = tensorboard_log_dir or f"./logs/{self.experiment_name}"
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging enabled: {log_dir}")
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        
        if self.use_tensorboard and self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step or 0)
    
    def log_model(self, model: Any, input_sample: Any) -> None:
        """
        Log model architecture.
        
        Args:
            model: Model to log
            input_sample: Sample input
        """
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.watch(model)
            except Exception as e:
                logger.warning(f"Could not watch model: {e}")
        
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_graph(model, input_sample)
            except Exception as e:
                logger.warning(f"Could not add graph: {e}")
    
    def finish(self) -> None:
        """Finish experiment tracking."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()


def create_tracker(
    use_wandb: bool = False,
    use_tensorboard: bool = True,
    **kwargs
) -> ExperimentTracker:
    """
    Create experiment tracker.
    
    Args:
        use_wandb: Enable W&B
        use_tensorboard: Enable TensorBoard
        **kwargs: Additional tracker arguments
        
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        **kwargs
    )



