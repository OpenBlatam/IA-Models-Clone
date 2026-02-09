"""
Experiment Tracking
Integration with TensorBoard and WandB
"""

import torch
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ExperimentTracker:
    """
    Unified experiment tracking interface
    Supports TensorBoard and WandB
    """
    
    def __init__(
        self,
        experiment_name: str,
        project_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of experiment
            project_name: Project name (for WandB)
            use_tensorboard: Enable TensorBoard
            use_wandb: Enable WandB
            wandb_config: WandB configuration
            log_dir: Log directory for TensorBoard
        """
        self.experiment_name = experiment_name
        self.project_name = project_name or "addiction-recovery-ai"
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # TensorBoard
        self.tb_writer = None
        if self.use_tensorboard:
            log_dir = log_dir or f"runs/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        
        # WandB
        self.wandb_run = None
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("WandB not available, install with: pip install wandb")
                self.use_wandb = False
            else:
                wandb_config = wandb_config or {}
                self.wandb_run = wandb.init(
                    project=self.project_name,
                    name=experiment_name,
                    config=wandb_config
                )
                logger.info(f"WandB run initialized: {self.wandb_run.name}")
        
        # Local metrics storage
        self.metrics_history: Dict[str, List[float]] = {}
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log scalar value
        
        Args:
            tag: Metric name
            value: Metric value
            step: Step number
        """
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        
        # WandB
        if self.wandb_run:
            wandb.log({tag: value}, step=step)
        
        # Local storage
        if tag not in self.metrics_history:
            self.metrics_history[tag] = []
        self.metrics_history[tag].append(value)
    
    def log_scalars(self, tag: str, scalars: Dict[str, float], step: int):
        """
        Log multiple scalars
        
        Args:
            tag: Base tag name
            scalars: Dictionary of scalar values
            step: Step number
        """
        # TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalars(tag, scalars, step)
        
        # WandB
        if self.wandb_run:
            wandb.log({f"{tag}/{k}": v for k, v in scalars.items()}, step=step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int, bins: int = 100):
        """
        Log histogram
        
        Args:
            tag: Tag name
            values: Tensor values
            step: Step number
            bins: Number of bins
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step, bins=bins)
        
        if self.wandb_run:
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """
        Log model graph
        
        Args:
            model: Model to log
            input_sample: Sample input
        """
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, input_sample)
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """
        Log hyperparameters
        
        Args:
            hparams: Hyperparameters dictionary
            metrics: Optional metrics to associate
        """
        if self.tb_writer:
            self.tb_writer.add_hparams(hparams, metrics or {})
        
        if self.wandb_run:
            wandb.config.update(hparams)
            if metrics:
                wandb.log(metrics)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int, dataformats: str = "CHW"):
        """
        Log image
        
        Args:
            tag: Tag name
            image: Image tensor
            step: Step number
            dataformats: Image format (CHW, HWC, etc.)
        """
        if self.tb_writer:
            self.tb_writer.add_image(tag, image, step, dataformats=dataformats)
        
        if self.wandb_run:
            wandb.log({tag: wandb.Image(image.cpu().numpy())}, step=step)
    
    def log_text(self, tag: str, text: str, step: int):
        """
        Log text
        
        Args:
            tag: Tag name
            text: Text to log
            step: Step number
        """
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
        
        if self.wandb_run:
            wandb.log({tag: text}, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log multiple metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Step number
            prefix: Prefix for metric names
        """
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)
    
    def get_metrics_history(self, tag: str) -> List[float]:
        """
        Get metrics history
        
        Args:
            tag: Metric tag
            
        Returns:
            List of metric values
        """
        return self.metrics_history.get(tag, [])
    
    def close(self):
        """Close trackers"""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_run:
            wandb.finish()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def create_tracker(
    experiment_name: str,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    **kwargs
) -> ExperimentTracker:
    """Factory for experiment tracker"""
    return ExperimentTracker(
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        **kwargs
    )













