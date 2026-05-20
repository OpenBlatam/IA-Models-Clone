"""
Experiment Tracking with TensorBoard and Weights & Biases
"""

import torch
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install: pip install tensorboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install: pip install wandb")


class ExperimentTracker:
    """Experiment tracking with multiple backends"""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "runs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None
    ):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Experiment name
            log_dir: Log directory
            use_tensorboard: Use TensorBoard
            use_wandb: Use Weights & Biases
            wandb_project: W&B project name
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.tensorboard_writer = None
        self.wandb_run = None
        
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            log_path = self.log_dir / experiment_name
            self.tensorboard_writer = SummaryWriter(str(log_path))
            logger.info(f"TensorBoard initialized: {log_path}")
        
        if use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=wandb_project or "addiction-recovery-ai",
                name=experiment_name,
                reinit=True
            )
            logger.info("Weights & Biases initialized")
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """
        Log metric
        
        Args:
            metric_name: Metric name
            value: Metric value
            step: Optional step number
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(metric_name, value, step or 0)
        
        if self.wandb_run:
            wandb.log({metric_name: value}, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """
        Log histogram
        
        Args:
            tag: Tag name
            values: Values tensor
            step: Optional step number
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(tag, values, step or 0)
        
        if self.wandb_run:
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """
        Log model graph
        
        Args:
            model: Model to log
            input_sample: Sample input
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.add_graph(model, input_sample)
    
    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters
        
        Args:
            hparams: Hyperparameters dictionary
            metrics: Metrics dictionary
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparams, metrics)
        
        if self.wandb_run:
            wandb.config.update(hparams)
            wandb.log(metrics)
    
    def close(self):
        """Close trackers"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        logger.info("Experiment tracking closed")


class TrainingLogger:
    """Enhanced training logger with experiment tracking"""
    
    def __init__(
        self,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False
    ):
        """
        Initialize training logger
        
        Args:
            experiment_name: Experiment name
            use_tensorboard: Use TensorBoard
            use_wandb: Use Weights & Biases
        """
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb
        )
        self.step = 0
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        **kwargs
    ):
        """
        Log epoch metrics
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Optional validation loss
            **kwargs: Additional metrics
        """
        metrics = {"train_loss": train_loss}
        
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        
        metrics.update(kwargs)
        
        self.tracker.log_metrics(metrics, step=epoch)
        self.step = epoch
    
    def log_batch(
        self,
        batch_loss: float,
        learning_rate: Optional[float] = None,
        **kwargs
    ):
        """
        Log batch metrics
        
        Args:
            batch_loss: Batch loss
            learning_rate: Optional learning rate
            **kwargs: Additional metrics
        """
        metrics = {"batch_loss": batch_loss}
        
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate
        
        metrics.update(kwargs)
        
        self.tracker.log_metrics(metrics, step=self.step)
        self.step += 1
    
    def close(self):
        """Close logger"""
        self.tracker.close()

