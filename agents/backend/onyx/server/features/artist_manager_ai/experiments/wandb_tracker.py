"""
Weights & Biases Tracker
========================

Integration with Weights & Biases for experiment tracking.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


class WandBTracker:
    """
    Weights & Biases experiment tracker.
    
    Provides integration with W&B for experiment tracking,
    visualization, and collaboration.
    """
    
    def __init__(self, project_name: str = "artist-manager-ai"):
        """
        Initialize W&B tracker.
        
        Args:
            project_name: W&B project name
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        
        self.project_name = project_name
        self.run = None
        self._logger = logger
    
    def init(
        self,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize W&B run.
        
        Args:
            experiment_name: Experiment name
            config: Configuration dictionary
        """
        self.run = wandb.init(
            project=self.project_name,
            name=experiment_name,
            config=config or {}
        )
        self._logger.info(f"Initialized W&B run: {experiment_name}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log metric to W&B.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number
        """
        if self.run:
            self.run.log({name: value}, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        if self.run:
            self.run.log(metrics, step=step)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hyperparameters: Hyperparameter dictionary
        """
        if self.run:
            self.run.config.update(hyperparameters)
    
    def finish(self):
        """Finish W&B run."""
        if self.run:
            self.run.finish()
            self._logger.info("Finished W&B run")




