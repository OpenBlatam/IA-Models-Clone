"""
Weights & Biases Logger Module
===============================

W&B integration for experiment tracking.

Author: BUL System
Date: 2024
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install with: pip install wandb")


class WandBLogger:
    """
    Weights & Biases logger for experiment tracking.
    
    Provides integration with W&B for experiment management
    and visualization.
    
    Example:
        >>> logger = WandBLogger(project="my-project", name="exp1")
        >>> logger.log_metric("loss", 0.5, step=100)
        >>> logger.finish()
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            name: Experiment name
            config: Configuration dictionary
            enabled: Whether logging is enabled
        """
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases not available")
            enabled = False
        
        self.enabled = enabled
        self.project = project
        self.name = name
        
        if self.enabled:
            wandb.init(
                project=project,
                name=name,
                config=config or {}
            )
            logger.info(f"W&B logging enabled for project: {project}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric to W&B.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if self.enabled:
            if step is not None:
                wandb.log({name: value}, step=step)
            else:
                wandb.log({name: value})
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number
        """
        if self.enabled:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        if self.enabled:
            wandb.config.update(config)
    
    def finish(self) -> None:
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()
            logger.info("W&B run finished")

