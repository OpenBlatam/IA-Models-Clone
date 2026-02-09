"""
Experiment Tracking
Supports Weights & Biases and TensorBoard.
"""

import logging
import torch
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Experiment tracker supporting multiple backends.
    """
    
    def __init__(
        self,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        **kwargs
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.project_name = project_name
        self.run_name = run_name
        
        # Initialize TensorBoard
        self.tb_writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = log_dir or "./logs/tensorboard"
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard initialized: {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")
                self.use_tensorboard = False
        
        # Initialize Weights & Biases
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=project_name or "ml-experiments",
                    name=run_name,
                    **kwargs
                )
                self.wandb_run = wandb.run
                logger.info("Weights & Biases initialized")
            except ImportError:
                logger.warning("Weights & Biases not available")
                self.use_wandb = False
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                self.use_wandb = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step or 0)
        
        if self.use_wandb and self.wandb_run:
            if step is not None:
                metrics["step"] = step
            self.wandb_run.log(metrics)
        
        # Also log to standard logger
        logger.info(f"Metrics (step {step}): {metrics}")
    
    def log_model(self, model: Any, input_shape: tuple):
        """
        Log model architecture.
        
        Args:
            model: Model to log
            input_shape: Input shape for visualization
        """
        if self.use_tensorboard and self.tb_writer:
            try:
                # Try to log model graph
                dummy_input = torch.zeros(input_shape)
                self.tb_writer.add_graph(model, dummy_input)
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")
        
        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                # Log model summary
                wandb.watch(model)
            except Exception as e:
                logger.warning(f"Could not watch model: {e}")
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            config: Configuration dictionary
        """
        if self.use_tensorboard and self.tb_writer:
            # TensorBoard doesn't have a direct hyperparameter logging
            # Log as text or use hparams plugin
            pass
        
        if self.use_wandb and self.wandb_run:
            self.wandb_run.config.update(config)
    
    def finish(self):
        """Finish tracking and close writers."""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()
        
        if self.use_wandb and self.wandb_run:
            import wandb
            wandb.finish()
        
        logger.info("Experiment tracking finished")

