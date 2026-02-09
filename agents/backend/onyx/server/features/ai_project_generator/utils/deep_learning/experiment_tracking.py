"""Experiment Tracking"""

def generate_experiment_tracking_code() -> str:
    return '''"""
Experiment Tracking
===================

Tracking de experimentos con WandB y TensorBoard.
"""

import wandb
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracker de experimentos."""
    
    def __init__(self, use_wandb: bool = True, use_tensorboard: bool = True, project_name: str = "deep-learning-project"):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.project_name = project_name
        self.writer = None
        
        if use_wandb:
            wandb.init(project=project_name)
        
        if use_tensorboard:
            self.writer = SummaryWriter(f"logs/{project_name}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log métricas."""
        if self.use_wandb:
            wandb.log(metrics, step=step)
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def log_model(self, model, input_shape):
        """Log arquitectura del modelo."""
        if self.use_wandb:
            wandb.watch(model)
        if self.writer:
            # Log model graph
            dummy_input = torch.randn(1, *input_shape)
            self.writer.add_graph(model, dummy_input)
    
    def finish(self):
        """Finaliza tracking."""
        if self.use_wandb:
            wandb.finish()
        if self.writer:
            self.writer.close()
'''

