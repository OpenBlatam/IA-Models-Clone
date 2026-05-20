"""
Experiment Tracking for Manufacturing
=====================================

Tracking de experimentos para modelos de manufactura.
"""

import logging
from typing import Dict, Any, Optional

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

logger = logging.getLogger(__name__)


class ManufacturingExperimentTracker:
    """Tracker de experimentos para manufactura."""
    
    def __init__(self, experiment_name: str, use_wandb: bool = False):
        """
        Inicializar tracker.
        
        Args:
            experiment_name: Nombre del experimento
            use_wandb: Usar Weights & Biases (si no, TensorBoard)
        """
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        if self.use_wandb:
            wandb.init(project="manufacturing-ai", name=experiment_name)
            self.writer = None
        elif TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
        else:
            self.writer = None
            logger.warning("No experiment tracking available")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Loggear métricas.
        
        Args:
            metrics: Diccionario de métricas
            step: Paso (opcional)
        """
        if self.use_wandb:
            wandb.log(metrics, step=step)
        elif self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, global_step=step)
    
    def log_quality_metrics(
        self,
        pass_rate: float,
        avg_score: float,
        defects_count: int,
        step: Optional[int] = None
    ):
        """Loggear métricas de calidad."""
        self.log_metrics({
            "quality/pass_rate": pass_rate,
            "quality/avg_score": avg_score,
            "quality/defects_count": defects_count
        }, step=step)
    
    def log_production_metrics(
        self,
        production_rate: float,
        efficiency: float,
        downtime: float,
        step: Optional[int] = None
    ):
        """Loggear métricas de producción."""
        self.log_metrics({
            "production/rate": production_rate,
            "production/efficiency": efficiency,
            "production/downtime": downtime
        }, step=step)
    
    def log_optimization_result(
        self,
        improvement: float,
        process_id: str,
        step: Optional[int] = None
    ):
        """Loggear resultado de optimización."""
        self.log_metrics({
            f"optimization/{process_id}/improvement": improvement
        }, step=step)
    
    def finish(self):
        """Finalizar tracking."""
        if self.use_wandb:
            wandb.finish()
        elif self.writer:
            self.writer.close()

