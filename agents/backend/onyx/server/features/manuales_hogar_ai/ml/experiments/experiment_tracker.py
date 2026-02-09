"""
Experiment Tracker
=================

Tracking avanzado de experimentos.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

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

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracker avanzado de experimentos."""
    
    def __init__(
        self,
        experiment_name: str,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        output_dir: str = "./experiments"
    ):
        """
        Inicializar tracker.
        
        Args:
            experiment_name: Nombre del experimento
            use_wandb: Usar Weights & Biases
            use_tensorboard: Usar TensorBoard
            output_dir: Directorio de salida
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        
        self.wandb_run = None
        self.tensorboard_writer = None
        
        if self.use_wandb:
            self.wandb_run = wandb.init(project="manuales-hogar-ai", name=experiment_name)
        
        if self.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(str(self.output_dir / "tensorboard"))
        
        self.metrics_history: List[Dict[str, Any]] = []
        self._logger = logger
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Registrar hiperparámetros."""
        # Guardar en archivo
        with open(self.output_dir / "hyperparameters.json", "w") as f:
            json.dump(hyperparams, f, indent=2)
        
        # WandB
        if self.use_wandb:
            wandb.config.update(hyperparams)
        
        # TensorBoard
        if self.use_tensorboard:
            for key, value in hyperparams.items():
                self.tensorboard_writer.add_text(f"hyperparams/{key}", str(value), 0)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        """Registrar métricas."""
        metrics["step"] = step
        metrics["timestamp"] = datetime.now().isoformat()
        self.metrics_history.append(metrics.copy())
        
        # Guardar en archivo
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # WandB
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, step)
    
    def log_model(self, model_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Registrar modelo."""
        model_info = {
            "path": model_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        with open(self.output_dir / "models.json", "a") as f:
            f.write(json.dumps(model_info) + "\n")
        
        if self.use_wandb:
            wandb.log_artifact(model_path, name="model")
    
    def finish(self):
        """Finalizar experimento."""
        if self.use_wandb:
            wandb.finish()
        
        if self.use_tensorboard:
            self.tensorboard_writer.close()
        
        self._logger.info(f"Experimento {self.experiment_name} finalizado")




