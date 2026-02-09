"""
Routing Experiment Tracker
==========================

Sistema de tracking de experimentos usando wandb y tensorboard.
Sigue mejores prácticas de experiment tracking en ML.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. WandB tracking will be disabled.")

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("tensorboard not available. TensorBoard tracking will be disabled.")


class ExperimentTracker:
    """Tracker de experimentos profesional."""
    
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        tensorboard_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Inicializar tracker de experimentos.
        
        Args:
            project_name: Nombre del proyecto
            experiment_name: Nombre del experimento (auto-generado si None)
            use_wandb: Usar WandB
            use_tensorboard: Usar TensorBoard
            tensorboard_dir: Directorio para TensorBoard
            config: Configuración del experimento
            tags: Tags para el experimento
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.tags = tags or []
        
        # WandB
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=self.experiment_name,
                    config=self.config,
                    tags=self.tags
                )
                logger.info(f"WandB initialized: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.wandb_run = None
        
        # TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                if tensorboard_dir:
                    log_dir = Path(tensorboard_dir) / self.experiment_name
                else:
                    log_dir = Path("runs") / self.experiment_name
                
                log_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
                logger.info(f"TensorBoard initialized: {log_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.tensorboard_writer = None
        
        # Historial local
        self.history: List[Dict[str, Any]] = []
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Registrar métricas.
        
        Args:
            metrics: Diccionario de métricas
            step: Paso/epoch (auto-incrementa si None)
        """
        if step is None:
            step = len(self.history)
        
        # Agregar timestamp
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.history.append(log_entry)
        
        # WandB
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")
        
        # TensorBoard
        if self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, step)
            except Exception as e:
                logger.warning(f"Failed to log to TensorBoard: {e}")
    
    def log_model(self, model_path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Registrar modelo.
        
        Args:
            model_path: Ruta al modelo
            metadata: Metadatos del modelo
        """
        if self.wandb_run:
            try:
                artifact = wandb.Artifact('model', type='model')
                artifact.add_file(model_path)
                if metadata:
                    artifact.metadata = metadata
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to log model to WandB: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """Actualizar configuración."""
        self.config.update(config)
        if self.wandb_run:
            try:
                wandb.config.update(config)
            except Exception as e:
                logger.warning(f"Failed to update WandB config: {e}")
    
    def finish(self):
        """Finalizar tracking."""
        if self.wandb_run:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish WandB: {e}")
        
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
            except Exception as e:
                logger.warning(f"Failed to close TensorBoard: {e}")
    
    def save_history(self, filepath: str):
        """Guardar historial local."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"History saved to {filepath}")


