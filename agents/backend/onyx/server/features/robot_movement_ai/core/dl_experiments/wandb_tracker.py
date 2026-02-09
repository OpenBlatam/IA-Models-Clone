"""
Weights & Biases Tracker
========================

Tracker para Weights & Biases.
"""

import logging
from typing import Dict, Any, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class WandBTracker(ExperimentTracker):
    """
    Tracker para Weights & Biases.
    
    Usa wandb para tracking de experimentos.
    """
    
    def __init__(self, experiment_name: str, project_name: Optional[str] = None):
        """
        Inicializar tracker.
        
        Args:
            experiment_name: Nombre del experimento
            project_name: Nombre del proyecto (opcional)
        """
        super().__init__(experiment_name, project_name)
        
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases not available. Install with: pip install wandb")
        
        self.run = None
    
    def init(self, config: Dict[str, Any]):
        """Inicializar Weights & Biases."""
        if not WANDB_AVAILABLE:
            raise ImportError("Weights & Biases is required. Install with: pip install wandb")
        
        # Inicializar run
        wandb.init(
            project=self.project_name or "robot-movement-ai",
            name=self.experiment_name,
            config=config
        )
        
        self.run = wandb.run
        self.initialized = True
        logger.info(f"W&B tracker initialized: {self.experiment_name}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Loggear métrica."""
        if not self.initialized or self.run is None:
            logger.warning("Tracker not initialized")
            return
        
        wandb.log({name: value}, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Loggear múltiples métricas."""
        if not self.initialized or self.run is None:
            logger.warning("Tracker not initialized")
            return
        
        wandb.log(metrics, step=step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Loggear hiperparámetros."""
        if not self.initialized or self.run is None:
            logger.warning("Tracker not initialized")
            return
        
        wandb.config.update(params)
    
    def log_model(self, model, artifact_name: Optional[str] = None):
        """Loggear modelo como artifact."""
        if not self.initialized or self.run is None:
            logger.warning("Tracker not initialized")
            return
        
        try:
            import torch
            artifact_name = artifact_name or f"{self.experiment_name}_model"
            
            # Guardar modelo temporalmente
            model_path = f"/tmp/{artifact_name}.pt"
            torch.save(model.state_dict(), model_path)
            
            # Crear artifact
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
            logger.info(f"Logged model as artifact: {artifact_name}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def log_image(self, name: str, image, step: Optional[int] = None):
        """
        Loggear imagen.
        
        Args:
            name: Nombre de la imagen
            image: Imagen (numpy array o PIL Image)
            step: Paso (opcional)
        """
        if not self.initialized or self.run is None:
            logger.warning("Tracker not initialized")
            return
        
        wandb.log({name: wandb.Image(image)}, step=step)
    
    def finish(self):
        """Finalizar tracking."""
        if self.run:
            wandb.finish()
            logger.info("W&B tracker finished")


