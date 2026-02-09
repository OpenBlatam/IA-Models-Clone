"""
TensorBoard Tracker
===================

Tracker para TensorBoard.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from .tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class TensorBoardTracker(ExperimentTracker):
    """
    Tracker para TensorBoard.
    
    Usa SummaryWriter de PyTorch para logging.
    """
    
    def __init__(self, experiment_name: str, project_name: Optional[str] = None):
        """
        Inicializar tracker.
        
        Args:
            experiment_name: Nombre del experimento
            project_name: Nombre del proyecto (opcional)
        """
        super().__init__(experiment_name, project_name)
        
        if not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        
        self.writer = None
        self.log_dir = None
    
    def init(self, config: Dict[str, Any]):
        """Inicializar TensorBoard."""
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is required. Install with: pip install tensorboard")
        
        # Crear directorio de logs
        if self.project_name:
            self.log_dir = f"runs/{self.project_name}/{self.experiment_name}"
        else:
            self.log_dir = f"runs/{self.experiment_name}"
        
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Loggear configuración
        self.log_hyperparameters(config)
        
        self.initialized = True
        logger.info(f"TensorBoard tracker initialized: {self.log_dir}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Loggear métrica."""
        if not self.initialized or self.writer is None:
            logger.warning("Tracker not initialized")
            return
        
        self.writer.add_scalar(name, value, global_step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Loggear múltiples métricas."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Loggear hiperparámetros."""
        if not self.initialized or self.writer is None:
            logger.warning("Tracker not initialized")
            return
        
        # Convertir a formato compatible
        hparams = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                hparams[key] = value
            else:
                hparams[key] = str(value)
        
        self.writer.add_hparams(hparams, {})
    
    def log_model(self, model, artifact_name: Optional[str] = None):
        """Loggear modelo (grafo)."""
        if not self.initialized or self.writer is None:
            logger.warning("Tracker not initialized")
            return
        
        try:
            # Intentar loggear el grafo del modelo
            # Nota: Esto requiere un ejemplo de input
            logger.info("Model graph logging requires example input")
        except Exception as e:
            logger.warning(f"Could not log model graph: {e}")
    
    def log_image(self, tag: str, image, step: Optional[int] = None):
        """
        Loggear imagen.
        
        Args:
            tag: Tag de la imagen
            image: Imagen (numpy array o tensor)
            step: Paso (opcional)
        """
        if not self.initialized or self.writer is None:
            logger.warning("Tracker not initialized")
            return
        
        self.writer.add_image(tag, image, global_step=step)
    
    def finish(self):
        """Finalizar tracking."""
        if self.writer:
            self.writer.close()
            logger.info("TensorBoard tracker finished")


