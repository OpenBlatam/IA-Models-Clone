"""
Experiment Tracking Service - Tracking de experimentos
======================================================

Sistema profesional para tracking de experimentos usando WandB y TensorBoard.
Sigue mejores prácticas de experiment tracking.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import tracking libraries
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available")


class TrackingBackend(str, Enum):
    """Backends de tracking"""
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    BOTH = "both"
    NONE = "none"


@dataclass
class ExperimentConfig:
    """Configuración de experimento"""
    name: str
    project: str = "default-project"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    backend: TrackingBackend = TrackingBackend.WANDB
    resume: Optional[str] = None  # For WandB resume
    log_dir: str = "./logs/tensorboard"  # For TensorBoard


@dataclass
class MetricLog:
    """Log de métrica"""
    name: str
    value: float
    step: int
    timestamp: datetime = field(default_factory=datetime.now)


class ExperimentTrackingService:
    """Servicio de tracking de experimentos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.experiments: Dict[str, Any] = {}
        self.wandb_runs: Dict[str, Any] = {}
        self.tensorboard_writers: Dict[str, Any] = {}
        logger.info(
            f"ExperimentTrackingService initialized "
            f"(WandB: {WANDB_AVAILABLE}, TensorBoard: {TENSORBOARD_AVAILABLE})"
        )
    
    def start_experiment(
        self,
        experiment_id: str,
        config: ExperimentConfig
    ) -> Dict[str, Any]:
        """
        Iniciar nuevo experimento.
        
        Args:
            experiment_id: ID único del experimento
            config: Configuración del experimento
        
        Returns:
            Información del experimento iniciado
        """
        experiment_data = {
            "id": experiment_id,
            "config": config,
            "started_at": datetime.now(),
            "metrics": [],
        }
        
        # Initialize WandB
        if config.backend in [TrackingBackend.WANDB, TrackingBackend.BOTH]:
            if not WANDB_AVAILABLE:
                logger.warning("WandB requested but not available")
            else:
                try:
                    wandb.init(
                        project=config.project,
                        name=config.name,
                        tags=config.tags,
                        notes=config.notes,
                        config=config.config,
                        resume=config.resume or "allow",
                        id=experiment_id,
                    )
                    self.wandb_runs[experiment_id] = wandb.run
                    logger.info(f"WandB run started: {wandb.run.id}")
                except Exception as e:
                    logger.error(f"Error starting WandB run: {e}", exc_info=True)
        
        # Initialize TensorBoard
        if config.backend in [TrackingBackend.TENSORBOARD, TrackingBackend.BOTH]:
            if not TENSORBOARD_AVAILABLE:
                logger.warning("TensorBoard requested but not available")
            else:
                try:
                    log_dir = Path(config.log_dir) / experiment_id
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    writer = SummaryWriter(log_dir=str(log_dir))
                    self.tensorboard_writers[experiment_id] = writer
                    logger.info(f"TensorBoard writer created: {log_dir}")
                except Exception as e:
                    logger.error(f"Error creating TensorBoard writer: {e}", exc_info=True)
        
        self.experiments[experiment_id] = experiment_data
        
        logger.info(f"Experiment {experiment_id} started")
        return experiment_data
    
    def log_metric(
        self,
        experiment_id: str,
        name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Registrar métrica.
        
        Args:
            experiment_id: ID del experimento
            name: Nombre de la métrica
            value: Valor de la métrica
            step: Paso (opcional, auto-incrementa si None)
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        
        # Auto-increment step if not provided
        if step is None:
            step = len(experiment["metrics"])
        
        metric_log = MetricLog(name=name, value=value, step=step)
        experiment["metrics"].append(metric_log)
        
        # Log to WandB
        if experiment_id in self.wandb_runs:
            try:
                wandb.log({name: value}, step=step)
            except Exception as e:
                logger.error(f"Error logging to WandB: {e}")
        
        # Log to TensorBoard
        if experiment_id in self.tensorboard_writers:
            try:
                writer = self.tensorboard_writers[experiment_id]
                writer.add_scalar(name, value, step)
            except Exception as e:
                logger.error(f"Error logging to TensorBoard: {e}")
    
    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Registrar múltiples métricas.
        
        Args:
            experiment_id: ID del experimento
            metrics: Diccionario de métricas
            step: Paso (opcional)
        """
        for name, value in metrics.items():
            self.log_metric(experiment_id, name, value, step)
    
    def log_image(
        self,
        experiment_id: str,
        name: str,
        image: Any,  # PIL Image or numpy array
        step: Optional[int] = None
    ) -> None:
        """
        Registrar imagen.
        
        Args:
            experiment_id: ID del experimento
            name: Nombre de la imagen
            image: Imagen (PIL Image o numpy array)
            step: Paso (opcional)
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return
        
        if step is None:
            step = len(experiment["metrics"])
        
        # Log to WandB
        if experiment_id in self.wandb_runs:
            try:
                wandb.log({name: wandb.Image(image)}, step=step)
            except Exception as e:
                logger.error(f"Error logging image to WandB: {e}")
        
        # Log to TensorBoard
        if experiment_id in self.tensorboard_writers:
            try:
                writer = self.tensorboard_writers[experiment_id]
                writer.add_image(name, image, step)
            except Exception as e:
                logger.error(f"Error logging image to TensorBoard: {e}")
    
    def log_model(
        self,
        experiment_id: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar modelo.
        
        Args:
            experiment_id: ID del experimento
            model_path: Ruta al modelo
            metadata: Metadatos adicionales
        """
        # Log to WandB
        if experiment_id in self.wandb_runs:
            try:
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(model_path)
                if metadata:
                    artifact.metadata = metadata
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.error(f"Error logging model to WandB: {e}")
    
    def finish_experiment(self, experiment_id: str) -> None:
        """Finalizar experimento"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return
        
        experiment["completed_at"] = datetime.now()
        
        # Finish WandB run
        if experiment_id in self.wandb_runs:
            try:
                wandb.finish()
                del self.wandb_runs[experiment_id]
            except Exception as e:
                logger.error(f"Error finishing WandB run: {e}")
        
        # Close TensorBoard writer
        if experiment_id in self.tensorboard_writers:
            try:
                writer = self.tensorboard_writers[experiment_id]
                writer.close()
                del self.tensorboard_writers[experiment_id]
            except Exception as e:
                logger.error(f"Error closing TensorBoard writer: {e}")
        
        logger.info(f"Experiment {experiment_id} finished")
