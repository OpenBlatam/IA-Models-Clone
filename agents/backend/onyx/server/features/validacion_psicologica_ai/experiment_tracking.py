"""
Sistema de Experiment Tracking
==============================
Tracking de experimentos con wandb y tensorboard
"""

from typing import Dict, Any, List, Optional
import structlog
import os

logger = structlog.get_logger()

# Intentar importar wandb y tensorboard
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("tensorboard not available")


class ExperimentTracker:
    """Tracker de experimentos"""
    
    def __init__(
        self,
        project_name: str = "psychological-validation",
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        log_dir: str = "./logs"
    ):
        """
        Inicializar tracker
        
        Args:
            project_name: Nombre del proyecto
            use_wandb: Usar wandb
            use_tensorboard: Usar tensorboard
            log_dir: Directorio de logs
        """
        self.project_name = project_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.log_dir = log_dir
        
        self.wandb_run = None
        self.tensorboard_writer = None
        
        if self.use_wandb:
            try:
                wandb.init(project=project_name, mode="online" if os.getenv("WANDB_API_KEY") else "disabled")
                self.wandb_run = wandb.run
                logger.info("Wandb initialized", project=project_name)
            except Exception as e:
                logger.warning("Could not initialize wandb", error=str(e))
                self.use_wandb = False
        
        if self.use_tensorboard:
            try:
                os.makedirs(log_dir, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
                logger.info("TensorBoard initialized", log_dir=log_dir)
            except Exception as e:
                logger.warning("Could not initialize tensorboard", error=str(e))
                self.use_tensorboard = False
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Registrar métricas
        
        Args:
            metrics: Diccionario de métricas
            step: Paso del entrenamiento (opcional)
        """
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                logger.warning("Error logging to wandb", error=str(e))
        
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    self.tensorboard_writer.add_scalar(key, value, step or 0)
            except Exception as e:
                logger.warning("Error logging to tensorboard", error=str(e))
    
    def log_model(
        self,
        model_name: str,
        model_config: Dict[str, Any]
    ) -> None:
        """
        Registrar modelo
        
        Args:
            model_name: Nombre del modelo
            model_config: Configuración del modelo
        """
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.config.update(model_config)
            except Exception as e:
                logger.warning("Error logging model to wandb", error=str(e))
    
    def log_hyperparameters(
        self,
        hyperparameters: Dict[str, Any]
    ) -> None:
        """
        Registrar hiperparámetros
        
        Args:
            hyperparameters: Diccionario de hiperparámetros
        """
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.config.update(hyperparameters)
            except Exception as e:
                logger.warning("Error logging hyperparameters to wandb", error=str(e))
    
    def log_artifacts(
        self,
        file_path: str,
        artifact_name: str
    ) -> None:
        """
        Registrar artefactos
        
        Args:
            file_path: Ruta del archivo
            artifact_name: Nombre del artefacto
        """
        if self.use_wandb and self.wandb_run:
            try:
                artifact = wandb.Artifact(artifact_name, type="model")
                artifact.add_file(file_path)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning("Error logging artifact to wandb", error=str(e))
    
    def finish(self) -> None:
        """Finalizar tracking"""
        if self.use_wandb and self.wandb_run:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning("Error finishing wandb", error=str(e))
        
        if self.use_tensorboard and self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
            except Exception as e:
                logger.warning("Error closing tensorboard", error=str(e))


# Instancia global del tracker
experiment_tracker = ExperimentTracker()




