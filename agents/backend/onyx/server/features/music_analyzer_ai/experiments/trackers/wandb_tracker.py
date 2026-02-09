"""
Weights & Biases Tracker
Integration with WandB for experiment tracking
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available")

from .base_tracker import BaseExperimentTracker


class WandBTracker(BaseExperimentTracker):
    """WandB experiment tracker"""
    
    def __init__(
        self,
        experiment_name: str,
        project_name: Optional[str] = None,
        entity: Optional[str] = None
    ):
        super().__init__(experiment_name, project_name)
        self.entity = entity
        self.run = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize WandB run"""
        if not WANDB_AVAILABLE:
            logger.warning("WandB not available, tracker disabled")
            return False
        
        try:
            self.run = wandb.init(
                project=self.project_name or "music_analyzer_ai",
                name=self.experiment_name,
                entity=self.entity,
                config=config or {}
            )
            self.initialized = True
            logger.info(f"WandB tracker initialized: {self.experiment_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing WandB: {str(e)}")
            return False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to WandB"""
        if not self.initialized or self.run is None:
            return
        
        try:
            if step is not None:
                self.run.log(metrics, step=step)
            else:
                self.run.log(metrics)
        except Exception as e:
            logger.error(f"Error logging metrics to WandB: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters to WandB"""
        if not self.initialized or self.run is None:
            return
        
        try:
            self.run.config.update(params)
        except Exception as e:
            logger.error(f"Error logging params to WandB: {str(e)}")
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model artifact to WandB"""
        if not self.initialized or self.run is None:
            return
        
        try:
            artifact = wandb.Artifact(model_name, type="model")
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
        except Exception as e:
            logger.error(f"Error logging model to WandB: {str(e)}")
    
    def finish(self):
        """Finish WandB run"""
        if self.run is not None:
            self.run.finish()



