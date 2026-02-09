"""
Experiment Tracking System - Sistema de seguimiento de experimentos
====================================================================
Integración con WandB y TensorBoard para tracking de experimentos
"""

import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Sistema de seguimiento de experimentos"""
    
    def __init__(
        self,
        project_name: str = "3d-prototype-ai",
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        log_dir: str = "./logs/experiments"
    ):
        self.project_name = project_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_run = None
        self.tensorboard_writer = None
        self.current_experiment_id = None
        
        if self.use_wandb:
            try:
                wandb.init(project=project_name, mode="online" if os.getenv("WANDB_API_KEY") else "disabled")
                self.wandb_run = wandb.run
                logger.info("WandB initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.use_wandb = False
        
        if self.use_tensorboard:
            try:
                self.tensorboard_writer = SummaryWriter(log_dir=str(self.log_dir))
                logger.info("TensorBoard initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False
    
    def start_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """Inicia un nuevo experimento"""
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_experiment_id = experiment_id
        
        if self.use_wandb and self.wandb_run:
            wandb.config.update(config)
            if tags:
                wandb.run.tags = tags
            wandb.run.name = experiment_name
        
        logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Registra métricas"""
        if self.use_wandb and self.wandb_run:
            wandb.log(metrics, step=step)
        
        if self.use_tensorboard and self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step or 0)
    
    def log_model(self, model_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Registra un modelo"""
        if self.use_wandb and self.wandb_run:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(model_path)
            if metadata:
                artifact.metadata = metadata
            wandb.log_artifact(artifact)
    
    def log_image(self, image_path: str, caption: Optional[str] = None, step: Optional[int] = None):
        """Registra una imagen"""
        if self.use_wandb and self.wandb_run:
            wandb.log({"images": [wandb.Image(image_path, caption=caption)]}, step=step)
        
        if self.use_tensorboard and self.tensorboard_writer:
            from PIL import Image
            import numpy as np
            img = Image.open(image_path)
            img_array = np.array(img)
            self.tensorboard_writer.add_image("images", img_array, step or 0, dataformats="HWC")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Registra hiperparámetros"""
        if self.use_wandb and self.wandb_run:
            wandb.config.update(hyperparams)
    
    def finish_experiment(self):
        """Finaliza el experimento"""
        if self.use_wandb and self.wandb_run:
            wandb.finish()
        
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        logger.info(f"Finished experiment: {self.current_experiment_id}")
        self.current_experiment_id = None




