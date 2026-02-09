"""
Experiment Tracker - Sistema de tracking de experimentos (W&B/TensorBoard)
============================================================================
"""

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuración de experimento"""
    name: str
    project: str = "research-paper-code-improver"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_api_key: Optional[str] = None
    tensorboard_log_dir: str = "./logs/tensorboard"


class ExperimentTracker:
    """Tracker de experimentos con soporte para W&B y TensorBoard"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.wandb_run = None
        self.tensorboard_writer = None
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Inicializar W&B
        if config.use_wandb:
            self._init_wandb()
        
        # Inicializar TensorBoard
        if config.use_tensorboard:
            self._init_tensorboard()
    
    def _init_wandb(self):
        """Inicializa Weights & Biases"""
        try:
            import wandb
            
            if self.config.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.config.wandb_api_key
            
            self.wandb_run = wandb.init(
                project=self.config.project,
                name=self.config.name,
                tags=self.config.tags,
                notes=self.config.notes
            )
            logger.info("W&B inicializado")
        except ImportError:
            logger.warning("wandb no instalado, usando solo TensorBoard")
        except Exception as e:
            logger.error(f"Error inicializando W&B: {e}")
    
    def _init_tensorboard(self):
        """Inicializa TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)
            log_path = os.path.join(
                self.config.tensorboard_log_dir,
                self.config.name,
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            self.tensorboard_writer = SummaryWriter(log_dir=log_path)
            logger.info(f"TensorBoard inicializado: {log_path}")
        except ImportError:
            logger.warning("tensorboard no instalado")
        except Exception as e:
            logger.error(f"Error inicializando TensorBoard: {e}")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Registra una métrica"""
        metric_data = {
            "key": key,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics_history.append(metric_data)
        
        # Log a W&B
        if self.wandb_run:
            try:
                self.wandb_run.log({key: value}, step=step)
            except Exception as e:
                logger.error(f"Error logging a W&B: {e}")
        
        # Log a TensorBoard
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_scalar(key, value, step or 0)
            except Exception as e:
                logger.error(f"Error logging a TensorBoard: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Registra múltiples métricas"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Registra hiperparámetros"""
        # Log a W&B
        if self.wandb_run:
            try:
                self.wandb_run.config.update(hyperparams)
            except Exception as e:
                logger.error(f"Error logging hyperparams a W&B: {e}")
        
        # Log a TensorBoard
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.add_hparams(hyperparams, {})
            except Exception as e:
                logger.error(f"Error logging hyperparams a TensorBoard: {e}")
    
    def log_image(self, key: str, image, step: Optional[int] = None):
        """Registra una imagen"""
        # Log a W&B
        if self.wandb_run:
            try:
                import wandb
                self.wandb_run.log({key: wandb.Image(image)}, step=step)
            except Exception as e:
                logger.error(f"Error logging image a W&B: {e}")
        
        # Log a TensorBoard
        if self.tensorboard_writer:
            try:
                import torch
                if not isinstance(image, torch.Tensor):
                    import numpy as np
                    from PIL import Image
                    if isinstance(image, Image.Image):
                        image = np.array(image)
                    image = torch.from_numpy(image).permute(2, 0, 1) if len(image.shape) == 3 else image
                self.tensorboard_writer.add_image(key, image, step or 0)
            except Exception as e:
                logger.error(f"Error logging image a TensorBoard: {e}")
    
    def finish(self):
        """Finaliza el tracking"""
        if self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception as e:
                logger.error(f"Error finalizando W&B: {e}")
        
        if self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
            except Exception as e:
                logger.error(f"Error cerrando TensorBoard: {e}")
        
        logger.info("Experiment tracking finalizado")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Obtiene el historial de métricas"""
        return self.metrics_history




