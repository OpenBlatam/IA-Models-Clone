"""
Experiment Tracking Module
==========================

Módulo para tracking de experimentos usando wandb y tensorboard.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

# Wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb no disponible")

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("tensorboard no disponible")


class ExperimentTracker:
    """
    Tracker de experimentos para deep learning.
    """
    
    def __init__(
        self,
        project_name: str = "routing_ai",
        experiment_name: Optional[str] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        log_dir: str = "./logs"
    ):
        """
        Inicializar tracker.
        
        Args:
            project_name: Nombre del proyecto
            experiment_name: Nombre del experimento
            use_wandb: Usar wandb
            use_tensorboard: Usar tensorboard
            log_dir: Directorio de logs
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir
        
        # Crear directorio de logs
        os.makedirs(log_dir, exist_ok=True)
        
        # Inicializar wandb
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=self.experiment_name,
                    dir=log_dir
                )
                logger.info(f"Wandb inicializado: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"Error inicializando wandb: {e}")
        
        # Inicializar tensorboard
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                tb_dir = os.path.join(log_dir, "tensorboard", self.experiment_name)
                self.tensorboard_writer = SummaryWriter(tb_dir)
                logger.info(f"TensorBoard inicializado: {tb_dir}")
            except Exception as e:
                logger.warning(f"Error inicializando tensorboard: {e}")
        
        # Historial local
        self.history: List[Dict[str, Any]] = []
    
    def log_config(self, config: Dict[str, Any]):
        """
        Loggear configuración.
        
        Args:
            config: Diccionario de configuración
        """
        if self.wandb_run:
            self.wandb_run.config.update(config)
        
        # Guardar en archivo
        config_path = os.path.join(self.log_dir, f"{self.experiment_name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ):
        """
        Loggear métricas.
        
        Args:
            metrics: Diccionario de métricas
            step: Paso actual
            epoch: Época actual
        """
        step = step or len(self.history)
        
        # Wandb
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        
        # TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)
        
        # Historial local
        entry = {"step": step, "epoch": epoch, **metrics}
        self.history.append(entry)
    
    def log_model(
        self,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Loggear modelo.
        
        Args:
            model_path: Ruta al modelo
            metadata: Metadata adicional
        """
        if self.wandb_run:
            artifact = wandb.Artifact(
                name=f"model_{self.experiment_name}",
                type="model"
            )
            artifact.add_file(model_path)
            if metadata:
                artifact.metadata = metadata
            self.wandb_run.log_artifact(artifact)
    
    def log_image(
        self,
        image_path: str,
        caption: str = "",
        step: Optional[int] = None
    ):
        """
        Loggear imagen.
        
        Args:
            image_path: Ruta a la imagen
            caption: Captión
            step: Paso actual
        """
        step = step or len(self.history)
        
        if self.wandb_run:
            self.wandb_run.log({caption: wandb.Image(image_path)}, step=step)
        
        if self.tensorboard_writer:
            from PIL import Image
            import numpy as np
            img = Image.open(image_path)
            img_array = np.array(img)
            self.tensorboard_writer.add_image(caption, img_array, step, dataformats="HWC")
    
    def log_histogram(
        self,
        name: str,
        values: Any,
        step: Optional[int] = None
    ):
        """
        Loggear histograma.
        
        Args:
            name: Nombre del histograma
            values: Valores
            step: Paso actual
        """
        step = step or len(self.history)
        
        if self.wandb_run:
            self.wandb_run.log({name: wandb.Histogram(values)}, step=step)
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(name, values, step)
    
    def finish(self):
        """Finalizar tracking."""
        if self.wandb_run:
            self.wandb_run.finish()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Guardar historial
        history_path = os.path.join(self.log_dir, f"{self.experiment_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Tracking finalizado: {self.experiment_name}")




