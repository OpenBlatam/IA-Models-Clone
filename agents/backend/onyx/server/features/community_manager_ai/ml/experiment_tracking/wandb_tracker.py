"""
WandB Tracker - Experiment Tracking
====================================

Tracking avanzado de experimentos con Weights & Biases.
"""

import logging
import wandb
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class WandBTracker:
    """Tracker de experimentos con WandB"""
    
    def __init__(
        self,
        project_name: str = "community-manager-ai",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None
    ):
        """
        Inicializar tracker
        
        Args:
            project_name: Nombre del proyecto
            entity: Entidad de WandB
            config: Configuración del experimento
            tags: Tags del experimento
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        
        # Inicializar WandB
        try:
            wandb.init(
                project=project_name,
                entity=entity,
                config=self.config,
                tags=self.tags
            )
            logger.info(f"WandB inicializado para proyecto {project_name}")
        except Exception as e:
            logger.warning(f"Error inicializando WandB: {e}")
            self.wandb_available = False
        else:
            self.wandb_available = True
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Loggear métricas
        
        Args:
            metrics: Dict con métricas
            step: Paso actual
        """
        if self.wandb_available:
            wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str, name: str = "model"):
        """
        Loggear modelo
        
        Args:
            model_path: Ruta del modelo
            name: Nombre del artefacto
        """
        if self.wandb_available:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)
    
    def log_image(self, image, caption: str = "", step: Optional[int] = None):
        """
        Loggear imagen
        
        Args:
            image: Imagen (PIL Image o numpy array)
            caption: Caption de la imagen
            step: Paso actual
        """
        if self.wandb_available:
            wandb.log({caption: wandb.Image(image)}, step=step)
    
    def log_histogram(self, name: str, values, step: Optional[int] = None):
        """
        Loggear histograma
        
        Args:
            name: Nombre del histograma
            values: Valores
            step: Paso actual
        """
        if self.wandb_available:
            wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def log_table(self, table_name: str, columns: list, data: list):
        """
        Loggear tabla
        
        Args:
            table_name: Nombre de la tabla
            columns: Columnas
            data: Datos
        """
        if self.wandb_available:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({table_name: table})
    
    def finish(self):
        """Finalizar run"""
        if self.wandb_available:
            wandb.finish()




