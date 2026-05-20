"""
Experiment Tracking - Utilidades de Tracking de Experimentos
============================================================

Utilidades para tracking de experimentos con TensorBoard y Weights & Biases.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Intentar importar tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    _has_tensorboard = True
except ImportError:
    _has_tensorboard = False

# Intentar importar wandb
try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False


class ExperimentTracker:
    """
    Tracker de experimentos con soporte para múltiples backends.
    """
    
    def __init__(
        self,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        log_dir: str = "./logs"
    ):
        """
        Inicializar tracker.
        
        Args:
            experiment_name: Nombre del experimento
            use_tensorboard: Usar TensorBoard
            use_wandb: Usar Weights & Biases
            wandb_project: Proyecto de W&B
            log_dir: Directorio de logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard and _has_tensorboard:
            tb_dir = self.log_dir / "tensorboard" / experiment_name
            self.tb_writer = SummaryWriter(str(tb_dir))
            logger.info(f"TensorBoard initialized: {tb_dir}")
        
        # Weights & Biases
        self.wandb_run = None
        if use_wandb and _has_wandb:
            wandb.init(
                project=wandb_project or "ml-experiments",
                name=experiment_name
            )
            self.wandb_run = wandb.run
            logger.info(f"W&B initialized: {experiment_name}")
        
        # Historial local
        self.history: Dict[str, List[float]] = {}
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Registrar métrica.
        
        Args:
            name: Nombre de métrica
            value: Valor
            step: Paso (opcional)
        """
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(value)
        
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step or len(self.history[name]))
        
        if self.wandb_run:
            wandb.log({name: value}, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Registrar múltiples métricas.
        
        Args:
            metrics: Diccionario de métricas
            step: Paso (opcional)
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Registrar hiperparámetros.
        
        Args:
            hyperparams: Diccionario de hiperparámetros
        """
        if self.tb_writer:
            self.tb_writer.add_hparams(hyperparams, {})
        
        if self.wandb_run:
            wandb.config.update(hyperparams)
        
        # Guardar localmente
        config_path = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def log_image(self, name: str, image: Any, step: Optional[int] = None) -> None:
        """
        Registrar imagen.
        
        Args:
            name: Nombre de imagen
            image: Imagen (PIL, numpy, tensor)
            step: Paso (opcional)
        """
        if self.tb_writer:
            self.tb_writer.add_image(name, image, step)
        
        if self.wandb_run:
            wandb.log({name: wandb.Image(image)}, step=step)
    
    def log_model_graph(self, model: Any, input_sample: Any) -> None:
        """
        Registrar grafo del modelo.
        
        Args:
            model: Modelo PyTorch
            input_sample: Muestra de entrada
        """
        if self.tb_writer:
            self.tb_writer.add_graph(model, input_sample)
    
    def close(self) -> None:
        """Cerrar tracker"""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        # Guardar historial
        history_path = self.log_dir / f"{self.experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Experiment tracking closed: {self.experiment_name}")


class ExperimentManager:
    """
    Gestor de múltiples experimentos.
    """
    
    def __init__(self, base_dir: str = "./experiments"):
        """
        Inicializar gestor.
        
        Args:
            base_dir: Directorio base
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentTracker] = {}
    
    def create_experiment(
        self,
        name: str,
        **kwargs
    ) -> ExperimentTracker:
        """
        Crear nuevo experimento.
        
        Args:
            name: Nombre del experimento
            **kwargs: Argumentos para ExperimentTracker
            
        Returns:
            ExperimentTracker
        """
        tracker = ExperimentTracker(
            experiment_name=name,
            log_dir=str(self.base_dir),
            **kwargs
        )
        self.experiments[name] = tracker
        return tracker
    
    def get_experiment(self, name: str) -> Optional[ExperimentTracker]:
        """
        Obtener experimento.
        
        Args:
            name: Nombre del experimento
            
        Returns:
            ExperimentTracker o None
        """
        return self.experiments.get(name)
    
    def list_experiments(self) -> List[str]:
        """
        Listar experimentos.
        
        Returns:
            Lista de nombres de experimentos
        """
        return list(self.experiments.keys())
    
    def close_all(self) -> None:
        """Cerrar todos los experimentos"""
        for tracker in self.experiments.values():
            tracker.close()




