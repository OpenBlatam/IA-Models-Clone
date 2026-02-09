"""
Hyperparameter Optimization
===========================

Optimización de hiperparámetros usando Optuna.
"""

import logging
from typing import Dict, Any, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna no disponible, optimización de hiperparámetros deshabilitada")


class HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros usando Optuna.
    """
    
    def __init__(
        self,
        study_name: str = "routing_optimization",
        direction: str = "minimize",
        n_trials: int = 50,
        timeout: Optional[float] = None
    ):
        """
        Inicializar optimizador.
        
        Args:
            study_name: Nombre del estudio
            direction: Dirección de optimización ("minimize" o "maximize")
            n_trials: Número de trials
            timeout: Timeout en segundos (opcional)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna no disponible. Instalar con: pip install optuna")
        
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
    
    def suggest_model_config(self, trial: Trial) -> Dict[str, Any]:
        """
        Sugerir configuración de modelo.
        
        Args:
            trial: Trial de Optuna
            
        Returns:
            Configuración sugerida
        """
        return {
            "input_dim": 20,  # Fijo
            "hidden_dims": [
                trial.suggest_int("hidden_dim_1", 64, 256, step=32),
                trial.suggest_int("hidden_dim_2", 128, 512, step=64),
                trial.suggest_int("hidden_dim_3", 64, 256, step=32)
            ],
            "output_dim": 4,  # Fijo
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "activation": trial.suggest_categorical("activation", ["relu", "elu", "gelu"]),
            "use_batch_norm": trial.suggest_categorical("use_batch_norm", [True, False]),
            "use_attention": trial.suggest_categorical("use_attention", [True, False])
        }
    
    def suggest_training_config(self, trial: Trial) -> Dict[str, Any]:
        """
        Sugerir configuración de entrenamiento.
        
        Args:
            trial: Trial de Optuna
            
        Returns:
            Configuración sugerida
        """
        return {
            "epochs": 50,  # Reducido para optimización
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
            "scheduler": trial.suggest_categorical("scheduler", ["reduce_on_plateau", "cosine", None]),
            "gradient_clip_norm": trial.suggest_float("gradient_clip_norm", 0.5, 2.0)
        }
    
    def optimize(
        self,
        objective_fn: Callable[[Trial], float],
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> optuna.Study:
        """
        Ejecutar optimización.
        
        Args:
            objective_fn: Función objetivo que toma un Trial y retorna un score
            n_trials: Número de trials (opcional, usa self.n_trials si None)
            timeout: Timeout en segundos (opcional)
            
        Returns:
            Estudio de Optuna
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        self.study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Optimización completada. Mejor trial: {self.study.best_trial.number}")
        logger.info(f"Mejor valor: {self.study.best_value:.4f}")
        logger.info(f"Mejores parámetros: {self.study.best_params}")
        
        return self.study
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Obtener mejores parámetros.
        
        Returns:
            Diccionario con mejores parámetros
        """
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """
        Obtener mejor valor.
        
        Returns:
            Mejor valor encontrado
        """
        return self.study.best_value
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plotear historial de optimización.
        
        Args:
            save_path: Ruta donde guardar (opcional)
        """
        try:
            import optuna.visualization as vis
            
            fig = vis.plot_optimization_history(self.study)
            if save_path:
                fig.write_image(save_path)
            else:
                fig.show()
        except Exception as e:
            logger.warning(f"Error ploteando historial: {e}")


def create_objective_function(
    train_loader,
    val_loader,
    model_factory_fn,
    device: str = "cuda"
) -> Callable:
    """
    Crear función objetivo para Optuna.
    
    Args:
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        model_factory_fn: Función que crea modelo desde config
        device: Dispositivo
        
    Returns:
        Función objetivo
    """
    def objective(trial):
        import torch
        import torch.nn as nn
        from ..routing_training.trainer import RouteTrainer, TrainingConfig
        
        # Sugerir configuraciones
        model_config = HyperparameterOptimizer(None).suggest_model_config(trial)
        training_config = HyperparameterOptimizer(None).suggest_training_config(trial)
        
        # Crear modelo
        model = model_factory_fn(model_config)
        model = model.to(device)
        
        # Crear entrenador
        trainer = RouteTrainer(
            model=model,
            config=TrainingConfig(**training_config),
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Entrenar (solo unas pocas épocas para optimización)
        history = trainer.train()
        
        # Retornar mejor pérdida de validación
        return history.get("best_val_loss", float('inf'))
    
    return objective


