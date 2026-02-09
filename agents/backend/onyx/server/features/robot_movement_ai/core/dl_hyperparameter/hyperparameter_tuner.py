"""
Hyperparameter Tuner - Modular Hyperparameter Tuning
====================================================

Tuning modular de hiperparámetros usando diferentes estrategias.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import torch
import torch.nn as nn
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Espacio de hiperparámetros."""
    learning_rate: tuple = (1e-5, 1e-2)
    batch_size: List[int] = None
    hidden_dim: List[int] = None
    num_layers: List[int] = None
    dropout: tuple = (0.0, 0.5)
    weight_decay: tuple = (1e-6, 1e-3)
    
    def __post_init__(self):
        """Inicialización post-construcción."""
        if self.batch_size is None:
            self.batch_size = [16, 32, 64, 128]
        if self.hidden_dim is None:
            self.hidden_dim = [128, 256, 512]
        if self.num_layers is None:
            self.num_layers = [2, 4, 6, 8]


class HyperparameterTuner(ABC):
    """Clase base para tuners."""
    
    @abstractmethod
    def tune(
        self,
        model_fn: Callable,
        train_loader: Any,
        val_loader: Any,
        space: HyperparameterSpace,
        num_trials: int = 10
    ) -> Dict[str, Any]:
        """Ejecutar tuning."""
        pass


class RandomSearchTuner(HyperparameterTuner):
    """Random search para hiperparámetros."""
    
    def tune(
        self,
        model_fn: Callable,
        train_loader: Any,
        val_loader: Any,
        space: HyperparameterSpace,
        num_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Ejecutar random search.
        
        Args:
            model_fn: Función que crea el modelo
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            space: Espacio de hiperparámetros
            num_trials: Número de trials
            
        Returns:
            Mejores hiperparámetros
        """
        import random
        
        best_params = None
        best_score = float('inf')
        results = []
        
        for trial in range(num_trials):
            # Sample hiperparámetros
            params = {
                'learning_rate': random.uniform(*space.learning_rate),
                'batch_size': random.choice(space.batch_size),
                'hidden_dim': random.choice(space.hidden_dim),
                'num_layers': random.choice(space.num_layers),
                'dropout': random.uniform(*space.dropout),
                'weight_decay': random.uniform(*space.weight_decay)
            }
            
            logger.info(f"Trial {trial + 1}/{num_trials}: {params}")
            
            # Crear y entrenar modelo
            try:
                model = model_fn(**params)
                score = self._train_and_evaluate(model, train_loader, val_loader, params)
                
                results.append({
                    'params': params,
                    'score': score
                })
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"New best score: {best_score:.4f}")
            
            except Exception as e:
                logger.error(f"Trial {trial + 1} failed: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def _train_and_evaluate(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        params: Dict[str, Any]
    ) -> float:
        """Entrenar y evaluar modelo."""
        # Implementación simplificada
        # En producción, usar Trainer completo
        from ..dl_training.trainer import Trainer
        from ..dl_evaluation.evaluator import Evaluator
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Entrenar por pocas épocas para tuning rápido
        trainer.train(num_epochs=5)
        
        # Evaluar
        evaluator = Evaluator(model)
        results = evaluator.evaluate(val_loader, metrics=['mse'])
        
        return results.get('mse', float('inf'))


class OptunaTuner(HyperparameterTuner):
    """Tuner usando Optuna."""
    
    def tune(
        self,
        model_fn: Callable,
        train_loader: Any,
        val_loader: Any,
        space: HyperparameterSpace,
        num_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Ejecutar tuning con Optuna.
        
        Args:
            model_fn: Función que crea el modelo
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            space: Espacio de hiperparámetros
            num_trials: Número de trials
            
        Returns:
            Mejores hiperparámetros
        """
        try:
            import optuna
            
            def objective(trial):
                # Suggest hiperparámetros
                params = {
                    'learning_rate': trial.suggest_loguniform('lr', *space.learning_rate),
                    'batch_size': trial.suggest_categorical('batch_size', space.batch_size),
                    'hidden_dim': trial.suggest_categorical('hidden_dim', space.hidden_dim),
                    'num_layers': trial.suggest_categorical('num_layers', space.num_layers),
                    'dropout': trial.suggest_uniform('dropout', *space.dropout),
                    'weight_decay': trial.suggest_loguniform('weight_decay', *space.weight_decay)
                }
                
                # Crear y entrenar modelo
                model = model_fn(**params)
                score = self._train_and_evaluate(model, train_loader, val_loader, params)
                
                return score
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=num_trials)
            
            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'study': study
            }
        except ImportError:
            raise ImportError("optuna not available. Install with: pip install optuna")
    
    def _train_and_evaluate(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        params: Dict[str, Any]
    ) -> float:
        """Entrenar y evaluar modelo."""
        from ..dl_training.trainer import Trainer
        from ..dl_evaluation.evaluator import Evaluator
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        trainer.train(num_epochs=5)
        
        evaluator = Evaluator(model)
        results = evaluator.evaluate(val_loader, metrics=['mse'])
        
        return results.get('mse', float('inf'))


class HyperparameterTunerFactory:
    """Factory para tuners."""
    
    _tuners = {
        'random': RandomSearchTuner,
        'optuna': OptunaTuner
    }
    
    @classmethod
    def get_tuner(cls, tuner_type: str) -> HyperparameterTuner:
        """
        Obtener tuner por tipo.
        
        Args:
            tuner_type: Tipo de tuner
            
        Returns:
            Tuner
        """
        if tuner_type not in cls._tuners:
            raise ValueError(f"Unknown tuner type: {tuner_type}")
        
        return cls._tuners[tuner_type]()
    
    @classmethod
    def register_tuner(cls, tuner_type: str, tuner_class: type):
        """Registrar nuevo tuner."""
        cls._tuners[tuner_type] = tuner_class








