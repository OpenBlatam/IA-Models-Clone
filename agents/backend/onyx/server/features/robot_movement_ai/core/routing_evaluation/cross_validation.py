"""
Cross Validation
================

Validación cruzada para evaluación robusta.
"""

import torch
from typing import List, Dict, Any, Optional
from sklearn.model_selection import KFold
import numpy as np
import logging

from ..routing_data import RouteDataset
from ..routing_training import RouteTrainer, TrainingConfig
from ..routing_models.base_model import BaseRouteModel
from .evaluator import ModelEvaluator, EvaluationMetrics

logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Validador cruzado base.
    """
    
    def __init__(self, n_splits: int = 5):
        """
        Inicializar validador.
        
        Args:
            n_splits: Número de folds
        """
        self.n_splits = n_splits
    
    def validate(
        self,
        model_factory: callable,
        dataset: RouteDataset,
        training_config: TrainingConfig
    ) -> List[EvaluationMetrics]:
        """
        Ejecutar validación cruzada.
        
        Args:
            model_factory: Función que crea modelo
            dataset: Dataset
            training_config: Configuración de entrenamiento
            
        Returns:
            Lista de métricas por fold
        """
        raise NotImplementedError


class KFoldCrossValidator(CrossValidator):
    """
    Validación cruzada K-Fold.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Inicializar K-Fold validator.
        
        Args:
            n_splits: Número de folds
            shuffle: Mezclar datos
            random_state: Semilla aleatoria
        """
        super(KFoldCrossValidator, self).__init__(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    def validate(
        self,
        model_factory: callable,
        dataset: RouteDataset,
        training_config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Ejecutar K-Fold cross validation.
        
        Args:
            model_factory: Función que crea modelo
            dataset: Dataset
            training_config: Configuración de entrenamiento
            
        Returns:
            Resultados de validación cruzada
        """
        # Convertir dataset a arrays para KFold
        features = [dataset[i][0].numpy() for i in range(len(dataset))]
        targets = [dataset[i][1].numpy() for i in range(len(dataset))]
        
        features_array = np.array(features)
        targets_array = np.array(targets)
        
        fold_metrics = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(self.kfold.split(features_array)):
            logger.info(f"Fold {fold_idx + 1}/{self.n_splits}")
            
            # Crear datasets para este fold
            train_features = [features_array[i] for i in train_indices]
            train_targets = [targets_array[i] for i in train_indices]
            val_features = [features_array[i] for i in val_indices]
            val_targets = [targets_array[i] for i in val_indices]
            
            train_dataset = RouteDataset(train_features, train_targets)
            val_dataset = RouteDataset(val_features, val_targets)
            
            # Crear modelo
            model = model_factory()
            
            # Crear data loaders
            from ..routing_data import RouteDataLoader
            train_loader, val_loader = RouteDataLoader.create_train_val_loaders(
                train_dataset,
                val_dataset,
                batch_size=training_config.batch_size
            )
            
            # Entrenar
            trainer = RouteTrainer(
                model=model,
                config=training_config,
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            history = trainer.train()
            
            # Evaluar
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(model, val_loader)
            
            fold_metrics.append(metrics)
        
        # Calcular estadísticas agregadas
        return self._aggregate_results(fold_metrics)
    
    def _aggregate_results(
        self,
        fold_metrics: List[EvaluationMetrics]
    ) -> Dict[str, Any]:
        """
        Agregar resultados de todos los folds.
        
        Args:
            fold_metrics: Métricas por fold
            
        Returns:
            Resultados agregados
        """
        r2_scores = [m.r2 for m in fold_metrics]
        mse_scores = [m.mse for m in fold_metrics]
        mae_scores = [m.mae for m in fold_metrics]
        
        return {
            "fold_metrics": fold_metrics,
            "mean_r2": np.mean(r2_scores),
            "std_r2": np.std(r2_scores),
            "mean_mse": np.mean(mse_scores),
            "std_mse": np.std(mse_scores),
            "mean_mae": np.mean(mae_scores),
            "std_mae": np.std(mae_scores),
            "best_fold": np.argmax(r2_scores),
            "worst_fold": np.argmin(r2_scores)
        }

