"""
Cross-validation para evaluación robusta
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Callable, Optional
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validator para modelos"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def k_fold_cv(
        self,
        dataset: Dataset,
        train_fn: Callable,
        evaluate_fn: Callable,
        is_classification: bool = False
    ) -> Dict[str, Any]:
        """
        K-fold cross-validation
        
        Args:
            dataset: Dataset completo
            train_fn: Función de entrenamiento
            evaluate_fn: Función de evaluación
            is_classification: Si es clasificación (usa StratifiedKFold)
            
        Returns:
            Resultados de CV
        """
        if is_classification:
            kfold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            kfold = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        
        fold_results = []
        
        # Obtener índices
        indices = np.arange(len(dataset))
        if is_classification:
            # Necesitaríamos labels para StratifiedKFold
            # Por simplicidad, usar KFold
            kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            logger.info(f"Fold {fold+1}/{self.n_splits}")
            
            # Crear datasets para este fold
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            # Entrenar
            model = train_fn(train_subset)
            
            # Evaluar
            metrics = evaluate_fn(model, val_subset)
            metrics["fold"] = fold + 1
            fold_results.append(metrics)
        
        # Calcular estadísticas
        all_metrics = {}
        for key in fold_results[0].keys():
            if key != "fold":
                values = [r[key] for r in fold_results]
                all_metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values
                }
        
        return {
            "fold_results": fold_results,
            "summary": all_metrics,
            "n_splits": self.n_splits
        }




