"""
Cross Validation System - Sistema de validación cruzada
========================================================
"""

import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class CVStrategy(Enum):
    """Estrategias de validación cruzada"""
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"
    LEAVE_ONE_OUT = "leave_one_out"


@dataclass
class CVResult:
    """Resultado de validación cruzada"""
    fold: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None


class CrossValidation:
    """Sistema de validación cruzada"""
    
    def __init__(self, strategy: CVStrategy = CVStrategy.K_FOLD, n_splits: int = 5):
        self.strategy = strategy
        self.n_splits = n_splits
        self.results: List[CVResult] = []
    
    def k_fold_split(
        self,
        dataset: Dataset,
        k: Optional[int] = None
    ) -> List[Dict[str, List[int]]]:
        """Divide dataset en k folds"""
        k = k or self.n_splits
        total_size = len(dataset)
        fold_size = total_size // k
        
        splits = []
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else total_size
            
            val_indices = indices[start:end]
            train_indices = indices[:start] + indices[end:]
            
            splits.append({
                "train": train_indices,
                "val": val_indices
            })
        
        return splits
    
    def stratified_k_fold_split(
        self,
        dataset: Dataset,
        labels: List[int],
        k: Optional[int] = None
    ) -> List[Dict[str, List[int]]]:
        """Divide dataset en k folds estratificados"""
        try:
            from sklearn.model_selection import StratifiedKFold
            
            k = k or self.n_splits
            skf = StratifiedKFold(n_splits=k, shuffle=True)
            splits = []
            
            for train_idx, val_idx in skf.split(range(len(dataset)), labels):
                splits.append({
                    "train": train_idx.tolist(),
                    "val": val_idx.tolist()
                })
            
            return splits
        except ImportError:
            logger.warning("sklearn no disponible, usando k_fold normal")
            return self.k_fold_split(dataset, k)
    
    def cross_validate(
        self,
        dataset: Dataset,
        train_fn: Callable,
        eval_fn: Callable,
        labels: Optional[List[int]] = None
    ) -> List[CVResult]:
        """Ejecuta validación cruzada"""
        # Obtener splits
        if self.strategy == CVStrategy.STRATIFIED_K_FOLD and labels:
            splits = self.stratified_k_fold_split(dataset, labels)
        else:
            splits = self.k_fold_split(dataset)
        
        results = []
        
        for fold, split in enumerate(splits):
            logger.info(f"Ejecutando fold {fold + 1}/{len(splits)}")
            
            # Crear datasets para este fold
            train_dataset = torch.utils.data.Subset(dataset, split["train"])
            val_dataset = torch.utils.data.Subset(dataset, split["val"])
            
            # Entrenar
            train_metrics = train_fn(train_dataset, fold)
            
            # Evaluar
            val_metrics = eval_fn(val_dataset, fold)
            
            result = CVResult(
                fold=fold,
                train_metrics=train_metrics,
                val_metrics=val_metrics
            )
            results.append(result)
        
        self.results = results
        return results
    
    def get_cv_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de validación cruzada"""
        if not self.results:
            return {}
        
        # Agregar métricas
        train_metrics_agg = {}
        val_metrics_agg = {}
        
        for result in self.results:
            for key, value in result.train_metrics.items():
                if key not in train_metrics_agg:
                    train_metrics_agg[key] = []
                train_metrics_agg[key].append(value)
            
            for key, value in result.val_metrics.items():
                if key not in val_metrics_agg:
                    val_metrics_agg[key] = []
                val_metrics_agg[key].append(value)
        
        # Calcular estadísticas
        summary = {
            "n_folds": len(self.results),
            "train_metrics": {
                key: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for key, values in train_metrics_agg.items()
            },
            "val_metrics": {
                key: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for key, values in val_metrics_agg.items()
            }
        }
        
        return summary




