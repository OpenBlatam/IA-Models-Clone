"""
Cross-Validation Utils - Utilidades de Validación Cruzada
==========================================================

Utilidades para validación cruzada y evaluación de modelos.
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from collections import defaultdict

logger = logging.getLogger(__name__)

# Intentar importar sklearn
try:
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    logger.warning("sklearn not available, some CV functions will be limited")


class CrossValidator:
    """
    Validador cruzado para modelos PyTorch.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratify: bool = False
    ):
        """
        Inicializar validador cruzado.
        
        Args:
            n_splits: Número de folds
            shuffle: Mezclar datos
            random_state: Semilla aleatoria
            stratify: Usar estratificación (requiere sklearn)
        """
        if not _has_sklearn:
            raise ImportError("sklearn is required for CrossValidator")
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        if stratify:
            self.cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        else:
            self.cv = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generar splits de validación cruzada.
        
        Args:
            X: Features
            y: Targets (opcional, requerido para estratificación)
            
        Returns:
            Lista de tuplas (train_indices, val_indices)
        """
        if y is not None:
            return list(self.cv.split(X, y))
        else:
            return list(self.cv.split(X))
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        train_fn: Callable,
        eval_fn: Callable,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Evaluar modelo con validación cruzada.
        
        Args:
            model: Modelo PyTorch
            dataset: Dataset completo
            train_fn: Función de entrenamiento
            eval_fn: Función de evaluación
            device: Dispositivo
            
        Returns:
            Diccionario con resultados
        """
        X = np.arange(len(dataset))
        y = np.array([dataset[i][1] for i in range(len(dataset))])
        
        results = defaultdict(list)
        
        for fold, (train_idx, val_idx) in enumerate(self.split(X, y)):
            logger.info(f"Fold {fold + 1}/{self.n_splits}")
            
            # Crear datasets para este fold
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            
            # Entrenar modelo
            train_fn(model, train_dataset, fold)
            
            # Evaluar modelo
            val_results = eval_fn(model, val_dataset, device)
            
            for key, value in val_results.items():
                results[key].append(value)
        
        # Calcular promedios
        summary = {}
        for key, values in results.items():
            summary[f"{key}_mean"] = np.mean(values)
            summary[f"{key}_std"] = np.std(values)
            summary[f"{key}_values"] = values
        
        return summary


class TimeSeriesCrossValidator:
    """
    Validador cruzado para series temporales.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Inicializar validador para series temporales.
        
        Args:
            n_splits: Número de folds
            max_train_size: Tamaño máximo de entrenamiento
            test_size: Tamaño de test
            gap: Gap entre train y test
        """
        if not _has_sklearn:
            raise ImportError("sklearn is required for TimeSeriesCrossValidator")
        
        self.cv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=gap
        )
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generar splits para series temporales.
        
        Args:
            X: Features
            
        Returns:
            Lista de tuplas (train_indices, val_indices)
        """
        return list(self.cv.split(X))


class GroupKFold:
    """
    K-Fold con grupos (para evitar data leakage).
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
        """
        Inicializar GroupKFold.
        
        Args:
            n_splits: Número de folds
            shuffle: Mezclar grupos
            random_state: Semilla aleatoria
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generar splits con grupos.
        
        Args:
            X: Features
            y: Targets (opcional)
            groups: Grupos para evitar leakage
            
        Returns:
            Lista de tuplas (train_indices, val_indices)
        """
        if groups is None:
            raise ValueError("groups is required for GroupKFold")
        
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_groups)
        
        fold_size = n_groups // self.n_splits
        remainder = n_groups % self.n_splits
        
        splits = []
        start = 0
        
        for fold in range(self.n_splits):
            fold_size_adj = fold_size + (1 if fold < remainder else 0)
            end = start + fold_size_adj
            
            test_groups = unique_groups[start:end]
            train_groups = np.concatenate([unique_groups[:start], unique_groups[end:]])
            
            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx = np.where(np.isin(groups, test_groups))[0]
            
            splits.append((train_idx, test_idx))
            start = end
        
        return splits


def k_fold_cv(
    model: torch.nn.Module,
    dataset: Dataset,
    train_fn: Callable,
    eval_fn: Callable,
    n_splits: int = 5,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Realizar K-Fold cross-validation.
    
    Args:
        model: Modelo PyTorch
        dataset: Dataset completo
        train_fn: Función de entrenamiento
        eval_fn: Función de evaluación
        n_splits: Número de folds
        device: Dispositivo
        
    Returns:
        Resultados de validación cruzada
    """
    validator = CrossValidator(n_splits=n_splits)
    return validator.evaluate(model, dataset, train_fn, eval_fn, device)


def stratified_k_fold_cv(
    model: torch.nn.Module,
    dataset: Dataset,
    train_fn: Callable,
    eval_fn: Callable,
    n_splits: int = 5,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Realizar Stratified K-Fold cross-validation.
    
    Args:
        model: Modelo PyTorch
        dataset: Dataset completo
        train_fn: Función de entrenamiento
        eval_fn: Función de evaluación
        n_splits: Número de folds
        device: Dispositivo
        
    Returns:
        Resultados de validación cruzada
    """
    validator = CrossValidator(n_splits=n_splits, stratify=True)
    return validator.evaluate(model, dataset, train_fn, eval_fn, device)




