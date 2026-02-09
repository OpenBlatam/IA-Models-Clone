"""
Routing Cross-Validation
=========================

Sistema de cross-validation para modelos de routing.
Implementa k-fold, stratified, y time-series cross-validation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Cross-validation features will be limited.")


@dataclass
class CrossValidationResult:
    """Resultado de cross-validation."""
    fold: int
    train_loss: float
    val_loss: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    model_state: Optional[Any] = None


class CrossValidator:
    """Validador cruzado profesional."""
    
    def __init__(
        self,
        n_splits: int = 5,
        cv_type: str = "kfold",
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Inicializar cross-validator.
        
        Args:
            n_splits: Número de folds
            cv_type: Tipo de CV ('kfold', 'stratified', 'timeseries')
            shuffle: Mezclar datos
            random_state: Semilla aleatoria
        """
        self.n_splits = n_splits
        self.cv_type = cv_type.lower()
        self.shuffle = shuffle
        self.random_state = random_state
        
        if SKLEARN_AVAILABLE:
            if self.cv_type == "kfold":
                self.cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            elif self.cv_type == "stratified":
                self.cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            elif self.cv_type == "timeseries":
                self.cv = TimeSeriesSplit(n_splits=n_splits)
            else:
                raise ValueError(f"Unknown CV type: {cv_type}")
        else:
            self.cv = None
            logger.warning("Using simple k-fold without sklearn")
    
    def cross_validate(
        self,
        train_func: Callable,
        data: List[Any],
        labels: Optional[List[Any]] = None,
        **kwargs
    ) -> List[CrossValidationResult]:
        """
        Ejecutar cross-validation.
        
        Args:
            train_func: Función de entrenamiento
            data: Datos de entrenamiento
            labels: Labels (para stratified CV)
            **kwargs: Argumentos adicionales para train_func
        
        Returns:
            Lista de resultados por fold
        """
        if self.cv is None:
            return self._simple_kfold(train_func, data, labels, **kwargs)
        
        results = []
        
        if self.cv_type == "stratified" and labels is not None:
            splits = self.cv.split(data, labels)
        else:
            splits = self.cv.split(data)
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Fold {fold + 1}/{self.n_splits}")
            
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]
            
            # Entrenar modelo
            result = train_func(train_data, val_data, fold=fold, **kwargs)
            results.append(result)
        
        return results
    
    def _simple_kfold(
        self,
        train_func: Callable,
        data: List[Any],
        labels: Optional[List[Any]],
        **kwargs
    ) -> List[CrossValidationResult]:
        """K-fold simple sin sklearn."""
        n = len(data)
        fold_size = n // self.n_splits
        results = []
        
        for fold in range(self.n_splits):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n
            
            val_idx = list(range(val_start, val_end))
            train_idx = [i for i in range(n) if i not in val_idx]
            
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]
            
            result = train_func(train_data, val_data, fold=fold, **kwargs)
            results.append(result)
        
        return results
    
    def aggregate_results(
        self,
        results: List[CrossValidationResult]
    ) -> Dict[str, Any]:
        """
        Agregar resultados de cross-validation.
        
        Args:
            results: Lista de resultados
        
        Returns:
            Estadísticas agregadas
        """
        train_losses = [r.train_loss for r in results]
        val_losses = [r.val_loss for r in results]
        
        return {
            'mean_train_loss': np.mean(train_losses),
            'std_train_loss': np.std(train_losses),
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'best_fold': int(np.argmin(val_losses)),
            'best_val_loss': float(np.min(val_losses)),
            'folds': len(results)
        }

