"""
Data Preprocessing Pipeline - Pipeline de preprocesamiento de datos
====================================================================
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Métodos de normalización"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    UNIT_VECTOR = "unit_vector"


@dataclass
class PreprocessingConfig:
    """Configuración de preprocesamiento"""
    normalize: bool = True
    normalization_method: NormalizationMethod = NormalizationMethod.Z_SCORE
    handle_missing: bool = True
    missing_strategy: str = "mean"  # "mean", "median", "zero", "drop"
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation"
    feature_scaling: bool = True
    categorical_encoding: str = "one_hot"  # "one_hot", "label", "embedding"


class DataPreprocessor:
    """Preprocesador de datos"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.stats: Dict[str, Any] = {}
        self.fitted = False
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """Ajusta el preprocesador a los datos"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        self.stats = {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
            "median": np.median(data, axis=0),
            "q25": np.percentile(data, 25, axis=0),
            "q75": np.percentile(data, 75, axis=0)
        }
        
        self.fitted = True
        logger.info("Preprocesador ajustado a los datos")
    
    def transform(
        self,
        data: Union[torch.Tensor, np.ndarray],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Transforma los datos"""
        if not self.fitted:
            raise ValueError("Preprocesador no ha sido ajustado. Llama a fit() primero.")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data_np = data.numpy()
        else:
            data_np = data.copy()
        
        # Manejar valores faltantes
        if self.config.handle_missing:
            data_np = self._handle_missing(data_np)
        
        # Manejar outliers
        if self.config.handle_outliers:
            data_np = self._handle_outliers(data_np)
        
        # Normalizar
        if self.config.normalize:
            data_np = self._normalize(data_np)
        
        if return_tensor and not is_tensor:
            return torch.tensor(data_np, dtype=torch.float32)
        elif not return_tensor and is_tensor:
            return data_np
        
        return data_np if not return_tensor else torch.tensor(data_np, dtype=torch.float32)
    
    def fit_transform(
        self,
        data: Union[torch.Tensor, np.ndarray],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Ajusta y transforma los datos"""
        self.fit(data)
        return self.transform(data, return_tensor)
    
    def _handle_missing(self, data: np.ndarray) -> np.ndarray:
        """Maneja valores faltantes"""
        if self.config.missing_strategy == "mean":
            missing_mask = np.isnan(data)
            data[missing_mask] = np.nanmean(data, axis=0)[np.newaxis, :]
        elif self.config.missing_strategy == "median":
            missing_mask = np.isnan(data)
            data[missing_mask] = np.nanmedian(data, axis=0)[np.newaxis, :]
        elif self.config.missing_strategy == "zero":
            data = np.nan_to_num(data, nan=0.0)
        elif self.config.missing_strategy == "drop":
            data = data[~np.isnan(data).any(axis=1)]
        
        return data
    
    def _handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """Maneja outliers"""
        if self.config.outlier_method == "iqr":
            q1 = self.stats["q25"]
            q3 = self.stats["q75"]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            data[outlier_mask] = np.clip(data[outlier_mask], lower_bound, upper_bound)
        
        elif self.config.outlier_method == "zscore":
            z_scores = np.abs((data - self.stats["mean"]) / (self.stats["std"] + 1e-8))
            outlier_mask = z_scores > 3
            data[outlier_mask] = self.stats["mean"][np.newaxis, :]
        
        return data
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normaliza los datos"""
        if self.config.normalization_method == NormalizationMethod.Z_SCORE:
            return (data - self.stats["mean"]) / (self.stats["std"] + 1e-8)
        
        elif self.config.normalization_method == NormalizationMethod.MIN_MAX:
            return (data - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + 1e-8)
        
        elif self.config.normalization_method == NormalizationMethod.ROBUST:
            return (data - self.stats["median"]) / (self.stats["q75"] - self.stats["q25"] + 1e-8)
        
        elif self.config.normalization_method == NormalizationMethod.UNIT_VECTOR:
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            return data / (norms + 1e-8)
        
        return data
    
    def inverse_transform(
        self,
        data: Union[torch.Tensor, np.ndarray],
        return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Transformación inversa"""
        if not self.fitted:
            raise ValueError("Preprocesador no ha sido ajustado")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data_np = data.numpy()
        else:
            data_np = data.copy()
        
        # Desnormalizar
        if self.config.normalize:
            if self.config.normalization_method == NormalizationMethod.Z_SCORE:
                data_np = data_np * self.stats["std"] + self.stats["mean"]
            elif self.config.normalization_method == NormalizationMethod.MIN_MAX:
                data_np = data_np * (self.stats["max"] - self.stats["min"]) + self.stats["min"]
        
        if return_tensor and not is_tensor:
            return torch.tensor(data_np, dtype=torch.float32)
        elif not return_tensor and is_tensor:
            return data_np
        
        return data_np if not return_tensor else torch.tensor(data_np, dtype=torch.float32)




