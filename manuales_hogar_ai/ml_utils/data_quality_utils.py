"""
Data Quality Utils - Utilidades de Calidad de Datos
====================================================

Utilidades para evaluar y mejorar la calidad de datos.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Reporte de calidad de datos."""
    total_samples: int
    missing_values: Dict[str, int]
    outliers: Dict[str, int]
    duplicates: int
    class_distribution: Dict[int, int]
    data_types: Dict[str, str]
    statistics: Dict[str, Dict[str, float]]


class DataQualityChecker:
    """
    Verificador de calidad de datos.
    """
    
    def __init__(self):
        """Inicializar verificador."""
        pass
    
    def check_dataset(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> DataQualityReport:
        """
        Verificar calidad de dataset.
        
        Args:
            data: Datos
            labels: Labels (opcional)
            feature_names: Nombres de features (opcional)
            
        Returns:
            Reporte de calidad
        """
        total_samples = len(data)
        
        # Valores faltantes
        missing_values = {}
        if feature_names:
            for i, name in enumerate(feature_names):
                missing = np.isnan(data[:, i]).sum() if len(data.shape) > 1 else np.isnan(data).sum()
                if missing > 0:
                    missing_values[name] = int(missing)
        else:
            if len(data.shape) > 1:
                for i in range(data.shape[1]):
                    missing = np.isnan(data[:, i]).sum()
                    if missing > 0:
                        missing_values[f"feature_{i}"] = int(missing)
        
        # Outliers (usando IQR)
        outliers = {}
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
                feature_data = data[:, i]
                q1 = np.percentile(feature_data, 25)
                q3 = np.percentile(feature_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_count = ((feature_data < lower_bound) | (feature_data > upper_bound)).sum()
                if outlier_count > 0:
                    feature_name = feature_names[i] if feature_names else f"feature_{i}"
                    outliers[feature_name] = int(outlier_count)
        
        # Duplicados
        if len(data.shape) == 2:
            duplicates = len(data) - len(np.unique(data, axis=0))
        else:
            duplicates = len(data) - len(np.unique(data))
        
        # Distribución de clases
        class_distribution = {}
        if labels is not None:
            class_distribution = dict(Counter(labels))
        
        # Tipos de datos
        data_types = {}
        if feature_names:
            for i, name in enumerate(feature_names):
                data_types[name] = str(data.dtype)
        else:
            data_types['data'] = str(data.dtype)
        
        # Estadísticas
        statistics = {}
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
                feature_data = data[:, i]
                feature_name = feature_names[i] if feature_names else f"feature_{i}"
                statistics[feature_name] = {
                    'mean': float(np.mean(feature_data)),
                    'std': float(np.std(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data)),
                    'median': float(np.median(feature_data))
                }
        
        return DataQualityReport(
            total_samples=total_samples,
            missing_values=missing_values,
            outliers=outliers,
            duplicates=duplicates,
            class_distribution=class_distribution,
            data_types=data_types,
            statistics=statistics
        )
    
    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detectar drift de datos.
        
        Args:
            reference_data: Datos de referencia
            current_data: Datos actuales
            threshold: Umbral de drift
            
        Returns:
            Información de drift
        """
        drift_info = {
            'has_drift': False,
            'feature_drifts': {}
        }
        
        if len(reference_data.shape) > 1 and len(current_data.shape) > 1:
            for i in range(reference_data.shape[1]):
                ref_feature = reference_data[:, i]
                curr_feature = current_data[:, i]
                
                ref_mean = np.mean(ref_feature)
                curr_mean = np.mean(curr_feature)
                
                drift_ratio = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-10)
                
                if drift_ratio > threshold:
                    drift_info['has_drift'] = True
                    drift_info['feature_drifts'][f"feature_{i}"] = {
                        'drift_ratio': float(drift_ratio),
                        'reference_mean': float(ref_mean),
                        'current_mean': float(curr_mean)
                    }
        
        return drift_info


class DataCleaner:
    """
    Limpiador de datos.
    """
    
    @staticmethod
    def remove_missing(
        data: np.ndarray,
        strategy: str = "drop"
    ) -> np.ndarray:
        """
        Remover valores faltantes.
        
        Args:
            data: Datos
            strategy: Estrategia ('drop', 'mean', 'median')
            
        Returns:
            Datos limpios
        """
        if strategy == "drop":
            if len(data.shape) > 1:
                return data[~np.isnan(data).any(axis=1)]
            else:
                return data[~np.isnan(data)]
        
        elif strategy == "mean":
            if len(data.shape) > 1:
                for i in range(data.shape[1]):
                    col = data[:, i]
                    col[np.isnan(col)] = np.nanmean(col)
            else:
                data[np.isnan(data)] = np.nanmean(data)
            return data
        
        elif strategy == "median":
            if len(data.shape) > 1:
                for i in range(data.shape[1]):
                    col = data[:, i]
                    col[np.isnan(col)] = np.nanmedian(col)
            else:
                data[np.isnan(data)] = np.nanmedian(data)
            return data
        
        return data
    
    @staticmethod
    def remove_outliers(
        data: np.ndarray,
        method: str = "iqr"
    ) -> np.ndarray:
        """
        Remover outliers.
        
        Args:
            data: Datos
            method: Método ('iqr', 'zscore')
            
        Returns:
            Datos sin outliers
        """
        if method == "iqr":
            if len(data.shape) > 1:
                mask = np.ones(len(data), dtype=bool)
                for i in range(data.shape[1]):
                    col = data[:, i]
                    q1 = np.percentile(col, 25)
                    q3 = np.percentile(col, 75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    mask &= (col >= lower) & (col <= upper)
                return data[mask]
            else:
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                return data[(data >= lower) & (data <= upper)]
        
        elif method == "zscore":
            if len(data.shape) > 1:
                z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
                mask = (z_scores < 3).all(axis=1)
                return data[mask]
            else:
                z_scores = np.abs((data - np.mean(data)) / np.std(data))
                return data[z_scores < 3]
        
        return data




