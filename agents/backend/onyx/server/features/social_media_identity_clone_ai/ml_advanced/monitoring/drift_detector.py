"""
Detección de drift en datos y modelos
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from scipy import stats
import torch

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detector de drift en datos y modelos"""
    
    def __init__(self):
        pass
    
    def detect_data_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        method: str = "ks_test"  # "ks_test", "wasserstein", "mmd"
    ) -> Dict[str, Any]:
        """
        Detecta drift en datos
        
        Args:
            reference_data: Datos de referencia
            current_data: Datos actuales
            method: Método de detección
            
        Returns:
            Resultados de detección de drift
        """
        try:
            if method == "ks_test":
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(reference_data, current_data)
                has_drift = p_value < 0.05
                
                return {
                    "has_drift": has_drift,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "method": method
                }
            
            elif method == "wasserstein":
                # Wasserstein distance
                from scipy.stats import wasserstein_distance
                distance = wasserstein_distance(reference_data, current_data)
                
                # Threshold (ajustar según dominio)
                threshold = np.std(reference_data) * 2
                has_drift = distance > threshold
                
                return {
                    "has_drift": has_drift,
                    "distance": float(distance),
                    "threshold": float(threshold),
                    "method": method
                }
            
            else:
                raise ValueError(f"Método {method} no soportado")
                
        except Exception as e:
            logger.error(f"Error detectando drift: {e}")
            return {"has_drift": False, "error": str(e)}
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Detecta drift en predicciones"""
        # Calcular distribución de predicciones
        ref_dist = np.histogram(reference_predictions, bins=20)[0]
        curr_dist = np.histogram(current_predictions, bins=20)[0]
        
        # Chi-square test
        chi2, p_value = stats.chisquare(curr_dist, ref_dist)
        has_drift = p_value < 0.05
        
        return {
            "has_drift": has_drift,
            "chi2": float(chi2),
            "p_value": float(p_value),
            "reference_mean": float(np.mean(reference_predictions)),
            "current_mean": float(np.mean(current_predictions)),
            "mean_difference": float(np.mean(current_predictions) - np.mean(reference_predictions))
        }
    
    def detect_concept_drift(
        self,
        reference_accuracy: float,
        current_accuracy: float,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Detecta concept drift basado en accuracy"""
        accuracy_drop = reference_accuracy - current_accuracy
        has_drift = accuracy_drop > threshold
        
        return {
            "has_drift": has_drift,
            "reference_accuracy": reference_accuracy,
            "current_accuracy": current_accuracy,
            "accuracy_drop": accuracy_drop,
            "threshold": threshold
        }




