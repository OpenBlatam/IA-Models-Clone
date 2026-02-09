"""
Training Metrics
================

Cálculo de métricas para entrenamiento y evaluación.
"""

import torch
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RouteMetrics:
    """Métricas de ruta."""
    mse: float
    mae: float
    rmse: float
    r2: float
    mape: float


class MetricsCalculator:
    """
    Calculador de métricas para modelos de enrutamiento.
    """
    
    @staticmethod
    def calculate(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calcular métricas.
        
        Args:
            predictions: Predicciones [batch_size, output_dim]
            targets: Targets [batch_size, output_dim]
            
        Returns:
            Diccionario de métricas
        """
        # Convertir a numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # MSE
        mse = np.mean((pred_np - target_np) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_np - target_np))
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # R²
        ss_res = np.sum((target_np - pred_np) ** 2)
        ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((target_np - pred_np) / (target_np + 1e-8))) * 100
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape)
        }
    
    @staticmethod
    def calculate_per_output(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Calcular métricas por cada output.
        
        Args:
            predictions: Predicciones [batch_size, output_dim]
            targets: Targets [batch_size, output_dim]
            
        Returns:
            Diccionario de métricas por output
        """
        output_names = ["time", "cost", "load", "probability"]
        metrics_per_output = {}
        
        for i, name in enumerate(output_names):
            pred_col = predictions[:, i]
            target_col = targets[:, i]
            
            metrics = MetricsCalculator.calculate(
                pred_col.unsqueeze(1),
                target_col.unsqueeze(1)
            )
            
            metrics_per_output[name] = metrics
        
        return metrics_per_output


