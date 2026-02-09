"""
Model Evaluator
===============

Evaluador avanzado de modelos con métricas completas.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuración de evaluación."""
    batch_size: int = 32
    device: Optional[str] = None
    compute_confidence_intervals: bool = True
    num_bootstrap_samples: int = 1000
    detailed_metrics: bool = True


@dataclass
class EvaluationMetrics:
    """Métricas de evaluación."""
    mse: float
    mae: float
    rmse: float
    r2: float
    mape: float
    per_output_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0


class ModelEvaluator:
    """
    Evaluador avanzado de modelos.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Inicializar evaluador.
        
        Args:
            config: Configuración (opcional)
        """
        self.config = config or EvaluationConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        compute_detailed: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluar modelo.
        
        Args:
            model: Modelo a evaluar
            dataloader: DataLoader de evaluación
            compute_detailed: Calcular métricas detalladas
            
        Returns:
            Métricas de evaluación
        """
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_targets = []
        inference_times = []
        
        import time
        
        with torch.no_grad():
            for batch_features, batch_targets, _ in dataloader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                # Medir tiempo de inferencia
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.time()
                predictions = model(batch_features)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(batch_targets.cpu())
        
        # Concatenar
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Calcular métricas básicas
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # Métricas por output
        if compute_detailed:
            metrics.per_output_metrics = self._calculate_per_output_metrics(
                all_predictions, all_targets
            )
        
        # Intervalos de confianza
        if self.config.compute_confidence_intervals:
            metrics.confidence_intervals = self._bootstrap_confidence_intervals(
                all_predictions, all_targets
            )
        
        # Métricas de rendimiento
        avg_inference_time = np.mean(inference_times)
        batch_size = dataloader.batch_size
        metrics.inference_time_ms = avg_inference_time
        metrics.throughput_samples_per_sec = (batch_size / avg_inference_time) * 1000 if avg_inference_time > 0 else 0
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> EvaluationMetrics:
        """Calcular métricas básicas."""
        # MSE
        mse = np.mean((predictions - targets) ** 2)
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # MAPE
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        return EvaluationMetrics(
            mse=float(mse),
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
            mape=float(mape)
        )
    
    def _calculate_per_output_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calcular métricas por cada output."""
        output_names = ["time", "cost", "load", "probability"]
        metrics_per_output = {}
        
        for i, name in enumerate(output_names):
            pred_col = predictions[:, i]
            target_col = targets[:, i]
            
            mse = np.mean((pred_col - target_col) ** 2)
            mae = np.mean(np.abs(pred_col - target_col))
            rmse = np.sqrt(mse)
            
            ss_res = np.sum((target_col - pred_col) ** 2)
            ss_tot = np.sum((target_col - np.mean(target_col)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            metrics_per_output[name] = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2)
            }
        
        return metrics_per_output
    
    def _bootstrap_confidence_intervals(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calcular intervalos de confianza usando bootstrap.
        
        Args:
            predictions: Predicciones
            targets: Targets
            confidence: Nivel de confianza
            
        Returns:
            Diccionario con intervalos de confianza
        """
        n_samples = len(predictions)
        n_bootstrap = self.config.num_bootstrap_samples
        
        # Bootstrap samples de R²
        r2_samples = []
        mse_samples = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            pred_boot = predictions[indices]
            target_boot = targets[indices]
            
            # R²
            ss_res = np.sum((target_boot - pred_boot) ** 2)
            ss_tot = np.sum((target_boot - np.mean(target_boot)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_samples.append(r2)
            
            # MSE
            mse = np.mean((pred_boot - target_boot) ** 2)
            mse_samples.append(mse)
        
        # Calcular percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            "r2": (
                np.percentile(r2_samples, lower_percentile),
                np.percentile(r2_samples, upper_percentile)
            ),
            "mse": (
                np.percentile(mse_samples, lower_percentile),
                np.percentile(mse_samples, upper_percentile)
            )
        }
    
    def compare_with_baseline(
        self,
        model_metrics: EvaluationMetrics,
        baseline_metrics: EvaluationMetrics
    ) -> Dict[str, Any]:
        """
        Comparar modelo con baseline.
        
        Args:
            model_metrics: Métricas del modelo
            baseline_metrics: Métricas del baseline
            
        Returns:
            Comparación
        """
        return {
            "r2_improvement": model_metrics.r2 - baseline_metrics.r2,
            "mse_reduction": (baseline_metrics.mse - model_metrics.mse) / baseline_metrics.mse,
            "mae_reduction": (baseline_metrics.mae - model_metrics.mae) / baseline_metrics.mae,
            "relative_improvement": {
                "r2": (model_metrics.r2 - baseline_metrics.r2) / abs(baseline_metrics.r2) if baseline_metrics.r2 != 0 else 0,
                "mse": (baseline_metrics.mse - model_metrics.mse) / baseline_metrics.mse if baseline_metrics.mse != 0 else 0
            }
        }

