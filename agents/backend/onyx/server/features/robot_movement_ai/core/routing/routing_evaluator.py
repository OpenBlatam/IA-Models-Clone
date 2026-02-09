"""
Routing Model Evaluator
=======================

Sistema de evaluación profesional para modelos de routing.
Implementa métricas estándar y visualizaciones.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Evaluation features will be disabled.")


@dataclass
class EvaluationMetrics:
    """Métricas de evaluación."""
    loss: float
    mse: float
    mae: float
    rmse: float
    r2_score: float
    mean_absolute_percentage_error: float
    timestamp: float = field(default_factory=time.time)
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class ModelEvaluator:
    """Evaluador profesional para modelos de routing."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Inicializar evaluador.
        
        Args:
            model: Modelo PyTorch a evaluar
            device: Dispositivo (CPU/GPU)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelEvaluator")
        
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> EvaluationMetrics:
        """
        Evaluar modelo en un DataLoader.
        
        Args:
            data_loader: DataLoader con datos de evaluación
            criterion: Función de pérdida (usa MSE si None)
        
        Returns:
            EvaluationMetrics con todas las métricas
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = [b.to(self.device) for b in batch[:-1]]
                    targets = batch[-1].to(self.device)
                else:
                    inputs = batch.get('features', batch.get('input')).to(self.device)
                    targets = batch.get('target', batch.get('labels')).to(self.device)
                
                # Forward pass
                if isinstance(inputs, list):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                
                # Loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Guardar predicciones y targets
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenar todas las predicciones
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calcular métricas
        metrics = self._compute_metrics(predictions, targets, total_loss / num_batches)
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        loss: float
    ) -> EvaluationMetrics:
        """Calcular todas las métricas."""
        # Flatten si es necesario
        pred_flat = predictions.flatten()
        targ_flat = targets.flatten()
        
        # MSE (ya calculado como loss)
        mse = np.mean((pred_flat - targ_flat) ** 2)
        
        # MAE
        mae = np.mean(np.abs(pred_flat - targ_flat))
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # R² Score
        ss_res = np.sum((targ_flat - pred_flat) ** 2)
        ss_tot = np.sum((targ_flat - np.mean(targ_flat)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-8))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((targ_flat - pred_flat) / (targ_flat + 1e-8))) * 100
        
        return EvaluationMetrics(
            loss=loss,
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2_score=r2_score,
            mean_absolute_percentage_error=mape
        )
    
    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        """
        Hacer predicciones.
        
        Args:
            inputs: Tensor de entrada
        
        Returns:
            Array de predicciones
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            return outputs.cpu().numpy()
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, EvaluationMetrics]:
        """
        Comparar múltiples modelos.
        
        Args:
            models: Diccionario de modelos {name: model}
            data_loader: DataLoader con datos de evaluación
            criterion: Función de pérdida
        
        Returns:
            Diccionario de métricas por modelo
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            evaluator = ModelEvaluator(model, self.device)
            metrics = evaluator.evaluate(data_loader, criterion)
            results[name] = metrics
        
        return results


