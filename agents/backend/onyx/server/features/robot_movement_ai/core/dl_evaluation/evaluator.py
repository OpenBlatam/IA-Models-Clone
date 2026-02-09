"""
Model Evaluation Module
======================

Módulo modular para evaluación de modelos con métricas diversas.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluador de modelos con múltiples métricas.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_amp: bool = True
    ):
        """
        Inicializar evaluador.
        
        Args:
            model: Modelo PyTorch
            device: Dispositivo (CPU/GPU)
            use_amp: Usar mixed precision
        """
        self.model = model
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.use_amp = use_amp and torch.cuda.is_available()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluar modelo en dataset.
        
        Args:
            data_loader: DataLoader con datos
            metrics: Lista de métricas a calcular
            
        Returns:
            Diccionario con métricas
        """
        if metrics is None:
            metrics = ['mse', 'mae', 'r2']
        
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_to_device(batch)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self._predict_batch(batch)
                else:
                    predictions = self._predict_batch(batch)
                
                targets = batch.get('trajectory', batch.get('target'))
                
                if predictions is not None and targets is not None:
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
        
        if not all_predictions:
            logger.warning("No predictions generated")
            return {}
        
        # Concatenar todas las predicciones
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calcular métricas
        results = {}
        for metric in metrics:
            if metric == 'mse':
                results['mse'] = mean_squared_error(targets.flatten(), predictions.flatten())
            elif metric == 'mae':
                results['mae'] = mean_absolute_error(targets.flatten(), predictions.flatten())
            elif metric == 'r2':
                results['r2'] = r2_score(targets.flatten(), predictions.flatten())
            elif metric == 'rmse':
                mse = mean_squared_error(targets.flatten(), predictions.flatten())
                results['rmse'] = np.sqrt(mse)
            elif metric == 'accuracy':
                # Para clasificación
                pred_classes = np.argmax(predictions, axis=-1)
                true_classes = np.argmax(targets, axis=-1) if len(targets.shape) > 1 else targets
                results['accuracy'] = accuracy_score(true_classes, pred_classes)
        
        return results
    
    def _predict_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Predecir batch."""
        trajectory = batch.get('trajectory')
        if trajectory is None:
            raise ValueError("Batch must contain 'trajectory' key")
        
        if hasattr(self.model, 'forward'):
            return self.model(trajectory)
        else:
            raise ValueError("Model must have forward method")
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Mover batch a device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, (list, tuple)):
                device_batch[key] = [
                    v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                device_batch[key] = value
        return device_batch
    
    def predict(
        self,
        data_loader: DataLoader,
        return_targets: bool = False
    ) -> tuple:
        """
        Generar predicciones.
        
        Args:
            data_loader: DataLoader con datos
            return_targets: Retornar targets también
            
        Returns:
            Tuple de (predictions, targets) si return_targets=True, else predictions
        """
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = self._move_to_device(batch)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = self._predict_batch(batch)
                else:
                    predictions = self._predict_batch(batch)
                
                all_predictions.append(predictions.cpu().numpy())
                
                if return_targets:
                    targets = batch.get('trajectory', batch.get('target'))
                    if targets is not None:
                        all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        
        if return_targets:
            if all_targets:
                targets = np.concatenate(all_targets, axis=0)
                return predictions, targets
            else:
                return predictions, None
        else:
            return predictions
