"""
Ensemble Utils - Utilidades de Ensembles
=========================================

Utilidades para crear y gestionar ensembles de modelos.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable, Union
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelEnsemble(nn.Module):
    """
    Ensemble de modelos con diferentes estrategias de combinación.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        method: str = "average",
        weights: Optional[List[float]] = None
    ):
        """
        Inicializar ensemble.
        
        Args:
            models: Lista de modelos
            method: Método de combinación ('average', 'weighted', 'voting', 'stacking')
            weights: Pesos para weighted average (opcional)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        self.weights = weights
        
        if weights and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        if weights and method == "weighted":
            # Normalizar pesos
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del ensemble.
        
        Args:
            x: Inputs
            
        Returns:
            Predicciones combinadas
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        if self.method == "average":
            return predictions.mean(dim=0)
        
        elif self.method == "weighted":
            if self.weights is None:
                raise ValueError("Weights required for weighted method")
            weights_tensor = torch.tensor(self.weights, device=predictions.device)
            weights_tensor = weights_tensor.view(-1, 1, 1)
            return (predictions * weights_tensor).sum(dim=0)
        
        elif self.method == "voting":
            # Voting para clasificación
            class_predictions = predictions.argmax(dim=-1)
            return torch.mode(class_predictions, dim=0)[0]
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predecir probabilidades.
        
        Args:
            x: Inputs
            
        Returns:
            Probabilidades
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class StackingEnsemble:
    """
    Stacking ensemble con meta-learner.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: nn.Module
    ):
        """
        Inicializar stacking ensemble.
        
        Args:
            base_models: Modelos base
            meta_model: Meta-modelo
        """
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
    
    def fit_meta_model(
        self,
        X_train: DataLoader,
        y_train: torch.Tensor,
        X_val: DataLoader,
        y_val: torch.Tensor,
        epochs: int = 10
    ):
        """
        Entrenar meta-modelo.
        
        Args:
            X_train: DataLoader de entrenamiento
            y_train: Targets de entrenamiento
            X_val: DataLoader de validación
            y_val: Targets de validación
            epochs: Número de épocas
        """
        # Generar predicciones de modelos base
        base_predictions_train = self._get_base_predictions(X_train)
        base_predictions_val = self._get_base_predictions(X_val)
        
        # Entrenar meta-modelo
        optimizer = torch.optim.Adam(self.meta_model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.meta_model.train()
            optimizer.zero_grad()
            
            meta_outputs = self.meta_model(base_predictions_train)
            loss = criterion(meta_outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Validación
            self.meta_model.eval()
            with torch.no_grad():
                val_outputs = self.meta_model(base_predictions_val)
                val_loss = criterion(val_outputs, y_val)
            
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    def _get_base_predictions(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Obtener predicciones de modelos base.
        
        Args:
            dataloader: DataLoader
            
        Returns:
            Predicciones combinadas
        """
        all_predictions = []
        
        for model in self.base_models:
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    
                    pred = model(x)
                    predictions.append(pred)
            
            all_predictions.append(torch.cat(predictions))
        
        return torch.stack(all_predictions, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predecir con stacking.
        
        Args:
            x: Inputs
            
        Returns:
            Predicciones
        """
        # Predicciones de modelos base
        base_predictions = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                base_predictions.append(pred)
        
        base_predictions = torch.stack(base_predictions, dim=1)
        
        # Predicción del meta-modelo
        self.meta_model.eval()
        with torch.no_grad():
            return self.meta_model(base_predictions)


class BaggingEnsemble:
    """
    Bagging ensemble con bootstrap sampling.
    """
    
    def __init__(
        self,
        model_class: type,
        n_models: int = 5,
        model_kwargs: Optional[Dict] = None
    ):
        """
        Inicializar bagging ensemble.
        
        Args:
            model_class: Clase del modelo
            n_models: Número de modelos
            model_kwargs: Argumentos para el modelo
        """
        self.model_class = model_class
        self.n_models = n_models
        self.model_kwargs = model_kwargs or {}
        self.models = []
    
    def fit(
        self,
        dataloader: DataLoader,
        train_fn: Callable,
        epochs: int = 10
    ):
        """
        Entrenar modelos con bootstrap.
        
        Args:
            dataloader: DataLoader completo
            train_fn: Función de entrenamiento
            epochs: Número de épocas
        """
        # Obtener todos los datos
        all_data = []
        for batch in dataloader:
            all_data.append(batch)
        
        # Crear y entrenar modelos
        for i in range(self.n_models):
            logger.info(f"Training model {i + 1}/{self.n_models}")
            
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(all_data), size=len(all_data), replace=True)
            bootstrap_data = [all_data[idx] for idx in bootstrap_indices]
            
            # Crear modelo
            model = self.model_class(**self.model_kwargs)
            
            # Entrenar
            train_fn(model, bootstrap_data, epochs)
            
            self.models.append(model)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predecir con bagging.
        
        Args:
            x: Inputs
            
        Returns:
            Predicciones promedio
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        return predictions.mean(dim=0)


def create_ensemble(
    models: List[nn.Module],
    method: str = "average",
    weights: Optional[List[float]] = None
) -> ModelEnsemble:
    """
    Crear ensemble de modelos.
    
    Args:
        models: Lista de modelos
        method: Método de combinación
        weights: Pesos (opcional)
        
    Returns:
        ModelEnsemble
    """
    return ModelEnsemble(models, method, weights)




