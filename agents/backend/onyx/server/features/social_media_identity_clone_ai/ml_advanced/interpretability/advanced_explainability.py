"""
Explicabilidad avanzada con SHAP y LIME
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP explainer (simplificado)"""
    
    def __init__(self, model: nn.Module, background_data: torch.Tensor):
        self.model = model
        self.background_data = background_data
        self.model.eval()
    
    def explain(
        self,
        instance: torch.Tensor,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Explica predicción usando SHAP
        
        Args:
            instance: Instancia a explicar
            num_samples: Número de muestras
            
        Returns:
            SHAP values
        """
        # SHAP simplificado usando sampling
        shap_values = []
        
        with torch.no_grad():
            # Predicción base (promedio sobre background)
            base_prediction = self._predict_batch(self.background_data).mean()
            
            # Predicción de instancia
            instance_prediction = self._predict(instance)
            
            # Calcular contribuciones aproximadas
            for i in range(instance.size(0)):
                # Crear instancia con feature i removida
                masked_instance = instance.clone()
                masked_instance[i] = self.background_data[:, i].mean()
                
                masked_prediction = self._predict(masked_instance)
                contribution = instance_prediction - masked_prediction
                shap_values.append(contribution.item())
        
        return {
            "shap_values": shap_values,
            "base_value": base_prediction.item(),
            "prediction": instance_prediction.item()
        }
    
    def _predict(self, instance: torch.Tensor) -> torch.Tensor:
        """Predicción de instancia"""
        with torch.no_grad():
            if instance.dim() == 1:
                instance = instance.unsqueeze(0)
            outputs = self.model(instance)
            if hasattr(outputs, 'logits'):
                return torch.softmax(outputs.logits, dim=-1).max()
            return outputs.max()
    
    def _predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Predicción de batch"""
        with torch.no_grad():
            outputs = self.model(batch)
            if hasattr(outputs, 'logits'):
                return torch.softmax(outputs.logits, dim=-1).max(dim=1)[0]
            return outputs.max(dim=1)[0]


class LIMEExplainer:
    """LIME explainer (simplificado)"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    def explain(
        self,
        instance: torch.Tensor,
        num_samples: int = 1000,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explica predicción usando LIME
        
        Args:
            instance: Instancia a explicar
            num_samples: Número de muestras
            num_features: Número de features a explicar
            
        Returns:
            LIME explanation
        """
        # Generar muestras perturbadas
        perturbations = self._generate_perturbations(instance, num_samples)
        
        # Predecir en muestras perturbadas
        predictions = []
        for pert in perturbations:
            pred = self._predict(pert)
            predictions.append(pred.item())
        
        predictions = np.array(predictions)
        
        # Calcular pesos (distancia a instancia original)
        distances = np.linalg.norm(perturbations - instance.cpu().numpy(), axis=1)
        weights = np.exp(-distances ** 2 / (2 * 0.5 ** 2))  # Kernel gaussiano
        
        # Ajustar modelo lineal local
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(perturbations, predictions, sample_weight=weights)
        
        # Obtener importancia de features
        feature_importance = np.abs(model.coef_)
        top_features = np.argsort(feature_importance)[-num_features:][::-1]
        
        return {
            "feature_importance": feature_importance.tolist(),
            "top_features": top_features.tolist(),
            "intercept": float(model.intercept_),
            "prediction": float(self._predict(instance).item())
        }
    
    def _generate_perturbations(
        self,
        instance: torch.Tensor,
        num_samples: int
    ) -> np.ndarray:
        """Genera muestras perturbadas"""
        instance_np = instance.cpu().numpy()
        perturbations = []
        
        for _ in range(num_samples):
            # Perturbación aleatoria
            noise = np.random.normal(0, 0.1, instance_np.shape)
            perturbation = instance_np + noise
            perturbations.append(perturbation)
        
        return np.array(perturbations)
    
    def _predict(self, instance: torch.Tensor) -> torch.Tensor:
        """Predicción"""
        with torch.no_grad():
            if instance.dim() == 1:
                instance = instance.unsqueeze(0)
            outputs = self.model(instance)
            if hasattr(outputs, 'logits'):
                return torch.softmax(outputs.logits, dim=-1).max()
            return outputs.max()




