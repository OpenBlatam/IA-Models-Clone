"""
Explainability Utils - Utilidades de Explicabilidad Avanzada
=============================================================

Utilidades para explicabilidad e interpretabilidad de modelos.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable

logger = logging.getLogger(__name__)

# Intentar importar bibliotecas opcionales
try:
    import matplotlib.pyplot as plt
    _has_matplotlib = True
except ImportError:
    _has_matplotlib = False
    logger.warning("matplotlib not available, plotting features will be limited")

try:
    import shap
    _has_shap = True
except ImportError:
    _has_shap = False
    logger.warning("SHAP not available, some explainability features will be limited")

logger = logging.getLogger(__name__)

# Intentar importar bibliotecas opcionales
try:
    import shap
    _has_shap = True
except ImportError:
    _has_shap = False
    logger.warning("SHAP not available, some explainability features will be limited")


class SHAPExplainer:
    """
    Wrapper para SHAP explainer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        device: str = "cuda"
    ):
        """
        Inicializar SHAP explainer.
        
        Args:
            model: Modelo a explicar
            background_data: Datos de fondo
            device: Dispositivo
        """
        if not _has_shap:
            raise ImportError("SHAP is required for SHAPExplainer")
        
        self.model = model.to(device)
        self.device = device
        
        # Convertir a numpy para SHAP
        self.background_data = background_data.cpu().numpy()
        
        # Crear función wrapper
        def model_wrapper(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x_tensor)
                return outputs.cpu().numpy()
        
        self.explainer = shap.DeepExplainer(model_wrapper, self.background_data)
    
    def explain(
        self,
        inputs: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explicar predicciones.
        
        Args:
            inputs: Inputs a explicar
            feature_names: Nombres de features (opcional)
            
        Returns:
            Explicaciones SHAP
        """
        inputs_np = inputs.cpu().numpy()
        shap_values = self.explainer.shap_values(inputs_np)
        
        return {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'base_value': self.explainer.expected_value
        }
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot summary de SHAP values.
        
        Args:
            shap_values: SHAP values
            feature_names: Nombres de features
            save_path: Ruta para guardar (opcional)
        """
        if not _has_shap:
            raise ImportError("SHAP is required for plotting")
        if not _has_matplotlib:
            raise ImportError("matplotlib is required for plotting")
        
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class LIMEExplainer:
    """
    LIME explainer para modelos.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda"
    ):
        """
        Inicializar LIME explainer.
        
        Args:
            model: Modelo a explicar
            device: Dispositivo
        """
        self.model = model.to(device)
        self.device = device
    
    def explain_instance(
        self,
        instance: torch.Tensor,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Explicar instancia.
        
        Args:
            instance: Instancia a explicar
            num_features: Número de features a mostrar
            num_samples: Número de muestras para LIME
            
        Returns:
            Explicación
        """
        # Implementación simplificada de LIME
        instance = instance.to(self.device)
        
        # Generar muestras perturbadas
        perturbations = torch.randn(num_samples, *instance.shape).to(self.device)
        perturbations = perturbations * 0.1 + instance
        
        # Predecir
        self.model.eval()
        with torch.no_grad():
            original_pred = self.model(instance.unsqueeze(0))
            perturbed_preds = self.model(perturbations)
        
        # Calcular importancia (simplificado)
        importance = torch.abs(perturbed_preds - original_pred).mean(dim=0)
        
        # Obtener top features
        top_indices = importance.argsort(descending=True)[:num_features]
        
        return {
            'importance': importance.cpu().numpy(),
            'top_features': top_indices.cpu().tolist(),
            'prediction': original_pred.cpu().numpy()
        }


class FeatureImportanceAnalyzer:
    """
    Analizador de importancia de features.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar analizador.
        
        Args:
            model: Modelo
        """
        self.model = model
    
    def permutation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        metric: Callable = None,
        n_repeats: int = 10
    ) -> Dict[int, float]:
        """
        Calcular importancia por permutación.
        
        Args:
            X: Features
            y: Targets
            metric: Función de métrica
            n_repeats: Número de repeticiones
            
        Returns:
            Importancia por feature
        """
        if metric is None:
            def metric(y_true, y_pred):
                return (y_true == y_pred).float().mean()
        
        self.model.eval()
        
        # Baseline
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_score = metric(y, baseline_pred.argmax(dim=1))
        
        # Importancia por feature
        importance = {}
        
        for feature_idx in range(X.shape[1]):
            scores = []
            
            for _ in range(n_repeats):
                X_permuted = X.clone()
                # Permutar feature
                perm_indices = torch.randperm(len(X))
                X_permuted[:, feature_idx] = X_permuted[perm_indices, feature_idx]
                
                with torch.no_grad():
                    pred = self.model(X_permuted)
                    score = metric(y, pred.argmax(dim=1))
                    scores.append(score.item())
            
            importance[feature_idx] = baseline_score - np.mean(scores)
        
        return importance
    
    def ablation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        metric: Callable = None
    ) -> Dict[int, float]:
        """
        Calcular importancia por ablación.
        
        Args:
            X: Features
            y: Targets
            metric: Función de métrica
            
        Returns:
            Importancia por feature
        """
        if metric is None:
            def metric(y_true, y_pred):
                return (y_true == y_pred).float().mean()
        
        self.model.eval()
        
        # Baseline
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_score = metric(y, baseline_pred.argmax(dim=1))
        
        # Importancia por feature
        importance = {}
        
        for feature_idx in range(X.shape[1]):
            X_ablated = X.clone()
            # Ablar feature (poner a cero)
            X_ablated[:, feature_idx] = 0
            
            with torch.no_grad():
                pred = self.model(X_ablated)
                score = metric(y, pred.argmax(dim=1))
                importance[feature_idx] = baseline_score - score
        
        return importance

