"""
Model Interpretability Service - Interpretabilidad de modelos
==============================================================

Sistema para interpretar y explicar predicciones de modelos.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Try to import interpretability libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available")


@dataclass
class FeatureImportance:
    """Importancia de features"""
    feature_name: str
    importance: float
    contribution: float


@dataclass
class ModelExplanation:
    """Explicación del modelo"""
    prediction: Any
    feature_importances: List[FeatureImportance]
    shap_values: Optional[np.ndarray] = None
    explanation_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelInterpretabilityService:
    """Servicio de interpretabilidad de modelos"""
    
    def __init__(self):
        """Inicializar servicio"""
        logger.info("ModelInterpretabilityService initialized")
    
    def calculate_feature_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        method: str = "permutation"
    ) -> List[FeatureImportance]:
        """Calcular importancia de features"""
        if method == "permutation":
            # Permutation importance
            baseline_score = self._predict_and_score(model, X)
            
            importances = []
            for i in range(X.shape[1]):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_score = self._predict_and_score(model, X_permuted)
                
                importance = baseline_score - permuted_score
                feature_name = feature_names[i] if feature_names else f"feature_{i}"
                importances.append(FeatureImportance(
                    feature_name=feature_name,
                    importance=importance,
                    contribution=importance
                ))
            
            return sorted(importances, key=lambda x: abs(x.importance), reverse=True)
        
        elif method == "gradient":
            # Gradient-based importance
            if not TORCH_AVAILABLE:
                return []
            
            importances = []
            X_tensor = torch.FloatTensor(X)
            X_tensor.requires_grad = True
            
            output = model(X_tensor)
            output.backward(torch.ones_like(output))
            
            gradients = X_tensor.grad.numpy()
            
            for i in range(X.shape[1]):
                importance = np.mean(np.abs(gradients[:, i]))
                feature_name = feature_names[i] if feature_names else f"feature_{i}"
                importances.append(FeatureImportance(
                    feature_name=feature_name,
                    importance=float(importance),
                    contribution=float(importance)
                ))
            
            return sorted(importances, key=lambda x: abs(x.importance), reverse=True)
        
        return []
    
    def explain_prediction_shap(
        self,
        model: Any,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        instance_idx: int = 0
    ) -> ModelExplanation:
        """Explicar predicción usando SHAP"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using fallback")
            return self._explain_fallback(model, X, instance_idx)
        
        try:
            # Use background data or sample
            if X_background is None:
                X_background = X[:min(100, len(X))]
            
            # Create explainer
            explainer = shap.Explainer(model, X_background)
            
            # Explain instance
            shap_values = explainer(X[instance_idx:instance_idx+1])
            
            # Get prediction
            prediction = model.predict(X[instance_idx:instance_idx+1])[0]
            
            # Extract feature importances
            feature_importances = [
                FeatureImportance(
                    feature_name=f"feature_{i}",
                    importance=float(shap_values.values[0, i]),
                    contribution=float(shap_values.values[0, i])
                )
                for i in range(X.shape[1])
            ]
            
            return ModelExplanation(
                prediction=prediction,
                feature_importances=sorted(feature_importances, key=lambda x: abs(x.importance), reverse=True),
                shap_values=shap_values.values,
            )
            
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            return self._explain_fallback(model, X, instance_idx)
    
    def _explain_fallback(
        self,
        model: Any,
        X: np.ndarray,
        instance_idx: int
    ) -> ModelExplanation:
        """Fallback explanation method"""
        prediction = model.predict(X[instance_idx:instance_idx+1])[0]
        
        # Simple feature importance
        feature_importances = [
            FeatureImportance(
                feature_name=f"feature_{i}",
                importance=float(X[instance_idx, i]),
                contribution=float(X[instance_idx, i])
            )
            for i in range(X.shape[1])
        ]
        
        return ModelExplanation(
            prediction=prediction,
            feature_importances=feature_importances,
        )
    
    def _predict_and_score(self, model: Any, X: np.ndarray) -> float:
        """Helper para predecir y obtener score"""
        try:
            predictions = model.predict(X)
            # Return mean prediction as score
            return float(np.mean(predictions))
        except:
            return 0.0
    
    def generate_explanation_text(
        self,
        explanation: ModelExplanation,
        top_k: int = 5
    ) -> str:
        """Generar texto explicativo"""
        top_features = explanation.feature_importances[:top_k]
        
        text = f"Prediction: {explanation.prediction}\n\n"
        text += "Top contributing features:\n"
        
        for i, feature in enumerate(top_features, 1):
            text += f"{i}. {feature.feature_name}: {feature.contribution:.4f}\n"
        
        return text




