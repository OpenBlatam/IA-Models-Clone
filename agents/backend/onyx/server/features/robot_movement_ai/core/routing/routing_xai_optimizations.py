"""
Optimizaciones de Explainable AI (XAI) para Routing.

Este módulo implementa técnicas de explicabilidad para hacer
transparentes las decisiones del sistema de routing.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ExplanationMethod(Enum):
    """Métodos de explicación."""
    SHAP = "shap"
    LIME = "lime"
    GRADIENT_BASED = "gradient"
    ATTENTION = "attention"
    FEATURE_IMPORTANCE = "feature_importance"


@dataclass
class Explanation:
    """Explicación de una decisión."""
    route_id: str
    method: ExplanationMethod
    feature_contributions: Dict[str, float]
    decision_factors: List[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class FeatureImportance:
    """Importancia de características."""
    feature_name: str
    importance_score: float
    contribution: float
    direction: str  # "positive" or "negative"


class SHAPExplainer:
    """Explicador SHAP (SHapley Additive exPlanations)."""
    
    def __init__(self):
        self.explanations: List[Explanation] = []
        self.feature_importance_history: Dict[str, List[float]] = {}
    
    def explain(self, route: Dict[str, Any], model_output: float,
               features: Dict[str, float]) -> Explanation:
        """Explicar decisión usando SHAP."""
        # Simulación de valores SHAP
        feature_contributions = {}
        total_contribution = 0.0
        
        for feature_name, feature_value in features.items():
            # Calcular contribución SHAP (simplificado)
            contribution = feature_value * np.random.uniform(-1, 1)
            feature_contributions[feature_name] = contribution
            total_contribution += abs(contribution)
        
        # Normalizar
        if total_contribution > 0:
            feature_contributions = {
                k: v / total_contribution 
                for k, v in feature_contributions.items()
            }
        
        # Identificar factores de decisión
        decision_factors = []
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for feature_name, contribution in sorted_features[:3]:
            direction = "increased" if contribution > 0 else "decreased"
            decision_factors.append(
                f"{feature_name} {direction} route cost by {abs(contribution):.2%}"
            )
        
        explanation = Explanation(
            route_id=route.get("route_id", "unknown"),
            method=ExplanationMethod.SHAP,
            feature_contributions=feature_contributions,
            decision_factors=decision_factors,
            confidence=0.85
        )
        
        self.explanations.append(explanation)
        
        # Actualizar historial
        for feature_name, contribution in feature_contributions.items():
            if feature_name not in self.feature_importance_history:
                self.feature_importance_history[feature_name] = []
            self.feature_importance_history[feature_name].append(abs(contribution))
        
        return explanation
    
    def get_feature_importance(self) -> List[FeatureImportance]:
        """Obtener importancia promedio de características."""
        importances = []
        
        for feature_name, history in self.feature_importance_history.items():
            avg_importance = np.mean(history) if history else 0.0
            importances.append(FeatureImportance(
                feature_name=feature_name,
                importance_score=avg_importance,
                contribution=avg_importance,
                direction="positive" if avg_importance > 0 else "negative"
            ))
        
        return sorted(importances, key=lambda x: x.importance_score, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "method": "shap",
            "total_explanations": len(self.explanations),
            "features_tracked": len(self.feature_importance_history)
        }


class LIMEExplainer:
    """Explicador LIME (Local Interpretable Model-agnostic Explanations)."""
    
    def __init__(self):
        self.explanations: List[Explanation] = []
        self.local_models: Dict[str, Any] = {}
    
    def explain(self, route: Dict[str, Any], model_output: float,
               features: Dict[str, float], num_samples: int = 100) -> Explanation:
        """Explicar decisión usando LIME."""
        # Generar muestras locales
        samples = []
        for _ in range(num_samples):
            sample = {}
            for feature_name in features:
                # Perturbar características
                sample[feature_name] = features[feature_name] + np.random.normal(0, 0.1)
            samples.append(sample)
        
        # Entrenar modelo local (simplificado)
        feature_contributions = {}
        for feature_name in features:
            # Calcular correlación con output
            contributions = [s[feature_name] for s in samples]
            correlation = np.corrcoef(contributions, [model_output] * len(contributions))[0, 1]
            feature_contributions[feature_name] = correlation if not np.isnan(correlation) else 0.0
        
        # Normalizar
        total = sum(abs(v) for v in feature_contributions.values())
        if total > 0:
            feature_contributions = {k: v / total for k, v in feature_contributions.items()}
        
        # Factores de decisión
        decision_factors = []
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for feature_name, contribution in sorted_features[:3]:
            direction = "positive" if contribution > 0 else "negative"
            decision_factors.append(
                f"{feature_name} has {direction} local impact"
            )
        
        explanation = Explanation(
            route_id=route.get("route_id", "unknown"),
            method=ExplanationMethod.LIME,
            feature_contributions=feature_contributions,
            decision_factors=decision_factors,
            confidence=0.80
        )
        
        self.explanations.append(explanation)
        return explanation
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "method": "lime",
            "total_explanations": len(self.explanations),
            "local_models": len(self.local_models)
        }


class AttentionExplainer:
    """Explicador basado en atención."""
    
    def __init__(self):
        self.attention_weights: Dict[str, np.ndarray] = {}
        self.explanations: List[Explanation] = []
    
    def explain(self, route: Dict[str, Any], attention_weights: np.ndarray,
               nodes: List[Dict[str, Any]]) -> Explanation:
        """Explicar usando pesos de atención."""
        # Convertir pesos de atención a contribuciones
        feature_contributions = {}
        
        for i, node in enumerate(nodes):
            node_id = node.get("node_id", f"node_{i}")
            weight = attention_weights[i] if i < len(attention_weights) else 0.0
            feature_contributions[node_id] = float(weight)
        
        # Normalizar
        total = sum(feature_contributions.values())
        if total > 0:
            feature_contributions = {k: v / total for k, v in feature_contributions.items()}
        
        # Factores de decisión
        decision_factors = []
        sorted_nodes = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for node_id, weight in sorted_nodes[:3]:
            decision_factors.append(
                f"Node {node_id} has attention weight {weight:.3f}"
            )
        
        explanation = Explanation(
            route_id=route.get("route_id", "unknown"),
            method=ExplanationMethod.ATTENTION,
            feature_contributions=feature_contributions,
            decision_factors=decision_factors,
            confidence=0.90
        )
        
        self.explanations.append(explanation)
        self.attention_weights[route.get("route_id", "unknown")] = attention_weights
        
        return explanation
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "method": "attention",
            "total_explanations": len(self.explanations),
            "attention_weights_tracked": len(self.attention_weights)
        }


class XAIOptimizer:
    """Optimizador principal de XAI."""
    
    def __init__(self, method: ExplanationMethod = ExplanationMethod.SHAP,
                 enable_xai: bool = True):
        self.enable_xai = enable_xai
        self.method = method
        
        if method == ExplanationMethod.SHAP:
            self.explainer = SHAPExplainer()
        elif method == ExplanationMethod.LIME:
            self.explainer = LIMEExplainer()
        elif method == ExplanationMethod.ATTENTION:
            self.explainer = AttentionExplainer()
        else:
            self.explainer = SHAPExplainer()
        
        self.explanations_generated = 0
        self.explanation_cache: Dict[str, Explanation] = {}
    
    def explain_route(self, route: Dict[str, Any], model_output: float,
                     features: Dict[str, float], 
                     attention_weights: Optional[np.ndarray] = None,
                     nodes: Optional[List[Dict[str, Any]]] = None) -> Optional[Explanation]:
        """Explicar decisión de ruta."""
        if not self.enable_xai:
            return None
        
        route_id = route.get("route_id", "unknown")
        
        # Verificar cache
        if route_id in self.explanation_cache:
            return self.explanation_cache[route_id]
        
        try:
            if isinstance(self.explainer, AttentionExplainer) and attention_weights is not None and nodes:
                explanation = self.explainer.explain(route, attention_weights, nodes)
            else:
                explanation = self.explainer.explain(route, model_output, features)
            
            self.explanations_generated += 1
            self.explanation_cache[route_id] = explanation
            
            return explanation
        except Exception as e:
            logger.warning(f"XAI explanation failed: {e}")
            return None
    
    def get_feature_importance(self) -> List[FeatureImportance]:
        """Obtener importancia de características."""
        if not self.enable_xai or not isinstance(self.explainer, SHAPExplainer):
            return []
        
        return self.explainer.get_feature_importance()
    
    def get_explanation_summary(self, route_id: str) -> Dict[str, Any]:
        """Obtener resumen de explicación."""
        if route_id not in self.explanation_cache:
            return {}
        
        explanation = self.explanation_cache[route_id]
        
        return {
            "route_id": route_id,
            "method": explanation.method.value,
            "top_factors": explanation.decision_factors[:3],
            "confidence": explanation.confidence,
            "timestamp": explanation.timestamp
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        if not self.enable_xai:
            return {
                "xai_enabled": False
            }
        
        stats = self.explainer.get_stats()
        stats["xai_enabled"] = True
        stats["method"] = self.method.value
        stats["explanations_generated"] = self.explanations_generated
        stats["cached_explanations"] = len(self.explanation_cache)
        
        return stats


