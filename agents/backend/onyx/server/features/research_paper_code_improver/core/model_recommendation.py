"""
Model Recommendation System - Sistema de recomendación de modelos
===================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Tipos de tarea"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"


@dataclass
class ModelRecommendation:
    """Recomendación de modelo"""
    model_name: str
    model_type: str
    confidence: float
    reasoning: str
    estimated_accuracy: float
    estimated_latency_ms: float
    use_cases: List[str] = field(default_factory=list)


class ModelRecommendationSystem:
    """Sistema de recomendación de modelos"""
    
    def __init__(self):
        self.recommendations: List[ModelRecommendation] = []
        self.model_database: Dict[str, Dict[str, Any]] = {
            "resnet18": {
                "type": "classification",
                "accuracy": 0.69,
                "latency_ms": 5.0,
                "size_mb": 45.0,
                "use_cases": ["Image classification", "Transfer learning"]
            },
            "bert-base": {
                "type": "text_classification",
                "accuracy": 0.85,
                "latency_ms": 50.0,
                "size_mb": 440.0,
                "use_cases": ["Text classification", "Sentiment analysis"]
            },
            "gpt2": {
                "type": "text_generation",
                "accuracy": 0.0,  # Perplexity-based
                "latency_ms": 100.0,
                "size_mb": 500.0,
                "use_cases": ["Text generation", "Language modeling"]
            }
        }
    
    def recommend_model(
        self,
        task_type: TaskType,
        constraints: Dict[str, Any],
        dataset_size: Optional[int] = None
    ) -> List[ModelRecommendation]:
        """Recomienda modelos"""
        recommendations = []
        
        # Filtrar modelos por tipo de tarea
        candidate_models = self._filter_by_task(task_type)
        
        # Aplicar constraints
        filtered_models = self._apply_constraints(candidate_models, constraints)
        
        # Generar recomendaciones
        for model_name, model_info in filtered_models.items():
            confidence = self._calculate_confidence(model_info, constraints, dataset_size)
            reasoning = self._generate_reasoning(model_info, constraints)
            
            recommendation = ModelRecommendation(
                model_name=model_name,
                model_type=model_info["type"],
                confidence=confidence,
                reasoning=reasoning,
                estimated_accuracy=model_info.get("accuracy", 0.0),
                estimated_latency_ms=model_info.get("latency_ms", 0.0),
                use_cases=model_info.get("use_cases", [])
            )
            
            recommendations.append(recommendation)
        
        # Ordenar por confianza
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        self.recommendations.extend(recommendations)
        return recommendations[:5]  # Top 5
    
    def _filter_by_task(self, task_type: TaskType) -> Dict[str, Dict[str, Any]]:
        """Filtra modelos por tipo de tarea"""
        task_mapping = {
            TaskType.CLASSIFICATION: ["classification"],
            TaskType.TEXT_CLASSIFICATION: ["text_classification"],
            TaskType.TEXT_GENERATION: ["text_generation"],
            TaskType.REGRESSION: ["regression", "classification"],
            TaskType.OBJECT_DETECTION: ["object_detection"],
            TaskType.SEGMENTATION: ["segmentation"]
        }
        
        allowed_types = task_mapping.get(task_type, [])
        
        return {
            name: info
            for name, info in self.model_database.items()
            if info["type"] in allowed_types
        }
    
    def _apply_constraints(
        self,
        models: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Aplica constraints"""
        filtered = {}
        
        for name, info in models.items():
            # Constraint de latencia
            if "max_latency_ms" in constraints:
                if info.get("latency_ms", float('inf')) > constraints["max_latency_ms"]:
                    continue
            
            # Constraint de tamaño
            if "max_size_mb" in constraints:
                if info.get("size_mb", float('inf')) > constraints["max_size_mb"]:
                    continue
            
            # Constraint de accuracy mínima
            if "min_accuracy" in constraints:
                if info.get("accuracy", 0.0) < constraints["min_accuracy"]:
                    continue
            
            filtered[name] = info
        
        return filtered
    
    def _calculate_confidence(
        self,
        model_info: Dict[str, Any],
        constraints: Dict[str, Any],
        dataset_size: Optional[int]
    ) -> float:
        """Calcula confianza en recomendación"""
        confidence = 0.5  # Base
        
        # Aumentar si cumple constraints
        if "max_latency_ms" in constraints:
            if model_info.get("latency_ms", 0) <= constraints["max_latency_ms"]:
                confidence += 0.2
        
        if "min_accuracy" in constraints:
            if model_info.get("accuracy", 0) >= constraints["min_accuracy"]:
                confidence += 0.2
        
        # Ajustar por tamaño de dataset
        if dataset_size:
            if dataset_size > 10000:
                confidence += 0.1
        
        return min(1.0, confidence)
    
    def _generate_reasoning(
        self,
        model_info: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Genera razonamiento para recomendación"""
        reasons = []
        
        if model_info.get("accuracy", 0) > 0.8:
            reasons.append("Alta precisión")
        
        if model_info.get("latency_ms", 0) < 50:
            reasons.append("Baja latencia")
        
        if model_info.get("size_mb", 0) < 100:
            reasons.append("Modelo compacto")
        
        return "; ".join(reasons) if reasons else "Modelo general"




