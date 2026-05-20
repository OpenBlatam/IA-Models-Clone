"""
ML Predictor - Sistema de machine learning para predicciones
============================================================
"""

import logging
from typing import Dict, List, Any, Optional
import math

logger = logging.getLogger(__name__)


class MLPredictor:
    """Sistema de predicciones usando ML (simulado)"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.training_data: List[Dict[str, Any]] = []
    
    def predict_cost(self, product_type: str, num_parts: int, 
                    complexity: str, materials_count: int) -> Dict[str, Any]:
        """
        Predice el costo de un prototipo
        
        En producción, esto usaría un modelo ML entrenado
        """
        # Modelo simple basado en reglas
        base_cost = 50.0
        
        # Factores
        type_multiplier = {
            "licuadora": 1.0,
            "estufa": 1.5,
            "maquina": 2.0,
            "electrodomestico": 1.3,
            "herramienta": 1.2
        }.get(product_type.lower(), 1.0)
        
        complexity_multiplier = {
            "fácil": 0.8,
            "media": 1.0,
            "difícil": 1.5
        }.get(complexity.lower(), 1.0)
        
        parts_factor = num_parts * 10
        materials_factor = materials_count * 5
        
        predicted_cost = (
            base_cost * type_multiplier * complexity_multiplier +
            parts_factor + materials_factor
        )
        
        # Rango de confianza (simulado)
        confidence = 0.75
        lower_bound = predicted_cost * 0.8
        upper_bound = predicted_cost * 1.2
        
        return {
            "predicted_cost": round(predicted_cost, 2),
            "confidence": confidence,
            "range": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2)
            },
            "factors": {
                "product_type": type_multiplier,
                "complexity": complexity_multiplier,
                "parts": num_parts,
                "materials": materials_count
            }
        }
    
    def predict_build_time(self, num_parts: int, complexity: str,
                          num_steps: int) -> Dict[str, Any]:
        """Predice el tiempo de construcción"""
        base_time = 60  # minutos
        
        complexity_multiplier = {
            "fácil": 0.7,
            "media": 1.0,
            "difícil": 1.5
        }.get(complexity.lower(), 1.0)
        
        predicted_minutes = (
            base_time * complexity_multiplier +
            num_parts * 15 +
            num_steps * 10
        )
        
        predicted_hours = predicted_minutes / 60
        
        return {
            "predicted_minutes": round(predicted_minutes, 0),
            "predicted_hours": round(predicted_hours, 1),
            "formatted": f"{int(predicted_hours)}-{int(predicted_hours) + 1} horas",
            "confidence": 0.70
        }
    
    def predict_feasibility(self, cost: float, complexity: str,
                           user_experience: str) -> Dict[str, Any]:
        """Predice la viabilidad"""
        # Score base
        score = 100.0
        
        # Reducir por costo
        if cost > 500:
            score -= 20
        elif cost > 300:
            score -= 10
        
        # Reducir por complejidad
        if complexity.lower() == "difícil":
            score -= 15
        elif complexity.lower() == "media":
            score -= 5
        
        # Ajustar por experiencia
        experience_multiplier = {
            "principiante": 0.7,
            "intermedio": 1.0,
            "avanzado": 1.2
        }.get(user_experience.lower(), 1.0)
        
        score *= experience_multiplier
        score = max(0, min(100, score))
        
        if score >= 80:
            level = "Muy Alta"
        elif score >= 60:
            level = "Alta"
        elif score >= 40:
            level = "Media"
        elif score >= 20:
            level = "Baja"
        else:
            level = "Muy Baja"
        
        return {
            "predicted_score": round(score, 1),
            "predicted_level": level,
            "confidence": 0.65,
            "factors": {
                "cost_impact": cost > 300,
                "complexity_impact": complexity.lower() == "difícil",
                "experience_impact": user_experience.lower() == "principiante"
            }
        }
    
    def recommend_optimizations(self, prototype_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recomienda optimizaciones usando ML"""
        recommendations = []
        
        cost = prototype_data.get("total_cost_estimate", 0)
        materials = prototype_data.get("materials", [])
        
        # Análisis de materiales costosos
        if materials:
            avg_cost = sum(m.get("total_price", 0) for m in materials) / len(materials)
            expensive_materials = [
                m for m in materials
                if m.get("total_price", 0) > avg_cost * 2
            ]
            
            if expensive_materials:
                recommendations.append({
                    "type": "cost_optimization",
                    "priority": "high",
                    "message": f"Considera alternativas para {len(expensive_materials)} materiales costosos",
                    "potential_savings": sum(m.get("total_price", 0) * 0.2 for m in expensive_materials)
                })
        
        # Análisis de complejidad
        num_parts = len(prototype_data.get("cad_parts", []))
        if num_parts > 10:
            recommendations.append({
                "type": "complexity_reduction",
                "priority": "medium",
                "message": "El prototipo tiene muchas partes. Considera simplificar el diseño",
                "suggestion": "Combinar partes relacionadas puede reducir costos y tiempo"
            })
        
        return recommendations
    
    def train_model(self, training_data: List[Dict[str, Any]]):
        """Entrena un modelo (simulado)"""
        self.training_data = training_data
        logger.info(f"Modelo entrenado con {len(training_data)} ejemplos")
        # En producción, aquí se entrenaría un modelo real (scikit-learn, tensorflow, etc.)




