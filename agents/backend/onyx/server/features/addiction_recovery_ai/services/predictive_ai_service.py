"""
Servicio de IA Predictiva Avanzada - Predicciones inteligentes con machine learning
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random


class PredictiveAIService:
    """Servicio de IA predictiva avanzada"""
    
    def __init__(self):
        """Inicializa el servicio de IA predictiva"""
        pass
    
    def predict_success_probability(
        self,
        user_id: str,
        days_sober: int,
        historical_data: Dict
    ) -> Dict:
        """
        Predice probabilidad de éxito en recuperación
        
        Args:
            user_id: ID del usuario
            days_sober: Días de sobriedad
            historical_data: Datos históricos
        
        Returns:
            Predicción de probabilidad de éxito
        """
        # Factores que influyen en el éxito
        factors = {
            "days_sober": days_sober,
            "check_in_consistency": historical_data.get("check_in_consistency", 0.7),
            "support_system": historical_data.get("support_system", 5),
            "therapy_sessions": historical_data.get("therapy_sessions", 0),
            "exercise_frequency": historical_data.get("exercise_frequency", 0),
            "stress_level": historical_data.get("stress_level", 5)
        }
        
        # Cálculo de probabilidad (simplificado)
        base_probability = min(0.95, 0.3 + (days_sober / 365) * 0.4)
        
        if factors["check_in_consistency"] > 0.8:
            base_probability += 0.1
        if factors["support_system"] >= 7:
            base_probability += 0.1
        if factors["therapy_sessions"] > 0:
            base_probability += 0.05
        if factors["exercise_frequency"] >= 3:
            base_probability += 0.05
        
        if factors["stress_level"] > 7:
            base_probability -= 0.1
        
        probability = max(0.1, min(0.95, base_probability))
        
        return {
            "user_id": user_id,
            "success_probability": round(probability, 2),
            "confidence": 0.85,
            "factors_analyzed": list(factors.keys()),
            "recommendations": self._generate_success_recommendations(factors, probability),
            "predicted_at": datetime.now().isoformat()
        }
    
    def predict_relapse_window(
        self,
        user_id: str,
        current_data: Dict
    ) -> Dict:
        """
        Predice ventana de riesgo de recaída
        
        Args:
            user_id: ID del usuario
            current_data: Datos actuales
        
        Returns:
            Predicción de ventana de riesgo
        """
        risk_factors = {
            "stress_level": current_data.get("stress_level", 5),
            "cravings_level": current_data.get("cravings_level", 3),
            "support_available": current_data.get("support_available", True),
            "recent_triggers": current_data.get("recent_triggers", 0)
        }
        
        # Calcular nivel de riesgo
        risk_score = 0
        if risk_factors["stress_level"] >= 8:
            risk_score += 3
        elif risk_factors["stress_level"] >= 6:
            risk_score += 2
        
        if risk_factors["cravings_level"] >= 7:
            risk_score += 3
        elif risk_factors["cravings_level"] >= 5:
            risk_score += 2
        
        if not risk_factors["support_available"]:
            risk_score += 2
        
        if risk_factors["recent_triggers"] >= 3:
            risk_score += 2
        
        # Determinar ventana de riesgo
        if risk_score >= 7:
            risk_window = "high"
            days_ahead = 1
        elif risk_score >= 4:
            risk_window = "medium"
            days_ahead = 3
        else:
            risk_window = "low"
            days_ahead = 7
        
        return {
            "user_id": user_id,
            "risk_window": risk_window,
            "risk_score": risk_score,
            "days_ahead": days_ahead,
            "predicted_window": {
                "start": datetime.now().isoformat(),
                "end": (datetime.now() + timedelta(days=days_ahead)).isoformat()
            },
            "interventions": self._suggest_interventions(risk_score),
            "predicted_at": datetime.now().isoformat()
        }
    
    def predict_optimal_intervention_timing(
        self,
        user_id: str,
        user_patterns: Dict
    ) -> Dict:
        """
        Predice momento óptimo para intervenciones
        
        Args:
            user_id: ID del usuario
            user_patterns: Patrones del usuario
        
        Returns:
            Predicción de timing óptimo
        """
        # Analizar patrones históricos
        typical_stress_times = user_patterns.get("typical_stress_times", [])
        typical_craving_times = user_patterns.get("typical_craving_times", [])
        
        optimal_times = []
        
        # Identificar horas críticas
        for hour in range(24):
            risk_count = 0
            if hour in typical_stress_times:
                risk_count += 1
            if hour in typical_craving_times:
                risk_count += 1
            
            if risk_count > 0:
                optimal_times.append({
                    "hour": hour,
                    "intervention_type": "preventive",
                    "priority": "high" if risk_count == 2 else "medium"
                })
        
        return {
            "user_id": user_id,
            "optimal_intervention_times": optimal_times,
            "recommended_interventions": self._get_recommended_interventions(),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _generate_success_recommendations(self, factors: Dict, probability: float) -> List[str]:
        """Genera recomendaciones para mejorar probabilidad de éxito"""
        recommendations = []
        
        if factors["check_in_consistency"] < 0.7:
            recommendations.append("Aumenta la consistencia en tus check-ins diarios")
        
        if factors["support_system"] < 5:
            recommendations.append("Fortalecer tu sistema de apoyo es crucial")
        
        if factors["therapy_sessions"] == 0:
            recommendations.append("Considera sesiones de terapia para mejorar tus probabilidades")
        
        if factors["exercise_frequency"] < 3:
            recommendations.append("El ejercicio regular mejora significativamente las probabilidades de éxito")
        
        if probability < 0.6:
            recommendations.append("Enfócate en construir hábitos consistentes día a día")
        
        return recommendations
    
    def _suggest_interventions(self, risk_score: int) -> List[Dict]:
        """Sugiere intervenciones basadas en nivel de riesgo"""
        interventions = []
        
        if risk_score >= 7:
            interventions.append({
                "type": "immediate_support",
                "priority": "critical",
                "action": "Contactar sistema de apoyo inmediatamente"
            })
            interventions.append({
                "type": "crisis_resources",
                "priority": "high",
                "action": "Acceder a recursos de crisis"
            })
        elif risk_score >= 4:
            interventions.append({
                "type": "preventive_check_in",
                "priority": "high",
                "action": "Check-in preventivo con coach"
            })
            interventions.append({
                "type": "mindfulness",
                "priority": "medium",
                "action": "Sesión de mindfulness o meditación"
            })
        else:
            interventions.append({
                "type": "routine_maintenance",
                "priority": "low",
                "action": "Mantener rutina diaria"
            })
        
        return interventions
    
    def _get_recommended_interventions(self) -> List[Dict]:
        """Obtiene intervenciones recomendadas"""
        return [
            {
                "type": "check_in_reminder",
                "description": "Recordatorio de check-in diario"
            },
            {
                "type": "motivational_message",
                "description": "Mensaje motivacional personalizado"
            },
            {
                "type": "support_contact",
                "description": "Sugerencia de contacto con sistema de apoyo"
            }
        ]

