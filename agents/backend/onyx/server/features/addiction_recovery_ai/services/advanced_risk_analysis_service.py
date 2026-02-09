"""
Servicio de Análisis de Riesgo Avanzado - Sistema completo de análisis de riesgo
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedRiskAnalysisService:
    """Servicio de análisis de riesgo avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de análisis de riesgo"""
        pass
    
    def perform_risk_assessment(
        self,
        user_id: str,
        current_data: Dict,
        historical_data: List[Dict]
    ) -> Dict:
        """
        Realiza evaluación de riesgo completa
        
        Args:
            user_id: ID del usuario
            current_data: Datos actuales
            historical_data: Datos históricos
        
        Returns:
            Evaluación de riesgo
        """
        risk_factors = self._identify_risk_factors(current_data, historical_data)
        protective_factors = self._identify_protective_factors(current_data, historical_data)
        
        risk_score = self._calculate_risk_score(risk_factors, protective_factors)
        
        return {
            "user_id": user_id,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "recommendations": self._generate_risk_recommendations(risk_score, risk_factors),
            "interventions": self._suggest_interventions(risk_score),
            "assessed_at": datetime.now().isoformat()
        }
    
    def predict_relapse_probability(
        self,
        user_id: str,
        time_horizon: int = 7,
        current_data: Dict = None
    ) -> Dict:
        """
        Predice probabilidad de recaída
        
        Args:
            user_id: ID del usuario
            time_horizon: Horizonte de tiempo en días
            current_data: Datos actuales
        
        Returns:
            Predicción de probabilidad
        """
        # Factores de riesgo
        stress_level = current_data.get("stress_level", 5) if current_data else 5
        cravings_level = current_data.get("cravings_level", 3) if current_data else 3
        support_available = current_data.get("support_available", True) if current_data else True
        
        # Calcular probabilidad base
        base_probability = 0.1
        
        if stress_level >= 8:
            base_probability += 0.3
        elif stress_level >= 6:
            base_probability += 0.15
        
        if cravings_level >= 7:
            base_probability += 0.25
        elif cravings_level >= 5:
            base_probability += 0.1
        
        if not support_available:
            base_probability += 0.15
        
        probability = min(0.95, base_probability)
        
        return {
            "user_id": user_id,
            "time_horizon_days": time_horizon,
            "relapse_probability": round(probability, 2),
            "confidence": 0.75,
            "factors_considered": ["stress", "cravings", "support"],
            "predicted_at": datetime.now().isoformat()
        }
    
    def analyze_risk_trends(
        self,
        user_id: str,
        risk_assessments: List[Dict]
    ) -> Dict:
        """
        Analiza tendencias de riesgo
        
        Args:
            user_id: ID del usuario
            risk_assessments: Evaluaciones de riesgo históricas
        
        Returns:
            Análisis de tendencias
        """
        if not risk_assessments or len(risk_assessments) < 2:
            return {
                "user_id": user_id,
                "trend": "insufficient_data"
            }
        
        risk_scores = [assessment.get("risk_score", 0) for assessment in risk_assessments]
        
        trend = "improving" if risk_scores[-1] < risk_scores[0] else "worsening"
        
        return {
            "user_id": user_id,
            "trend": trend,
            "current_risk": risk_scores[-1],
            "average_risk": round(statistics.mean(risk_scores), 2),
            "peak_risk": max(risk_scores),
            "lowest_risk": min(risk_scores),
            "data_points": len(risk_scores),
            "generated_at": datetime.now().isoformat()
        }
    
    def _identify_risk_factors(self, current: Dict, historical: List[Dict]) -> List[Dict]:
        """Identifica factores de riesgo"""
        factors = []
        
        if current.get("stress_level", 5) >= 7:
            factors.append({
                "factor": "high_stress",
                "severity": "high",
                "description": "Nivel de estrés elevado"
            })
        
        if current.get("cravings_level", 3) >= 6:
            factors.append({
                "factor": "high_cravings",
                "severity": "high",
                "description": "Deseos intensos"
            })
        
        return factors
    
    def _identify_protective_factors(self, current: Dict, historical: List[Dict]) -> List[Dict]:
        """Identifica factores protectores"""
        factors = []
        
        if current.get("support_available", False):
            factors.append({
                "factor": "support_system",
                "strength": "high",
                "description": "Sistema de apoyo disponible"
            })
        
        if current.get("therapy_sessions", 0) > 0:
            factors.append({
                "factor": "therapy",
                "strength": "medium",
                "description": "Participación en terapia"
            })
        
        return factors
    
    def _calculate_risk_score(self, risk_factors: List[Dict], protective_factors: List[Dict]) -> float:
        """Calcula puntuación de riesgo"""
        base_score = 5.0
        
        # Aumentar por factores de riesgo
        for factor in risk_factors:
            if factor.get("severity") == "high":
                base_score += 2.0
            elif factor.get("severity") == "medium":
                base_score += 1.0
        
        # Reducir por factores protectores
        for factor in protective_factors:
            if factor.get("strength") == "high":
                base_score -= 1.5
            elif factor.get("strength") == "medium":
                base_score -= 0.75
        
        return max(0, min(10, round(base_score, 2)))
    
    def _get_risk_level(self, score: float) -> str:
        """Obtiene nivel de riesgo"""
        if score >= 8:
            return "critical"
        elif score >= 6:
            return "high"
        elif score >= 4:
            return "medium"
        else:
            return "low"
    
    def _generate_risk_recommendations(self, score: float, factors: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en riesgo"""
        recommendations = []
        
        if score >= 8:
            recommendations.append("⚠️ Riesgo crítico detectado. Contacta tu sistema de apoyo inmediatamente.")
        elif score >= 6:
            recommendations.append("Riesgo alto. Considera aumentar contacto con tu sistema de apoyo.")
        
        return recommendations
    
    def _suggest_interventions(self, score: float) -> List[Dict]:
        """Sugiere intervenciones"""
        interventions = []
        
        if score >= 8:
            interventions.append({
                "type": "immediate_support",
                "priority": "critical",
                "action": "Contactar sistema de apoyo inmediatamente"
            })
        elif score >= 6:
            interventions.append({
                "type": "preventive_check_in",
                "priority": "high",
                "action": "Check-in preventivo con coach"
            })
        
        return interventions

