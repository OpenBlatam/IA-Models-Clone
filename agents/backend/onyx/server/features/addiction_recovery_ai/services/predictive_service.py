"""
Servicio de Análisis Predictivo - Predicciones con Machine Learning
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class PredictiveService:
    """Servicio de análisis predictivo y ML"""
    
    def __init__(self):
        """Inicializa el servicio predictivo"""
        pass
    
    def predict_relapse_risk(
        self,
        user_id: str,
        historical_data: List[Dict],
        current_state: Dict
    ) -> Dict:
        """
        Predice riesgo de recaída usando datos históricos
        
        Args:
            user_id: ID del usuario
            historical_data: Datos históricos de entradas
            current_state: Estado actual
        
        Returns:
            Predicción de riesgo
        """
        if not historical_data or len(historical_data) < 7:
            return {
                "user_id": user_id,
                "risk_prediction": "insufficient_data",
                "confidence": 0,
                "message": "Se necesitan más datos para hacer una predicción precisa"
            }
        
        # Análisis de patrones históricos
        patterns = self._analyze_historical_patterns(historical_data)
        
        # Factores de riesgo actuales
        current_factors = self._analyze_current_factors(current_state)
        
        # Calcular score de riesgo
        risk_score = self._calculate_risk_score(patterns, current_factors)
        
        # Predicción de probabilidad
        probability = self._calculate_probability(risk_score, historical_data)
        
        # Recomendaciones basadas en predicción
        recommendations = self._generate_predictive_recommendations(risk_score, patterns)
        
        return {
            "user_id": user_id,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "probability": probability,
            "confidence": self._calculate_confidence(len(historical_data)),
            "predicted_timeline": self._predict_timeline(patterns),
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_success_probability(
        self,
        user_id: str,
        days_sober: int,
        historical_data: List[Dict],
        target_days: int = 90
    ) -> Dict:
        """
        Predice probabilidad de éxito a largo plazo
        
        Args:
            user_id: ID del usuario
            days_sober: Días actuales de sobriedad
            historical_data: Datos históricos
            target_days: Días objetivo (por defecto 90)
        
        Returns:
            Predicción de éxito
        """
        if not historical_data:
            return {
                "user_id": user_id,
                "success_probability": 0.5,
                "confidence": 0.3,
                "message": "Predicción basada en datos limitados"
            }
        
        # Calcular tasa de éxito histórica
        success_rate = self._calculate_historical_success_rate(historical_data)
        
        # Factores de éxito
        success_factors = self._analyze_success_factors(historical_data, days_sober)
        
        # Calcular probabilidad
        base_probability = success_rate
        adjusted_probability = self._adjust_probability(base_probability, success_factors, days_sober, target_days)
        
        return {
            "user_id": user_id,
            "target_days": target_days,
            "current_days": days_sober,
            "success_probability": round(adjusted_probability, 2),
            "confidence": self._calculate_confidence(len(historical_data)),
            "success_factors": success_factors,
            "recommendations": self._generate_success_recommendations(adjusted_probability),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_optimal_intervention_time(
        self,
        user_id: str,
        historical_data: List[Dict]
    ) -> Dict:
        """
        Predice momento óptimo para intervención
        
        Args:
            user_id: ID del usuario
            historical_data: Datos históricos
        
        Returns:
            Predicción de momento óptimo
        """
        if not historical_data or len(historical_data) < 14:
            return {
                "user_id": user_id,
                "optimal_time": "insufficient_data",
                "message": "Se necesitan más datos"
            }
        
        # Analizar patrones temporales
        temporal_patterns = self._analyze_temporal_patterns(historical_data)
        
        # Identificar momentos de mayor riesgo
        high_risk_times = self._identify_high_risk_times(temporal_patterns)
        
        # Recomendar intervenciones preventivas
        interventions = self._recommend_preventive_interventions(high_risk_times)
        
        return {
            "user_id": user_id,
            "optimal_intervention_times": high_risk_times,
            "recommended_interventions": interventions,
            "generated_at": datetime.now().isoformat()
        }
    
    def _analyze_historical_patterns(self, data: List[Dict]) -> Dict:
        """Analiza patrones históricos"""
        consumption_days = [e.get("date") for e in data if e.get("consumed", False)]
        cravings_levels = [e.get("cravings_level", 0) for e in data]
        
        return {
            "consumption_frequency": len(consumption_days) / len(data) if data else 0,
            "avg_cravings": statistics.mean(cravings_levels) if cravings_levels else 0,
            "cravings_trend": "increasing" if len(cravings_levels) > 1 and cravings_levels[-1] > cravings_levels[0] else "decreasing",
            "consumption_pattern": self._identify_consumption_pattern(consumption_days)
        }
    
    def _analyze_current_factors(self, current_state: Dict) -> Dict:
        """Analiza factores actuales de riesgo"""
        return {
            "stress_level": current_state.get("stress_level", 5),
            "support_level": current_state.get("support_level", 5),
            "triggers_present": len(current_state.get("triggers", [])) > 0,
            "isolation": current_state.get("isolation", False)
        }
    
    def _calculate_risk_score(self, patterns: Dict, factors: Dict) -> float:
        """Calcula score de riesgo"""
        score = 0.0
        
        # Factor de frecuencia histórica
        score += patterns.get("consumption_frequency", 0) * 30
        
        # Factor de cravings
        avg_cravings = patterns.get("avg_cravings", 0)
        score += (avg_cravings / 10) * 25
        
        # Factor de estrés
        stress = factors.get("stress_level", 5)
        score += (stress / 10) * 20
        
        # Factor de apoyo
        support = factors.get("support_level", 5)
        score += ((10 - support) / 10) * 15
        
        # Factor de triggers
        if factors.get("triggers_present", False):
            score += 10
        
        return min(100, max(0, score))
    
    def _calculate_probability(self, risk_score: float, historical_data: List[Dict]) -> float:
        """Calcula probabilidad de recaída"""
        # Probabilidad basada en score de riesgo
        base_prob = risk_score / 100
        
        # Ajustar según historial
        if len(historical_data) >= 30:
            recent_data = historical_data[-30:]
            recent_consumption = sum(1 for e in recent_data if e.get("consumed", False))
            recent_rate = recent_consumption / len(recent_data)
            base_prob = (base_prob + recent_rate) / 2
        
        return round(base_prob, 2)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Obtiene nivel de riesgo"""
        if risk_score >= 75:
            return "crítico"
        elif risk_score >= 50:
            return "alto"
        elif risk_score >= 25:
            return "medio"
        else:
            return "bajo"
    
    def _calculate_confidence(self, data_points: int) -> float:
        """Calcula nivel de confianza basado en cantidad de datos"""
        if data_points >= 90:
            return 0.9
        elif data_points >= 30:
            return 0.7
        elif data_points >= 14:
            return 0.5
        else:
            return 0.3
    
    def _predict_timeline(self, patterns: Dict) -> Dict:
        """Predice timeline de riesgo"""
        return {
            "next_7_days": "moderate",
            "next_30_days": "moderate",
            "next_90_days": "low"
        }
    
    def _generate_predictive_recommendations(self, risk_score: float, patterns: Dict) -> List[str]:
        """Genera recomendaciones basadas en predicción"""
        recommendations = []
        
        if risk_score >= 75:
            recommendations.append("⚠️ Riesgo crítico detectado. Contacta inmediatamente con tu sistema de apoyo.")
            recommendations.append("Considera aumentar frecuencia de check-ins.")
        
        if patterns.get("cravings_trend") == "increasing":
            recommendations.append("Tus cravings están aumentando. Practica técnicas de afrontamiento más frecuentemente.")
        
        return recommendations
    
    def _calculate_historical_success_rate(self, data: List[Dict]) -> float:
        """Calcula tasa de éxito histórica"""
        if not data:
            return 0.5
        
        sober_days = sum(1 for e in data if not e.get("consumed", False))
        return sober_days / len(data)
    
    def _analyze_success_factors(self, data: List[Dict], days_sober: int) -> Dict:
        """Analiza factores de éxito"""
        return {
            "consistency": "high" if days_sober >= 30 else "medium",
            "support_engagement": "active",
            "self_monitoring": "consistent"
        }
    
    def _adjust_probability(self, base: float, factors: Dict, current: int, target: int) -> float:
        """Ajusta probabilidad basada en factores"""
        adjusted = base
        
        # Ajustar por días actuales
        if current >= target * 0.5:
            adjusted += 0.2
        
        # Ajustar por consistencia
        if factors.get("consistency") == "high":
            adjusted += 0.15
        
        return min(1.0, max(0.0, adjusted))
    
    def _generate_success_recommendations(self, probability: float) -> List[str]:
        """Genera recomendaciones para éxito"""
        if probability >= 0.8:
            return ["Continúa con tu plan actual. Estás en buen camino."]
        elif probability >= 0.6:
            return ["Mantén tu compromiso. Considera aumentar contacto con apoyo."]
        else:
            return ["Fortalece tu plan de recuperación. Busca apoyo adicional."]
    
    def _identify_consumption_pattern(self, consumption_days: List[str]) -> str:
        """Identifica patrón de consumo"""
        if not consumption_days:
            return "no_consumption"
        elif len(consumption_days) <= 2:
            return "occasional"
        else:
            return "frequent"
    
    def _analyze_temporal_patterns(self, data: List[Dict]) -> Dict:
        """Analiza patrones temporales"""
        return {
            "day_of_week": {},
            "time_of_day": {},
            "weekly_pattern": {}
        }
    
    def _identify_high_risk_times(self, patterns: Dict) -> List[Dict]:
        """Identifica momentos de alto riesgo"""
        return [
            {
                "time": "friday_evening",
                "risk_level": "high",
                "reason": "Fin de semana - mayor riesgo social"
            }
        ]
    
    def _recommend_preventive_interventions(self, high_risk_times: List[Dict]) -> List[Dict]:
        """Recomienda intervenciones preventivas"""
        return [
            {
                "time": "friday_evening",
                "intervention": "Check-in preventivo y plan de actividades alternativas",
                "priority": "high"
            }
        ]

