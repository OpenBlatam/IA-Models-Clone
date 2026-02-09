"""
Servicio de Análisis de Resiliencia Avanzado - Sistema completo de análisis de resiliencia
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class ResilienceAnalysisService:
    """Servicio de análisis de resiliencia avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de resiliencia"""
        pass
    
    def assess_resilience(
        self,
        user_id: str,
        resilience_data: Dict
    ) -> Dict:
        """
        Evalúa resiliencia
        
        Args:
            user_id: ID del usuario
            resilience_data: Datos de resiliencia
        
        Returns:
            Evaluación de resiliencia
        """
        return {
            "user_id": user_id,
            "assessment_id": f"resilience_{datetime.now().timestamp()}",
            "resilience_score": self._calculate_resilience_score(resilience_data),
            "resilience_factors": self._analyze_resilience_factors(resilience_data),
            "strengths": self._identify_strengths(resilience_data),
            "areas_for_improvement": self._identify_improvement_areas(resilience_data),
            "recommendations": self._generate_resilience_recommendations(resilience_data),
            "assessed_at": datetime.now().isoformat()
        }
    
    def track_resilience_over_time(
        self,
        user_id: str,
        resilience_assessments: List[Dict]
    ) -> Dict:
        """
        Rastrea resiliencia a lo largo del tiempo
        
        Args:
            user_id: ID del usuario
            resilience_assessments: Evaluaciones de resiliencia
        
        Returns:
            Análisis de tendencias de resiliencia
        """
        if not resilience_assessments or len(resilience_assessments) < 2:
            return {
                "user_id": user_id,
                "analysis": "insufficient_data"
            }
        
        scores = [a.get("resilience_score", 0) for a in resilience_assessments]
        
        return {
            "user_id": user_id,
            "total_assessments": len(resilience_assessments),
            "average_score": round(statistics.mean(scores), 2) if scores else 0,
            "trend": self._calculate_resilience_trend(scores),
            "improvement_rate": self._calculate_improvement_rate(scores),
            "resilience_level": self._determine_resilience_level(statistics.mean(scores) if scores else 0),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_resilience_outcome(
        self,
        user_id: str,
        current_resilience: Dict,
        historical_data: List[Dict]
    ) -> Dict:
        """
        Predice resultado de resiliencia
        
        Args:
            user_id: ID del usuario
            current_resilience: Resiliencia actual
            historical_data: Datos históricos
        
        Returns:
            Predicción de resultado
        """
        outcome_probability = self._calculate_outcome_probability(current_resilience, historical_data)
        
        return {
            "user_id": user_id,
            "predicted_outcome": "high_resilience" if outcome_probability > 0.7 else "moderate_resilience",
            "outcome_probability": round(outcome_probability, 3),
            "confidence": 0.80,
            "key_factors": self._identify_key_factors(current_resilience),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _calculate_resilience_score(self, data: Dict) -> float:
        """Calcula puntuación de resiliencia"""
        base_score = 5.0
        
        # Factores de resiliencia
        social_support = data.get("social_support", 5)
        coping_skills = data.get("coping_skills", 5)
        self_efficacy = data.get("self_efficacy", 5)
        optimism = data.get("optimism", 5)
        
        avg_factors = (social_support + coping_skills + self_efficacy + optimism) / 4
        base_score = avg_factors
        
        return round(max(1, min(10, base_score)), 2)
    
    def _analyze_resilience_factors(self, data: Dict) -> Dict:
        """Analiza factores de resiliencia"""
        return {
            "social_support": data.get("social_support", 5),
            "coping_skills": data.get("coping_skills", 5),
            "self_efficacy": data.get("self_efficacy", 5),
            "optimism": data.get("optimism", 5),
            "adaptability": data.get("adaptability", 5)
        }
    
    def _identify_strengths(self, data: Dict) -> List[str]:
        """Identifica fortalezas"""
        strengths = []
        
        if data.get("social_support", 5) >= 7:
            strengths.append("Fuerte red de apoyo social")
        
        if data.get("coping_skills", 5) >= 7:
            strengths.append("Buenas habilidades de afrontamiento")
        
        return strengths
    
    def _identify_improvement_areas(self, data: Dict) -> List[str]:
        """Identifica áreas de mejora"""
        areas = []
        
        if data.get("social_support", 5) < 5:
            areas.append("Fortalecer red de apoyo social")
        
        if data.get("coping_skills", 5) < 5:
            areas.append("Desarrollar habilidades de afrontamiento")
        
        return areas
    
    def _generate_resilience_recommendations(self, data: Dict) -> List[str]:
        """Genera recomendaciones de resiliencia"""
        recommendations = []
        
        score = self._calculate_resilience_score(data)
        
        if score < 6:
            recommendations.append("Participa en grupos de apoyo para fortalecer tu resiliencia")
            recommendations.append("Practica técnicas de mindfulness y meditación")
        
        return recommendations
    
    def _calculate_resilience_trend(self, scores: List[float]) -> str:
        """Calcula tendencia de resiliencia"""
        if len(scores) < 2:
            return "stable"
        
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0
        
        if avg_second > avg_first * 1.1:
            return "improving"
        elif avg_second < avg_first * 0.9:
            return "declining"
        return "stable"
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """Calcula tasa de mejora"""
        if len(scores) < 2:
            return 0.0
        
        first_score = scores[0]
        last_score = scores[-1]
        
        if first_score > 0:
            improvement = ((last_score - first_score) / first_score) * 100
            return round(improvement, 2)
        
        return 0.0
    
    def _determine_resilience_level(self, score: float) -> str:
        """Determina nivel de resiliencia"""
        if score >= 8:
            return "high"
        elif score >= 6:
            return "moderate"
        else:
            return "low"
    
    def _calculate_outcome_probability(self, current: Dict, history: List[Dict]) -> float:
        """Calcula probabilidad de resultado"""
        base_probability = 0.5
        
        current_score = self._calculate_resilience_score(current)
        if current_score >= 7:
            base_probability += 0.2
        
        if history:
            historical_scores = [self._calculate_resilience_score(h) for h in history]
            avg_historical = statistics.mean(historical_scores) if historical_scores else 0
            if avg_historical >= 6:
                base_probability += 0.1
        
        return min(1.0, base_probability)
    
    def _identify_key_factors(self, current: Dict) -> List[str]:
        """Identifica factores clave"""
        factors = []
        
        if current.get("social_support", 5) >= 7:
            factors.append("Alto apoyo social")
        
        if current.get("coping_skills", 5) >= 7:
            factors.append("Buenas habilidades de afrontamiento")
        
        return factors

