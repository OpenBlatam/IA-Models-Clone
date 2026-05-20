"""
Servicio de Análisis de Bienestar Integral - Sistema completo de análisis de bienestar
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class ComprehensiveWellnessAnalysisService:
    """Servicio de análisis de bienestar integral"""
    
    def __init__(self):
        """Inicializa el servicio de bienestar"""
        pass
    
    def assess_comprehensive_wellness(
        self,
        user_id: str,
        wellness_data: Dict
    ) -> Dict:
        """
        Evalúa bienestar integral
        
        Args:
            user_id: ID del usuario
            wellness_data: Datos de bienestar
        
        Returns:
            Evaluación de bienestar integral
        """
        return {
            "user_id": user_id,
            "assessment_id": f"wellness_{datetime.now().timestamp()}",
            "overall_wellness_score": self._calculate_overall_score(wellness_data),
            "physical_wellness": self._assess_physical_wellness(wellness_data),
            "mental_wellness": self._assess_mental_wellness(wellness_data),
            "emotional_wellness": self._assess_emotional_wellness(wellness_data),
            "social_wellness": self._assess_social_wellness(wellness_data),
            "spiritual_wellness": self._assess_spiritual_wellness(wellness_data),
            "recommendations": self._generate_wellness_recommendations(wellness_data),
            "assessed_at": datetime.now().isoformat()
        }
    
    def track_wellness_trends(
        self,
        user_id: str,
        wellness_assessments: List[Dict]
    ) -> Dict:
        """
        Rastrea tendencias de bienestar
        
        Args:
            user_id: ID del usuario
            wellness_assessments: Evaluaciones de bienestar
        
        Returns:
            Análisis de tendencias
        """
        if not wellness_assessments or len(wellness_assessments) < 2:
            return {
                "user_id": user_id,
                "analysis": "insufficient_data"
            }
        
        scores = [a.get("overall_wellness_score", 0) for a in wellness_assessments]
        
        return {
            "user_id": user_id,
            "total_assessments": len(wellness_assessments),
            "average_score": round(statistics.mean(scores), 2) if scores else 0,
            "trend": self._calculate_wellness_trend(scores),
            "improvement_areas": self._identify_improvement_areas(wellness_assessments),
            "strength_areas": self._identify_strength_areas(wellness_assessments),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_overall_score(self, data: Dict) -> float:
        """Calcula puntuación general de bienestar"""
        physical = data.get("physical_wellness", 5)
        mental = data.get("mental_wellness", 5)
        emotional = data.get("emotional_wellness", 5)
        social = data.get("social_wellness", 5)
        spiritual = data.get("spiritual_wellness", 5)
        
        overall = (physical + mental + emotional + social + spiritual) / 5
        
        return round(max(1, min(10, overall)), 2)
    
    def _assess_physical_wellness(self, data: Dict) -> Dict:
        """Evalúa bienestar físico"""
        return {
            "score": data.get("physical_wellness", 5),
            "factors": {
                "exercise": data.get("exercise_level", 5),
                "nutrition": data.get("nutrition_quality", 5),
                "sleep": data.get("sleep_quality", 5),
                "health": data.get("health_status", 5)
            }
        }
    
    def _assess_mental_wellness(self, data: Dict) -> Dict:
        """Evalúa bienestar mental"""
        return {
            "score": data.get("mental_wellness", 5),
            "factors": {
                "stress_management": data.get("stress_management", 5),
                "cognitive_function": data.get("cognitive_function", 5),
                "mental_clarity": data.get("mental_clarity", 5)
            }
        }
    
    def _assess_emotional_wellness(self, data: Dict) -> Dict:
        """Evalúa bienestar emocional"""
        return {
            "score": data.get("emotional_wellness", 5),
            "factors": {
                "emotional_stability": data.get("emotional_stability", 5),
                "mood": data.get("mood", 5),
                "emotional_expression": data.get("emotional_expression", 5)
            }
        }
    
    def _assess_social_wellness(self, data: Dict) -> Dict:
        """Evalúa bienestar social"""
        return {
            "score": data.get("social_wellness", 5),
            "factors": {
                "social_connections": data.get("social_connections", 5),
                "support_network": data.get("support_network", 5),
                "relationships": data.get("relationships_quality", 5)
            }
        }
    
    def _assess_spiritual_wellness(self, data: Dict) -> Dict:
        """Evalúa bienestar espiritual"""
        return {
            "score": data.get("spiritual_wellness", 5),
            "factors": {
                "purpose": data.get("sense_of_purpose", 5),
                "meaning": data.get("meaning_in_life", 5),
                "mindfulness": data.get("mindfulness_practice", 5)
            }
        }
    
    def _generate_wellness_recommendations(self, data: Dict) -> List[str]:
        """Genera recomendaciones de bienestar"""
        recommendations = []
        
        overall_score = self._calculate_overall_score(data)
        
        if overall_score < 6:
            recommendations.append("Considera un enfoque integral para mejorar tu bienestar")
        
        if data.get("physical_wellness", 5) < 5:
            recommendations.append("Mejora tu bienestar físico con ejercicio regular y nutrición adecuada")
        
        if data.get("mental_wellness", 5) < 5:
            recommendations.append("Practica técnicas de manejo de estrés y mindfulness")
        
        return recommendations
    
    def _calculate_wellness_trend(self, scores: List[float]) -> str:
        """Calcula tendencia de bienestar"""
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
    
    def _identify_improvement_areas(self, assessments: List[Dict]) -> List[str]:
        """Identifica áreas de mejora"""
        areas = []
        
        if assessments:
            latest = assessments[-1]
            if latest.get("physical_wellness", 5) < 5:
                areas.append("Bienestar físico")
            if latest.get("mental_wellness", 5) < 5:
                areas.append("Bienestar mental")
        
        return areas
    
    def _identify_strength_areas(self, assessments: List[Dict]) -> List[str]:
        """Identifica áreas de fortaleza"""
        strengths = []
        
        if assessments:
            latest = assessments[-1]
            if latest.get("physical_wellness", 5) >= 7:
                strengths.append("Bienestar físico")
            if latest.get("social_wellness", 5) >= 7:
                strengths.append("Bienestar social")
        
        return strengths

