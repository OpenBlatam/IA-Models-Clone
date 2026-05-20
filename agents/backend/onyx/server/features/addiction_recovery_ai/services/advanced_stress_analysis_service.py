"""
Servicio de Análisis de Estrés Avanzado - Sistema completo de análisis de estrés
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedStressAnalysisService:
    """Servicio de análisis de estrés avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de estrés"""
        pass
    
    def assess_stress(
        self,
        user_id: str,
        stress_data: Dict
    ) -> Dict:
        """
        Evalúa estrés
        
        Args:
            user_id: ID del usuario
            stress_data: Datos de estrés
        
        Returns:
            Evaluación de estrés
        """
        return {
            "user_id": user_id,
            "assessment_id": f"stress_{datetime.now().timestamp()}",
            "stress_level": stress_data.get("stress_level", 5),
            "stress_sources": self._identify_stress_sources(stress_data),
            "stress_indicators": self._analyze_stress_indicators(stress_data),
            "coping_resources": self._assess_coping_resources(stress_data),
            "recommendations": self._generate_stress_recommendations(stress_data),
            "assessed_at": datetime.now().isoformat()
        }
    
    def track_stress_over_time(
        self,
        user_id: str,
        stress_assessments: List[Dict]
    ) -> Dict:
        """
        Rastrea estrés a lo largo del tiempo
        
        Args:
            user_id: ID del usuario
            stress_assessments: Evaluaciones de estrés
        
        Returns:
            Análisis de tendencias de estrés
        """
        if not stress_assessments or len(stress_assessments) < 2:
            return {
                "user_id": user_id,
                "analysis": "insufficient_data"
            }
        
        stress_levels = [a.get("stress_level", 5) for a in stress_assessments]
        
        return {
            "user_id": user_id,
            "total_assessments": len(stress_assessments),
            "average_stress": round(statistics.mean(stress_levels), 2) if stress_levels else 0,
            "peak_stress": max(stress_levels) if stress_levels else 0,
            "trend": self._calculate_stress_trend(stress_levels),
            "stress_patterns": self._identify_stress_patterns(stress_assessments),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_stress_episode(
        self,
        user_id: str,
        current_state: Dict,
        stress_history: List[Dict]
    ) -> Dict:
        """
        Predice episodio de estrés
        
        Args:
            user_id: ID del usuario
            current_state: Estado actual
            stress_history: Historial de estrés
        
        Returns:
            Predicción de episodio
        """
        episode_probability = self._calculate_episode_probability(current_state, stress_history)
        
        return {
            "user_id": user_id,
            "episode_probability": round(episode_probability, 3),
            "risk_level": "high" if episode_probability >= 0.7 else "medium" if episode_probability >= 0.4 else "low",
            "warning_signs": self._identify_warning_signs(current_state),
            "prevention_strategies": self._generate_prevention_strategies(episode_probability),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _identify_stress_sources(self, data: Dict) -> List[Dict]:
        """Identifica fuentes de estrés"""
        sources = []
        
        work_stress = data.get("work_stress", 0)
        if work_stress >= 5:
            sources.append({
                "source": "work",
                "intensity": work_stress,
                "description": "Estrés relacionado con el trabajo"
            })
        
        relationship_stress = data.get("relationship_stress", 0)
        if relationship_stress >= 5:
            sources.append({
                "source": "relationships",
                "intensity": relationship_stress,
                "description": "Estrés relacionado con relaciones"
            })
        
        financial_stress = data.get("financial_stress", 0)
        if financial_stress >= 5:
            sources.append({
                "source": "financial",
                "intensity": financial_stress,
                "description": "Estrés financiero"
            })
        
        return sources
    
    def _analyze_stress_indicators(self, data: Dict) -> Dict:
        """Analiza indicadores de estrés"""
        return {
            "physical_symptoms": data.get("physical_symptoms", []),
            "emotional_symptoms": data.get("emotional_symptoms", []),
            "behavioral_changes": data.get("behavioral_changes", [])
        }
    
    def _assess_coping_resources(self, data: Dict) -> Dict:
        """Evalúa recursos de afrontamiento"""
        return {
            "social_support": data.get("social_support", 5),
            "coping_skills": data.get("coping_skills", 5),
            "self_care": data.get("self_care", 5),
            "overall_resources": round((data.get("social_support", 5) + data.get("coping_skills", 5) + data.get("self_care", 5)) / 3, 2)
        }
    
    def _generate_stress_recommendations(self, data: Dict) -> List[str]:
        """Genera recomendaciones de estrés"""
        recommendations = []
        
        stress_level = data.get("stress_level", 5)
        
        if stress_level >= 7:
            recommendations.append("⚠️ Nivel de estrés alto. Considera técnicas de relajación inmediatas")
            recommendations.append("Contacta tu sistema de apoyo")
        elif stress_level >= 5:
            recommendations.append("Practica técnicas de manejo de estrés")
            recommendations.append("Considera actividades de relajación")
        
        return recommendations
    
    def _calculate_stress_trend(self, levels: List[float]) -> str:
        """Calcula tendencia de estrés"""
        if len(levels) < 2:
            return "stable"
        
        first_half = levels[:len(levels)//2]
        second_half = levels[len(levels)//2:]
        
        avg_first = statistics.mean(first_half) if first_half else 0
        avg_second = statistics.mean(second_half) if second_half else 0
        
        if avg_second > avg_first * 1.1:
            return "increasing"
        elif avg_second < avg_first * 0.9:
            return "decreasing"
        return "stable"
    
    def _identify_stress_patterns(self, assessments: List[Dict]) -> Dict:
        """Identifica patrones de estrés"""
        return {
            "daily_pattern": "evening_peak",
            "weekly_pattern": "weekend_lower"
        }
    
    def _calculate_episode_probability(self, current: Dict, history: List[Dict]) -> float:
        """Calcula probabilidad de episodio"""
        base_probability = 0.3
        
        current_stress = current.get("stress_level", 5)
        if current_stress >= 7:
            base_probability += 0.3
        
        if history:
            recent_stress = [h.get("stress_level", 5) for h in history[-5:]]
            if recent_stress:
                avg_recent = statistics.mean(recent_stress)
                if avg_recent >= 6:
                    base_probability += 0.2
        
        return min(1.0, base_probability)
    
    def _identify_warning_signs(self, current: Dict) -> List[str]:
        """Identifica señales de advertencia"""
        signs = []
        
        if current.get("stress_level", 5) >= 7:
            signs.append("Nivel de estrés muy alto")
        
        if current.get("sleep_quality", 5) < 4:
            signs.append("Calidad de sueño afectada")
        
        return signs
    
    def _generate_prevention_strategies(self, probability: float) -> List[str]:
        """Genera estrategias de prevención"""
        strategies = []
        
        if probability >= 0.7:
            strategies.append("⚠️ Alto riesgo de episodio de estrés. Implementa técnicas de relajación inmediatas")
            strategies.append("Evita situaciones estresantes cuando sea posible")
        elif probability >= 0.4:
            strategies.append("Monitorea tu nivel de estrés regularmente")
            strategies.append("Practica técnicas de manejo de estrés preventivas")
        
        return strategies

