"""
Servicio de Análisis de Estado de Ánimo Avanzado - Sistema completo de análisis de estado de ánimo
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AdvancedMoodAnalysisService:
    """Servicio de análisis de estado de ánimo avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de estado de ánimo"""
        pass
    
    def analyze_mood_patterns(
        self,
        user_id: str,
        mood_data: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de estado de ánimo
        
        Args:
            user_id: ID del usuario
            mood_data: Datos de estado de ánimo
        
        Returns:
            Análisis de patrones
        """
        if not mood_data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "analysis_id": f"mood_{datetime.now().timestamp()}",
            "total_entries": len(mood_data),
            "mood_distribution": self._analyze_mood_distribution(mood_data),
            "mood_trends": self._analyze_mood_trends(mood_data),
            "mood_triggers": self._identify_mood_triggers(mood_data),
            "mood_stability": self._calculate_mood_stability(mood_data),
            "recommendations": self._generate_mood_recommendations(mood_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_mood_episode(
        self,
        user_id: str,
        current_state: Dict,
        mood_history: List[Dict]
    ) -> Dict:
        """
        Predice episodio de estado de ánimo
        
        Args:
            user_id: ID del usuario
            current_state: Estado actual
            mood_history: Historial de estado de ánimo
        
        Returns:
            Predicción de episodio
        """
        episode_probability = self._calculate_episode_probability(current_state, mood_history)
        
        return {
            "user_id": user_id,
            "episode_probability": round(episode_probability, 3),
            "predicted_mood": self._predict_mood(current_state, mood_history),
            "risk_factors": self._identify_risk_factors(current_state),
            "prevention_strategies": self._generate_prevention_strategies(episode_probability),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _analyze_mood_distribution(self, data: List[Dict]) -> Dict:
        """Analiza distribución de estado de ánimo"""
        mood_counts = defaultdict(int)
        
        for entry in data:
            mood = entry.get("mood", "neutral")
            mood_counts[mood] += 1
        
        return dict(mood_counts)
    
    def _analyze_mood_trends(self, data: List[Dict]) -> Dict:
        """Analiza tendencias de estado de ánimo"""
        if len(data) < 2:
            return {"trend": "stable"}
        
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        first_avg = statistics.mean([d.get("mood_score", 5) for d in first_half]) if first_half else 0
        second_avg = statistics.mean([d.get("mood_score", 5) for d in second_half]) if second_half else 0
        
        if second_avg > first_avg * 1.1:
            return {"trend": "improving", "change": round(second_avg - first_avg, 2)}
        elif second_avg < first_avg * 0.9:
            return {"trend": "declining", "change": round(second_avg - first_avg, 2)}
        return {"trend": "stable", "change": 0}
    
    def _identify_mood_triggers(self, data: List[Dict]) -> List[Dict]:
        """Identifica triggers de estado de ánimo"""
        triggers = []
        
        for entry in data:
            entry_triggers = entry.get("triggers", [])
            triggers.extend(entry_triggers)
        
        trigger_counts = defaultdict(int)
        for trigger in triggers:
            trigger_counts[trigger] += 1
        
        sorted_triggers = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"trigger": trigger, "frequency": count}
            for trigger, count in sorted_triggers[:5]
        ]
    
    def _calculate_mood_stability(self, data: List[Dict]) -> float:
        """Calcula estabilidad de estado de ánimo"""
        if len(data) < 2:
            return 0.5
        
        mood_scores = [d.get("mood_score", 5) for d in data]
        std_dev = statistics.stdev(mood_scores) if len(mood_scores) > 1 else 0
        
        # Menor desviación = mayor estabilidad
        stability = max(0, 1 - (std_dev / 10))
        
        return round(stability, 2)
    
    def _generate_mood_recommendations(self, data: List[Dict]) -> List[str]:
        """Genera recomendaciones de estado de ánimo"""
        recommendations = []
        
        stability = self._calculate_mood_stability(data)
        if stability < 0.6:
            recommendations.append("Considera técnicas de regulación emocional para mejorar la estabilidad del estado de ánimo")
        
        return recommendations
    
    def _calculate_episode_probability(self, current: Dict, history: List[Dict]) -> float:
        """Calcula probabilidad de episodio"""
        base_probability = 0.3
        
        current_mood = current.get("mood_score", 5)
        if current_mood < 4:
            base_probability += 0.3
        
        if history:
            recent_moods = [h.get("mood_score", 5) for h in history[-5:]]
            if recent_moods:
                avg_recent = statistics.mean(recent_moods)
                if avg_recent < 4:
                    base_probability += 0.2
        
        return min(1.0, base_probability)
    
    def _predict_mood(self, current: Dict, history: List[Dict]) -> str:
        """Predice estado de ánimo"""
        current_mood = current.get("mood_score", 5)
        
        if current_mood < 4:
            return "low"
        elif current_mood >= 7:
            return "high"
        else:
            return "neutral"
    
    def _identify_risk_factors(self, current: Dict) -> List[str]:
        """Identifica factores de riesgo"""
        factors = []
        
        if current.get("mood_score", 5) < 4:
            factors.append("Estado de ánimo bajo")
        
        if current.get("stress_level", 5) >= 7:
            factors.append("Estrés elevado")
        
        return factors
    
    def _generate_prevention_strategies(self, probability: float) -> List[str]:
        """Genera estrategias de prevención"""
        strategies = []
        
        if probability >= 0.7:
            strategies.append("⚠️ Alto riesgo de episodio de estado de ánimo bajo. Contacta tu sistema de apoyo")
            strategies.append("Practica técnicas de regulación emocional")
        elif probability >= 0.4:
            strategies.append("Monitorea tu estado de ánimo regularmente")
            strategies.append("Mantén actividades que mejoren tu estado de ánimo")
        
        return strategies

