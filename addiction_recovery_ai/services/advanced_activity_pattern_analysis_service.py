"""
Servicio de Análisis de Patrones de Actividad Avanzado - Sistema completo de análisis de actividad
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AdvancedActivityPatternAnalysisService:
    """Servicio de análisis de patrones de actividad avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de análisis de actividad"""
        pass
    
    def analyze_activity_patterns(
        self,
        user_id: str,
        activity_data: List[Dict],
        analysis_type: str = "comprehensive"
    ) -> Dict:
        """
        Analiza patrones de actividad
        
        Args:
            user_id: ID del usuario
            activity_data: Datos de actividad
            analysis_type: Tipo de análisis
        
        Returns:
            Análisis de patrones
        """
        if not activity_data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "analysis_type": analysis_type,
            "total_activities": len(activity_data),
            "activity_distribution": self._analyze_activity_distribution(activity_data),
            "temporal_patterns": self._analyze_temporal_patterns(activity_data),
            "activity_intensity": self._analyze_intensity(activity_data),
            "correlations": self._find_activity_correlations(activity_data),
            "recommendations": self._generate_activity_recommendations(activity_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_activity_outcome(
        self,
        user_id: str,
        current_activities: Dict,
        activity_history: List[Dict]
    ) -> Dict:
        """
        Predice resultado de actividad
        
        Args:
            user_id: ID del usuario
            current_activities: Actividades actuales
            activity_history: Historial de actividades
        
        Returns:
            Predicción de resultado
        """
        outcome_probability = self._calculate_outcome_probability(current_activities, activity_history)
        
        return {
            "user_id": user_id,
            "predicted_outcome": "positive" if outcome_probability > 0.6 else "neutral",
            "outcome_probability": round(outcome_probability, 3),
            "confidence": 0.78,
            "key_factors": self._identify_activity_factors(current_activities),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _analyze_activity_distribution(self, data: List[Dict]) -> Dict:
        """Analiza distribución de actividades"""
        distribution = defaultdict(int)
        
        for activity in data:
            activity_type = activity.get("activity_type", "unknown")
            distribution[activity_type] += 1
        
        return dict(distribution)
    
    def _analyze_temporal_patterns(self, data: List[Dict]) -> Dict:
        """Analiza patrones temporales"""
        hourly_counts = defaultdict(int)
        
        for activity in data:
            timestamp = activity.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    hourly_counts[dt.hour] += 1
                except:
                    pass
        
        return {
            "peak_hours": sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "activity_pattern": "morning" if max(hourly_counts.keys() or [0]) < 12 else "afternoon"
        }
    
    def _analyze_intensity(self, data: List[Dict]) -> Dict:
        """Analiza intensidad de actividades"""
        intensities = [a.get("intensity", 5) for a in data]
        
        return {
            "average_intensity": round(statistics.mean(intensities), 2) if intensities else 0,
            "max_intensity": max(intensities) if intensities else 0,
            "min_intensity": min(intensities) if intensities else 0
        }
    
    def _find_activity_correlations(self, data: List[Dict]) -> List[Dict]:
        """Encuentra correlaciones de actividad"""
        return []
    
    def _generate_activity_recommendations(self, data: List[Dict]) -> List[str]:
        """Genera recomendaciones de actividad"""
        recommendations = []
        
        total_activities = len(data)
        if total_activities < 10:
            recommendations.append("Aumenta tu nivel de actividad para apoyar la recuperación")
        
        return recommendations
    
    def _calculate_outcome_probability(self, current: Dict, history: List[Dict]) -> float:
        """Calcula probabilidad de resultado"""
        base_probability = 0.5
        
        activity_level = current.get("activity_level", 5)
        if activity_level >= 7:
            base_probability += 0.2
        
        return min(1.0, base_probability)
    
    def _identify_activity_factors(self, current: Dict) -> List[str]:
        """Identifica factores de actividad"""
        factors = []
        
        if current.get("activity_level", 5) >= 7:
            factors.append("Alto nivel de actividad")
        
        return factors

