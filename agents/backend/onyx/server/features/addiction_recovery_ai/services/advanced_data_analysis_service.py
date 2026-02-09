"""
Servicio de Análisis de Datos Avanzado - Sistema completo de análisis de datos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics
from collections import defaultdict


class AdvancedDataAnalysisService:
    """Servicio de análisis de datos avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de análisis"""
        pass
    
    def perform_comprehensive_analysis(
        self,
        user_id: str,
        data: List[Dict],
        analysis_type: str = "full"
    ) -> Dict:
        """
        Realiza análisis comprensivo
        
        Args:
            user_id: ID del usuario
            data: Datos a analizar
            analysis_type: Tipo de análisis
        
        Returns:
            Análisis comprensivo
        """
        if not data:
            return {
                "user_id": user_id,
                "error": "No data available"
            }
        
        analysis = {
            "user_id": user_id,
            "analysis_type": analysis_type,
            "summary": self._generate_summary(data),
            "trends": self._identify_trends(data),
            "patterns": self._identify_patterns(data),
            "correlations": self._find_correlations(data),
            "anomalies": self._detect_anomalies(data),
            "predictions": self._generate_predictions(data),
            "recommendations": self._generate_recommendations(data),
            "generated_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def analyze_behavioral_patterns(
        self,
        user_id: str,
        behavioral_data: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de comportamiento
        
        Args:
            user_id: ID del usuario
            behavioral_data: Datos de comportamiento
        
        Returns:
            Análisis de patrones
        """
        patterns = {
            "user_id": user_id,
            "most_active_times": self._find_most_active_times(behavioral_data),
            "trigger_patterns": self._identify_trigger_patterns(behavioral_data),
            "response_patterns": self._identify_response_patterns(behavioral_data),
            "success_factors": self._identify_success_factors(behavioral_data),
            "risk_factors": self._identify_risk_factors(behavioral_data),
            "generated_at": datetime.now().isoformat()
        }
        
        return patterns
    
    def generate_insights_report(
        self,
        user_id: str,
        historical_data: List[Dict]
    ) -> Dict:
        """
        Genera reporte de insights
        
        Args:
            user_id: ID del usuario
            historical_data: Datos históricos
        
        Returns:
            Reporte de insights
        """
        insights = []
        
        # Analizar mejoras
        if len(historical_data) >= 7:
            recent_avg = statistics.mean([d.get("mood_score", 5) for d in historical_data[-7:]])
            older_avg = statistics.mean([d.get("mood_score", 5) for d in historical_data[:7]])
            
            if recent_avg > older_avg:
                insights.append({
                    "type": "positive",
                    "title": "Mejora en Estado de Ánimo",
                    "description": f"Tu estado de ánimo ha mejorado {round(recent_avg - older_avg, 2)} puntos en la última semana"
                })
        
        return {
            "user_id": user_id,
            "insights": insights,
            "total_insights": len(insights),
            "generated_at": datetime.now().isoformat()
        }
    
    def _generate_summary(self, data: List[Dict]) -> Dict:
        """Genera resumen de datos"""
        return {
            "total_entries": len(data),
            "date_range": {
                "start": data[0].get("date") if data else None,
                "end": data[-1].get("date") if data else None
            },
            "key_metrics": {
                "average_mood": round(statistics.mean([d.get("mood_score", 5) for d in data]), 2) if data else 0,
                "average_cravings": round(statistics.mean([d.get("cravings_level", 3) for d in data]), 2) if data else 0
            }
        }
    
    def _identify_trends(self, data: List[Dict]) -> List[Dict]:
        """Identifica tendencias"""
        trends = []
        
        if len(data) >= 7:
            mood_trend = "improving" if data[-1].get("mood_score", 5) > data[0].get("mood_score", 5) else "declining"
            trends.append({
                "metric": "mood",
                "trend": mood_trend,
                "strength": "moderate"
            })
        
        return trends
    
    def _identify_patterns(self, data: List[Dict]) -> List[Dict]:
        """Identifica patrones"""
        patterns = []
        
        # Patrón de días de la semana
        weekday_data = defaultdict(list)
        for entry in data:
            date = datetime.fromisoformat(entry.get("date", datetime.now().isoformat()))
            weekday_data[date.weekday()].append(entry.get("mood_score", 5))
        
        for weekday, scores in weekday_data.items():
            if len(scores) >= 3:
                avg = statistics.mean(scores)
                patterns.append({
                    "type": "weekly",
                    "day": weekday,
                    "average_mood": round(avg, 2)
                })
        
        return patterns
    
    def _find_correlations(self, data: List[Dict]) -> List[Dict]:
        """Encuentra correlaciones"""
        correlations = []
        
        # Correlación entre ejercicio y estado de ánimo
        exercise_days = [d for d in data if d.get("exercise_done", False)]
        if exercise_days:
            exercise_mood = statistics.mean([d.get("mood_score", 5) for d in exercise_days])
            no_exercise_mood = statistics.mean([d.get("mood_score", 5) for d in data if not d.get("exercise_done", False)])
            
            if exercise_mood > no_exercise_mood:
                correlations.append({
                    "factor1": "exercise",
                    "factor2": "mood",
                    "correlation": "positive",
                    "strength": round(exercise_mood - no_exercise_mood, 2)
                })
        
        return correlations
    
    def _detect_anomalies(self, data: List[Dict]) -> List[Dict]:
        """Detecta anomalías"""
        anomalies = []
        
        if len(data) >= 7:
            mood_scores = [d.get("mood_score", 5) for d in data]
            mean = statistics.mean(mood_scores)
            std = statistics.stdev(mood_scores) if len(mood_scores) > 1 else 0
            
            for i, entry in enumerate(data):
                score = entry.get("mood_score", 5)
                if abs(score - mean) > 2 * std:
                    anomalies.append({
                        "date": entry.get("date"),
                        "metric": "mood_score",
                        "value": score,
                        "deviation": round(score - mean, 2)
                    })
        
        return anomalies
    
    def _generate_predictions(self, data: List[Dict]) -> Dict:
        """Genera predicciones"""
        if len(data) < 7:
            return {
                "available": False,
                "message": "Insufficient data"
            }
        
        recent_trend = data[-1].get("mood_score", 5) - data[-7].get("mood_score", 5)
        
        return {
            "available": True,
            "predicted_mood_7_days": round(data[-1].get("mood_score", 5) + recent_trend, 2),
            "confidence": 0.7
        }
    
    def _generate_recommendations(self, data: List[Dict]) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        avg_exercise = statistics.mean([1 if d.get("exercise_done", False) else 0 for d in data]) if data else 0
        if avg_exercise < 0.3:
            recommendations.append("Aumenta la frecuencia de ejercicio para mejorar tu bienestar")
        
        return recommendations
    
    def _find_most_active_times(self, data: List[Dict]) -> List[int]:
        """Encuentra horas más activas"""
        hour_counts = defaultdict(int)
        for entry in data:
            hour = entry.get("hour", 12)
            hour_counts[hour] += 1
        
        return sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _identify_trigger_patterns(self, data: List[Dict]) -> List[Dict]:
        """Identifica patrones de triggers"""
        return []
    
    def _identify_response_patterns(self, data: List[Dict]) -> List[Dict]:
        """Identifica patrones de respuesta"""
        return []
    
    def _identify_success_factors(self, data: List[Dict]) -> List[str]:
        """Identifica factores de éxito"""
        return []
    
    def _identify_risk_factors(self, data: List[Dict]) -> List[str]:
        """Identifica factores de riesgo"""
        return []

