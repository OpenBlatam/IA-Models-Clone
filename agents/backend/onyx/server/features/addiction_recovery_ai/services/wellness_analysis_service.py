"""
Servicio de Análisis de Bienestar General - Análisis completo de bienestar
"""

from typing import Dict, List, Optional
from datetime import datetime
import statistics


class WellnessAnalysisService:
    """Servicio de análisis de bienestar"""
    
    def __init__(self):
        """Inicializa el servicio de análisis de bienestar"""
        pass
    
    def calculate_wellness_score(
        self,
        user_id: str,
        metrics: Dict
    ) -> Dict:
        """
        Calcula puntuación de bienestar general
        
        Args:
            user_id: ID del usuario
            metrics: Métricas de bienestar
        
        Returns:
            Puntuación de bienestar
        """
        # Componentes del bienestar
        physical_score = self._calculate_physical_wellness(metrics)
        mental_score = self._calculate_mental_wellness(metrics)
        social_score = self._calculate_social_wellness(metrics)
        emotional_score = self._calculate_emotional_wellness(metrics)
        
        # Puntuación general (promedio ponderado)
        overall_score = (
            physical_score * 0.25 +
            mental_score * 0.30 +
            social_score * 0.20 +
            emotional_score * 0.25
        )
        
        return {
            "user_id": user_id,
            "overall_score": round(overall_score, 2),
            "components": {
                "physical": round(physical_score, 2),
                "mental": round(mental_score, 2),
                "social": round(social_score, 2),
                "emotional": round(emotional_score, 2)
            },
            "level": self._get_wellness_level(overall_score),
            "recommendations": self._generate_wellness_recommendations(
                physical_score, mental_score, social_score, emotional_score
            ),
            "calculated_at": datetime.now().isoformat()
        }
    
    def analyze_wellness_trends(
        self,
        user_id: str,
        historical_data: List[Dict]
    ) -> Dict:
        """
        Analiza tendencias de bienestar
        
        Args:
            user_id: ID del usuario
            historical_data: Datos históricos
        
        Returns:
            Análisis de tendencias
        """
        if not historical_data or len(historical_data) < 2:
            return {
                "user_id": user_id,
                "trend": "insufficient_data",
                "message": "Se necesitan al menos 2 puntos de datos"
            }
        
        scores = [entry.get("wellness_score", 0) for entry in historical_data]
        
        # Calcular tendencia
        if len(scores) >= 2:
            trend_direction = "improving" if scores[-1] > scores[0] else "declining"
            trend_strength = abs(scores[-1] - scores[0]) / max(scores) if max(scores) > 0 else 0
        else:
            trend_direction = "stable"
            trend_strength = 0
        
        return {
            "user_id": user_id,
            "trend": trend_direction,
            "trend_strength": round(trend_strength, 2),
            "average_score": round(statistics.mean(scores), 2),
            "current_score": scores[-1] if scores else 0,
            "best_score": max(scores) if scores else 0,
            "worst_score": min(scores) if scores else 0,
            "data_points": len(scores),
            "generated_at": datetime.now().isoformat()
        }
    
    def get_wellness_insights(
        self,
        user_id: str,
        current_metrics: Dict,
        historical_metrics: List[Dict]
    ) -> Dict:
        """
        Obtiene insights de bienestar
        
        Args:
            user_id: ID del usuario
            current_metrics: Métricas actuales
            historical_metrics: Métricas históricas
        
        Returns:
            Insights de bienestar
        """
        insights = []
        
        # Comparar con histórico
        if historical_metrics:
            avg_sleep = statistics.mean([m.get("sleep_hours", 7) for m in historical_metrics])
            current_sleep = current_metrics.get("sleep_hours", 7)
            
            if current_sleep > avg_sleep:
                insights.append({
                    "type": "positive",
                    "category": "sleep",
                    "message": f"Tu sueño ha mejorado ({current_sleep:.1f}h vs {avg_sleep:.1f}h promedio)"
                })
        
        # Detectar áreas de mejora
        if current_metrics.get("exercise_days", 0) < 3:
            insights.append({
                "type": "improvement",
                "category": "exercise",
                "message": "Aumentar ejercicio puede mejorar tu bienestar general"
            })
        
        return {
            "user_id": user_id,
            "insights": insights,
            "total_insights": len(insights),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_physical_wellness(self, metrics: Dict) -> float:
        """Calcula bienestar físico"""
        sleep_score = min(10, (metrics.get("sleep_hours", 7) / 8) * 10)
        exercise_score = min(10, (metrics.get("exercise_days", 0) / 5) * 10)
        nutrition_score = metrics.get("nutrition_score", 5)
        
        return (sleep_score + exercise_score + nutrition_score) / 3
    
    def _calculate_mental_wellness(self, metrics: Dict) -> float:
        """Calcula bienestar mental"""
        stress_score = 10 - metrics.get("stress_level", 5)
        anxiety_score = 10 - metrics.get("anxiety_level", 5)
        focus_score = metrics.get("focus_score", 5)
        
        return (stress_score + anxiety_score + focus_score) / 3
    
    def _calculate_social_wellness(self, metrics: Dict) -> float:
        """Calcula bienestar social"""
        support_score = metrics.get("support_system_score", 5)
        social_connections = min(10, metrics.get("social_connections", 0))
        community_involvement = metrics.get("community_involvement", 5)
        
        return (support_score + social_connections + community_involvement) / 3
    
    def _calculate_emotional_wellness(self, metrics: Dict) -> float:
        """Calcula bienestar emocional"""
        mood_score = metrics.get("mood_score", 5)
        emotional_stability = metrics.get("emotional_stability", 5)
        self_esteem = metrics.get("self_esteem_score", 5)
        
        return (mood_score + emotional_stability + self_esteem) / 3
    
    def _get_wellness_level(self, score: float) -> str:
        """Obtiene nivel de bienestar"""
        if score >= 8:
            return "excellent"
        elif score >= 6:
            return "good"
        elif score >= 4:
            return "fair"
        else:
            return "needs_improvement"
    
    def _generate_wellness_recommendations(
        self,
        physical: float,
        mental: float,
        social: float,
        emotional: float
    ) -> List[str]:
        """Genera recomendaciones de bienestar"""
        recommendations = []
        
        if physical < 6:
            recommendations.append("Mejora tu bienestar físico: prioriza sueño, ejercicio y nutrición")
        
        if mental < 6:
            recommendations.append("Fortalece tu bienestar mental: practica mindfulness y manejo de estrés")
        
        if social < 6:
            recommendations.append("Construye conexiones sociales: participa en grupos de apoyo")
        
        if emotional < 6:
            recommendations.append("Cuida tu bienestar emocional: practica auto-compasión y expresión emocional")
        
        return recommendations

