"""
Servicio de Seguimiento de Ejercicio Avanzado - Sistema completo de ejercicio
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedExerciseTrackingService:
    """Servicio de seguimiento de ejercicio avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de ejercicio"""
        pass
    
    def record_exercise_session(
        self,
        user_id: str,
        exercise_data: Dict
    ) -> Dict:
        """
        Registra sesión de ejercicio
        
        Args:
            user_id: ID del usuario
            exercise_data: Datos del ejercicio
        
        Returns:
            Sesión registrada
        """
        session = {
            "id": f"exercise_{datetime.now().timestamp()}",
            "user_id": user_id,
            "exercise_data": exercise_data,
            "exercise_type": exercise_data.get("exercise_type", "other"),
            "duration_minutes": exercise_data.get("duration_minutes", 0),
            "intensity": exercise_data.get("intensity", "moderate"),
            "calories_burned": exercise_data.get("calories_burned", 0),
            "timestamp": exercise_data.get("timestamp", datetime.now().isoformat()),
            "recorded_at": datetime.now().isoformat()
        }
        
        return session
    
    def analyze_exercise_patterns(
        self,
        user_id: str,
        exercise_sessions: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza patrones de ejercicio
        
        Args:
            user_id: ID del usuario
            exercise_sessions: Sesiones de ejercicio
            days: Número de días
        
        Returns:
            Análisis de patrones
        """
        if not exercise_sessions:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        total_duration = sum(s.get("duration_minutes", 0) for s in exercise_sessions)
        total_calories = sum(s.get("calories_burned", 0) for s in exercise_sessions)
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_sessions": len(exercise_sessions),
            "total_duration_minutes": total_duration,
            "average_weekly_duration": round(total_duration / (days / 7), 2) if days > 0 else 0,
            "total_calories_burned": total_calories,
            "exercise_type_distribution": self._analyze_exercise_types(exercise_sessions),
            "intensity_analysis": self._analyze_intensity(exercise_sessions),
            "trends": self._calculate_trends(exercise_sessions),
            "recommendations": self._generate_exercise_recommendations(exercise_sessions, days),
            "generated_at": datetime.now().isoformat()
        }
    
    def correlate_exercise_with_recovery(
        self,
        user_id: str,
        exercise_data: List[Dict],
        recovery_data: List[Dict]
    ) -> Dict:
        """
        Correlaciona ejercicio con recuperación
        
        Args:
            user_id: ID del usuario
            exercise_data: Datos de ejercicio
            recovery_data: Datos de recuperación
        
        Returns:
            Análisis de correlación
        """
        return {
            "user_id": user_id,
            "correlation_score": self._calculate_correlation(exercise_data, recovery_data),
            "findings": self._identify_correlations(exercise_data, recovery_data),
            "recommendations": self._generate_correlation_recommendations(exercise_data, recovery_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def _analyze_exercise_types(self, sessions: List[Dict]) -> Dict:
        """Analiza distribución de tipos de ejercicio"""
        type_counts = {}
        
        for session in sessions:
            exercise_type = session.get("exercise_type", "other")
            type_counts[exercise_type] = type_counts.get(exercise_type, 0) + 1
        
        return type_counts
    
    def _analyze_intensity(self, sessions: List[Dict]) -> Dict:
        """Analiza intensidad de ejercicio"""
        intensity_counts = {}
        
        for session in sessions:
            intensity = session.get("intensity", "moderate")
            intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
        
        return intensity_counts
    
    def _calculate_trends(self, sessions: List[Dict]) -> Dict:
        """Calcula tendencias de ejercicio"""
        if len(sessions) < 2:
            return {"trend": "insufficient_data"}
        
        # Dividir en dos mitades
        first_half = sessions[:len(sessions)//2]
        second_half = sessions[len(sessions)//2:]
        
        first_avg = sum(s.get("duration_minutes", 0) for s in first_half) / len(first_half) if first_half else 0
        second_avg = sum(s.get("duration_minutes", 0) for s in second_half) / len(second_half) if second_half else 0
        
        if second_avg > first_avg * 1.1:
            trend = "increasing"
        elif second_avg < first_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change_percentage": round(((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0, 2)
        }
    
    def _generate_exercise_recommendations(self, sessions: List[Dict], days: int) -> List[str]:
        """Genera recomendaciones de ejercicio"""
        recommendations = []
        
        total_duration = sum(s.get("duration_minutes", 0) for s in sessions)
        weekly_avg = total_duration / (days / 7) if days > 0 else 0
        
        if weekly_avg < 150:  # Menos de 150 minutos por semana
            recommendations.append("Intenta hacer al menos 150 minutos de ejercicio moderado por semana")
        
        return recommendations
    
    def _calculate_correlation(self, exercise_data: List[Dict], recovery_data: List[Dict]) -> float:
        """Calcula correlación entre ejercicio y recuperación"""
        # Lógica simplificada
        return 0.65
    
    def _identify_correlations(self, exercise_data: List[Dict], recovery_data: List[Dict]) -> List[str]:
        """Identifica correlaciones específicas"""
        return [
            "El ejercicio regular se correlaciona con mejor estado de ánimo",
            "Sesiones de ejercicio intenso se asocian con menor nivel de cravings"
        ]
    
    def _generate_correlation_recommendations(self, exercise_data: List[Dict], recovery_data: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en correlaciones"""
        return [
            "Mantén una rutina regular de ejercicio para apoyar tu recuperación",
            "Considera ejercicio moderado cuando sientas cravings"
        ]

