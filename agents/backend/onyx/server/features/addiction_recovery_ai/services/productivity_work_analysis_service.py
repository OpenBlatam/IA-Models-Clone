"""
Servicio de Análisis de Productividad y Trabajo - Sistema completo de análisis laboral
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class ProductivityWorkAnalysisService:
    """Servicio de análisis de productividad y trabajo"""
    
    def __init__(self):
        """Inicializa el servicio de productividad"""
        pass
    
    def record_work_session(
        self,
        user_id: str,
        work_data: Dict
    ) -> Dict:
        """
        Registra sesión de trabajo
        
        Args:
            user_id: ID del usuario
            work_data: Datos de trabajo
        
        Returns:
            Sesión registrada
        """
        session = {
            "id": f"work_session_{datetime.now().timestamp()}",
            "user_id": user_id,
            "work_data": work_data,
            "duration_hours": work_data.get("duration_hours", 0),
            "productivity_score": work_data.get("productivity_score", 5),
            "stress_level": work_data.get("stress_level", 5),
            "recorded_at": datetime.now().isoformat()
        }
        
        return session
    
    def analyze_work_patterns(
        self,
        user_id: str,
        work_sessions: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza patrones de trabajo
        
        Args:
            user_id: ID del usuario
            work_sessions: Sesiones de trabajo
            days: Número de días
        
        Returns:
            Análisis de patrones
        """
        if not work_sessions:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        productivity_scores = [s.get("productivity_score", 5) for s in work_sessions]
        stress_levels = [s.get("stress_level", 5) for s in work_sessions]
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_sessions": len(work_sessions),
            "average_productivity": round(statistics.mean(productivity_scores), 2),
            "average_stress": round(statistics.mean(stress_levels), 2),
            "productivity_trend": self._calculate_trend(productivity_scores),
            "stress_trend": self._calculate_trend(stress_levels),
            "correlations": self._find_work_correlations(work_sessions),
            "recommendations": self._generate_work_recommendations(work_sessions),
            "generated_at": datetime.now().isoformat()
        }
    
    def detect_work_triggers(
        self,
        user_id: str,
        work_sessions: List[Dict]
    ) -> Dict:
        """
        Detecta triggers relacionados con trabajo
        
        Args:
            user_id: ID del usuario
            work_sessions: Sesiones de trabajo
        
        Returns:
            Triggers detectados
        """
        triggers = []
        
        for session in work_sessions:
            stress_level = session.get("stress_level", 5)
            if stress_level >= 8:
                triggers.append({
                    "session_id": session.get("id"),
                    "trigger_type": "high_work_stress",
                    "severity": "high",
                    "timestamp": session.get("recorded_at")
                })
        
        return {
            "user_id": user_id,
            "triggers_detected": triggers,
            "total_triggers": len(triggers),
            "recommendations": self._generate_trigger_recommendations(triggers),
            "generated_at": datetime.now().isoformat()
        }
    
    def calculate_recovery_work_balance(
        self,
        user_id: str,
        work_data: List[Dict],
        recovery_data: List[Dict]
    ) -> Dict:
        """
        Calcula balance trabajo-recuperación
        
        Args:
            user_id: ID del usuario
            work_data: Datos de trabajo
            recovery_data: Datos de recuperación
        
        Returns:
            Análisis de balance
        """
        return {
            "user_id": user_id,
            "work_hours": sum(w.get("duration_hours", 0) for w in work_data),
            "recovery_activities": len(recovery_data),
            "balance_score": self._calculate_balance_score(work_data, recovery_data),
            "recommendations": self._generate_balance_recommendations(work_data, recovery_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendencia"""
        if len(values) < 2:
            return "stable"
        
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.1:
            return "improving"
        elif second_half < first_half * 0.9:
            return "declining"
        return "stable"
    
    def _find_work_correlations(self, sessions: List[Dict]) -> List[Dict]:
        """Encuentra correlaciones de trabajo"""
        return []
    
    def _generate_work_recommendations(self, sessions: List[Dict]) -> List[str]:
        """Genera recomendaciones de trabajo"""
        recommendations = []
        
        avg_stress = statistics.mean([s.get("stress_level", 5) for s in sessions])
        if avg_stress >= 7:
            recommendations.append("Considera técnicas de manejo de estrés laboral")
        
        return recommendations
    
    def _generate_trigger_recommendations(self, triggers: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en triggers"""
        if triggers:
            return [
                "El estrés laboral puede ser un trigger. Considera técnicas de relajación",
                "Establece límites claros entre trabajo y recuperación"
            ]
        return []
    
    def _calculate_balance_score(self, work_data: List[Dict], recovery_data: List[Dict]) -> float:
        """Calcula puntuación de balance"""
        work_hours = sum(w.get("duration_hours", 0) for w in work_data)
        recovery_activities = len(recovery_data)
        
        # Lógica simplificada
        if work_hours > 40 and recovery_activities < 5:
            return 0.3  # Desbalanceado
        elif work_hours <= 40 and recovery_activities >= 5:
            return 0.8  # Balanceado
        
        return 0.5  # Neutral
    
    def _generate_balance_recommendations(self, work_data: List[Dict], recovery_data: List[Dict]) -> List[str]:
        """Genera recomendaciones de balance"""
        balance_score = self._calculate_balance_score(work_data, recovery_data)
        
        if balance_score < 0.5:
            return [
                "Prioriza actividades de recuperación",
                "Establece límites de tiempo de trabajo"
            ]
        
        return ["Mantén el balance entre trabajo y recuperación"]

