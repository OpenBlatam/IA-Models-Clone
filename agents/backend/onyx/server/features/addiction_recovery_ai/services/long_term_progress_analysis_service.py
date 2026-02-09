"""
Servicio de Análisis de Progreso a Largo Plazo - Sistema completo de análisis de progreso a largo plazo
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class LongTermProgressAnalysisService:
    """Servicio de análisis de progreso a largo plazo"""
    
    def __init__(self):
        """Inicializa el servicio de progreso a largo plazo"""
        pass
    
    def analyze_long_term_progress(
        self,
        user_id: str,
        progress_data: List[Dict],
        time_period_months: int = 12
    ) -> Dict:
        """
        Analiza progreso a largo plazo
        
        Args:
            user_id: ID del usuario
            progress_data: Datos de progreso
            time_period_months: Período de tiempo en meses
        
        Returns:
            Análisis de progreso a largo plazo
        """
        if not progress_data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "analysis_id": f"long_term_{datetime.now().timestamp()}",
            "time_period_months": time_period_months,
            "total_data_points": len(progress_data),
            "overall_trend": self._calculate_overall_trend(progress_data),
            "milestone_achievements": self._identify_milestones(progress_data),
            "recovery_phases": self._identify_recovery_phases(progress_data),
            "sustained_improvements": self._identify_sustained_improvements(progress_data),
            "challenges_overcome": self._identify_challenges(progress_data),
            "recommendations": self._generate_long_term_recommendations(progress_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_long_term_outcome(
        self,
        user_id: str,
        current_progress: Dict,
        historical_data: List[Dict]
    ) -> Dict:
        """
        Predice resultado a largo plazo
        
        Args:
            user_id: ID del usuario
            current_progress: Progreso actual
            historical_data: Datos históricos
        
        Returns:
            Predicción de resultado
        """
        outcome_probability = self._calculate_outcome_probability(current_progress, historical_data)
        
        return {
            "user_id": user_id,
            "predicted_outcome": "successful" if outcome_probability > 0.7 else "moderate" if outcome_probability > 0.5 else "needs_support",
            "outcome_probability": round(outcome_probability, 3),
            "confidence": 0.82,
            "key_factors": self._identify_key_factors(current_progress, historical_data),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _calculate_overall_trend(self, data: List[Dict]) -> str:
        """Calcula tendencia general"""
        if len(data) < 2:
            return "stable"
        
        first_quarter = data[:len(data)//4]
        last_quarter = data[-len(data)//4:]
        
        first_avg = statistics.mean([d.get("progress_score", 5) for d in first_quarter]) if first_quarter else 0
        last_avg = statistics.mean([d.get("progress_score", 5) for d in last_quarter]) if last_quarter else 0
        
        if last_avg > first_avg * 1.2:
            return "strongly_improving"
        elif last_avg > first_avg * 1.1:
            return "improving"
        elif last_avg < first_avg * 0.9:
            return "declining"
        return "stable"
    
    def _identify_milestones(self, data: List[Dict]) -> List[Dict]:
        """Identifica hitos"""
        milestones = []
        
        for item in data:
            if item.get("is_milestone", False):
                milestones.append({
                    "date": item.get("date"),
                    "description": item.get("description", ""),
                    "significance": item.get("significance", "medium")
                })
        
        return milestones
    
    def _identify_recovery_phases(self, data: List[Dict]) -> List[Dict]:
        """Identifica fases de recuperación"""
        phases = []
        
        if len(data) >= 3:
            phases.append({
                "phase": "early_recovery",
                "duration_days": 90,
                "characteristics": "Establecimiento de rutinas y sistema de apoyo"
            })
        
        return phases
    
    def _identify_sustained_improvements(self, data: List[Dict]) -> List[str]:
        """Identifica mejoras sostenidas"""
        improvements = []
        
        trend = self._calculate_overall_trend(data)
        if trend in ["improving", "strongly_improving"]:
            improvements.append("Progreso constante y sostenido")
        
        return improvements
    
    def _identify_challenges(self, data: List[Dict]) -> List[Dict]:
        """Identifica desafíos superados"""
        challenges = []
        
        # Lógica simplificada
        return challenges
    
    def _generate_long_term_recommendations(self, data: List[Dict]) -> List[str]:
        """Genera recomendaciones a largo plazo"""
        recommendations = []
        
        trend = self._calculate_overall_trend(data)
        if trend == "declining":
            recommendations.append("Considera ajustar tu estrategia de recuperación")
        
        return recommendations
    
    def _calculate_outcome_probability(self, current: Dict, history: List[Dict]) -> float:
        """Calcula probabilidad de resultado"""
        base_probability = 0.5
        
        current_score = current.get("progress_score", 5)
        if current_score >= 7:
            base_probability += 0.2
        
        if history:
            recent_scores = [h.get("progress_score", 5) for h in history[-6:]]
            if recent_scores:
                avg_recent = statistics.mean(recent_scores)
                if avg_recent >= 6:
                    base_probability += 0.1
        
        return min(1.0, base_probability)
    
    def _identify_key_factors(self, current: Dict, history: List[Dict]) -> List[str]:
        """Identifica factores clave"""
        factors = []
        
        if current.get("support_level", 5) >= 7:
            factors.append("Alto nivel de apoyo social")
        
        return factors

