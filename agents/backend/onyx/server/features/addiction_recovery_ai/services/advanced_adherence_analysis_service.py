"""
Servicio de Análisis de Adherencia Avanzado - Sistema completo de adherencia
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedAdherenceAnalysisService:
    """Servicio de análisis de adherencia avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de adherencia"""
        pass
    
    def calculate_adherence_rate(
        self,
        user_id: str,
        expected_actions: List[Dict],
        completed_actions: List[Dict],
        period_days: int = 30
    ) -> Dict:
        """
        Calcula tasa de adherencia
        
        Args:
            user_id: ID del usuario
            expected_actions: Acciones esperadas
            completed_actions: Acciones completadas
            period_days: Período en días
        
        Returns:
            Análisis de adherencia
        """
        total_expected = len(expected_actions)
        total_completed = len(completed_actions)
        
        adherence_rate = (total_completed / total_expected * 100) if total_expected > 0 else 0
        
        return {
            "user_id": user_id,
            "period_days": period_days,
            "total_expected": total_expected,
            "total_completed": total_completed,
            "adherence_rate": round(adherence_rate, 2),
            "adherence_level": self._determine_adherence_level(adherence_rate),
            "trend": self._calculate_adherence_trend(expected_actions, completed_actions),
            "barriers": self._identify_adherence_barriers(expected_actions, completed_actions),
            "recommendations": self._generate_adherence_recommendations(adherence_rate),
            "generated_at": datetime.now().isoformat()
        }
    
    def analyze_adherence_patterns(
        self,
        user_id: str,
        adherence_data: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de adherencia
        
        Args:
            user_id: ID del usuario
            adherence_data: Datos de adherencia
        
        Returns:
            Análisis de patrones
        """
        if not adherence_data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "total_records": len(adherence_data),
            "daily_patterns": self._analyze_daily_patterns(adherence_data),
            "weekly_patterns": self._analyze_weekly_patterns(adherence_data),
            "adherence_by_category": self._analyze_by_category(adherence_data),
            "predictors": self._identify_adherence_predictors(adherence_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_adherence_risk(
        self,
        user_id: str,
        current_state: Dict,
        historical_adherence: List[Dict]
    ) -> Dict:
        """
        Predice riesgo de no adherencia
        
        Args:
            user_id: ID del usuario
            current_state: Estado actual
            historical_adherence: Adherencia histórica
        
        Returns:
            Predicción de riesgo
        """
        risk_score = self._calculate_adherence_risk(current_state, historical_adherence)
        
        return {
            "user_id": user_id,
            "non_adherence_risk": round(risk_score, 3),
            "risk_level": self._determine_risk_level(risk_score),
            "risk_factors": self._identify_risk_factors(current_state),
            "interventions": self._suggest_interventions(risk_score),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _determine_adherence_level(self, rate: float) -> str:
        """Determina nivel de adherencia"""
        if rate >= 90:
            return "excellent"
        elif rate >= 75:
            return "good"
        elif rate >= 60:
            return "fair"
        elif rate >= 40:
            return "poor"
        else:
            return "very_poor"
    
    def _calculate_adherence_trend(self, expected: List[Dict], completed: List[Dict]) -> str:
        """Calcula tendencia de adherencia"""
        if len(completed) < 2:
            return "stable"
        
        # Lógica simplificada
        return "improving"
    
    def _identify_adherence_barriers(self, expected: List[Dict], completed: List[Dict]) -> List[str]:
        """Identifica barreras de adherencia"""
        barriers = []
        
        missed_count = len(expected) - len(completed)
        if missed_count > len(expected) * 0.2:
            barriers.append("Alto número de acciones perdidas")
        
        return barriers
    
    def _generate_adherence_recommendations(self, rate: float) -> List[str]:
        """Genera recomendaciones de adherencia"""
        recommendations = []
        
        if rate < 75:
            recommendations.append("Considera establecer recordatorios más frecuentes")
            recommendations.append("Revisa las barreras que pueden estar afectando tu adherencia")
        
        return recommendations
    
    def _analyze_daily_patterns(self, data: List[Dict]) -> Dict:
        """Analiza patrones diarios"""
        return {}
    
    def _analyze_weekly_patterns(self, data: List[Dict]) -> Dict:
        """Analiza patrones semanales"""
        return {}
    
    def _analyze_by_category(self, data: List[Dict]) -> Dict:
        """Analiza por categoría"""
        return {}
    
    def _identify_adherence_predictors(self, data: List[Dict]) -> List[str]:
        """Identifica predictores de adherencia"""
        return []
    
    def _calculate_adherence_risk(self, current: Dict, history: List[Dict]) -> float:
        """Calcula riesgo de no adherencia"""
        base_risk = 0.3
        
        recent_adherence = history[-7:] if len(history) >= 7 else history
        if recent_adherence:
            avg_rate = statistics.mean([h.get("adherence_rate", 0) for h in recent_adherence])
            if avg_rate < 70:
                base_risk += 0.3
        
        return min(1.0, base_risk)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determina nivel de riesgo"""
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _identify_risk_factors(self, current: Dict) -> List[str]:
        """Identifica factores de riesgo"""
        factors = []
        
        if current.get("stress_level", 5) >= 7:
            factors.append("Nivel de estrés elevado")
        
        return factors
    
    def _suggest_interventions(self, risk_score: float) -> List[str]:
        """Sugiere intervenciones"""
        interventions = []
        
        if risk_score >= 0.7:
            interventions.append("Intervención intensiva recomendada")
            interventions.append("Contacto diario con sistema de apoyo")
        
        return interventions

