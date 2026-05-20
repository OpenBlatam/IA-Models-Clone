"""
Servicio de Predicción de Éxito a Largo Plazo - Sistema completo de predicción
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class LongTermSuccessPredictionService:
    """Servicio de predicción de éxito a largo plazo"""
    
    def __init__(self):
        """Inicializa el servicio de predicción"""
        pass
    
    def predict_long_term_success(
        self,
        user_id: str,
        current_state: Dict,
        historical_data: List[Dict],
        prediction_horizon_years: int = 5
    ) -> Dict:
        """
        Predice éxito a largo plazo
        
        Args:
            user_id: ID del usuario
            current_state: Estado actual
            historical_data: Datos históricos
            prediction_horizon_years: Horizonte de predicción en años
        
        Returns:
            Predicción de éxito
        """
        success_probability = self._calculate_success_probability(
            current_state,
            historical_data,
            prediction_horizon_years
        )
        
        return {
            "user_id": user_id,
            "prediction_horizon_years": prediction_horizon_years,
            "success_probability": round(success_probability, 3),
            "confidence": self._calculate_confidence(historical_data),
            "key_factors": self._identify_key_success_factors(current_state, historical_data),
            "risk_factors": self._identify_risk_factors(current_state, historical_data),
            "recommendations": self._generate_success_recommendations(success_probability, current_state),
            "predicted_at": datetime.now().isoformat()
        }
    
    def predict_milestone_achievement(
        self,
        user_id: str,
        milestone: str,
        current_progress: Dict
    ) -> Dict:
        """
        Predice logro de hito
        
        Args:
            user_id: ID del usuario
            milestone: Hito a predecir
            current_progress: Progreso actual
        
        Returns:
            Predicción de logro
        """
        achievement_probability = self._calculate_milestone_probability(milestone, current_progress)
        
        return {
            "user_id": user_id,
            "milestone": milestone,
            "achievement_probability": round(achievement_probability, 3),
            "estimated_date": self._estimate_achievement_date(milestone, current_progress),
            "recommendations": self._generate_milestone_recommendations(milestone, achievement_probability),
            "predicted_at": datetime.now().isoformat()
        }
    
    def analyze_success_trajectory(
        self,
        user_id: str,
        historical_data: List[Dict]
    ) -> Dict:
        """
        Analiza trayectoria de éxito
        
        Args:
            user_id: ID del usuario
            historical_data: Datos históricos
        
        Returns:
            Análisis de trayectoria
        """
        return {
            "user_id": user_id,
            "trajectory": self._calculate_trajectory(historical_data),
            "trend": self._analyze_trend(historical_data),
            "projected_outcome": self._project_outcome(historical_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_success_probability(
        self,
        current_state: Dict,
        historical_data: List[Dict],
        horizon_years: int
    ) -> float:
        """Calcula probabilidad de éxito"""
        base_probability = 0.5
        
        # Factores positivos
        days_sober = current_state.get("days_sober", 0)
        if days_sober > 365:
            base_probability += 0.2
        elif days_sober > 90:
            base_probability += 0.15
        elif days_sober > 30:
            base_probability += 0.1
        
        support_level = current_state.get("support_level", 5)
        if support_level >= 8:
            base_probability += 0.15
        elif support_level >= 6:
            base_probability += 0.1
        
        # Factores negativos
        stress_level = current_state.get("stress_level", 5)
        if stress_level >= 8:
            base_probability -= 0.15
        elif stress_level >= 6:
            base_probability -= 0.1
        
        # Ajustar por horizonte
        if horizon_years > 3:
            base_probability *= 0.9
        
        return max(0.0, min(1.0, base_probability))
    
    def _calculate_confidence(self, historical_data: List[Dict]) -> float:
        """Calcula confianza en predicción"""
        if len(historical_data) < 30:
            return 0.5
        elif len(historical_data) < 90:
            return 0.7
        else:
            return 0.85
    
    def _identify_key_success_factors(self, current_state: Dict, historical_data: List[Dict]) -> List[str]:
        """Identifica factores clave de éxito"""
        factors = []
        
        days_sober = current_state.get("days_sober", 0)
        if days_sober > 90:
            factors.append("Sobriedad sostenida")
        
        support_level = current_state.get("support_level", 5)
        if support_level >= 7:
            factors.append("Sistema de apoyo fuerte")
        
        return factors
    
    def _identify_risk_factors(self, current_state: Dict, historical_data: List[Dict]) -> List[str]:
        """Identifica factores de riesgo"""
        risk_factors = []
        
        stress_level = current_state.get("stress_level", 5)
        if stress_level >= 7:
            risk_factors.append("Nivel de estrés elevado")
        
        return risk_factors
    
    def _generate_success_recommendations(
        self,
        success_probability: float,
        current_state: Dict
    ) -> List[str]:
        """Genera recomendaciones de éxito"""
        recommendations = []
        
        if success_probability < 0.6:
            recommendations.append("Fortalecer sistema de apoyo es crucial para éxito a largo plazo")
            recommendations.append("Considera terapia o counseling adicional")
        
        if current_state.get("support_level", 5) < 6:
            recommendations.append("Desarrollar red de apoyo más fuerte")
        
        return recommendations
    
    def _calculate_milestone_probability(self, milestone: str, progress: Dict) -> float:
        """Calcula probabilidad de lograr hito"""
        # Lógica simplificada
        days_sober = progress.get("days_sober", 0)
        
        milestone_days = {
            "30_days": 30,
            "90_days": 90,
            "1_year": 365,
            "5_years": 1825
        }
        
        target_days = milestone_days.get(milestone, 0)
        
        if days_sober >= target_days:
            return 1.0
        elif days_sober >= target_days * 0.8:
            return 0.8
        elif days_sober >= target_days * 0.5:
            return 0.6
        else:
            return 0.4
    
    def _estimate_achievement_date(self, milestone: str, progress: Dict) -> str:
        """Estima fecha de logro"""
        days_sober = progress.get("days_sober", 0)
        
        milestone_days = {
            "30_days": 30,
            "90_days": 90,
            "1_year": 365
        }
        
        target_days = milestone_days.get(milestone, 0)
        remaining_days = max(0, target_days - days_sober)
        
        estimated_date = datetime.now() + timedelta(days=remaining_days)
        return estimated_date.isoformat()
    
    def _generate_milestone_recommendations(self, milestone: str, probability: float) -> List[str]:
        """Genera recomendaciones de hito"""
        recommendations = []
        
        if probability < 0.7:
            recommendations.append("Mantén tu compromiso diario para alcanzar este hito")
        
        return recommendations
    
    def _calculate_trajectory(self, historical_data: List[Dict]) -> str:
        """Calcula trayectoria"""
        if len(historical_data) < 2:
            return "insufficient_data"
        
        # Lógica simplificada
        return "positive"
    
    def _analyze_trend(self, historical_data: List[Dict]) -> str:
        """Analiza tendencia"""
        if len(historical_data) < 7:
            return "stable"
        
        # Lógica simplificada
        return "improving"
    
    def _project_outcome(self, historical_data: List[Dict]) -> str:
        """Proyecta resultado"""
        return "favorable"

