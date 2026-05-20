"""
Servicio de Análisis de Patrones Temporales Avanzado - Sistema completo de análisis temporal
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class AdvancedTemporalPatternAnalysisService:
    """Servicio de análisis de patrones temporales avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de análisis temporal"""
        pass
    
    def analyze_daily_patterns(
        self,
        user_id: str,
        data: List[Dict],
        metric: str = "mood"
    ) -> Dict:
        """
        Analiza patrones diarios
        
        Args:
            user_id: ID del usuario
            data: Datos a analizar
            metric: Métrica a analizar
        
        Returns:
            Análisis de patrones diarios
        """
        if not data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        hourly_values = defaultdict(list)
        
        for record in data:
            timestamp = record.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    value = record.get(metric, 0)
                    hourly_values[dt.hour].append(value)
                except:
                    pass
        
        hourly_averages = {
            hour: statistics.mean(values) if values else 0
            for hour, values in hourly_values.items()
        }
        
        return {
            "user_id": user_id,
            "metric": metric,
            "hourly_patterns": hourly_averages,
            "peak_hours": sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)[:3],
            "low_hours": sorted(hourly_averages.items(), key=lambda x: x[1])[:3],
            "generated_at": datetime.now().isoformat()
        }
    
    def analyze_weekly_patterns(
        self,
        user_id: str,
        data: List[Dict],
        metric: str = "mood"
    ) -> Dict:
        """
        Analiza patrones semanales
        
        Args:
            user_id: ID del usuario
            data: Datos a analizar
            metric: Métrica a analizar
        
        Returns:
            Análisis de patrones semanales
        """
        if not data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        weekday_values = defaultdict(list)
        
        for record in data:
            timestamp = record.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    weekday = dt.weekday()  # 0 = Monday, 6 = Sunday
                    value = record.get(metric, 0)
                    weekday_values[weekday].append(value)
                except:
                    pass
        
        weekday_averages = {
            day: statistics.mean(values) if values else 0
            for day, values in weekday_values.items()
        }
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_named = {day_names[day]: avg for day, avg in weekday_averages.items()}
        
        return {
            "user_id": user_id,
            "metric": metric,
            "weekly_patterns": weekday_named,
            "best_day": max(weekday_named.items(), key=lambda x: x[1])[0] if weekday_named else None,
            "worst_day": min(weekday_named.items(), key=lambda x: x[1])[0] if weekday_named else None,
            "generated_at": datetime.now().isoformat()
        }
    
    def analyze_seasonal_patterns(
        self,
        user_id: str,
        data: List[Dict],
        metric: str = "mood"
    ) -> Dict:
        """
        Analiza patrones estacionales
        
        Args:
            user_id: ID del usuario
            data: Datos a analizar
            metric: Métrica a analizar
        
        Returns:
            Análisis de patrones estacionales
        """
        if not data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        seasonal_values = defaultdict(list)
        
        for record in data:
            timestamp = record.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    month = dt.month
                    season = self._get_season(month)
                    value = record.get(metric, 0)
                    seasonal_values[season].append(value)
                except:
                    pass
        
        seasonal_averages = {
            season: statistics.mean(values) if values else 0
            for season, values in seasonal_values.items()
        }
        
        return {
            "user_id": user_id,
            "metric": metric,
            "seasonal_patterns": seasonal_averages,
            "best_season": max(seasonal_averages.items(), key=lambda x: x[1])[0] if seasonal_averages else None,
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_temporal_risk(
        self,
        user_id: str,
        current_time: datetime,
        historical_patterns: Dict
    ) -> Dict:
        """
        Predice riesgo temporal
        
        Args:
            user_id: ID del usuario
            current_time: Tiempo actual
            historical_patterns: Patrones históricos
        
        Returns:
            Predicción de riesgo temporal
        """
        hour = current_time.hour
        weekday = current_time.weekday()
        
        risk_score = 0.5  # Base
        
        # Ajustar por hora
        if hour >= 18 and hour <= 23:  # Noche
            risk_score += 0.2
        
        # Ajustar por día de semana
        if weekday >= 4:  # Viernes, Sábado, Domingo
            risk_score += 0.1
        
        return {
            "user_id": user_id,
            "current_time": current_time.isoformat(),
            "predicted_risk_score": min(1.0, risk_score),
            "risk_level": "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.4 else "low",
            "temporal_factors": self._identify_temporal_factors(hour, weekday),
            "recommendations": self._generate_temporal_recommendations(risk_score),
            "predicted_at": datetime.now().isoformat()
        }
    
    def _get_season(self, month: int) -> str:
        """Obtiene estación del año"""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    def _identify_temporal_factors(self, hour: int, weekday: int) -> List[str]:
        """Identifica factores temporales"""
        factors = []
        
        if hour >= 18:
            factors.append("Hora nocturna")
        if weekday >= 4:
            factors.append("Fin de semana")
        
        return factors
    
    def _generate_temporal_recommendations(self, risk_score: float) -> List[str]:
        """Genera recomendaciones temporales"""
        recommendations = []
        
        if risk_score >= 0.7:
            recommendations.append("⚠️ Período de alto riesgo. Ten precaución extra")
        elif risk_score >= 0.4:
            recommendations.append("Considera tener tu sistema de apoyo disponible")
        
        return recommendations

