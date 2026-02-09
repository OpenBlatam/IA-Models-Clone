"""
Servicio de Análisis de Patrones Temporales - Análisis de patrones en el tiempo
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict


class TemporalPatternAnalysisService:
    """Servicio de análisis de patrones temporales"""
    
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
            data: Datos históricos
            metric: Métrica a analizar
        
        Returns:
            Análisis de patrones diarios
        """
        hourly_data = defaultdict(list)
        
        for entry in data:
            timestamp = datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat()))
            hour = timestamp.hour
            value = entry.get(metric, 0)
            hourly_data[hour].append(value)
        
        # Calcular promedios por hora
        hourly_averages = {}
        for hour in range(24):
            if hour in hourly_data:
                hourly_averages[hour] = sum(hourly_data[hour]) / len(hourly_data[hour])
            else:
                hourly_averages[hour] = 0
        
        # Identificar horas críticas
        critical_hours = []
        for hour, avg in hourly_averages.items():
            if metric == "cravings_level" and avg >= 6:
                critical_hours.append(hour)
            elif metric == "stress_level" and avg >= 7:
                critical_hours.append(hour)
        
        return {
            "user_id": user_id,
            "metric": metric,
            "hourly_averages": hourly_averages,
            "critical_hours": critical_hours,
            "peak_hour": max(hourly_averages.items(), key=lambda x: x[1])[0] if hourly_averages else None,
            "lowest_hour": min(hourly_averages.items(), key=lambda x: x[1])[0] if hourly_averages else None,
            "generated_at": datetime.now().isoformat()
        }
    
    def analyze_weekly_patterns(
        self,
        user_id: str,
        data: List[Dict],
        metric: str = "check_ins"
    ) -> Dict:
        """
        Analiza patrones semanales
        
        Args:
            user_id: ID del usuario
            data: Datos históricos
            metric: Métrica a analizar
        
        Returns:
            Análisis de patrones semanales
        """
        weekday_data = defaultdict(list)
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for entry in data:
            timestamp = datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat()))
            weekday = timestamp.weekday()
            value = entry.get(metric, 0)
            weekday_data[weekday].append(value)
        
        # Calcular promedios por día de la semana
        weekday_averages = {}
        for weekday in range(7):
            if weekday in weekday_data:
                weekday_averages[weekday_names[weekday]] = sum(weekday_data[weekday]) / len(weekday_data[weekday])
            else:
                weekday_averages[weekday_names[weekday]] = 0
        
        # Identificar días críticos
        critical_days = []
        for day, avg in weekday_averages.items():
            if metric == "cravings_level" and avg >= 6:
                critical_days.append(day)
        
        return {
            "user_id": user_id,
            "metric": metric,
            "weekday_averages": weekday_averages,
            "critical_days": critical_days,
            "best_day": max(weekday_averages.items(), key=lambda x: x[1])[0] if weekday_averages else None,
            "worst_day": min(weekday_averages.items(), key=lambda x: x[1])[0] if weekday_averages else None,
            "generated_at": datetime.now().isoformat()
        }
    
    def analyze_seasonal_patterns(
        self,
        user_id: str,
        data: List[Dict],
        metric: str = "relapse_risk"
    ) -> Dict:
        """
        Analiza patrones estacionales
        
        Args:
            user_id: ID del usuario
            data: Datos históricos
            metric: Métrica a analizar
        
        Returns:
            Análisis de patrones estacionales
        """
        monthly_data = defaultdict(list)
        month_names = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        for entry in data:
            timestamp = datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat()))
            month = timestamp.month
            value = entry.get(metric, 0)
            monthly_data[month].append(value)
        
        # Calcular promedios por mes
        monthly_averages = {}
        for month in range(1, 13):
            if month in monthly_data:
                monthly_averages[month_names[month - 1]] = sum(monthly_data[month]) / len(monthly_data[month])
            else:
                monthly_averages[month_names[month - 1]] = 0
        
        return {
            "user_id": user_id,
            "metric": metric,
            "monthly_averages": monthly_averages,
            "peak_month": max(monthly_averages.items(), key=lambda x: x[1])[0] if monthly_averages else None,
            "lowest_month": min(monthly_averages.items(), key=lambda x: x[1])[0] if monthly_averages else None,
            "generated_at": datetime.now().isoformat()
        }
    
    def detect_anomalies(
        self,
        user_id: str,
        data: List[Dict],
        metric: str = "mood"
    ) -> Dict:
        """
        Detecta anomalías en patrones temporales
        
        Args:
            user_id: ID del usuario
            data: Datos históricos
            metric: Métrica a analizar
        
        Returns:
            Anomalías detectadas
        """
        values = [entry.get(metric, 0) for entry in data]
        
        if not values:
            return {
                "user_id": user_id,
                "anomalies": [],
                "message": "Insufficient data"
            }
        
        # Calcular estadísticas
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        # Detectar valores atípicos (más de 2 desviaciones estándar)
        anomalies = []
        for i, entry in enumerate(data):
            value = entry.get(metric, 0)
            if abs(value - mean) > 2 * std_dev:
                anomalies.append({
                    "timestamp": entry.get("timestamp"),
                    "value": value,
                    "deviation": value - mean,
                    "severity": "high" if abs(value - mean) > 3 * std_dev else "medium"
                })
        
        return {
            "user_id": user_id,
            "metric": metric,
            "mean": round(mean, 2),
            "std_dev": round(std_dev, 2),
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "generated_at": datetime.now().isoformat()
        }
    
    def predict_future_patterns(
        self,
        user_id: str,
        historical_data: List[Dict],
        days_ahead: int = 7
    ) -> Dict:
        """
        Predice patrones futuros
        
        Args:
            user_id: ID del usuario
            historical_data: Datos históricos
            days_ahead: Días a predecir
        
        Returns:
            Predicción de patrones futuros
        """
        # Análisis simplificado de tendencias
        recent_data = historical_data[-7:] if len(historical_data) >= 7 else historical_data
        
        if not recent_data:
            return {
                "user_id": user_id,
                "predictions": [],
                "message": "Insufficient data for prediction"
            }
        
        # Calcular tendencia
        avg_recent = sum(entry.get("mood", 5) for entry in recent_data) / len(recent_data)
        
        predictions = []
        for day in range(days_ahead):
            predicted_date = datetime.now() + timedelta(days=day)
            predictions.append({
                "date": predicted_date.date().isoformat(),
                "predicted_mood": round(avg_recent, 2),
                "confidence": 0.7,
                "factors": ["historical_average"]
            })
        
        return {
            "user_id": user_id,
            "predictions": predictions,
            "days_ahead": days_ahead,
            "generated_at": datetime.now().isoformat()
        }

