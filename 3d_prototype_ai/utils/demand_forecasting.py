"""
Demand Forecasting - Sistema de predicción de demanda
======================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class DemandForecasting:
    """Sistema de predicción de demanda"""
    
    def __init__(self):
        self.historical_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.forecasts: Dict[str, Dict[str, Any]] = {}
    
    def record_demand(self, product_type: str, quantity: int, date: Optional[datetime] = None):
        """Registra demanda histórica"""
        if not date:
            date = datetime.now()
        
        demand_record = {
            "product_type": product_type,
            "quantity": quantity,
            "date": date.isoformat()
        }
        
        self.historical_data[product_type].append(demand_record)
        
        # Mantener solo últimos 2 años
        cutoff = datetime.now() - timedelta(days=730)
        self.historical_data[product_type] = [
            r for r in self.historical_data[product_type]
            if datetime.fromisoformat(r["date"]) > cutoff
        ]
    
    def forecast_demand(self, product_type: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Predice demanda futura"""
        historical = self.historical_data.get(product_type, [])
        
        if not historical:
            return {
                "product_type": product_type,
                "forecast": [],
                "confidence": 0.0,
                "message": "No hay datos históricos suficientes"
            }
        
        # Calcular promedio diario
        daily_demands = defaultdict(int)
        for record in historical:
            date = datetime.fromisoformat(record["date"]).date()
            daily_demands[date] += record["quantity"]
        
        if not daily_demands:
            return {
                "product_type": product_type,
                "forecast": [],
                "confidence": 0.0
            }
        
        avg_daily = sum(daily_demands.values()) / len(daily_demands)
        
        # Generar forecast (simplificado - en producción usaría modelos ML)
        forecast = []
        start_date = datetime.now().date()
        
        for i in range(days_ahead):
            forecast_date = start_date + timedelta(days=i)
            
            # Ajustar por día de la semana (simplificado)
            day_of_week = forecast_date.weekday()
            multiplier = 1.2 if day_of_week < 5 else 0.8  # Más demanda en semana
            
            predicted_demand = avg_daily * multiplier
            
            forecast.append({
                "date": forecast_date.isoformat(),
                "predicted_demand": round(predicted_demand, 2),
                "confidence": 0.75
            })
        
        total_forecast = sum(f["predicted_demand"] for f in forecast)
        
        forecast_result = {
            "product_type": product_type,
            "forecast": forecast,
            "total_forecast": total_forecast,
            "average_daily": round(avg_daily, 2),
            "confidence": 0.75,
            "generated_at": datetime.now().isoformat()
        }
        
        self.forecasts[product_type] = forecast_result
        
        return forecast_result
    
    def get_demand_trends(self, product_type: str, days: int = 90) -> Dict[str, Any]:
        """Obtiene tendencias de demanda"""
        cutoff = datetime.now() - timedelta(days=days)
        
        historical = [
            r for r in self.historical_data.get(product_type, [])
            if datetime.fromisoformat(r["date"]) > cutoff
        ]
        
        if not historical:
            return {"trend": "stable", "data": []}
        
        # Agrupar por semana
        weekly_demands = defaultdict(int)
        for record in historical:
            date = datetime.fromisoformat(record["date"]).date()
            week = date.isocalendar()[1]  # Número de semana
            weekly_demands[week] += record["quantity"]
        
        trend_data = [
            {
                "week": week,
                "total_demand": demand
            }
            for week, demand in sorted(weekly_demands.items())
        ]
        
        # Calcular tendencia
        if len(trend_data) >= 2:
            first_demand = trend_data[0]["total_demand"]
            last_demand = trend_data[-1]["total_demand"]
            
            if last_demand > first_demand * 1.1:
                trend = "increasing"
            elif last_demand < first_demand * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "product_type": product_type,
            "trend": trend,
            "data": trend_data,
            "period_days": days
        }




