"""
Demand Predictor - Predicción de Demanda
=======================================

Sistema de predicción de demanda usando ML para anticipar necesidades de recursos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class DemandForecast:
    """Pronóstico de demanda."""
    forecast_id: str
    resource_type: str
    predicted_value: float
    confidence: float
    timestamp: datetime
    time_horizon_minutes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class DemandPredictor:
    """Predictor de demanda con ML."""
    
    def __init__(self, history_window_hours: int = 24):
        self.history: Dict[str, deque] = {}
        self.forecasts: List[DemandForecast] = []
        self.history_window_hours = history_window_hours
        self._lock = asyncio.Lock()
    
    async def record_demand(
        self,
        resource_type: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Registrar demanda actual.
        
        Args:
            resource_type: Tipo de recurso (sessions, requests, etc.)
            value: Valor de demanda
            timestamp: Timestamp (opcional)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        async with self._lock:
            if resource_type not in self.history:
                self.history[resource_type] = deque(maxlen=10000)
            
            record = {
                "value": value,
                "timestamp": timestamp,
            }
            
            self.history[resource_type].append(record)
            
            # Limpiar registros antiguos
            cutoff = timestamp - timedelta(hours=self.history_window_hours)
            while (self.history[resource_type] and 
                   self.history[resource_type][0]["timestamp"] < cutoff):
                self.history[resource_type].popleft()
    
    async def predict_demand(
        self,
        resource_type: str,
        time_horizon_minutes: int = 5,
    ) -> Optional[DemandForecast]:
        """
        Predecir demanda futura.
        
        Args:
            resource_type: Tipo de recurso
            time_horizon_minutes: Horizonte de tiempo en minutos
        
        Returns:
            Pronóstico de demanda
        """
        history = self.history.get(resource_type)
        
        if not history or len(history) < 10:
            return None  # No hay suficientes datos
        
        # Calcular promedio móvil
        recent_values = [r["value"] for r in list(history)[-20:]]
        avg_value = statistics.mean(recent_values)
        
        # Calcular tendencia
        if len(recent_values) >= 10:
            first_half = statistics.mean(recent_values[:10])
            second_half = statistics.mean(recent_values[10:])
            trend = (second_half - first_half) / first_half if first_half > 0 else 0.0
        else:
            trend = 0.0
        
        # Predecir valor futuro
        predicted_value = avg_value * (1 + trend * (time_horizon_minutes / 60.0))
        
        # Calcular confianza basada en variabilidad
        if len(recent_values) > 1:
            std_dev = statistics.stdev(recent_values)
            cv = std_dev / avg_value if avg_value > 0 else 1.0
            confidence = max(0.1, min(1.0, 1.0 - cv))
        else:
            confidence = 0.5
        
        forecast = DemandForecast(
            forecast_id=f"forecast_{resource_type}_{datetime.now().timestamp()}",
            resource_type=resource_type,
            predicted_value=predicted_value,
            confidence=confidence,
            timestamp=datetime.now(),
            time_horizon_minutes=time_horizon_minutes,
            metadata={
                "trend": trend,
                "current_avg": avg_value,
                "history_size": len(history),
            },
        )
        
        self.forecasts.append(forecast)
        
        # Mantener solo últimos 1000 pronósticos
        if len(self.forecasts) > 1000:
            self.forecasts.pop(0)
        
        return forecast
    
    async def predict_multiple_resources(
        self,
        resource_types: List[str],
        time_horizon_minutes: int = 5,
    ) -> Dict[str, DemandForecast]:
        """Predecir demanda para múltiples recursos."""
        predictions = {}
        
        for resource_type in resource_types:
            forecast = await self.predict_demand(resource_type, time_horizon_minutes)
            if forecast:
                predictions[resource_type] = forecast
        
        return predictions
    
    def get_demand_history(
        self,
        resource_type: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de demanda."""
        history = self.history.get(resource_type, deque())
        
        return [
            {
                "value": r["value"],
                "timestamp": r["timestamp"].isoformat(),
            }
            for r in list(history)[-limit:]
        ]
    
    def get_forecast_history(
        self,
        resource_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de pronósticos."""
        forecasts = self.forecasts
        
        if resource_type:
            forecasts = [f for f in forecasts if f.resource_type == resource_type]
        
        return [
            {
                "forecast_id": f.forecast_id,
                "resource_type": f.resource_type,
                "predicted_value": f.predicted_value,
                "confidence": f.confidence,
                "timestamp": f.timestamp.isoformat(),
                "time_horizon_minutes": f.time_horizon_minutes,
            }
            for f in forecasts[-limit:]
        ]
















