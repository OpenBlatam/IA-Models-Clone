"""
Sistema de análisis predictivo avanzado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import statistics


@dataclass
class PredictiveInsight:
    """Insight predictivo"""
    metric: str
    current_value: float
    predicted_value_7d: float
    predicted_value_30d: float
    predicted_value_90d: float
    confidence: float
    risk_level: str  # "low", "medium", "high"
    recommendation: str
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "metric": self.metric,
            "current_value": self.current_value,
            "predicted_value_7d": self.predicted_value_7d,
            "predicted_value_30d": self.predicted_value_30d,
            "predicted_value_90d": self.predicted_value_90d,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "created_at": self.created_at
        }


class PredictiveAnalytics:
    """Sistema de análisis predictivo avanzado"""
    
    def __init__(self):
        """Inicializa el sistema"""
        pass
    
    def generate_predictive_insights(self, historical_data: List[Dict],
                                    metrics: List[str] = None) -> List[PredictiveInsight]:
        """
        Genera insights predictivos
        
        Args:
            historical_data: Datos históricos
            metrics: Lista de métricas a analizar
            
        Returns:
            Lista de insights predictivos
        """
        if metrics is None:
            metrics = ["overall_score", "hydration_score", "texture_score"]
        
        insights = []
        
        for metric in metrics:
            # Extraer valores históricos de la métrica
            values = self._extract_metric_values(historical_data, metric)
            
            if len(values) < 3:
                continue
            
            # Predicciones para diferentes períodos
            pred_7d = self._predict_value(values, days=7)
            pred_30d = self._predict_value(values, days=30)
            pred_90d = self._predict_value(values, days=90)
            
            # Calcular confianza
            confidence = self._calculate_confidence(values)
            
            # Determinar nivel de riesgo
            risk_level = self._determine_risk_level(values[-1], pred_30d)
            
            # Generar recomendación
            recommendation = self._generate_recommendation(metric, values[-1], pred_30d, risk_level)
            
            insight = PredictiveInsight(
                metric=metric,
                current_value=values[-1],
                predicted_value_7d=pred_7d,
                predicted_value_30d=pred_30d,
                predicted_value_90d=pred_90d,
                confidence=confidence,
                risk_level=risk_level,
                recommendation=recommendation
            )
            
            insights.append(insight)
        
        return insights
    
    def _extract_metric_values(self, historical_data: List[Dict], metric: str) -> List[float]:
        """Extrae valores de una métrica"""
        values = []
        
        for data_point in historical_data:
            value = data_point.get("quality_scores", {}).get(metric, None)
            if value is not None:
                values.append(float(value))
        
        return values
    
    def _predict_value(self, values: List[float], days: int) -> float:
        """Predice valor futuro"""
        if len(values) < 2:
            return values[-1] if values else 0.0
        
        # Regresión lineal simple
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calcular pendiente
        slope = np.polyfit(x, y, 1)[0]
        intercept = np.polyfit(x, y, 1)[1]
        
        # Predecir
        future_x = len(values) + (days / 7)  # Asumiendo datos semanales
        predicted = slope * future_x + intercept
        
        return float(max(0, min(100, predicted)))  # Limitar entre 0-100
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """Calcula confianza de la predicción"""
        if len(values) < 3:
            return 0.3
        
        # Confianza basada en:
        # 1. Cantidad de datos
        data_confidence = min(1.0, len(values) / 20.0)
        
        # 2. Consistencia (menor varianza = mayor confianza)
        variance = np.var(values)
        mean_value = np.mean(values)
        consistency = 1.0 / (1.0 + variance / max(mean_value, 1))
        
        # Confianza combinada
        confidence = (data_confidence * 0.4 + consistency * 0.6)
        
        return float(confidence)
    
    def _determine_risk_level(self, current_value: float, predicted_value: float) -> str:
        """Determina nivel de riesgo"""
        change = predicted_value - current_value
        
        if predicted_value < 40 or change < -10:
            return "high"
        elif predicted_value < 60 or change < -5:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendation(self, metric: str, current: float,
                                predicted: float, risk_level: str) -> str:
        """Genera recomendación"""
        if risk_level == "high":
            return f"Acción inmediata requerida. {metric} predicho en {predicted:.1f}. Consulta profesional recomendada."
        elif risk_level == "medium":
            return f"Monitorea {metric} de cerca. Predicción: {predicted:.1f}. Ajusta tu rutina si es necesario."
        else:
            return f"{metric} se mantiene estable. Predicción: {predicted:.1f}. Continúa con tu rutina actual."
    
    def get_risk_assessment(self, insights: List[PredictiveInsight]) -> Dict:
        """Obtiene evaluación de riesgo general"""
        if not insights:
            return {"overall_risk": "unknown"}
        
        high_risk = sum(1 for i in insights if i.risk_level == "high")
        medium_risk = sum(1 for i in insights if i.risk_level == "medium")
        low_risk = sum(1 for i in insights if i.risk_level == "low")
        
        if high_risk > 0:
            overall_risk = "high"
        elif medium_risk > 0:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "overall_risk": overall_risk,
            "high_risk_metrics": high_risk,
            "medium_risk_metrics": medium_risk,
            "low_risk_metrics": low_risk,
            "total_metrics": len(insights)
        }






