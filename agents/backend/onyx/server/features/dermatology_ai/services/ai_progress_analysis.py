"""
Sistema de análisis de progreso con IA
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class AIProgressInsight:
    """Insight de progreso con IA"""
    insight_type: str  # "improvement", "regression", "stability", "anomaly"
    metric: str
    description: str
    confidence: float
    recommendation: str
    timeframe: str
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "insight_type": self.insight_type,
            "metric": self.metric,
            "description": self.description,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "timeframe": self.timeframe
        }


@dataclass
class AIProgressReport:
    """Reporte de progreso con IA"""
    id: str
    user_id: str
    insights: List[AIProgressInsight]
    overall_trend: str
    predicted_outcome: Dict
    action_items: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "insights": [i.to_dict() for i in self.insights],
            "overall_trend": self.overall_trend,
            "predicted_outcome": self.predicted_outcome,
            "action_items": self.action_items,
            "created_at": self.created_at
        }


class AIProgressAnalysis:
    """Sistema de análisis de progreso con IA"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[AIProgressReport]] = {}  # user_id -> [reports]
        self.historical_data: Dict[str, List[Dict]] = {}  # user_id -> [data_points]
    
    def add_data_point(self, user_id: str, timestamp: str, metrics: Dict):
        """Agrega punto de datos"""
        data_point = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        if user_id not in self.historical_data:
            self.historical_data[user_id] = []
        
        self.historical_data[user_id].append(data_point)
    
    def analyze_progress(self, user_id: str, days: int = 90) -> AIProgressReport:
        """Analiza progreso con IA"""
        data_points = self.historical_data.get(user_id, [])
        
        if len(data_points) < 3:
            return AIProgressReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                insights=[],
                overall_trend="insufficient_data",
                predicted_outcome={},
                action_items=["Necesitas más datos para análisis"]
            )
        
        # Filtrar por período
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = [
            d for d in data_points
            if datetime.fromisoformat(d["timestamp"]) >= cutoff
        ]
        
        if len(recent_data) < 3:
            return AIProgressReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                insights=[],
                overall_trend="insufficient_data",
                predicted_outcome={},
                action_items=["Necesitas más datos en el período especificado"]
            )
        
        insights = []
        
        # Analizar cada métrica
        for metric_name in ["overall_score", "hydration_score", "texture_score"]:
            values = [d["metrics"].get(metric_name, 0) for d in recent_data]
            
            if not values or len(values) < 2:
                continue
            
            # Calcular tendencia
            first_value = values[0]
            last_value = values[-1]
            change = last_value - first_value
            change_percentage = (change / first_value * 100) if first_value > 0 else 0
            
            # Determinar tipo de insight
            if change_percentage > 10:
                insight_type = "improvement"
                description = f"{metric_name} mejoró {change_percentage:.1f}%"
                recommendation = f"Continúa con tu rutina actual para {metric_name}"
            elif change_percentage < -10:
                insight_type = "regression"
                description = f"{metric_name} empeoró {abs(change_percentage):.1f}%"
                recommendation = f"Revisa y ajusta tu rutina para {metric_name}"
            else:
                insight_type = "stability"
                description = f"{metric_name} se mantiene estable"
                recommendation = f"Considera intensificar tratamiento para {metric_name}"
            
            insight = AIProgressInsight(
                insight_type=insight_type,
                metric=metric_name,
                description=description,
                confidence=0.85,
                recommendation=recommendation,
                timeframe=f"{days} días"
            )
            insights.append(insight)
        
        # Tendencia general
        improvements = sum(1 for i in insights if i.insight_type == "improvement")
        regressions = sum(1 for i in insights if i.insight_type == "regression")
        
        if improvements > regressions:
            overall_trend = "improving"
        elif regressions > improvements:
            overall_trend = "declining"
        else:
            overall_trend = "stable"
        
        # Predicción de resultado
        predicted_outcome = {
            "next_30_days": "Continuación de tendencia actual",
            "confidence": 0.75,
            "expected_improvement": improvements * 2.0
        }
        
        # Items de acción
        action_items = []
        if overall_trend == "declining":
            action_items.append("Revisa tu rutina actual")
            action_items.append("Considera consultar con un dermatólogo")
        elif overall_trend == "improving":
            action_items.append("Mantén tu rutina actual")
            action_items.append("Continúa monitoreando el progreso")
        
        report = AIProgressReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            insights=insights,
            overall_trend=overall_trend,
            predicted_outcome=predicted_outcome,
            action_items=action_items
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def get_user_reports(self, user_id: str) -> List[AIProgressReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])






