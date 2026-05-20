"""
Sistema de análisis de progreso con machine learning avanzado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics
import numpy as np


@dataclass
class MLInsight:
    """Insight de ML"""
    insight_type: str
    metric: str
    description: str
    confidence: float
    data_points: int
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "insight_type": self.insight_type,
            "metric": self.metric,
            "description": self.description,
            "confidence": self.confidence,
            "data_points": self.data_points,
            "recommendation": self.recommendation
        }


@dataclass
class AdvancedMLReport:
    """Reporte de ML avanzado"""
    id: str
    user_id: str
    insights: List[MLInsight]
    correlations: Dict[str, float]
    patterns_detected: List[str]
    predictions: Dict[str, Dict]
    model_confidence: float
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
            "correlations": self.correlations,
            "patterns_detected": self.patterns_detected,
            "predictions": self.predictions,
            "model_confidence": self.model_confidence,
            "created_at": self.created_at
        }


class AdvancedMLAnalysis:
    """Sistema de análisis con ML avanzado"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[AdvancedMLReport]] = {}
        self.historical_data: Dict[str, List[Dict]] = {}
    
    def add_data_point(self, user_id: str, timestamp: str, metrics: Dict, context: Optional[Dict] = None):
        """Agrega punto de datos con contexto"""
        data_point = {
            "timestamp": timestamp,
            "metrics": metrics,
            "context": context or {}
        }
        
        if user_id not in self.historical_data:
            self.historical_data[user_id] = []
        
        self.historical_data[user_id].append(data_point)
        self.historical_data[user_id].sort(key=lambda x: x["timestamp"])
    
    def generate_ml_analysis(self, user_id: str, days: int = 90) -> AdvancedMLReport:
        """Genera análisis con ML avanzado"""
        data_points = self.historical_data.get(user_id, [])
        
        if len(data_points) < 5:
            return AdvancedMLReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                insights=[],
                correlations={},
                patterns_detected=[],
                predictions={},
                model_confidence=0.0
            )
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = [d for d in data_points if datetime.fromisoformat(d["timestamp"]) >= cutoff]
        
        if len(recent_data) < 5:
            return AdvancedMLReport(
                id=str(uuid.uuid4()),
                user_id=user_id,
                insights=[],
                correlations={},
                patterns_detected=[],
                predictions={},
                model_confidence=0.0
            )
        
        insights = []
        patterns = []
        metrics_data = {}
        
        for metric_name in ["overall_score", "hydration_score", "texture_score"]:
            values = [d["metrics"].get(metric_name, 0) for d in recent_data]
            if values:
                metrics_data[metric_name] = values
        
        for metric_name, values in metrics_data.items():
            if len(values) < 3:
                continue
            
            trend = self._detect_trend(values)
            if trend != "stable":
                insights.append(MLInsight(
                    insight_type="trend",
                    metric=metric_name,
                    description=f"Tendencia {trend} detectada",
                    confidence=0.75,
                    data_points=len(values),
                    recommendation=f"Continúa monitoreando {metric_name}"
                ))
                patterns.append(f"{trend}_trend_{metric_name}")
        
        correlations = {}
        if len(metrics_data) >= 2:
            metric_names = list(metrics_data.keys())
            for i, metric1 in enumerate(metric_names):
                for metric2 in metric_names[i+1:]:
                    corr = self._calculate_correlation(metrics_data[metric1], metrics_data[metric2])
                    if abs(corr) > 0.5:
                        correlations[f"{metric1}_{metric2}"] = corr
        
        predictions = {}
        for metric_name, values in metrics_data.items():
            if len(values) >= 3:
                future_value = self._predict_future(values, days=30)
                predictions[metric_name] = {
                    "current": values[-1],
                    "predicted_30_days": future_value,
                    "confidence": 0.70
                }
        
        model_confidence = min(0.95, 0.5 + (len(recent_data) / 100) * 0.45)
        
        report = AdvancedMLReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            insights=insights,
            correlations=correlations,
            patterns_detected=patterns,
            predictions=predictions,
            model_confidence=model_confidence
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        self.reports[user_id].append(report)
        return report
    
    def _detect_trend(self, values: List[float]) -> str:
        """Detecta tendencia"""
        if len(values) < 3:
            return "stable"
        x = np.array(range(len(values)))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.5:
            return "improving"
        elif slope < -0.5:
            return "declining"
        return "stable"
    
    def _detect_anomalies(self, values: List[float]) -> List[int]:
        """Detecta anomalías"""
        if len(values) < 3:
            return []
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        if std == 0:
            return []
        anomalies = []
        for i, value in enumerate(values):
            if abs((value - mean) / std) > 2:
                anomalies.append(i)
        return anomalies
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calcula correlación"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        correlation = np.corrcoef(np.array(x), np.array(y))[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _predict_future(self, values: List[float], days: int = 30) -> float:
        """Predice valor futuro"""
        if len(values) < 2:
            return values[-1] if values else 0.0
        n = len(values)
        x = np.array(range(n))
        y = np.array(values)
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0] * (n + days) + coeffs[1])
