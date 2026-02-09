"""
Model Monitoring Service - Monitoreo de modelos
================================================

Sistema para monitorear modelos en producción.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Registro de predicción"""
    timestamp: datetime
    input_hash: str
    prediction: Any
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Métricas de modelo en producción"""
    model_id: str
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    avg_confidence: float = 0.0
    predictions_per_hour: float = 0.0
    last_prediction: Optional[datetime] = None


class ModelMonitoringService:
    """Servicio de monitoreo de modelos"""
    
    def __init__(self, max_records: int = 10000):
        """Inicializar servicio"""
        self.predictions: Dict[str, deque] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.max_records = max_records
        logger.info("ModelMonitoringService initialized")
    
    def record_prediction(
        self,
        model_id: str,
        prediction: Any,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        input_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Registrar predicción"""
        if model_id not in self.predictions:
            self.predictions[model_id] = deque(maxlen=self.max_records)
            self.metrics[model_id] = ModelMetrics(model_id=model_id)
        
        record = PredictionRecord(
            timestamp=datetime.now(),
            input_hash=input_hash or "",
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        
        self.predictions[model_id].append(record)
        self._update_metrics(model_id)
    
    def _update_metrics(self, model_id: str) -> None:
        """Actualizar métricas del modelo"""
        records = list(self.predictions[model_id])
        if not records:
            return
        
        metrics = self.metrics[model_id]
        metrics.total_predictions = len(records)
        metrics.last_prediction = records[-1].timestamp
        
        # Latencia
        latencies = [r.latency_ms for r in records if r.latency_ms is not None]
        if latencies:
            metrics.avg_latency_ms = np.mean(latencies)
            metrics.p95_latency_ms = np.percentile(latencies, 95)
            metrics.p99_latency_ms = np.percentile(latencies, 99)
        
        # Confidence
        confidences = [r.confidence for r in records if r.confidence is not None]
        if confidences:
            metrics.avg_confidence = np.mean(confidences)
        
        # Predictions per hour
        if len(records) > 1:
            time_span = (records[-1].timestamp - records[0].timestamp).total_seconds() / 3600
            if time_span > 0:
                metrics.predictions_per_hour = len(records) / time_span
    
    def get_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Obtener métricas del modelo"""
        return self.metrics.get(model_id)
    
    def detect_drift(
        self,
        model_id: str,
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """Detectar drift en predicciones"""
        records = list(self.predictions.get(model_id, []))
        if len(records) < 10:
            return {"error": "Not enough data"}
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_records = [r for r in records if r.timestamp >= cutoff_time]
        old_records = [r for r in records if r.timestamp < cutoff_time]
        
        if len(recent_records) < 5 or len(old_records) < 5:
            return {"error": "Not enough data for comparison"}
        
        # Comparar confidences
        recent_conf = [r.confidence for r in recent_records if r.confidence is not None]
        old_conf = [r.confidence for r in old_records if r.confidence is not None]
        
        drift_detected = False
        if recent_conf and old_conf:
            recent_mean = np.mean(recent_conf)
            old_mean = np.mean(old_conf)
            drift_score = abs(recent_mean - old_mean) / (old_mean + 1e-8)
            drift_detected = drift_score > 0.2  # 20% change
        
        return {
            "drift_detected": drift_detected,
            "recent_avg_confidence": np.mean(recent_conf) if recent_conf else None,
            "old_avg_confidence": np.mean(old_conf) if old_conf else None,
            "drift_score": drift_score if recent_conf and old_conf else None,
        }
    
    def get_prediction_stats(
        self,
        model_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Obtener estadísticas de predicciones"""
        records = list(self.predictions.get(model_id, []))
        if not records:
            return {"error": "No predictions found"}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_records = [r for r in records if r.timestamp >= cutoff_time]
        
        return {
            "total_predictions": len(recent_records),
            "time_range_hours": hours,
            "predictions_per_hour": len(recent_records) / hours if hours > 0 else 0,
            "avg_latency_ms": np.mean([r.latency_ms for r in recent_records if r.latency_ms]) if recent_records else 0,
        }




