"""
Model Monitoring System - Sistema de monitoreo de modelos
==========================================================
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MonitoringMetric(Enum):
    """Métricas de monitoreo"""
    PREDICTION_DRIFT = "prediction_drift"
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class MonitoringAlert:
    """Alerta de monitoreo"""
    metric: MonitoringMetric
    threshold: float
    current_value: float
    severity: str = "warning"  # "info", "warning", "critical"
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""


class ModelMonitor:
    """Monitor de modelos"""
    
    def __init__(self):
        self.reference_data: Optional[np.ndarray] = None
        self.reference_predictions: Optional[np.ndarray] = None
        self.current_data: List[np.ndarray] = []
        self.current_predictions: List[np.ndarray] = []
        self.alerts: List[MonitoringAlert] = []
        self.thresholds: Dict[MonitoringMetric, float] = {
            MonitoringMetric.PREDICTION_DRIFT: 0.1,
            MonitoringMetric.DATA_DRIFT: 0.15,
            MonitoringMetric.PERFORMANCE_DEGRADATION: 0.2
        }
    
    def set_reference(
        self,
        data: np.ndarray,
        predictions: Optional[np.ndarray] = None
    ):
        """Establece datos de referencia"""
        self.reference_data = data
        self.reference_predictions = predictions
        logger.info("Datos de referencia establecidos")
    
    def check_prediction_drift(
        self,
        current_predictions: np.ndarray
    ) -> float:
        """Verifica drift en predicciones"""
        if self.reference_predictions is None:
            logger.warning("No hay predicciones de referencia")
            return 0.0
        
        # Calcular distribución de predicciones
        ref_dist = np.histogram(self.reference_predictions, bins=50)[0]
        ref_dist = ref_dist / ref_dist.sum()
        
        curr_dist = np.histogram(current_predictions, bins=50)[0]
        curr_dist = curr_dist / curr_dist.sum()
        
        # KL divergence como medida de drift
        kl_div = np.sum(ref_dist * np.log((ref_dist + 1e-10) / (curr_dist + 1e-10)))
        
        return float(kl_div)
    
    def check_data_drift(
        self,
        current_data: np.ndarray
    ) -> float:
        """Verifica drift en datos"""
        if self.reference_data is None:
            logger.warning("No hay datos de referencia")
            return 0.0
        
        # Calcular estadísticas
        ref_mean = np.mean(self.reference_data, axis=0)
        ref_std = np.std(self.reference_data, axis=0)
        
        curr_mean = np.mean(current_data, axis=0)
        curr_std = np.std(current_data, axis=0)
        
        # Distancia normalizada
        mean_diff = np.abs(curr_mean - ref_mean) / (ref_std + 1e-10)
        std_diff = np.abs(curr_std - ref_std) / (ref_std + 1e-10)
        
        drift_score = np.mean(mean_diff) + np.mean(std_diff)
        
        return float(drift_score)
    
    def check_performance_degradation(
        self,
        current_metrics: Dict[str, float],
        reference_metrics: Dict[str, float]
    ) -> float:
        """Verifica degradación de performance"""
        degradation = 0.0
        
        for metric_name in reference_metrics:
            if metric_name in current_metrics:
                ref_value = reference_metrics[metric_name]
                curr_value = current_metrics[metric_name]
                
                # Asumir que métricas más altas son mejores
                if ref_value > 0:
                    degradation = max(degradation, (ref_value - curr_value) / ref_value)
        
        return degradation
    
    def monitor(
        self,
        data: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> List[MonitoringAlert]:
        """Monitorea el modelo"""
        alerts = []
        
        # Prediction drift
        if predictions is not None:
            drift_score = self.check_prediction_drift(predictions)
            threshold = self.thresholds.get(MonitoringMetric.PREDICTION_DRIFT, 0.1)
            
            if drift_score > threshold:
                severity = "critical" if drift_score > threshold * 2 else "warning"
                alert = MonitoringAlert(
                    metric=MonitoringMetric.PREDICTION_DRIFT,
                    threshold=threshold,
                    current_value=drift_score,
                    severity=severity,
                    message=f"Prediction drift detectado: {drift_score:.4f}"
                )
                alerts.append(alert)
        
        # Data drift
        if data is not None:
            drift_score = self.check_data_drift(data)
            threshold = self.thresholds.get(MonitoringMetric.DATA_DRIFT, 0.15)
            
            if drift_score > threshold:
                severity = "critical" if drift_score > threshold * 2 else "warning"
                alert = MonitoringAlert(
                    metric=MonitoringMetric.DATA_DRIFT,
                    threshold=threshold,
                    current_value=drift_score,
                    severity=severity,
                    message=f"Data drift detectado: {drift_score:.4f}"
                )
                alerts.append(alert)
        
        # Performance degradation
        if metrics is not None and self.reference_predictions is not None:
            # En producción, comparar con métricas de referencia
            pass
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[MonitoringAlert]:
        """Obtiene alertas"""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def set_threshold(self, metric: MonitoringMetric, threshold: float):
        """Establece umbral para una métrica"""
        self.thresholds[metric] = threshold




