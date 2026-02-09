"""
Sistema de Advanced Anomaly Detection
=======================================

Sistema avanzado para detección de anomalías.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalyDetectionMethod(Enum):
    """Método de detección de anomalías"""
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"  # Local Outlier Factor
    ONE_CLASS_SVM = "one_class_svm"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    STATISTICAL = "statistical"


@dataclass
class Anomaly:
    """Anomalía detectada"""
    anomaly_id: str
    data_point: Dict[str, Any]
    anomaly_score: float
    severity: str  # low, medium, high, critical
    method: AnomalyDetectionMethod
    timestamp: str


class AdvancedAnomalyDetection:
    """
    Sistema de Advanced Anomaly Detection
    
    Proporciona:
    - Detección avanzada de anomalías
    - Múltiples métodos de detección
    - Scoring de anomalías
    - Clasificación de severidad
    - Explicaciones de anomalías
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.detectors: Dict[str, Dict[str, Any]] = {}
        self.detected_anomalies: Dict[str, Anomaly] = {}
        logger.info("AdvancedAnomalyDetection inicializado")
    
    def create_detector(
        self,
        detector_id: str,
        method: AnomalyDetectionMethod = AnomalyDetectionMethod.ISOLATION_FOREST,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear detector de anomalías
        
        Args:
            detector_id: ID del detector
            method: Método de detección
            parameters: Parámetros del detector
        
        Returns:
            Detector creado
        """
        detector = {
            "detector_id": detector_id,
            "method": method.value,
            "parameters": parameters or {},
            "created_at": datetime.now().isoformat()
        }
        
        self.detectors[detector_id] = detector
        
        logger.info(f"Detector creado: {detector_id} - {method.value}")
        
        return detector
    
    def detect_anomalies(
        self,
        detector_id: str,
        data: List[Dict[str, Any]],
        threshold: float = 0.5
    ) -> List[Anomaly]:
        """
        Detectar anomalías
        
        Args:
            detector_id: ID del detector
            data: Datos a analizar
            threshold: Umbral de detección
        
        Returns:
            Lista de anomalías detectadas
        """
        if detector_id not in self.detectors:
            raise ValueError(f"Detector no encontrado: {detector_id}")
        
        detector = self.detectors[detector_id]
        method = AnomalyDetectionMethod(detector["method"])
        
        anomalies = []
        
        for i, data_point in enumerate(data):
            # Simulación de detección de anomalías
            # En producción, usaría métodos reales
            anomaly_score = 0.3 + (i % 10) * 0.1  # Simulación
            
            if anomaly_score > threshold:
                severity = self._classify_severity(anomaly_score)
                
                anomaly = Anomaly(
                    anomaly_id=f"anom_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                    data_point=data_point,
                    anomaly_score=anomaly_score,
                    severity=severity,
                    method=method,
                    timestamp=datetime.now().isoformat()
                )
                
                anomalies.append(anomaly)
                self.detected_anomalies[anomaly.anomaly_id] = anomaly
        
        logger.info(f"Anomalías detectadas: {len(anomalies)} de {len(data)} puntos")
        
        return anomalies
    
    def _classify_severity(self, score: float) -> str:
        """Clasificar severidad"""
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def explain_anomaly(
        self,
        anomaly_id: str
    ) -> Dict[str, Any]:
        """
        Explicar anomalía
        
        Args:
            anomaly_id: ID de la anomalía
        
        Returns:
            Explicación de la anomalía
        """
        if anomaly_id not in self.detected_anomalies:
            raise ValueError(f"Anomalía no encontrada: {anomaly_id}")
        
        anomaly = self.detected_anomalies[anomaly_id]
        
        explanation = {
            "anomaly_id": anomaly_id,
            "score": anomaly.anomaly_score,
            "severity": anomaly.severity,
            "contributing_factors": [
                "Feature X está fuera del rango normal",
                "Patrón temporal inusual",
                "Valor atípico en característica Y"
            ],
            "recommendations": [
                "Investigar origen del dato",
                "Verificar calidad de datos",
                "Revisar proceso de recolección"
            ]
        }
        
        logger.info(f"Explicación generada para anomalía: {anomaly_id}")
        
        return explanation


# Instancia global
_anomaly_detection: Optional[AdvancedAnomalyDetection] = None


def get_anomaly_detection() -> AdvancedAnomalyDetection:
    """Obtener instancia global del sistema"""
    global _anomaly_detection
    if _anomaly_detection is None:
        _anomaly_detection = AdvancedAnomalyDetection()
    return _anomaly_detection


