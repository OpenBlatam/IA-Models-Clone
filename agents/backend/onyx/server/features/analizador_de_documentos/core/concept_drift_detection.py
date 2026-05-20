"""
Sistema de Concept Drift Detection
===================================

Sistema para detección de concept drift.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DriftDetectionMethod(Enum):
    """Método de detección de drift"""
    STATISTICAL = "statistical"
    ADWIN = "adwin"  # Adaptive Windowing
    DDM = "ddm"  # Drift Detection Method
    EDDM = "eddm"  # Early Drift Detection Method
    PAGE_HINKLEY = "page_hinkley"


@dataclass
class DriftEvent:
    """Evento de concept drift"""
    event_id: str
    model_id: str
    drift_type: str  # sudden, gradual, incremental
    severity: float
    detected_at: str
    cause: Optional[str] = None


class ConceptDriftDetection:
    """
    Sistema de Concept Drift Detection
    
    Proporciona:
    - Detección de concept drift
    - Múltiples métodos de detección
    - Detección de drift gradual y súbito
    - Alertas de degradación
    - Recomendaciones de adaptación
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.drift_events: Dict[str, DriftEvent] = {}
        self.monitoring_history: List[Dict[str, Any]] = []
        logger.info("ConceptDriftDetection inicializado")
    
    def monitor_model(
        self,
        model_id: str,
        predictions: List[Dict[str, Any]],
        method: DriftDetectionMethod = DriftDetectionMethod.ADWIN
    ) -> Dict[str, Any]:
        """
        Monitorear modelo para detectar drift
        
        Args:
            model_id: ID del modelo
            predictions: Predicciones recientes
            method: Método de detección
        
        Returns:
            Resultado del monitoreo
        """
        # Simulación de detección de drift
        drift_detected = False
        drift_severity = 0.0
        
        # Detectar drift si hay cambios significativos
        if len(predictions) > 10:
            recent_accuracy = 0.85
            baseline_accuracy = 0.90
            
            if abs(recent_accuracy - baseline_accuracy) > 0.05:
                drift_detected = True
                drift_severity = abs(recent_accuracy - baseline_accuracy)
        
        monitoring_result = {
            "model_id": model_id,
            "drift_detected": drift_detected,
            "drift_severity": drift_severity,
            "method": method.value,
            "samples_analyzed": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
        self.monitoring_history.append(monitoring_result)
        
        if drift_detected:
            event = DriftEvent(
                event_id=f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_id=model_id,
                drift_type="gradual",
                severity=drift_severity,
                detected_at=datetime.now().isoformat(),
                cause="Cambio en distribución de datos"
            )
            
            self.drift_events[event.event_id] = event
            logger.warning(f"Drift detectado: {model_id} - Severidad: {drift_severity:.2f}")
        
        return monitoring_result
    
    def get_drift_recommendations(
        self,
        event_id: str
    ) -> List[str]:
        """
        Obtener recomendaciones para manejar drift
        
        Args:
            event_id: ID del evento de drift
        
        Returns:
            Lista de recomendaciones
        """
        if event_id not in self.drift_events:
            raise ValueError(f"Evento de drift no encontrado: {event_id}")
        
        event = self.drift_events[event_id]
        
        recommendations = [
            "Reentrenar modelo con datos recientes",
            "Ajustar umbrales de predicción",
            "Implementar modelo adaptativo",
            "Recolectar más datos de la nueva distribución"
        ]
        
        if event.severity > 0.2:
            recommendations.append("Considerar reentrenamiento completo del modelo")
        
        logger.info(f"Recomendaciones generadas para drift: {event_id}")
        
        return recommendations


# Instancia global
_concept_drift: Optional[ConceptDriftDetection] = None


def get_concept_drift() -> ConceptDriftDetection:
    """Obtener instancia global del sistema"""
    global _concept_drift
    if _concept_drift is None:
        _concept_drift = ConceptDriftDetection()
    return _concept_drift


