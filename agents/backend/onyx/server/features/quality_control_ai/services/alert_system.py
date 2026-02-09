"""
Sistema de Alertas para Control de Calidad
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alerta generada"""
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str
    data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data
        }


class AlertSystem:
    """
    Sistema de alertas para notificaciones de calidad
    """
    
    def __init__(self, max_alerts: int = 1000):
        """
        Inicializar sistema de alertas
        
        Args:
            max_alerts: Máximo número de alertas a mantener
        """
        self.max_alerts = max_alerts
        self.alerts = deque(maxlen=max_alerts)
        self.callbacks: List[Callable[[Alert], None]] = []
        self.thresholds = {
            "quality_score_low": 40.0,
            "quality_score_warning": 60.0,
            "defect_count_critical": 10,
            "defect_count_warning": 5,
            "critical_defect_threshold": 1
        }
        
        logger.info("Alert System initialized")
    
    def check_inspection_result(self, result: Dict) -> List[Alert]:
        """
        Verificar resultado de inspección y generar alertas
        
        Args:
            result: Resultado de inspección
            
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        
        if not result.get("success"):
            alert = Alert(
                level=AlertLevel.ERROR,
                message="Inspection failed",
                timestamp=datetime.now(),
                source="inspection",
                data={"error": result.get("error")}
            )
            alerts.append(alert)
            self._add_alert(alert)
            return alerts
        
        quality_score = result.get("quality_score", 100)
        defects = result.get("defects", [])
        summary = result.get("summary", {})
        
        # Alerta por calidad baja
        if quality_score < self.thresholds["quality_score_low"]:
            alert = Alert(
                level=AlertLevel.CRITICAL,
                message=f"Quality score critically low: {quality_score:.1f}/100",
                timestamp=datetime.now(),
                source="quality_score",
                data={"quality_score": quality_score}
            )
            alerts.append(alert)
            self._add_alert(alert)
        
        elif quality_score < self.thresholds["quality_score_warning"]:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=f"Quality score below threshold: {quality_score:.1f}/100",
                timestamp=datetime.now(),
                source="quality_score",
                data={"quality_score": quality_score}
            )
            alerts.append(alert)
            self._add_alert(alert)
        
        # Alerta por defectos críticos
        critical_defects = [d for d in defects if d.get("severity") == "critical"]
        if len(critical_defects) >= self.thresholds["critical_defect_threshold"]:
            alert = Alert(
                level=AlertLevel.CRITICAL,
                message=f"Critical defects detected: {len(critical_defects)}",
                timestamp=datetime.now(),
                source="defects",
                data={"critical_count": len(critical_defects), "defects": critical_defects}
            )
            alerts.append(alert)
            self._add_alert(alert)
        
        # Alerta por número de defectos
        total_defects = len(defects)
        if total_defects >= self.thresholds["defect_count_critical"]:
            alert = Alert(
                level=AlertLevel.CRITICAL,
                message=f"High number of defects detected: {total_defects}",
                timestamp=datetime.now(),
                source="defects",
                data={"defect_count": total_defects}
            )
            alerts.append(alert)
            self._add_alert(alert)
        
        elif total_defects >= self.thresholds["defect_count_warning"]:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=f"Multiple defects detected: {total_defects}",
                timestamp=datetime.now(),
                source="defects",
                data={"defect_count": total_defects}
            )
            alerts.append(alert)
            self._add_alert(alert)
        
        # Alerta por estado rechazado
        if summary.get("status") == "rejected":
            alert = Alert(
                level=AlertLevel.CRITICAL,
                message="Product rejected - quality standards not met",
                timestamp=datetime.now(),
                source="status",
                data={"status": "rejected", "summary": summary}
            )
            alerts.append(alert)
            self._add_alert(alert)
        
        return alerts
    
    def _add_alert(self, alert: Alert):
        """Agregar alerta y notificar callbacks"""
        self.alerts.append(alert)
        
        # Notificar callbacks
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}", exc_info=True)
    
    def register_callback(self, callback: Callable[[Alert], None]):
        """
        Registrar callback para alertas
        
        Args:
            callback: Función que recibe Alert
        """
        self.callbacks.append(callback)
        logger.info("Alert callback registered")
    
    def set_threshold(self, key: str, value: float):
        """
        Establecer umbral de alerta
        
        Args:
            key: Clave del umbral
            value: Valor del umbral
        """
        if key in self.thresholds:
            self.thresholds[key] = value
            logger.info(f"Threshold {key} set to {value}")
        else:
            logger.warning(f"Unknown threshold key: {key}")
    
    def get_recent_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 50
    ) -> List[Alert]:
        """
        Obtener alertas recientes
        
        Args:
            level: Filtrar por nivel (opcional)
            limit: Límite de alertas
            
        Returns:
            Lista de alertas
        """
        alerts = list(self.alerts)
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts[-limit:]
    
    def get_alert_statistics(self) -> Dict:
        """Obtener estadísticas de alertas"""
        alerts = list(self.alerts)
        
        stats = {
            "total": len(alerts),
            "by_level": {
                "info": 0,
                "warning": 0,
                "error": 0,
                "critical": 0
            },
            "recent_critical": 0
        }
        
        for alert in alerts:
            stats["by_level"][alert.level.value] += 1
            
            # Contar críticas recientes (últimas 24 horas)
            if alert.level == AlertLevel.CRITICAL:
                time_diff = (datetime.now() - alert.timestamp).total_seconds()
                if time_diff < 86400:  # 24 horas
                    stats["recent_critical"] += 1
        
        return stats
    
    def clear_alerts(self):
        """Limpiar todas las alertas"""
        self.alerts.clear()
        logger.info("All alerts cleared")






