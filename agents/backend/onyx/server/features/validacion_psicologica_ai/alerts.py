"""
Sistema de Alertas para Validación Psicológica AI
==================================================
Alertas y notificaciones basadas en análisis
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import structlog

from .models import PsychologicalProfile, ValidationReport

logger = structlog.get_logger()


class AlertSeverity(str, Enum):
    """Severidad de alerta"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Tipo de alerta"""
    RISK_FACTOR = "risk_factor"
    SENTIMENT_SHIFT = "sentiment_shift"
    BEHAVIORAL_CHANGE = "behavioral_change"
    LOW_CONFIDENCE = "low_confidence"
    DATA_QUALITY = "data_quality"


class Alert:
    """Representa una alerta"""
    
    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.created_at = datetime.utcnow()
        self.id = f"{alert_type.value}_{self.created_at.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat()
        }


class AlertManager:
    """Gestor de alertas"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._alerts: List[Alert] = []
        self._handlers: Dict[AlertType, List[Callable[[Alert], None]]] = {}
        logger.info("AlertManager initialized")
    
    def register_handler(
        self,
        alert_type: AlertType,
        handler: Callable[[Alert], None]
    ) -> None:
        """
        Registrar handler para tipo de alerta
        
        Args:
            alert_type: Tipo de alerta
            handler: Función handler
        """
        if alert_type not in self._handlers:
            self._handlers[alert_type] = []
        self._handlers[alert_type].append(handler)
        logger.info("Alert handler registered", alert_type=alert_type.value)
    
    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Crear y procesar alerta
        
        Args:
            alert_type: Tipo de alerta
            severity: Severidad
            message: Mensaje
            details: Detalles adicionales
            
        Returns:
            Alerta creada
        """
        alert = Alert(alert_type, severity, message, details)
        self._alerts.append(alert)
        
        # Ejecutar handlers
        if alert_type in self._handlers:
            for handler in self._handlers[alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(
                        "Error executing alert handler",
                        error=str(e),
                        alert_type=alert_type.value
                    )
        
        logger.info(
            "Alert created",
            alert_id=alert.id,
            type=alert_type.value,
            severity=severity.value
        )
        
        return alert
    
    def analyze_profile(self, profile: PsychologicalProfile) -> List[Alert]:
        """
        Analizar perfil y generar alertas
        
        Args:
            profile: Perfil psicológico
            
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        
        # Alerta por baja confianza
        if profile.confidence_score < 0.5:
            alerts.append(self.create_alert(
                AlertType.LOW_CONFIDENCE,
                AlertSeverity.MEDIUM,
                f"Baja confianza en el análisis ({profile.confidence_score * 100:.1f}%)",
                {"confidence_score": profile.confidence_score}
            ))
        
        # Alertas por factores de riesgo
        if profile.risk_factors:
            for risk_factor in profile.risk_factors:
                severity = AlertSeverity.HIGH if "critical" in risk_factor.lower() else AlertSeverity.MEDIUM
                alerts.append(self.create_alert(
                    AlertType.RISK_FACTOR,
                    severity,
                    f"Factor de riesgo detectado: {risk_factor}",
                    {"risk_factor": risk_factor}
                ))
        
        # Alerta por neuroticismo alto
        neuroticism = profile.personality_traits.get("neuroticism", 0.5)
        if neuroticism > 0.7:
            alerts.append(self.create_alert(
                AlertType.RISK_FACTOR,
                AlertSeverity.MEDIUM,
                f"Neuroticismo elevado detectado ({neuroticism:.2f})",
                {"neuroticism": neuroticism, "threshold": 0.7}
            ))
        
        # Alerta por sentimiento negativo
        overall_sentiment = profile.emotional_state.get("overall_sentiment", "neutral")
        if overall_sentiment == "negative":
            stress_level = profile.emotional_state.get("stress_level", 0.0)
            severity = AlertSeverity.HIGH if stress_level > 0.7 else AlertSeverity.MEDIUM
            alerts.append(self.create_alert(
                AlertType.SENTIMENT_SHIFT,
                severity,
                "Sentimiento negativo detectado en el análisis",
                {
                    "sentiment": overall_sentiment,
                    "stress_level": stress_level
                }
            ))
        
        return alerts
    
    def compare_profiles(
        self,
        previous_profile: PsychologicalProfile,
        current_profile: PsychologicalProfile
    ) -> List[Alert]:
        """
        Comparar perfiles y detectar cambios significativos
        
        Args:
            previous_profile: Perfil anterior
            current_profile: Perfil actual
            
        Returns:
            Lista de alertas generadas
        """
        alerts = []
        
        # Detectar cambios significativos en sentimientos
        prev_sentiment = previous_profile.emotional_state.get("overall_sentiment", "neutral")
        curr_sentiment = current_profile.emotional_state.get("overall_sentiment", "neutral")
        
        if prev_sentiment != curr_sentiment:
            if prev_sentiment == "positive" and curr_sentiment == "negative":
                alerts.append(self.create_alert(
                    AlertType.SENTIMENT_SHIFT,
                    AlertSeverity.HIGH,
                    "Cambio significativo de sentimiento: de positivo a negativo",
                    {
                        "previous": prev_sentiment,
                        "current": curr_sentiment
                    }
                ))
            elif prev_sentiment == "neutral" and curr_sentiment == "negative":
                alerts.append(self.create_alert(
                    AlertType.SENTIMENT_SHIFT,
                    AlertSeverity.MEDIUM,
                    "Cambio de sentimiento: de neutral a negativo",
                    {
                        "previous": prev_sentiment,
                        "current": curr_sentiment
                    }
                ))
        
        # Detectar cambios en neuroticismo
        prev_neuro = previous_profile.personality_traits.get("neuroticism", 0.5)
        curr_neuro = current_profile.personality_traits.get("neuroticism", 0.5)
        
        if curr_neuro - prev_neuro > 0.2:
            alerts.append(self.create_alert(
                AlertType.BEHAVIORAL_CHANGE,
                AlertSeverity.MEDIUM,
                f"Incremento significativo en neuroticismo ({prev_neuro:.2f} → {curr_neuro:.2f})",
                {
                    "previous": prev_neuro,
                    "current": curr_neuro,
                    "change": curr_neuro - prev_neuro
                }
            ))
        
        return alerts
    
    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Obtener alertas filtradas
        
        Args:
            alert_type: Filtrar por tipo
            severity: Filtrar por severidad
            limit: Límite de resultados
            
        Returns:
            Lista de alertas
        """
        alerts = self._alerts
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Ordenar por fecha (más recientes primero)
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        return alerts[:limit]
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Obtener resumen de alertas"""
        total = len(self._alerts)
        
        by_type = {}
        by_severity = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0
        }
        
        for alert in self._alerts:
            # Por tipo
            alert_type = alert.alert_type.value
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            
            # Por severidad
            severity = alert.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_alerts": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_alerts": len([a for a in self._alerts if (datetime.utcnow() - a.created_at).days < 7])
        }


# Instancia global de alert manager
alert_manager = AlertManager()




