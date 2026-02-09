"""
Alert System - Sistema de alertas y notificaciones avanzadas
=============================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSystem:
    """
    Sistema de alertas y notificaciones.
    """
    
    def __init__(self, alerts_dir: str = "data/alerts"):
        """
        Inicializar sistema de alertas.
        
        Args:
            alerts_dir: Directorio para almacenar alertas
        """
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
    
    def create_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crea una nueva alerta.
        
        Args:
            title: Título de la alerta
            message: Mensaje de la alerta
            level: Nivel de alerta
            category: Categoría (opcional)
            metadata: Metadata adicional (opcional)
            
        Returns:
            Información de la alerta creada
        """
        import uuid
        
        alert_id = str(uuid.uuid4())
        
        alert = {
            "alert_id": alert_id,
            "title": title,
            "message": message,
            "level": level.value,
            "category": category or "general",
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "acknowledged": False,
            "acknowledged_at": None
        }
        
        self.active_alerts[alert_id] = alert
        self._save_alert(alert)
        
        logger.info(f"Alerta creada: {alert_id} ({level.value}) - {title}")
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, user_id: Optional[str] = None) -> bool:
        """
        Marca una alerta como reconocida.
        
        Args:
            alert_id: ID de la alerta
            user_id: ID del usuario que reconoce (opcional)
            
        Returns:
            True si se reconoció exitosamente
        """
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert["acknowledged"] = True
        alert["acknowledged_at"] = datetime.now().isoformat()
        alert["acknowledged_by"] = user_id
        
        self._save_alert(alert)
        
        logger.info(f"Alerta reconocida: {alert_id}")
        
        return True
    
    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        category: Optional[str] = None,
        unacknowledged_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Obtiene alertas activas.
        
        Args:
            level: Filtrar por nivel (opcional)
            category: Filtrar por categoría (opcional)
            unacknowledged_only: Solo no reconocidas
            
        Returns:
            Lista de alertas
        """
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [a for a in alerts if a["level"] == level.value]
        
        if category:
            alerts = [a for a in alerts if a.get("category") == category]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.get("acknowledged", False)]
        
        # Ordenar por fecha (más recientes primero)
        alerts.sort(key=lambda x: x["created_at"], reverse=True)
        
        return alerts
    
    def check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verifica umbrales y crea alertas si es necesario.
        
        Args:
            metrics: Métricas a verificar
            
        Returns:
            Lista de alertas creadas
        """
        alerts_created = []
        
        # Verificar reglas de umbral
        for rule in self.alert_rules:
            metric_name = rule.get("metric")
            threshold = rule.get("threshold")
            level = AlertLevel(rule.get("level", "warning"))
            
            if metric_name in metrics:
                value = metrics[metric_name]
                
                if rule.get("operator") == "gt" and value > threshold:
                    alert = self.create_alert(
                        title=f"Umbral excedido: {metric_name}",
                        message=f"{metric_name} = {value} (umbral: {threshold})",
                        level=level,
                        category="threshold",
                        metadata={"metric": metric_name, "value": value, "threshold": threshold}
                    )
                    alerts_created.append(alert)
        
        return alerts_created
    
    def add_alert_rule(
        self,
        metric: str,
        threshold: float,
        operator: str = "gt",
        level: AlertLevel = AlertLevel.WARNING
    ):
        """
        Agrega una regla de alerta.
        
        Args:
            metric: Nombre de la métrica
            threshold: Umbral
            operator: Operador (gt, lt, eq)
            level: Nivel de alerta
        """
        rule = {
            "metric": metric,
            "threshold": threshold,
            "operator": operator,
            "level": level.value
        }
        
        self.alert_rules.append(rule)
        logger.info(f"Regla de alerta agregada: {metric} {operator} {threshold}")
    
    def _save_alert(self, alert: Dict[str, Any]):
        """Guarda alerta en disco"""
        try:
            alert_file = self.alerts_dir / f"{alert['alert_id']}.json"
            with open(alert_file, "w", encoding="utf-8") as f:
                json.dump(alert, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando alerta: {e}")




