"""
Security Service - Integración con sistemas de seguridad
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityEventType(str, Enum):
    """Tipos de eventos de seguridad"""
    INTRUSION = "intrusion"
    FIRE = "fire"
    EMERGENCY = "emergency"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SYSTEM_FAILURE = "system_failure"


class SecurityService:
    """Servicio para sistemas de seguridad"""
    
    def __init__(self):
        self.systems: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_security_system(
        self,
        store_id: str,
        system_type: str,  # "camera", "alarm", "access_control", "fire_detection"
        location: str,
        capabilities: List[str]
    ) -> Dict[str, Any]:
        """Registrar sistema de seguridad"""
        
        system_id = f"sec_{store_id}_{len(self.systems.get(store_id, [])) + 1}"
        
        system = {
            "system_id": system_id,
            "store_id": store_id,
            "type": system_type,
            "location": location,
            "capabilities": capabilities,
            "is_active": True,
            "registered_at": datetime.now().isoformat(),
            "last_check": datetime.now().isoformat()
        }
        
        if store_id not in self.systems:
            self.systems[store_id] = {}
        
        self.systems[store_id][system_id] = system
        
        return system
    
    def record_security_event(
        self,
        system_id: str,
        event_type: SecurityEventType,
        severity: str = "medium",  # "low", "medium", "high", "critical"
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registrar evento de seguridad"""
        
        event = {
            "event_id": f"evt_{system_id}_{len(self.events.get(system_id, [])) + 1}",
            "system_id": system_id,
            "type": event_type.value,
            "severity": severity,
            "description": description,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
        
        if system_id not in self.events:
            self.events[system_id] = []
        
        self.events[system_id].append(event)
        
        # Crear alerta si es crítica
        if severity in ["high", "critical"]:
            self._create_alert(event)
        
        return event
    
    def _create_alert(self, event: Dict[str, Any]):
        """Crear alerta de seguridad"""
        store_id = self._get_store_id_from_system(event["system_id"])
        
        if not store_id:
            return
        
        alert = {
            "alert_id": f"alert_{store_id}_{len(self.alerts.get(store_id, [])) + 1}",
            "store_id": store_id,
            "event": event,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "notified": False
        }
        
        if store_id not in self.alerts:
            self.alerts[store_id] = []
        
        self.alerts[store_id].append(alert)
    
    def _get_store_id_from_system(self, system_id: str) -> Optional[str]:
        """Obtener store_id del sistema"""
        for store_id, systems in self.systems.items():
            if system_id in systems:
                return store_id
        return None
    
    def get_security_status(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Obtener estado de seguridad"""
        
        systems = self.systems.get(store_id, {})
        
        # Obtener eventos recientes (últimas 24 horas)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        recent_events = []
        for system_id in systems.keys():
            events = self.events.get(system_id, [])
            recent = [
                e for e in events
                if start_time <= datetime.fromisoformat(e["timestamp"]) <= end_time
            ]
            recent_events.extend(recent)
        
        # Contar por severidad
        severity_counts = {}
        for event in recent_events:
            severity = event["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "store_id": store_id,
            "systems_count": len(systems),
            "active_systems": len([s for s in systems.values() if s["is_active"]]),
            "recent_events_24h": len(recent_events),
            "severity_distribution": severity_counts,
            "active_alerts": len([a for a in self.alerts.get(store_id, []) if a["status"] == "active"]),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_active_alerts(self, store_id: str) -> List[Dict[str, Any]]:
        """Obtener alertas activas"""
        alerts = self.alerts.get(store_id, [])
        return [a for a in alerts if a["status"] == "active"]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Reconocer alerta"""
        for store_alerts in self.alerts.values():
            for alert in store_alerts:
                if alert["alert_id"] == alert_id:
                    alert["status"] = "acknowledged"
                    alert["acknowledged_at"] = datetime.now().isoformat()
                    return True
        return False




