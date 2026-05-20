"""
Manufacturing Monitor
=====================

Sistema de monitoreo en tiempo real de manufactura.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EquipmentStatus(Enum):
    """Estado de equipo."""
    IDLE = "idle"
    RUNNING = "running"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class ProductionMetrics:
    """Métricas de producción."""
    timestamp: str
    production_rate: float = 0.0  # unidades/hora
    efficiency: float = 0.0  # 0.0 a 1.0
    quality_rate: float = 0.0  # 0.0 a 1.0
    downtime: float = 0.0  # horas
    energy_consumption: float = 0.0  # kWh
    defects_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Equipment:
    """Equipo de manufactura."""
    equipment_id: str
    name: str
    equipment_type: str
    status: EquipmentStatus = EquipmentStatus.IDLE
    current_job: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())


class ManufacturingMonitor:
    """
    Monitor de manufactura.
    
    Monitorea producción, equipos y métricas en tiempo real.
    """
    
    def __init__(self):
        """Inicializar monitor."""
        self.equipment: Dict[str, Equipment] = {}
        self.metrics_history: List[ProductionMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
    
    def register_equipment(
        self,
        equipment_id: str,
        name: str,
        equipment_type: str
    ):
        """
        Registrar equipo.
        
        Args:
            equipment_id: ID del equipo
            name: Nombre
            equipment_type: Tipo de equipo
        """
        equipment = Equipment(
            equipment_id=equipment_id,
            name=name,
            equipment_type=equipment_type
        )
        
        self.equipment[equipment_id] = equipment
        logger.info(f"Registered equipment: {equipment_id}")
    
    def update_equipment_status(
        self,
        equipment_id: str,
        status: EquipmentStatus,
        current_job: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Actualizar estado de equipo.
        
        Args:
            equipment_id: ID del equipo
            status: Estado
            current_job: Trabajo actual (opcional)
            metrics: Métricas (opcional)
        """
        if equipment_id not in self.equipment:
            logger.error(f"Equipment not found: {equipment_id}")
            return
        
        equipment = self.equipment[equipment_id]
        equipment.status = status
        equipment.current_job = current_job
        equipment.last_update = datetime.now().isoformat()
        
        if metrics:
            equipment.metrics.update(metrics)
        
        # Generar alertas si es necesario
        if status == EquipmentStatus.ERROR:
            self._create_alert(equipment_id, "error", f"Equipment {equipment_id} in error state")
    
    def record_metrics(self, metrics: ProductionMetrics):
        """
        Registrar métricas de producción.
        
        Args:
            metrics: Métricas
        """
        self.metrics_history.append(metrics)
        
        # Mantener solo últimas 10000 métricas
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-10000:]
        
        # Generar alertas si métricas están fuera de rango
        if metrics.efficiency < 0.7:
            self._create_alert("production", "warning", f"Low efficiency: {metrics.efficiency:.2%}")
        
        if metrics.quality_rate < 0.95:
            self._create_alert("production", "warning", f"Low quality rate: {metrics.quality_rate:.2%}")
    
    def _create_alert(self, source: str, level: str, message: str):
        """Crear alerta."""
        alert = {
            "alert_id": str(uuid.uuid4()),
            "source": source,
            "level": level,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert [{level}]: {message}")
    
    def get_current_metrics(self) -> Optional[ProductionMetrics]:
        """Obtener métricas actuales."""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_equipment_status(self, equipment_id: str) -> Optional[Equipment]:
        """Obtener estado de equipo."""
        return self.equipment.get(equipment_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        status_counts = {}
        for equipment in self.equipment.values():
            status_counts[equipment.status.value] = status_counts.get(equipment.status.value, 0) + 1
        
        return {
            "total_equipment": len(self.equipment),
            "status_counts": status_counts,
            "total_metrics": len(self.metrics_history),
            "active_alerts": len([a for a in self.alerts if a["level"] == "error"])
        }

