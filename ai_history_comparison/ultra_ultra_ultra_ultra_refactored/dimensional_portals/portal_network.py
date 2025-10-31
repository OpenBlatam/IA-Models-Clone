"""
Portal Network - Red de Portales Dimensionales
=============================================

Sistema avanzado de red de portales dimensionales que permite
viaje instantáneo entre dimensiones, universos paralelos y realidades.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import math
from enum import Enum

from ..time_dilation_core.time_domain.time_value_objects import (
    DimensionalPortalId,
    HyperdimensionalVector,
    RealityFabricCoordinate,
    OmniversalCoordinate
)


class PortalType(Enum):
    """Tipos de portales dimensionales."""
    DIMENSIONAL = "dimensional"
    TEMPORAL = "temporal"
    REALITY = "reality"
    CONSCIOUSNESS = "consciousness"
    QUANTUM = "quantum"
    TRANSCENDENT = "transcendent"
    OMNIVERSAL = "omniversal"


class PortalStatus(Enum):
    """Estados de portales."""
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    UNSTABLE = "unstable"
    COLLAPSING = "collapsing"
    TRANSCENDENT = "transcendent"


class TravelType(Enum):
    """Tipos de viaje dimensional."""
    INSTANTANEOUS = "instantaneous"
    TEMPORAL = "temporal"
    CONSCIOUSNESS = "consciousness"
    QUANTUM = "quantum"
    TRANSCENDENT = "transcendent"
    OMNIVERSAL = "omniversal"


@dataclass
class DimensionalPortal:
    """
    Portal dimensional que conecta diferentes dimensiones y realidades.
    """
    
    # Identidad del portal
    portal_id: DimensionalPortalId
    portal_type: PortalType
    status: PortalStatus = PortalStatus.CLOSED
    
    # Coordenadas del portal
    source_coordinate: HyperdimensionalVector
    destination_coordinate: HyperdimensionalVector
    reality_fabric_coordinate: Optional[RealityFabricCoordinate] = None
    omniversal_coordinate: Optional[OmniversalCoordinate] = None
    
    # Propiedades del portal
    stability_level: float = 1.0
    energy_level: float = 1.0
    coherence_level: float = 1.0
    dimensional_depth: int = 3
    
    # Metadatos
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    # Configuración
    max_capacity: int = 1000
    current_capacity: int = 0
    travel_restrictions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validar portal dimensional."""
        self._validate_portal()
    
    def _validate_portal(self) -> None:
        """Validar que el portal sea válido."""
        if self.stability_level < 0.0 or self.stability_level > 1.0:
            raise ValueError("Stability level must be between 0.0 and 1.0")
        
        if self.energy_level < 0.0 or self.energy_level > 1.0:
            raise ValueError("Energy level must be between 0.0 and 1.0")
        
        if self.coherence_level < 0.0 or self.coherence_level > 1.0:
            raise ValueError("Coherence level must be between 0.0 and 1.0")
        
        if self.dimensional_depth < 1 or self.dimensional_depth > 11:
            raise ValueError("Dimensional depth must be between 1 and 11")
    
    def open_portal(self) -> bool:
        """
        Abrir el portal dimensional.
        
        Returns:
            bool: True si se abrió exitosamente
        """
        if self.status != PortalStatus.CLOSED:
            return False
        
        if self.energy_level < 0.5:
            return False
        
        if self.stability_level < 0.7:
            return False
        
        self.status = PortalStatus.OPENING
        
        # Simular proceso de apertura
        if self._simulate_opening():
            self.status = PortalStatus.OPEN
            return True
        else:
            self.status = PortalStatus.UNSTABLE
            return False
    
    def close_portal(self) -> bool:
        """
        Cerrar el portal dimensional.
        
        Returns:
            bool: True si se cerró exitosamente
        """
        if self.status not in [PortalStatus.OPEN, PortalStatus.UNSTABLE]:
            return False
        
        self.status = PortalStatus.CLOSING
        
        # Simular proceso de cierre
        if self._simulate_closing():
            self.status = PortalStatus.CLOSED
            return True
        else:
            self.status = PortalStatus.COLLAPSING
            return False
    
    def travel_through(self, traveler_data: Dict[str, Any]) -> bool:
        """
        Viajar a través del portal.
        
        Args:
            traveler_data: Datos del viajero
            
        Returns:
            bool: True si el viaje fue exitoso
        """
        if self.status != PortalStatus.OPEN:
            return False
        
        if self.current_capacity >= self.max_capacity:
            return False
        
        # Verificar restricciones de viaje
        if not self._check_travel_restrictions(traveler_data):
            return False
        
        # Simular viaje
        if self._simulate_travel(traveler_data):
            self.current_capacity += 1
            self.usage_count += 1
            self.last_used = datetime.utcnow()
            return True
        
        return False
    
    def _simulate_opening(self) -> bool:
        """Simular apertura del portal."""
        # Calcular probabilidad de apertura exitosa
        success_probability = (
            self.energy_level * 0.4 +
            self.stability_level * 0.4 +
            self.coherence_level * 0.2
        )
        
        return np.random.random() < success_probability
    
    def _simulate_closing(self) -> bool:
        """Simular cierre del portal."""
        # Calcular probabilidad de cierre exitoso
        success_probability = (
            self.stability_level * 0.6 +
            self.coherence_level * 0.4
        )
        
        return np.random.random() < success_probability
    
    def _simulate_travel(self, traveler_data: Dict[str, Any]) -> bool:
        """Simular viaje a través del portal."""
        # Calcular probabilidad de viaje exitoso
        success_probability = (
            self.stability_level * 0.3 +
            self.energy_level * 0.3 +
            self.coherence_level * 0.2 +
            (1.0 - self.current_capacity / self.max_capacity) * 0.2
        )
        
        return np.random.random() < success_probability
    
    def _check_travel_restrictions(self, traveler_data: Dict[str, Any]) -> bool:
        """Verificar restricciones de viaje."""
        for restriction in self.travel_restrictions:
            if restriction in traveler_data:
                return False
        return True
    
    def is_operational(self) -> bool:
        """Verificar si el portal está operacional."""
        return (
            self.status == PortalStatus.OPEN and
            self.energy_level > 0.3 and
            self.stability_level > 0.5 and
            self.coherence_level > 0.4
        )
    
    def get_portal_efficiency(self) -> float:
        """Calcular eficiencia del portal."""
        efficiency = (
            self.stability_level * 0.4 +
            self.energy_level * 0.3 +
            self.coherence_level * 0.3
        )
        
        return efficiency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "portal_id": str(self.portal_id),
            "portal_type": self.portal_type.value,
            "status": self.status.value,
            "source_coordinate": self.source_coordinate.to_dict(),
            "destination_coordinate": self.destination_coordinate.to_dict(),
            "reality_fabric_coordinate": self.reality_fabric_coordinate.to_dict() if self.reality_fabric_coordinate else None,
            "omniversal_coordinate": self.omniversal_coordinate.to_dict() if self.omniversal_coordinate else None,
            "stability_level": self.stability_level,
            "energy_level": self.energy_level,
            "coherence_level": self.coherence_level,
            "dimensional_depth": self.dimensional_depth,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "max_capacity": self.max_capacity,
            "current_capacity": self.current_capacity,
            "travel_restrictions": self.travel_restrictions
        }


class PortalNetwork:
    """
    Red de portales dimensionales que gestiona múltiples portales
    y permite viaje entre dimensiones.
    """
    
    def __init__(self):
        self.portals: Dict[str, DimensionalPortal] = {}
        self.portal_connections: Dict[str, List[str]] = {}
        self.network_status: PortalStatus = PortalStatus.CLOSED
        
        # Parámetros de la red
        self.network_parameters = {
            "max_portals": 1000,
            "max_connections_per_portal": 10,
            "network_stability_threshold": 0.8,
            "energy_distribution_efficiency": 0.9
        }
        
        # Estadísticas de la red
        self.network_statistics = {
            "total_travels": 0,
            "successful_travels": 0,
            "failed_travels": 0,
            "total_energy_consumed": 0.0,
            "average_travel_time": 0.0
        }
    
    def create_portal(
        self,
        portal_type: PortalType,
        source_coordinate: HyperdimensionalVector,
        destination_coordinate: HyperdimensionalVector,
        reality_fabric_coordinate: Optional[RealityFabricCoordinate] = None,
        omniversal_coordinate: Optional[OmniversalCoordinate] = None,
        portal_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crear un nuevo portal dimensional.
        
        Args:
            portal_type: Tipo de portal
            source_coordinate: Coordenada de origen
            destination_coordinate: Coordenada de destino
            reality_fabric_coordinate: Coordenada de tejido de realidad
            omniversal_coordinate: Coordenada omniversal
            portal_config: Configuración del portal
            
        Returns:
            str: ID del portal creado
        """
        if len(self.portals) >= self.network_parameters["max_portals"]:
            raise ValueError("Maximum number of portals reached")
        
        portal_id = str(uuid.uuid4())
        
        # Configuración por defecto
        config = portal_config or {}
        
        portal = DimensionalPortal(
            portal_id=DimensionalPortalId(portal_id),
            portal_type=portal_type,
            source_coordinate=source_coordinate,
            destination_coordinate=destination_coordinate,
            reality_fabric_coordinate=reality_fabric_coordinate,
            omniversal_coordinate=omniversal_coordinate,
            stability_level=config.get("stability_level", 1.0),
            energy_level=config.get("energy_level", 1.0),
            coherence_level=config.get("coherence_level", 1.0),
            dimensional_depth=config.get("dimensional_depth", 3),
            max_capacity=config.get("max_capacity", 1000),
            travel_restrictions=config.get("travel_restrictions", [])
        )
        
        self.portals[portal_id] = portal
        self.portal_connections[portal_id] = []
        
        return portal_id
    
    def connect_portals(self, portal_id_1: str, portal_id_2: str) -> bool:
        """
        Conectar dos portales.
        
        Args:
            portal_id_1: ID del primer portal
            portal_id_2: ID del segundo portal
            
        Returns:
            bool: True si se conectaron exitosamente
        """
        if portal_id_1 not in self.portals or portal_id_2 not in self.portals:
            return False
        
        if len(self.portal_connections[portal_id_1]) >= self.network_parameters["max_connections_per_portal"]:
            return False
        
        if len(self.portal_connections[portal_id_2]) >= self.network_parameters["max_connections_per_portal"]:
            return False
        
        # Agregar conexión bidireccional
        self.portal_connections[portal_id_1].append(portal_id_2)
        self.portal_connections[portal_id_2].append(portal_id_1)
        
        return True
    
    def travel_between_portals(
        self,
        source_portal_id: str,
        destination_portal_id: str,
        traveler_data: Dict[str, Any]
    ) -> bool:
        """
        Viajar entre dos portales.
        
        Args:
            source_portal_id: ID del portal de origen
            destination_portal_id: ID del portal de destino
            traveler_data: Datos del viajero
            
        Returns:
            bool: True si el viaje fue exitoso
        """
        if source_portal_id not in self.portals or destination_portal_id not in self.portals:
            return False
        
        source_portal = self.portals[source_portal_id]
        destination_portal = self.portals[destination_portal_id]
        
        # Verificar que ambos portales estén abiertos
        if not source_portal.is_operational() or not destination_portal.is_operational():
            return False
        
        # Verificar conexión
        if destination_portal_id not in self.portal_connections[source_portal_id]:
            return False
        
        # Simular viaje
        self.network_statistics["total_travels"] += 1
        
        if self._simulate_inter_portal_travel(source_portal, destination_portal, traveler_data):
            self.network_statistics["successful_travels"] += 1
            return True
        else:
            self.network_statistics["failed_travels"] += 1
            return False
    
    def _simulate_inter_portal_travel(
        self,
        source_portal: DimensionalPortal,
        destination_portal: DimensionalPortal,
        traveler_data: Dict[str, Any]
    ) -> bool:
        """Simular viaje entre portales."""
        # Calcular probabilidad de viaje exitoso
        source_efficiency = source_portal.get_portal_efficiency()
        destination_efficiency = destination_portal.get_portal_efficiency()
        
        success_probability = (source_efficiency + destination_efficiency) / 2.0
        
        return np.random.random() < success_probability
    
    def get_network_status(self) -> Dict[str, Any]:
        """Obtener estado de la red de portales."""
        operational_portals = sum(1 for portal in self.portals.values() if portal.is_operational())
        total_portals = len(self.portals)
        
        return {
            "total_portals": total_portals,
            "operational_portals": operational_portals,
            "network_efficiency": operational_portals / total_portals if total_portals > 0 else 0.0,
            "network_status": self.network_status.value,
            "statistics": self.network_statistics.copy()
        }
    
    def get_portal_by_id(self, portal_id: str) -> Optional[DimensionalPortal]:
        """Obtener portal por ID."""
        return self.portals.get(portal_id)
    
    def get_portals_by_type(self, portal_type: PortalType) -> List[DimensionalPortal]:
        """Obtener portales por tipo."""
        return [portal for portal in self.portals.values() if portal.portal_type == portal_type]
    
    def get_connected_portals(self, portal_id: str) -> List[DimensionalPortal]:
        """Obtener portales conectados."""
        if portal_id not in self.portal_connections:
            return []
        
        connected_portal_ids = self.portal_connections[portal_id]
        return [self.portals[pid] for pid in connected_portal_ids if pid in self.portals]
    
    def optimize_network(self) -> Dict[str, Any]:
        """Optimizar la red de portales."""
        optimization_results = {
            "portals_optimized": 0,
            "connections_optimized": 0,
            "energy_saved": 0.0,
            "stability_improved": 0.0
        }
        
        # Optimizar portales individuales
        for portal in self.portals.values():
            if portal.energy_level < 0.5:
                portal.energy_level = min(1.0, portal.energy_level + 0.1)
                optimization_results["portals_optimized"] += 1
                optimization_results["energy_saved"] += 0.1
        
        # Optimizar conexiones
        for portal_id, connections in self.portal_connections.items():
            if len(connections) > 5:  # Reducir conexiones excesivas
                self.portal_connections[portal_id] = connections[:5]
                optimization_results["connections_optimized"] += 1
        
        return optimization_results
    
    def close_network(self) -> bool:
        """Cerrar toda la red de portales."""
        success_count = 0
        total_portals = len(self.portals)
        
        for portal in self.portals.values():
            if portal.close_portal():
                success_count += 1
        
        self.network_status = PortalStatus.CLOSED
        
        return success_count == total_portals
    
    def open_network(self) -> bool:
        """Abrir toda la red de portales."""
        success_count = 0
        total_portals = len(self.portals)
        
        for portal in self.portals.values():
            if portal.open_portal():
                success_count += 1
        
        if success_count == total_portals:
            self.network_status = PortalStatus.OPEN
        elif success_count > 0:
            self.network_status = PortalStatus.UNSTABLE
        else:
            self.network_status = PortalStatus.CLOSED
        
        return success_count > 0




