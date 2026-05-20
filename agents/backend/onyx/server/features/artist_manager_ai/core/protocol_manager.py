"""
Protocol Manager
================

Gestión de protocolos de comportamiento para artistas.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ProtocolCategory(Enum):
    """Categorías de protocolos."""
    SOCIAL_MEDIA = "social_media"
    INTERVIEW = "interview"
    PUBLIC_APPEARANCE = "public_appearance"
    NETWORKING = "networking"
    CONTRACT = "contract"
    MEDIA = "media"
    GENERAL = "general"


class ProtocolPriority(Enum):
    """Prioridad del protocolo."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Protocol:
    """Protocolo de comportamiento."""
    id: str
    title: str
    description: str
    category: ProtocolCategory
    priority: ProtocolPriority
    rules: List[str]
    do_s: List[str] = None
    dont_s: List[str] = None
    context: Optional[str] = None
    applicable_events: List[str] = None  # IDs de eventos donde aplica
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.do_s is None:
            self.do_s = []
        if self.dont_s is None:
            self.dont_s = []
        if self.applicable_events is None:
            self.applicable_events = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['category'] = self.category.value
        data['priority'] = self.priority.value
        return data


@dataclass
class ProtocolCompliance:
    """Registro de cumplimiento de protocolo."""
    protocol_id: str
    event_id: Optional[str]
    checked_at: datetime
    is_compliant: bool
    notes: Optional[str] = None
    violations: List[str] = None
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['checked_at'] = self.checked_at.isoformat()
        return data


class ProtocolManager:
    """Gestor de protocolos para artistas."""
    
    def __init__(self, artist_id: str):
        """
        Inicializar gestor de protocolos.
        
        Args:
            artist_id: ID del artista
        """
        self.artist_id = artist_id
        self.protocols: Dict[str, Protocol] = {}
        self.compliance_records: List[ProtocolCompliance] = []
        self._logger = logger
    
    def add_protocol(self, protocol: Protocol) -> Protocol:
        """
        Agregar protocolo.
        
        Args:
            protocol: Protocolo a agregar
        
        Returns:
            Protocolo agregado
        """
        if protocol.id in self.protocols:
            raise ValueError(f"Protocol with id {protocol.id} already exists")
        
        self.protocols[protocol.id] = protocol
        self._logger.info(f"Added protocol {protocol.id} for artist {self.artist_id}")
        return protocol
    
    def get_protocol(self, protocol_id: str) -> Optional[Protocol]:
        """
        Obtener protocolo por ID.
        
        Args:
            protocol_id: ID del protocolo
        
        Returns:
            Protocolo o None si no existe
        """
        return self.protocols.get(protocol_id)
    
    def update_protocol(self, protocol_id: str, **updates) -> Protocol:
        """
        Actualizar protocolo.
        
        Args:
            protocol_id: ID del protocolo
            **updates: Campos a actualizar
        
        Returns:
            Protocolo actualizado
        """
        if protocol_id not in self.protocols:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        protocol = self.protocols[protocol_id]
        for key, value in updates.items():
            if hasattr(protocol, key):
                setattr(protocol, key, value)
        
        self._logger.info(f"Updated protocol {protocol_id} for artist {self.artist_id}")
        return protocol
    
    def delete_protocol(self, protocol_id: str) -> bool:
        """
        Eliminar protocolo.
        
        Args:
            protocol_id: ID del protocolo
        
        Returns:
            True si se eliminó, False si no existía
        """
        if protocol_id in self.protocols:
            del self.protocols[protocol_id]
            self._logger.info(f"Deleted protocol {protocol_id} for artist {self.artist_id}")
            return True
        return False
    
    def get_protocols_by_category(self, category: ProtocolCategory) -> List[Protocol]:
        """
        Obtener protocolos por categoría.
        
        Args:
            category: Categoría del protocolo
        
        Returns:
            Lista de protocolos de la categoría
        """
        return [
            protocol for protocol in self.protocols.values()
            if protocol.category == category
        ]
    
    def get_protocols_for_event(self, event_id: str) -> List[Protocol]:
        """
        Obtener protocolos aplicables a un evento.
        
        Args:
            event_id: ID del evento
        
        Returns:
            Lista de protocolos aplicables
        """
        return [
            protocol for protocol in self.protocols.values()
            if event_id in protocol.applicable_events or len(protocol.applicable_events) == 0
        ]
    
    def get_protocols_by_priority(self, priority: ProtocolPriority) -> List[Protocol]:
        """
        Obtener protocolos por prioridad.
        
        Args:
            priority: Prioridad del protocolo
        
        Returns:
            Lista de protocolos con la prioridad especificada
        """
        return [
            protocol for protocol in self.protocols.values()
            if protocol.priority == priority
        ]
    
    def record_compliance(
        self,
        protocol_id: str,
        is_compliant: bool,
        event_id: Optional[str] = None,
        notes: Optional[str] = None,
        violations: Optional[List[str]] = None
    ) -> ProtocolCompliance:
        """
        Registrar cumplimiento de protocolo.
        
        Args:
            protocol_id: ID del protocolo
            is_compliant: Si se cumplió el protocolo
            event_id: ID del evento relacionado (opcional)
            notes: Notas adicionales
            violations: Lista de violaciones (opcional)
        
        Returns:
            Registro de cumplimiento
        """
        if protocol_id not in self.protocols:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        compliance = ProtocolCompliance(
            protocol_id=protocol_id,
            event_id=event_id,
            checked_at=datetime.now(),
            is_compliant=is_compliant,
            notes=notes,
            violations=violations or []
        )
        
        self.compliance_records.append(compliance)
        self._logger.info(
            f"Recorded compliance for protocol {protocol_id}: {is_compliant} "
            f"for artist {self.artist_id}"
        )
        return compliance
    
    def get_compliance_history(
        self,
        protocol_id: Optional[str] = None,
        event_id: Optional[str] = None
    ) -> List[ProtocolCompliance]:
        """
        Obtener historial de cumplimiento.
        
        Args:
            protocol_id: ID de protocolo específico (opcional)
            event_id: ID de evento específico (opcional)
        
        Returns:
            Lista de registros de cumplimiento
        """
        records = self.compliance_records
        
        if protocol_id:
            records = [r for r in records if r.protocol_id == protocol_id]
        
        if event_id:
            records = [r for r in records if r.event_id == event_id]
        
        return sorted(records, key=lambda r: r.checked_at, reverse=True)
    
    def get_compliance_rate(self, protocol_id: str) -> float:
        """
        Obtener tasa de cumplimiento de un protocolo.
        
        Args:
            protocol_id: ID del protocolo
        
        Returns:
            Tasa de cumplimiento (0.0 a 1.0)
        """
        records = [
            record for record in self.compliance_records
            if record.protocol_id == protocol_id
        ]
        
        if not records:
            return 0.0
        
        compliant_count = sum(1 for record in records if record.is_compliant)
        return compliant_count / len(records)
    
    def get_all_protocols(self) -> List[Protocol]:
        """
        Obtener todos los protocolos.
        
        Returns:
            Lista de todos los protocolos
        """
        return list(self.protocols.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "artist_id": self.artist_id,
            "protocols": [protocol.to_dict() for protocol in self.get_all_protocols()],
            "total_protocols": len(self.protocols),
            "total_compliance_records": len(self.compliance_records)
        }




