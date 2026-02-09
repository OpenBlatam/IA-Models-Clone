"""
Domain Layer Improved
=====================

Capa de dominio mejorada con entidades ricas, value objects y domain events.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
import uuid

from .error_handling import DomainError, ErrorCode, ErrorContext


class Entity(ABC):
    """Clase base para entidades de dominio."""
    
    def __init__(self, entity_id: Optional[str] = None):
        """
        Inicializar entidad.
        
        Args:
            entity_id: ID de la entidad (se genera si no se proporciona)
        """
        self._id = entity_id or str(uuid.uuid4())
        self._domain_events: List[DomainEvent] = []
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
    
    @property
    def id(self) -> str:
        """Obtener ID de la entidad."""
        return self._id
    
    @property
    def created_at(self) -> datetime:
        """Obtener fecha de creación."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Obtener fecha de actualización."""
        return self._updated_at
    
    def _mark_updated(self):
        """Marcar entidad como actualizada."""
        self._updated_at = datetime.now()
    
    def add_domain_event(self, event: 'DomainEvent'):
        """Agregar evento de dominio."""
        self._domain_events.append(event)
    
    def get_domain_events(self) -> List['DomainEvent']:
        """Obtener eventos de dominio."""
        return self._domain_events.copy()
    
    def clear_domain_events(self):
        """Limpiar eventos de dominio."""
        self._domain_events.clear()
    
    def __eq__(self, other):
        """Comparar entidades por ID."""
        if not isinstance(other, Entity):
            return False
        return self._id == other._id
    
    def __hash__(self):
        """Hash basado en ID."""
        return hash(self._id)


class ValueObject(ABC):
    """Clase base para value objects."""
    
    def __eq__(self, other):
        """Comparar value objects por valor."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__
    
    def __hash__(self):
        """Hash basado en valores."""
        return hash(tuple(sorted(self.__dict__.items())))


class DomainEvent(ABC):
    """Clase base para eventos de dominio."""
    
    def __init__(self, occurred_at: Optional[datetime] = None):
        """
        Inicializar evento.
        
        Args:
            occurred_at: Fecha de ocurrencia (default: ahora)
        """
        self._occurred_at = occurred_at or datetime.now()
        self._event_id = str(uuid.uuid4())
    
    @property
    def occurred_at(self) -> datetime:
        """Obtener fecha de ocurrencia."""
        return self._occurred_at
    
    @property
    def event_id(self) -> str:
        """Obtener ID del evento."""
        return self._event_id


# Value Objects

@dataclass(frozen=True)
class Position(ValueObject):
    """Value object para posición 3D."""
    x: float
    y: float
    z: float
    
    def __post_init__(self):
        """Validar posición."""
        if not (-1000.0 <= self.x <= 1000.0):
            raise DomainError(
                f"X coordinate must be between -1000 and 1000, got {self.x}",
                ErrorCode.DOMAIN_VALIDATION_ERROR
            )
        if not (-1000.0 <= self.y <= 1000.0):
            raise DomainError(
                f"Y coordinate must be between -1000 and 1000, got {self.y}",
                ErrorCode.DOMAIN_VALIDATION_ERROR
            )
        if not (-1000.0 <= self.z <= 1000.0):
            raise DomainError(
                f"Z coordinate must be between -1000 and 1000, got {self.z}",
                ErrorCode.DOMAIN_VALIDATION_ERROR
            )
    
    def distance_to(self, other: 'Position') -> float:
        """Calcular distancia a otra posición."""
        import math
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convertir a diccionario."""
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True)
class Orientation(ValueObject):
    """Value object para orientación (quaternion)."""
    qx: float
    qy: float
    qz: float
    qw: float
    
    def __post_init__(self):
        """Validar quaternion."""
        norm = self.qx ** 2 + self.qy ** 2 + self.qz ** 2 + self.qw ** 2
        if abs(norm - 1.0) > 0.1:
            raise DomainError(
                f"Quaternion must be normalized, norm={norm}",
                ErrorCode.DOMAIN_VALIDATION_ERROR
            )
    
    def to_dict(self) -> Dict[str, float]:
        """Convertir a diccionario."""
        return {"qx": self.qx, "qy": self.qy, "qz": self.qz, "qw": self.qw}


class MovementStatus(Enum):
    """Estado de movimiento."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# Domain Events

class MovementStartedEvent(DomainEvent):
    """Evento: movimiento iniciado."""
    
    def __init__(self, movement_id: str, robot_id: str, **kwargs):
        super().__init__(**kwargs)
        self.movement_id = movement_id
        self.robot_id = robot_id


class MovementCompletedEvent(DomainEvent):
    """Evento: movimiento completado."""
    
    def __init__(self, movement_id: str, robot_id: str, duration: float, **kwargs):
        super().__init__(**kwargs)
        self.movement_id = movement_id
        self.robot_id = robot_id
        self.duration = duration


class MovementFailedEvent(DomainEvent):
    """Evento: movimiento fallido."""
    
    def __init__(self, movement_id: str, robot_id: str, error: str, **kwargs):
        super().__init__(**kwargs)
        self.movement_id = movement_id
        self.robot_id = robot_id
        self.error = error


# Domain Entities

class RobotMovement(Entity):
    """Entidad de dominio: movimiento de robot."""
    
    def __init__(
        self,
        robot_id: str,
        target_position: Position,
        target_orientation: Optional[Orientation] = None,
        movement_id: Optional[str] = None,
        **kwargs
    ):
        """
        Inicializar movimiento.
        
        Args:
            robot_id: ID del robot
            target_position: Posición objetivo
            target_orientation: Orientación objetivo (opcional)
            movement_id: ID del movimiento (se genera si no se proporciona)
        """
        super().__init__(movement_id)
        self._robot_id = robot_id
        self._target_position = target_position
        self._target_orientation = target_orientation
        self._status = MovementStatus.PENDING
        self._current_position: Optional[Position] = None
        self._trajectory: List[Position] = []
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._error_message: Optional[str] = None
    
    @property
    def robot_id(self) -> str:
        """Obtener ID del robot."""
        return self._robot_id
    
    @property
    def target_position(self) -> Position:
        """Obtener posición objetivo."""
        return self._target_position
    
    @property
    def target_orientation(self) -> Optional[Orientation]:
        """Obtener orientación objetivo."""
        return self._target_orientation
    
    @property
    def status(self) -> MovementStatus:
        """Obtener estado."""
        return self._status
    
    @property
    def current_position(self) -> Optional[Position]:
        """Obtener posición actual."""
        return self._current_position
    
    @property
    def trajectory(self) -> List[Position]:
        """Obtener trayectoria."""
        return self._trajectory.copy()
    
    def start(self):
        """Iniciar movimiento."""
        if self._status != MovementStatus.PENDING:
            raise DomainError(
                f"Cannot start movement in status {self._status.value}",
                ErrorCode.DOMAIN_BUSINESS_RULE_VIOLATION
            )
        
        self._status = MovementStatus.EXECUTING
        self._started_at = datetime.now()
        self._mark_updated()
        
        # Emitir evento
        self.add_domain_event(
            MovementStartedEvent(
                movement_id=self.id,
                robot_id=self._robot_id
            )
        )
    
    def complete(self, final_position: Position):
        """
        Completar movimiento.
        
        Args:
            final_position: Posición final alcanzada
        """
        if self._status != MovementStatus.EXECUTING:
            raise DomainError(
                f"Cannot complete movement in status {self._status.value}",
                ErrorCode.DOMAIN_BUSINESS_RULE_VIOLATION
            )
        
        self._status = MovementStatus.COMPLETED
        self._current_position = final_position
        self._completed_at = datetime.now()
        self._mark_updated()
        
        # Calcular duración
        duration = (self._completed_at - self._started_at).total_seconds()
        
        # Emitir evento
        self.add_domain_event(
            MovementCompletedEvent(
                movement_id=self.id,
                robot_id=self._robot_id,
                duration=duration
            )
        )
    
    def fail(self, error_message: str):
        """
        Marcar movimiento como fallido.
        
        Args:
            error_message: Mensaje de error
        """
        if self._status not in [MovementStatus.PENDING, MovementStatus.EXECUTING]:
            raise DomainError(
                f"Cannot fail movement in status {self._status.value}",
                ErrorCode.DOMAIN_BUSINESS_RULE_VIOLATION
            )
        
        self._status = MovementStatus.FAILED
        self._error_message = error_message
        self._completed_at = datetime.now()
        self._mark_updated()
        
        # Emitir evento
        self.add_domain_event(
            MovementFailedEvent(
                movement_id=self.id,
                robot_id=self._robot_id,
                error=error_message
            )
        )
    
    def update_position(self, position: Position):
        """
        Actualizar posición actual.
        
        Args:
            position: Nueva posición
        """
        if self._status != MovementStatus.EXECUTING:
            return
        
        self._current_position = position
        self._trajectory.append(position)
        self._mark_updated()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "robot_id": self._robot_id,
            "target_position": self._target_position.to_dict(),
            "target_orientation": self._target_orientation.to_dict() if self._target_orientation else None,
            "status": self._status.value,
            "current_position": self._current_position.to_dict() if self._current_position else None,
            "trajectory": [p.to_dict() for p in self._trajectory],
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "completed_at": self._completed_at.isoformat() if self._completed_at else None,
            "error_message": self._error_message,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat()
        }


class Robot(Entity):
    """Entidad de dominio: robot."""
    
    def __init__(
        self,
        robot_id: str,
        brand: str,
        model: str,
        current_position: Optional[Position] = None,
        current_orientation: Optional[Orientation] = None,
        **kwargs
    ):
        """
        Inicializar robot.
        
        Args:
            robot_id: ID del robot
            brand: Marca del robot
            model: Modelo del robot
            current_position: Posición actual
            current_orientation: Orientación actual
        """
        super().__init__(robot_id)
        self._brand = brand
        self._model = model
        self._current_position = current_position
        self._current_orientation = current_orientation
        self._is_connected = False
        self._active_movements: Set[str] = set()
    
    @property
    def brand(self) -> str:
        """Obtener marca."""
        return self._brand
    
    @property
    def model(self) -> str:
        """Obtener modelo."""
        return self._model
    
    @property
    def current_position(self) -> Optional[Position]:
        """Obtener posición actual."""
        return self._current_position
    
    @property
    def current_orientation(self) -> Optional[Orientation]:
        """Obtener orientación actual."""
        return self._current_orientation
    
    @property
    def is_connected(self) -> bool:
        """Verificar si está conectado."""
        return self._is_connected
    
    @property
    def has_active_movements(self) -> bool:
        """Verificar si tiene movimientos activos."""
        return len(self._active_movements) > 0
    
    def connect(self):
        """Conectar robot."""
        if self._is_connected:
            raise DomainError(
                f"Robot {self.id} is already connected",
                ErrorCode.DOMAIN_BUSINESS_RULE_VIOLATION
            )
        self._is_connected = True
        self._mark_updated()
    
    def disconnect(self):
        """Desconectar robot."""
        if self.has_active_movements:
            raise DomainError(
                f"Cannot disconnect robot {self.id} with active movements",
                ErrorCode.DOMAIN_BUSINESS_RULE_VIOLATION
            )
        self._is_connected = False
        self._mark_updated()
    
    def update_position(self, position: Position, orientation: Optional[Orientation] = None):
        """
        Actualizar posición del robot.
        
        Args:
            position: Nueva posición
            orientation: Nueva orientación (opcional)
        """
        if not self._is_connected:
            raise DomainError(
                f"Cannot update position of disconnected robot {self.id}",
                ErrorCode.DOMAIN_BUSINESS_RULE_VIOLATION
            )
        
        self._current_position = position
        if orientation:
            self._current_orientation = orientation
        self._mark_updated()
    
    def register_movement(self, movement_id: str):
        """
        Registrar movimiento activo.
        
        Args:
            movement_id: ID del movimiento
        """
        self._active_movements.add(movement_id)
        self._mark_updated()
    
    def unregister_movement(self, movement_id: str):
        """
        Desregistrar movimiento activo.
        
        Args:
            movement_id: ID del movimiento
        """
        self._active_movements.discard(movement_id)
        self._mark_updated()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "brand": self._brand,
            "model": self._model,
            "current_position": self._current_position.to_dict() if self._current_position else None,
            "current_orientation": self._current_orientation.to_dict() if self._current_orientation else None,
            "is_connected": self._is_connected,
            "active_movements": list(self._active_movements),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat()
        }




