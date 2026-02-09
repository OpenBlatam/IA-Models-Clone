"""
Application Layer
=================

Capa de aplicación con use cases, commands, queries y DTOs.
Implementa el patrón CQRS (Command Query Responsibility Segregation).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable
from datetime import datetime

from .domain_improved import (
    Robot,
    RobotMovement,
    Position,
    Orientation,
    MovementStatus,
    DomainEvent
)
from .error_handling import (
    ApplicationError,
    DomainError,
    ErrorCode,
    ErrorContext,
    handle_error
)


# ============================================================================
# Repository Interfaces (from Domain Layer)
# ============================================================================

@runtime_checkable
class IRobotRepository(Protocol):
    """Interface para repositorio de robots."""
    
    async def find_by_id(self, robot_id: str) -> Optional[Robot]:
        """Encontrar robot por ID."""
        ...
    
    async def save(self, robot: Robot) -> None:
        """Guardar robot."""
        ...
    
    async def find_all(self) -> List[Robot]:
        """Encontrar todos los robots."""
        ...


@runtime_checkable
class IMovementRepository(Protocol):
    """Interface para repositorio de movimientos."""
    
    async def find_by_id(self, movement_id: str) -> Optional[RobotMovement]:
        """Encontrar movimiento por ID."""
        ...
    
    async def save(self, movement: RobotMovement) -> None:
        """Guardar movimiento."""
        ...
    
    async def find_by_robot_id(
        self,
        robot_id: str,
        limit: int = 100
    ) -> List[RobotMovement]:
        """Encontrar movimientos por robot ID."""
        ...


# ============================================================================
# Commands (Write Operations)
# ============================================================================

@dataclass
class MoveRobotCommand:
    """Comando para mover robot."""
    robot_id: str
    target_x: float
    target_y: float
    target_z: float
    target_qx: Optional[float] = None
    target_qy: Optional[float] = None
    target_qz: Optional[float] = None
    target_qw: Optional[float] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class ConnectRobotCommand:
    """Comando para conectar robot."""
    robot_id: str
    user_id: Optional[str] = None


@dataclass
class DisconnectRobotCommand:
    """Comando para desconectar robot."""
    robot_id: str
    user_id: Optional[str] = None


# ============================================================================
# Queries (Read Operations)
# ============================================================================

@dataclass
class GetRobotStatusQuery:
    """Query para obtener estado del robot."""
    robot_id: str


@dataclass
class GetMovementHistoryQuery:
    """Query para obtener historial de movimientos."""
    robot_id: str
    limit: int = 100
    status: Optional[MovementStatus] = None


# ============================================================================
# DTOs (Data Transfer Objects)
# ============================================================================

@dataclass
class MovementResultDTO:
    """DTO para resultado de movimiento."""
    movement_id: str
    robot_id: str
    status: str
    target_position: Dict[str, float]
    current_position: Optional[Dict[str, float]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    
    @classmethod
    def from_domain(cls, movement: RobotMovement) -> 'MovementResultDTO':
        """Crear DTO desde entidad de dominio."""
        return cls(
            movement_id=movement.id,
            robot_id=movement.robot_id,
            status=movement.status.value,
            target_position=movement.target_position.to_dict(),
            current_position=movement.current_position.to_dict() if movement.current_position else None,
            started_at=movement._started_at.isoformat() if movement._started_at else None,
            completed_at=movement._completed_at.isoformat() if movement._completed_at else None,
            error_message=movement._error_message
        )


@dataclass
class RobotStatusDTO:
    """DTO para estado del robot."""
    robot_id: str
    brand: str
    model: str
    is_connected: bool
    current_position: Optional[Dict[str, float]] = None
    current_orientation: Optional[Dict[str, float]] = None
    has_active_movements: bool = False
    active_movements_count: int = 0
    
    @classmethod
    def from_domain(cls, robot: Robot) -> 'RobotStatusDTO':
        """Crear DTO desde entidad de dominio."""
        return cls(
            robot_id=robot.id,
            brand=robot.brand,
            model=robot.model,
            is_connected=robot.is_connected,
            current_position=robot.current_position.to_dict() if robot.current_position else None,
            current_orientation=robot.current_orientation.to_dict() if robot.current_orientation else None,
            has_active_movements=robot.has_active_movements,
            active_movements_count=len(robot._active_movements)
        )


# ============================================================================
# Use Cases
# ============================================================================

class UseCase(ABC):
    """Clase base para casos de uso."""
    
    def __init__(self):
        """Inicializar caso de uso."""
        pass


class MoveRobotUseCase(UseCase):
    """Caso de uso: mover robot."""
    
    def __init__(
        self,
        robot_repository: IRobotRepository,
        movement_repository: IMovementRepository,
        event_publisher: Optional['IEventPublisher'] = None
    ):
        """
        Inicializar caso de uso.
        
        Args:
            robot_repository: Repositorio de robots
            movement_repository: Repositorio de movimientos
            event_publisher: Publicador de eventos (opcional)
        """
        self.robot_repository = robot_repository
        self.movement_repository = movement_repository
        self.event_publisher = event_publisher
    
    async def execute(self, command: MoveRobotCommand) -> MovementResultDTO:
        """
        Ejecutar caso de uso.
        
        Args:
            command: Comando de movimiento
            
        Returns:
            Resultado del movimiento
            
        Raises:
            ApplicationError: Si hay error en la aplicación
            DomainError: Si hay error de dominio
        """
        context = ErrorContext(
            user_id=command.user_id,
            request_id=command.request_id,
            robot_id=command.robot_id,
            operation="move_robot"
        )
        
        try:
            # 1. Obtener robot
            robot = await self.robot_repository.find_by_id(command.robot_id)
            if not robot:
                raise ApplicationError(
                    f"Robot {command.robot_id} not found",
                    ErrorCode.APPLICATION_NOT_FOUND,
                    context=context
                )
            
            # 2. Validar que esté conectado
            if not robot.is_connected:
                raise ApplicationError(
                    f"Robot {command.robot_id} is not connected",
                    ErrorCode.ROBOT_NOT_CONNECTED,
                    context=context
                )
            
            # 3. Crear posición objetivo
            target_position = Position(
                x=command.target_x,
                y=command.target_y,
                z=command.target_z
            )
            
            # 4. Crear orientación objetivo si se proporciona
            target_orientation = None
            if all([command.target_qx, command.target_qy, command.target_qz, command.target_qw]):
                target_orientation = Orientation(
                    qx=command.target_qx,
                    qy=command.target_qy,
                    qz=command.target_qz,
                    qw=command.target_qw
                )
            
            # 5. Crear movimiento
            movement = RobotMovement(
                robot_id=command.robot_id,
                target_position=target_position,
                target_orientation=target_orientation
            )
            
            # 6. Iniciar movimiento (lógica de dominio)
            movement.start()
            
            # 7. Registrar movimiento en robot
            robot.register_movement(movement.id)
            
            # 8. Persistir cambios
            await self.movement_repository.save(movement)
            await self.robot_repository.save(robot)
            
            # 9. Publicar eventos de dominio
            if self.event_publisher:
                for event in movement.get_domain_events():
                    await self.event_publisher.publish(event)
                movement.clear_domain_events()
            
            return MovementResultDTO.from_domain(movement)
        
        except (ApplicationError, DomainError):
            raise
        except Exception as e:
            error_details = handle_error(e, context)
            raise ApplicationError(
                f"Error moving robot: {error_details.message}",
                ErrorCode.APPLICATION_VALIDATION_ERROR,
                context=context,
                original_error=e
            )


class GetRobotStatusUseCase(UseCase):
    """Caso de uso: obtener estado del robot."""
    
    def __init__(self, robot_repository: IRobotRepository):
        """
        Inicializar caso de uso.
        
        Args:
            robot_repository: Repositorio de robots
        """
        self.robot_repository = robot_repository
    
    async def execute(self, query: GetRobotStatusQuery) -> RobotStatusDTO:
        """
        Ejecutar caso de uso.
        
        Args:
            query: Query de estado
            
        Returns:
            Estado del robot
            
        Raises:
            ApplicationError: Si el robot no existe
        """
        context = ErrorContext(
            robot_id=query.robot_id,
            operation="get_robot_status"
        )
        
        try:
            robot = await self.robot_repository.find_by_id(query.robot_id)
            if not robot:
                raise ApplicationError(
                    f"Robot {query.robot_id} not found",
                    ErrorCode.APPLICATION_NOT_FOUND,
                    context=context
                )
            
            return RobotStatusDTO.from_domain(robot)
        
        except ApplicationError:
            raise
        except Exception as e:
            error_details = handle_error(e, context)
            raise ApplicationError(
                f"Error getting robot status: {error_details.message}",
                ErrorCode.APPLICATION_VALIDATION_ERROR,
                context=context,
                original_error=e
            )


class GetMovementHistoryUseCase(UseCase):
    """Caso de uso: obtener historial de movimientos."""
    
    def __init__(self, movement_repository: IMovementRepository):
        """
        Inicializar caso de uso.
        
        Args:
            movement_repository: Repositorio de movimientos
        """
        self.movement_repository = movement_repository
    
    async def execute(self, query: GetMovementHistoryQuery) -> List[MovementResultDTO]:
        """
        Ejecutar caso de uso.
        
        Args:
            query: Query de historial
            
        Returns:
            Lista de movimientos
            
        Raises:
            ApplicationError: Si hay error
        """
        context = ErrorContext(
            robot_id=query.robot_id,
            operation="get_movement_history"
        )
        
        try:
            movements = await self.movement_repository.find_by_robot_id(
                query.robot_id,
                limit=query.limit
            )
            
            # Filtrar por estado si se especifica
            if query.status:
                movements = [m for m in movements if m.status == query.status]
            
            return [MovementResultDTO.from_domain(m) for m in movements]
        
        except Exception as e:
            error_details = handle_error(e, context)
            raise ApplicationError(
                f"Error getting movement history: {error_details.message}",
                ErrorCode.APPLICATION_VALIDATION_ERROR,
                context=context,
                original_error=e
            )


# ============================================================================
# Event Publisher Interface
# ============================================================================

@runtime_checkable
class IEventPublisher(Protocol):
    """Interface para publicador de eventos."""
    
    async def publish(self, event: DomainEvent) -> None:
        """Publicar evento."""
        ...




