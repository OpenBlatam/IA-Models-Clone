"""
Custom Exceptions
=================

Excepciones personalizadas para el sistema de routing y movimiento robótico.
Todas las excepciones incluyen contexto adicional para mejor debugging.
"""

from typing import Optional, Dict, Any
import traceback
from functools import wraps


def ErrorCode(description: str):
    """
    Decorador para anotar excepciones con códigos de error y descripciones.
    
    Args:
        description: Descripción del error que se usará en el constructor.
    
    Usage:
        @ErrorCode(description="Invalid input provided")
        class MyException(Exception):
            def __init__(self):
                super().__init__(description)
    """
    def decorator(cls):
        # Almacenar la descripción en la clase
        cls._error_description = description
        return cls
    return decorator


@ErrorCode(description="Base exception for all robot system errors")
class BaseRobotException(Exception):
    """Excepción base para todos los errores del sistema."""
    
    def __init__(
        self,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Inicializar excepción base.
        
        Args:
            error_code: Código de error único
            details: Detalles adicionales del error
            cause: Excepción que causó este error
        """
        # Usar la descripción del decorador @ErrorCode en lugar de cualquier argumento message
        description = getattr(self.__class__, '_error_description', "Base exception for all robot system errors")
        super().__init__(description)
        self.message = description
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.traceback = traceback.format_exc() if cause else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir excepción a diccionario para serialización."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "traceback": self.traceback
        }
    
    def __str__(self) -> str:
        """Representación en string de la excepción."""
        base = f"{self.__class__.__name__}: {self.message}"
        if self.error_code:
            base = f"[{self.error_code}] {base}"
        if self.details:
            base += f" | Details: {self.details}"
        return base


@ErrorCode(description="Base exception for routing errors")
class RoutingError(BaseRobotException):
    """Excepción base para errores de routing."""
    pass


@ErrorCode(description="Route not found in the routing system")
class RouteNotFoundError(RoutingError):
    """Error cuando no se encuentra una ruta."""
    pass


@ErrorCode(description="Invalid node provided in routing operation")
class InvalidNodeError(RoutingError):
    """Error cuando un nodo es inválido."""
    pass


@ErrorCode(description="Invalid routing strategy specified")
class InvalidStrategyError(RoutingError):
    """Error cuando una estrategia es inválida."""
    pass


@ErrorCode(description="Base exception for model-related errors")
class ModelError(BaseRobotException):
    """Excepción base para errores de modelos."""
    pass


@ErrorCode(description="Model not found in the system")
class ModelNotFoundError(ModelError):
    """Error cuando un modelo no se encuentra."""
    pass


@ErrorCode(description="Error occurred while loading the model")
class ModelLoadError(ModelError):
    """Error al cargar un modelo."""
    pass


@ErrorCode(description="Error occurred during model inference")
class ModelInferenceError(ModelError):
    """Error durante inferencia."""
    pass


@ErrorCode(description="Base exception for training-related errors")
class TrainingError(BaseRobotException):
    """Excepción base para errores de entrenamiento."""
    pass


@ErrorCode(description="Validation error occurred")
class ValidationError(BaseRobotException):
    """Error de validación."""
    pass


@ErrorCode(description="Invalid obstacle provided in validation")
class InvalidObstacleError(ValidationError):
    """Error cuando un obstáculo es inválido."""
    pass


@ErrorCode(description="Configuration error in the system")
class ConfigurationError(BaseRobotException):
    """Error de configuración."""
    pass


@ErrorCode(description="Base exception for robot movement errors")
class RobotMovementError(BaseRobotException):
    """Excepción base para errores de movimiento del robot."""
    pass


@ErrorCode(description="Error related to trajectory operations")
class TrajectoryError(RobotMovementError):
    """Error relacionado con trayectorias."""
    pass


@ErrorCode(description="Trajectory is empty and cannot be processed")
class TrajectoryEmptyError(TrajectoryError):
    """Error cuando una trayectoria está vacía."""
    pass


@ErrorCode(description="Trajectory contains collisions and cannot be executed")
class TrajectoryCollisionError(TrajectoryError):
    """Error cuando una trayectoria tiene colisiones."""
    pass


@ErrorCode(description="Trajectory is invalid and cannot be used")
class TrajectoryInvalidError(TrajectoryError):
    """Error cuando una trayectoria es inválida."""
    pass


@ErrorCode(description="Inverse kinematics calculation error")
class IKError(RobotMovementError):
    """Error de cinemática inversa."""
    pass


@ErrorCode(description="Error connecting to the robot")
class RobotConnectionError(RobotMovementError):
    """Error de conexión con el robot."""
    pass


@ErrorCode(description="Robot is not connected to the system")
class RobotNotConnectedError(RobotConnectionError):
    """Error cuando el robot no está conectado."""
    pass


@ErrorCode(description="Robot movement is already in progress")
class RobotMovementInProgressError(RobotMovementError):
    """Error cuando ya hay un movimiento en progreso."""
    pass


@ErrorCode(description="Algorithm not found in the routing system")
class AlgorithmNotFoundError(RoutingError):
    """Error cuando no se encuentra un algoritmo."""
    pass


@ErrorCode(description="Base exception for robot safety errors")
class SafetyError(RobotMovementError):
    """Error relacionado con seguridad del robot."""
    pass


@ErrorCode(description="Collision detected in robot movement")
class CollisionDetectedError(SafetyError):
    """Error cuando se detecta una colisión."""
    pass


@ErrorCode(description="Safety limit exceeded in robot operation")
class SafetyLimitExceededError(SafetyError):
    """Error cuando se excede un límite de seguridad."""
    pass


@ErrorCode(description="Emergency stop activated in robot system")
class EmergencyStopError(SafetyError):
    """Error cuando se activa parada de emergencia."""
    pass