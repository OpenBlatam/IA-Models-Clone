"""
Exceptions for Trajectory Optimization
=======================================
Custom exceptions for trajectory optimization errors.
"""

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


@ErrorCode(description="Base exception for trajectory optimization errors")
class TrajectoryOptimizationError(Exception):
    """Base exception for trajectory optimization errors"""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Base exception for trajectory optimization errors")
        super().__init__(message)
        self.message = message


@ErrorCode(description="Trajectory is empty and cannot be processed")
class TrajectoryEmptyError(TrajectoryOptimizationError):
    """Raised when trajectory is empty"""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Trajectory is empty and cannot be processed")
        super().__init__()
        self.message = message


@ErrorCode(description="Trajectory contains collisions and cannot be executed")
class TrajectoryCollisionError(TrajectoryOptimizationError):
    """Raised when trajectory has collisions"""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Trajectory contains collisions and cannot be executed")
        super().__init__()
        self.message = message


@ErrorCode(description="Trajectory is invalid and cannot be used")
class TrajectoryInvalidError(TrajectoryOptimizationError):
    """Raised when trajectory is invalid"""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Trajectory is invalid and cannot be used")
        super().__init__()
        self.message = message


@ErrorCode(description="No path can be found for the given trajectory optimization")
class PathNotFoundError(TrajectoryOptimizationError):
    """Raised when no path can be found"""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "No path can be found for the given trajectory optimization")
        super().__init__()
        self.message = message


@ErrorCode(description="Trajectory optimization operation timed out")
class OptimizationTimeoutError(TrajectoryOptimizationError):
    """Raised when optimization times out"""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Trajectory optimization operation timed out")
        super().__init__()
        self.message = message


@ErrorCode(description="Invalid algorithm specified for trajectory optimization")
class InvalidAlgorithmError(TrajectoryOptimizationError):
    """Raised when invalid algorithm is specified"""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Invalid algorithm specified for trajectory optimization")
        super().__init__()
        self.message = message


@ErrorCode(description="Algorithm not found in trajectory optimization system")
class AlgorithmNotFoundError(TrajectoryOptimizationError):
    """Raised when algorithm is not found"""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Algorithm not found in trajectory optimization system")
        super().__init__()
        self.message = message

