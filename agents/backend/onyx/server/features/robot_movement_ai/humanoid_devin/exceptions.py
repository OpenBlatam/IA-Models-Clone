"""
Exceptions for Humanoid Devin Robot (Optimizado)
================================================

Excepciones personalizadas para el robot humanoide.
Incluye jerarquía de excepciones y mensajes de error informativos.
Soporta tanto uso directo como integración con FastAPI.
"""

try:
    from fastapi import HTTPException, status
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    HTTPException = Exception
    status = None


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


@ErrorCode(description="Base exception for humanoid robot errors")
class HumanoidRobotError(Exception):
    """
    Excepción base para errores del robot humanoide.
    
    Esta es la excepción principal que deben capturar los usuarios del sistema.
    """
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Base exception for humanoid robot errors")
        super().__init__(message)
        self.message = message


@ErrorCode(description="Error connecting or disconnecting from the robot")
class RobotConnectionError(HumanoidRobotError):
    """Error al conectar o desconectar del robot."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error connecting or disconnecting from the robot")
        super().__init__()
        self.message = message


@ErrorCode(description="Error controlling the robot (movement, joints, etc.)")
class RobotControlError(HumanoidRobotError):
    """Error al controlar el robot (movimiento, articulaciones, etc.)."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error controlling the robot (movement, joints, etc.)")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in robot configuration")
class RobotConfigurationError(HumanoidRobotError):
    """Error en la configuración del robot."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in robot configuration")
        super().__init__()
        self.message = message


@ErrorCode(description="Robot is not initialized")
class RobotNotInitializedError(HumanoidRobotError):
    """Error cuando el robot no está inicializado."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Robot is not initialized")
        super().__init__()
        self.message = message


@ErrorCode(description="Invalid command provided to the robot")
class InvalidCommandError(HumanoidRobotError):
    """Error cuando un comando es inválido."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Invalid command provided to the robot")
        super().__init__()
        self.message = message


@ErrorCode(description="Error during robot movement")
class MovementError(HumanoidRobotError):
    """Error durante el movimiento del robot."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error during robot movement")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in ROS 2 integration")
class ROS2IntegrationError(HumanoidRobotError):
    """Error en la integración con ROS 2."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in ROS 2 integration")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in MoveIt 2 integration")
class MoveIt2Error(HumanoidRobotError):
    """Error en la integración con MoveIt 2."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in MoveIt 2 integration")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in Nav2 integration")
class Nav2Error(HumanoidRobotError):
    """Error en la integración con Nav2."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in Nav2 integration")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in vision processing")
class VisionError(HumanoidRobotError):
    """Error en el procesamiento de visión."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in vision processing")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in point cloud processing")
class PointCloudError(HumanoidRobotError):
    """Error en el procesamiento de nubes de puntos."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in point cloud processing")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in AI models (TensorFlow, PyTorch)")
class AIModelError(HumanoidRobotError):
    """Error en modelos de IA (TensorFlow, PyTorch)."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in AI models (TensorFlow, PyTorch)")
        super().__init__()
        self.message = message


@ErrorCode(description="Model error (alias for compatibility)")
class ModelError(AIModelError):
    """Alias para compatibilidad con código existente."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Model error (alias for compatibility)")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in Poppy robot integration")
class PoppyError(HumanoidRobotError):
    """Error en la integración con robot Poppy."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in Poppy robot integration")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in iCub robot integration")
class ICubError(HumanoidRobotError):
    """Error en la integración con robot iCub."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in iCub robot integration")
        super().__init__()
        self.message = message


@ErrorCode(description="Error in trajectory generation or execution")
class TrajectoryError(HumanoidRobotError):
    """Error en la generación o ejecución de trayectorias."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Error in trajectory generation or execution")
        super().__init__()
        self.message = message


@ErrorCode(description="Parameter validation error")
class ValidationError(HumanoidRobotError):
    """Error de validación de parámetros."""
    
    def __init__(self):
        """Initialize exception with description from @ErrorCode decorator."""
        message = getattr(self.__class__, '_error_description', "Parameter validation error")
        super().__init__()
        self.message = message


# Clases compatibles con FastAPI (si está disponible)
if FASTAPI_AVAILABLE:
    @ErrorCode(description="Failed to connect to robot")
    class HTTPRobotConnectionError(RobotConnectionError, HTTPException):
        """Error de conexión con el robot (compatible con FastAPI)."""
        
        def __init__(self):
            """Initialize exception with description from @ErrorCode decorator."""
            detail = getattr(self.__class__, '_error_description', "Failed to connect to robot")
            super().__init__(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail
            )
    
    @ErrorCode(description="Robot not initialized")
    class HTTPRobotNotInitializedError(RobotNotInitializedError, HTTPException):
        """Error cuando el robot no está inicializado (compatible con FastAPI)."""
        
        def __init__(self):
            """Initialize exception with description from @ErrorCode decorator."""
            detail = getattr(self.__class__, '_error_description', "Robot not initialized")
            super().__init__(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail
            )
    
    @ErrorCode(description="Invalid command")
    class HTTPInvalidCommandError(InvalidCommandError, HTTPException):
        """Error cuando un comando es inválido (compatible con FastAPI)."""
        
        def __init__(self):
            """Initialize exception with description from @ErrorCode decorator."""
            detail = getattr(self.__class__, '_error_description', "Invalid command")
            super().__init__(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail
            )
    
    @ErrorCode(description="Movement failed")
    class HTTPMovementError(MovementError, HTTPException):
        """Error durante el movimiento del robot (compatible con FastAPI)."""
        
        def __init__(self):
            """Initialize exception with description from @ErrorCode decorator."""
            detail = getattr(self.__class__, '_error_description', "Movement failed")
            super().__init__(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=detail
            )
    
    @ErrorCode(description="Model error")
    class HTTPModelError(AIModelError, HTTPException):
        """Error relacionado con modelos de IA (compatible con FastAPI)."""
        
        def __init__(self):
            """Initialize exception with description from @ErrorCode decorator."""
            detail = getattr(self.__class__, '_error_description', "Model error")
            super().__init__(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=detail
            )
    
    @ErrorCode(description="Vision processing error")
    class HTTPVisionError(VisionError, HTTPException):
        """Error en procesamiento de visión (compatible con FastAPI)."""
        
        def __init__(self):
            """Initialize exception with description from @ErrorCode decorator."""
            detail = getattr(self.__class__, '_error_description', "Vision processing error")
            super().__init__(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=detail
            )

