"""
Sistema de validación avanzado para Robot Movement AI v2.0
Validaciones personalizadas y decorators
"""

from typing import Any, Callable, Optional, List, Dict
from functools import wraps
from pydantic import BaseModel, validator, Field
from datetime import datetime


class ValidationError(Exception):
    """Excepción de validación personalizada"""
    def __init__(self, message: str, field: Optional[str] = None, errors: Optional[List[Dict]] = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.errors = errors or []


def validate_input(validator_func: Callable):
    """
    Decorator para validar entrada de funciones
    
    Args:
        validator_func: Función de validación que retorna (is_valid, error_message)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            is_valid, error = validator_func(*args, **kwargs)
            if not is_valid:
                raise ValidationError(error)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            is_valid, error = validator_func(*args, **kwargs)
            if not is_valid:
                raise ValidationError(error)
            return func(*args, **kwargs)
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class PositionValidator:
    """Validador para posiciones de robot"""
    
    MAX_COORDINATE = 1000.0
    MIN_COORDINATE = -1000.0
    
    @staticmethod
    def validate_coordinate(value: float, name: str = "coordinate") -> tuple[bool, Optional[str]]:
        """Validar coordenada"""
        if not isinstance(value, (int, float)):
            return False, f"{name} must be a number"
        
        if value < PositionValidator.MIN_COORDINATE or value > PositionValidator.MAX_COORDINATE:
            return False, f"{name} must be between {PositionValidator.MIN_COORDINATE} and {PositionValidator.MAX_COORDINATE}"
        
        return True, None
    
    @staticmethod
    def validate_position(x: float, y: float, z: float) -> tuple[bool, Optional[str]]:
        """Validar posición completa"""
        for coord, name in [(x, "x"), (y, "y"), (z, "z")]:
            is_valid, error = PositionValidator.validate_coordinate(coord, name)
            if not is_valid:
                return False, error
        return True, None


class RobotIDValidator:
    """Validador para IDs de robot"""
    
    MIN_LENGTH = 1
    MAX_LENGTH = 100
    ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    
    @staticmethod
    def validate(robot_id: str) -> tuple[bool, Optional[str]]:
        """Validar ID de robot"""
        if not isinstance(robot_id, str):
            return False, "robot_id must be a string"
        
        if len(robot_id) < RobotIDValidator.MIN_LENGTH:
            return False, f"robot_id must be at least {RobotIDValidator.MIN_LENGTH} characters"
        
        if len(robot_id) > RobotIDValidator.MAX_LENGTH:
            return False, f"robot_id must be at most {RobotIDValidator.MAX_LENGTH} characters"
        
        if not all(c in RobotIDValidator.ALLOWED_CHARS for c in robot_id):
            return False, "robot_id contains invalid characters"
        
        return True, None


class SpeedValidator:
    """Validador para velocidades"""
    
    MIN_SPEED = 0.0
    MAX_SPEED = 100.0
    
    @staticmethod
    def validate(speed: float) -> tuple[bool, Optional[str]]:
        """Validar velocidad"""
        if not isinstance(speed, (int, float)):
            return False, "speed must be a number"
        
        if speed < SpeedValidator.MIN_SPEED or speed > SpeedValidator.MAX_SPEED:
            return False, f"speed must be between {SpeedValidator.MIN_SPEED} and {SpeedValidator.MAX_SPEED}"
        
        return True, None


class MoveCommandValidator(BaseModel):
    """Validador Pydantic para comandos de movimiento"""
    robot_id: str = Field(..., min_length=1, max_length=100)
    target_x: float = Field(..., ge=-1000.0, le=1000.0)
    target_y: float = Field(..., ge=-1000.0, le=1000.0)
    target_z: float = Field(..., ge=-1000.0, le=1000.0)
    speed: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    @validator('robot_id')
    def validate_robot_id(cls, v):
        """Validar formato de robot_id"""
        is_valid, error = RobotIDValidator.validate(v)
        if not is_valid:
            raise ValueError(error)
        return v
    
    @validator('target_x', 'target_y', 'target_z')
    def validate_coordinates(cls, v):
        """Validar coordenadas"""
        is_valid, error = PositionValidator.validate_coordinate(v)
        if not is_valid:
            raise ValueError(error)
        return v
    
    @validator('speed')
    def validate_speed(cls, v):
        """Validar velocidad"""
        if v is not None:
            is_valid, error = SpeedValidator.validate(v)
            if not is_valid:
                raise ValueError(error)
        return v


def validate_move_command(func: Callable) -> Callable:
    """Decorator para validar comandos de movimiento"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Extraer argumentos
        command_data = {
            'robot_id': kwargs.get('robot_id') or (args[0] if args else None),
            'target_x': kwargs.get('target_x') or (args[1] if len(args) > 1 else None),
            'target_y': kwargs.get('target_y') or (args[2] if len(args) > 2 else None),
            'target_z': kwargs.get('target_z') or (args[3] if len(args) > 3 else None),
            'speed': kwargs.get('speed') or (args[4] if len(args) > 4 else None),
        }
        
        # Validar
        try:
            validated = MoveCommandValidator(**command_data)
            kwargs.update(validated.dict())
        except Exception as e:
            raise ValidationError(f"Invalid move command: {e}")
        
        return await func(*args, **kwargs)
    
    return async_wrapper
