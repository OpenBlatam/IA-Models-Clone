from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import logging
import time
import asyncio
from typing import (
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np
import functools
import inspect
from contextlib import contextmanager
from .error_handling import (
        import psutil
        import psutil
from typing import Any, List, Dict, Optional
"""
🚀 EARLY RETURNS - GUARD CLAUSE PATTERN
======================================

Sistema de early returns para evitar if statements anidados profundos.
Implementa el patrón "guard clause" para mejorar legibilidad y mantenibilidad.
"""

    Any, Optional, Union, Dict, List, Tuple, Callable, 
    TypeVar, Generic, Protocol, runtime_checkable
)

    AIVideoError, ErrorCategory, ErrorSeverity, ErrorContext,
    ValidationError, SystemError, ConfigurationError
)

# =============================================================================
# EARLY RETURN TYPES
# =============================================================================

class ReturnType(Enum):
    """Tipos de early returns."""
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    SUCCESS = auto()
    SKIP = auto()

@dataclass
class EarlyReturnResult:
    """Resultado de early return."""
    should_return: bool
    return_type: ReturnType
    message: str
    value: Any = None
    details: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# EARLY RETURN DECORATORS
# =============================================================================

def early_return_on_error(
    error_conditions: List[Callable],
    default_return: Any = None,
    log_errors: bool = True
):
    """Decorador para early returns en condiciones de error."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Verificar condiciones de error al inicio
            for condition in error_conditions:
                try:
                    if condition(*args, **kwargs):
                        if log_errors:
                            logging.error(f"Early return en {func.__name__}: condición de error cumplida")
                        return default_return
                except Exception as e:
                    if log_errors:
                        logging.error(f"Error en condición de early return: {e}")
                    return default_return
            
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Verificar condiciones de error al inicio
            for condition in error_conditions:
                try:
                    if asyncio.iscoroutinefunction(condition):
                        should_return = await condition(*args, **kwargs)
                    else:
                        should_return = condition(*args, **kwargs)
                    
                    if should_return:
                        if log_errors:
                            logging.error(f"Early return en {func.__name__}: condición de error cumplida")
                        return default_return
                except Exception as e:
                    if log_errors:
                        logging.error(f"Error en condición de early return: {e}")
                    return default_return
            
            return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

def early_return_on_condition(
    condition: Callable,
    return_value: Any,
    return_type: ReturnType = ReturnType.SUCCESS,
    log_return: bool = False
):
    """Decorador para early return en condición específica."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Verificar condición al inicio
            try:
                if condition(*args, **kwargs):
                    if log_return:
                        logging.info(f"Early return en {func.__name__}: {return_type.name}")
                    return return_value
            except Exception as e:
                logging.error(f"Error en condición de early return: {e}")
            
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Verificar condición al inicio
            try:
                if asyncio.iscoroutinefunction(condition):
                    should_return = await condition(*args, **kwargs)
                else:
                    should_return = condition(*args, **kwargs)
                
                if should_return:
                    if log_return:
                        logging.info(f"Early return en {func.__name__}: {return_type.name}")
                    return return_value
            except Exception as e:
                logging.error(f"Error en condición de early return: {e}")
            
            return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

# =============================================================================
# EARLY RETURN CONDITIONS
# =============================================================================

class EarlyReturnConditions:
    """Condiciones para early returns."""
    
    @staticmethod
    def is_none(value: Any) -> bool:
        """Verificar si valor es None."""
        return value is None
    
    @staticmethod
    def is_empty(data: Any) -> bool:
        """Verificar si datos están vacíos."""
        if data is None:
            return True
        
        if isinstance(data, (str, list, tuple, dict, set)):
            return len(data) == 0
        
        if isinstance(data, np.ndarray):
            return data.size == 0
        
        return False
    
    @staticmethod
    def file_not_exists(file_path: Union[str, Path]) -> bool:
        """Verificar si archivo no existe."""
        return not Path(file_path).exists()
    
    @staticmethod
    def invalid_batch_size(batch_size: int, max_size: int = 32) -> bool:
        """Verificar si batch size es inválido."""
        return batch_size <= 0 or batch_size > max_size
    
    @staticmethod
    def invalid_dimensions(width: int, height: int, max_dim: int = 4096) -> bool:
        """Verificar si dimensiones son inválidas."""
        return width <= 0 or height <= 0 or width > max_dim or height > max_dim
    
    @staticmethod
    def insufficient_memory(required_mb: float) -> bool:
        """Verificar si hay memoria insuficiente."""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        return available_memory < required_mb
    
    @staticmethod
    def system_overloaded(max_cpu_percent: float = 90.0) -> bool:
        """Verificar si sistema está sobrecargado."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return cpu_percent > max_cpu_percent
    
    @staticmethod
    def invalid_quality(quality: float) -> bool:
        """Verificar si calidad es inválida."""
        return quality < 0.0 or quality > 1.0
    
    @staticmethod
    def invalid_format(file_path: Union[str, Path], valid_formats: set) -> bool:
        """Verificar si formato es inválido."""
        return Path(file_path).suffix.lower() not in valid_formats
    
    @staticmethod
    def data_corrupted(data: np.ndarray) -> bool:
        """Verificar si datos están corruptos."""
        return np.isnan(data).any() or np.isinf(data).any()
    
    @staticmethod
    def model_not_loaded(model_name: str, loaded_models: set) -> bool:
        """Verificar si modelo no está cargado."""
        return model_name not in loaded_models

# =============================================================================
# EARLY RETURN HELPERS
# =============================================================================

def return_if_none(value: Any, return_value: Any = None, message: str = "Value is None") -> Optional[Any]:
    """Early return si valor es None."""
    if value is None:
        logging.warning(f"Early return: {message}")
        return return_value
    return None

def return_if_empty(data: Any, return_value: Any = None, message: str = "Data is empty") -> Optional[Any]:
    """Early return si datos están vacíos."""
    if EarlyReturnConditions.is_empty(data):
        logging.warning(f"Early return: {message}")
        return return_value
    return None

def return_if_file_not_exists(file_path: Union[str, Path], return_value: Any = None, message: str = "File not found") -> Optional[Any]:
    """Early return si archivo no existe."""
    if EarlyReturnConditions.file_not_exists(file_path):
        logging.error(f"Early return: {message} - {file_path}")
        return return_value
    return None

def return_if_invalid_batch_size(batch_size: int, max_size: int = 32, return_value: Any = None) -> Optional[Any]:
    """Early return si batch size es inválido."""
    if EarlyReturnConditions.invalid_batch_size(batch_size, max_size):
        logging.error(f"Early return: Invalid batch size {batch_size}, max allowed: {max_size}")
        return return_value
    return None

def return_if_insufficient_memory(required_mb: float, return_value: Any = None) -> Optional[Any]:
    """Early return si hay memoria insuficiente."""
    if EarlyReturnConditions.insufficient_memory(required_mb):
        logging.error(f"Early return: Insufficient memory, required: {required_mb}MB")
        return return_value
    return None

def return_if_system_overloaded(max_cpu_percent: float = 90.0, return_value: Any = None) -> Optional[Any]:
    """Early return si sistema está sobrecargado."""
    if EarlyReturnConditions.system_overloaded(max_cpu_percent):
        logging.warning(f"Early return: System overloaded, CPU > {max_cpu_percent}%")
        return return_value
    return None

def return_if_invalid_quality(quality: float, return_value: Any = None) -> Optional[Any]:
    """Early return si calidad es inválida."""
    if EarlyReturnConditions.invalid_quality(quality):
        logging.error(f"Early return: Invalid quality {quality}, must be between 0 and 1")
        return return_value
    return None

def return_if_data_corrupted(data: np.ndarray, return_value: Any = None) -> Optional[Any]:
    """Early return si datos están corruptos."""
    if EarlyReturnConditions.data_corrupted(data):
        logging.error("Early return: Data contains NaN or infinite values")
        return return_value
    return None

# =============================================================================
# EARLY RETURN CONTEXT MANAGERS
# =============================================================================

@contextmanager
def early_return_context(condition: Callable, return_value: Any = None, message: str = "Early return condition met"):
    """Context manager para early returns."""
    if condition():
        logging.warning(f"Early return: {message}")
        yield return_value
        return
    
    try:
        yield None
    except Exception as e:
        logging.error(f"Error in early return context: {e}")
        yield return_value

@contextmanager
def validation_context(validators: List[Callable], return_value: Any = None):
    """Context manager para validación con early returns."""
    for validator in validators:
        try:
            if not validator():
                logging.error(f"Early return: Validation failed - {validator.__name__}")
                yield return_value
                return
        except Exception as e:
            logging.error(f"Early return: Validation error - {e}")
            yield return_value
            return
    
    try:
        yield None
    except Exception as e:
        logging.error(f"Error in validation context: {e}")
        yield return_value

# =============================================================================
# EARLY RETURN PATTERNS
# =============================================================================

class EarlyReturnPatterns:
    """Patrones comunes de early returns."""
    
    @staticmethod
    def validate_inputs(*args, **kwargs) -> Optional[Any]:
        """Patrón para validar inputs con early returns."""
        # Validar argumentos posicionales
        for i, arg in enumerate(args):
            if arg is None:
                logging.error(f"Early return: Argument {i} is None")
                return None
        
        # Validar argumentos nombrados
        for key, value in kwargs.items():
            if value is None:
                logging.error(f"Early return: Keyword argument '{key}' is None")
                return None
        
        return None  # Continuar si todo es válido
    
    @staticmethod
    def validate_file_operations(file_path: Union[str, Path], operation: str = "read") -> Optional[Any]:
        """Patrón para validar operaciones de archivo."""
        # Verificar que archivo existe
        if not Path(file_path).exists():
            logging.error(f"Early return: File not found for {operation} - {file_path}")
            return None
        
        # Verificar permisos según operación
        if operation == "read" and not os.access(file_path, os.R_OK):
            logging.error(f"Early return: File not readable - {file_path}")
            return None
        
        if operation == "write" and not os.access(file_path, os.W_OK):
            logging.error(f"Early return: File not writable - {file_path}")
            return None
        
        return None  # Continuar si todo es válido
    
    @staticmethod
    def validate_system_resources(required_memory_mb: float = 0, max_cpu_percent: float = 90.0) -> Optional[Any]:
        """Patrón para validar recursos del sistema."""
        # Verificar memoria
        if required_memory_mb > 0:
            if EarlyReturnConditions.insufficient_memory(required_memory_mb):
                logging.error(f"Early return: Insufficient memory, required: {required_memory_mb}MB")
                return None
        
        # Verificar CPU
        if EarlyReturnConditions.system_overloaded(max_cpu_percent):
            logging.warning(f"Early return: System overloaded, CPU > {max_cpu_percent}%")
            return None
        
        return None  # Continuar si todo es válido
    
    @staticmethod
    def validate_data_integrity(data: np.ndarray, expected_shape: Optional[Tuple] = None) -> Optional[Any]:
        """Patrón para validar integridad de datos."""
        # Verificar que no sea None
        if data is None:
            logging.error("Early return: Data is None")
            return None
        
        # Verificar que sea NumPy array
        if not isinstance(data, np.ndarray):
            logging.error("Early return: Data is not NumPy array")
            return None
        
        # Verificar que no esté vacío
        if data.size == 0:
            logging.error("Early return: Data is empty")
            return None
        
        # Verificar forma si se especifica
        if expected_shape and data.shape != expected_shape:
            logging.error(f"Early return: Data shape {data.shape} != expected {expected_shape}")
            return None
        
        # Verificar que no contenga valores corruptos
        if EarlyReturnConditions.data_corrupted(data):
            logging.error("Early return: Data contains NaN or infinite values")
            return None
        
        return None  # Continuar si todo es válido

# =============================================================================
# EARLY RETURN EXAMPLES
# =============================================================================

def process_video_with_early_returns(video_path: str, batch_size: int, quality: float) -> Dict[str, Any]:
    """
    Procesar video usando early returns para evitar if statements anidados.
    
    Patrón: Validar condiciones al inicio y retornar temprano si no se cumplen.
    """
    # Early return: validar video_path
    if video_path is None:
        logging.error("Early return: video_path is None")
        return {"error": "video_path is required"}
    
    if not Path(video_path).exists():
        logging.error(f"Early return: Video file not found - {video_path}")
        return {"error": "Video file not found"}
    
    # Early return: validar batch_size
    if batch_size <= 0 or batch_size > 32:
        logging.error(f"Early return: Invalid batch size {batch_size}")
        return {"error": "Invalid batch size"}
    
    # Early return: validar quality
    if quality < 0.0 or quality > 1.0:
        logging.error(f"Early return: Invalid quality {quality}")
        return {"error": "Quality must be between 0 and 1"}
    
    # Early return: verificar memoria
    if EarlyReturnConditions.insufficient_memory(1024.0):  # 1GB
        logging.error("Early return: Insufficient memory")
        return {"error": "Insufficient memory"}
    
    # Early return: verificar sistema
    if EarlyReturnConditions.system_overloaded(90.0):
        logging.warning("Early return: System overloaded")
        return {"error": "System overloaded"}
    
    # Si llegamos aquí, todas las validaciones pasaron
    # Procesar video
    print(f"✅ Procesando video: {video_path}")
    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Quality: {quality}")
    
    # Simular procesamiento
    time.sleep(1)
    
    return {
        "success": True,
        "video_path": video_path,
        "batch_size": batch_size,
        "quality": quality,
        "processed": True
    }

def load_model_with_early_returns(model_path: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cargar modelo usando early returns.
    
    Patrón: Validar cada condición al inicio y retornar temprano.
    """
    # Early return: validar model_path
    if model_path is None:
        return {"error": "model_path is required"}
    
    if not Path(model_path).exists():
        return {"error": f"Model file not found: {model_path}"}
    
    # Early return: validar model_config
    if model_config is None:
        return {"error": "model_config is required"}
    
    if not isinstance(model_config, dict):
        return {"error": "model_config must be a dictionary"}
    
    required_keys = {"model_type", "batch_size"}
    if not all(key in model_config for key in required_keys):
        return {"error": f"model_config missing required keys: {required_keys}"}
    
    # Early return: validar batch_size en config
    batch_size = model_config.get("batch_size")
    if batch_size <= 0 or batch_size > 64:
        return {"error": f"Invalid batch_size in config: {batch_size}"}
    
    # Early return: verificar memoria para modelo
    if EarlyReturnConditions.insufficient_memory(2048.0):  # 2GB
        return {"error": "Insufficient memory for model loading"}
    
    # Si llegamos aquí, todas las validaciones pasaron
    print(f"✅ Cargando modelo: {model_path}")
    print(f"✅ Configuración: {model_config}")
    
    # Simular carga de modelo
    time.sleep(2)
    
    return {
        "success": True,
        "model_path": model_path,
        "config": model_config,
        "loaded": True
    }

def process_data_with_early_returns(data: np.ndarray, operation: str = "normalize") -> np.ndarray:
    """
    Procesar datos usando early returns.
    
    Patrón: Validar integridad de datos al inicio.
    """
    # Early return: validar data
    if data is None:
        logging.error("Early return: data is None")
        return np.array([])
    
    if not isinstance(data, np.ndarray):
        logging.error("Early return: data is not NumPy array")
        return np.array([])
    
    if data.size == 0:
        logging.error("Early return: data is empty")
        return np.array([])
    
    # Early return: verificar datos corruptos
    if np.isnan(data).any() or np.isinf(data).any():
        logging.error("Early return: data contains NaN or infinite values")
        return np.array([])
    
    # Early return: validar operación
    valid_operations = {"normalize", "scale", "filter", "transform"}
    if operation not in valid_operations:
        logging.error(f"Early return: Invalid operation '{operation}'")
        return np.array([])
    
    # Si llegamos aquí, todas las validaciones pasaron
    print(f"✅ Procesando datos: {data.shape}")
    print(f"✅ Operación: {operation}")
    
    # Simular procesamiento
    if operation == "normalize":
        result = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif operation == "scale":
        result = data * 2.0
    else:
        result = data
    
    return result

# =============================================================================
# DECORATOR EXAMPLES
# =============================================================================

@early_return_on_error([
    lambda video_path, **kwargs: video_path is None,
    lambda video_path, **kwargs: not Path(video_path).exists(),
    lambda batch_size, **kwargs: batch_size <= 0 or batch_size > 32,
    lambda quality, **kwargs: quality < 0.0 or quality > 1.0
], default_return={"error": "Validation failed"})
def process_video_decorated(video_path: str, batch_size: int, quality: float) -> Dict[str, Any]:
    """Procesar video usando decorador de early returns."""
    print(f"✅ Procesando video: {video_path}")
    time.sleep(1)
    return {"success": True, "video_path": video_path}

@early_return_on_condition(
    condition=lambda data, **kwargs: data is None or data.size == 0,
    return_value=np.array([]),
    return_type=ReturnType.ERROR
)
def process_data_decorated(data: np.ndarray) -> np.ndarray:
    """Procesar datos usando decorador de early returns."""
    print(f"✅ Procesando datos: {data.shape}")
    return data * 2.0

# =============================================================================
# ASYNC EXAMPLES
# =============================================================================

async def async_process_video_with_early_returns(video_path: str, quality: float) -> Dict[str, Any]:
    """
    Procesar video de forma asíncrona usando early returns.
    """
    # Early return: validar video_path
    if video_path is None:
        return {"error": "video_path is required"}
    
    if not Path(video_path).exists():
        return {"error": f"Video file not found: {video_path}"}
    
    # Early return: validar quality
    if quality < 0.0 or quality > 1.0:
        return {"error": f"Invalid quality: {quality}"}
    
    # Early return: verificar memoria
    if EarlyReturnConditions.insufficient_memory(1024.0):
        return {"error": "Insufficient memory"}
    
    # Si llegamos aquí, todas las validaciones pasaron
    print(f"✅ Procesando video async: {video_path}")
    
    # Simular procesamiento asíncrono
    await asyncio.sleep(2)
    
    return {
        "success": True,
        "video_path": video_path,
        "quality": quality,
        "processed": True
    }

@early_return_on_error([
    lambda model_path, **kwargs: model_path is None,
    lambda model_path, **kwargs: not Path(model_path).exists(),
    lambda batch_size, **kwargs: batch_size <= 0 or batch_size > 64
], default_return={"error": "Model loading failed"})
async def async_load_model_decorated(model_path: str, batch_size: int) -> Dict[str, Any]:
    """Cargar modelo de forma asíncrona usando decorador."""
    print(f"✅ Cargando modelo async: {model_path}")
    await asyncio.sleep(1)
    return {"success": True, "model_path": model_path}

# =============================================================================
# COMPLEX EXAMPLES
# =============================================================================

class VideoProcessor:
    """Procesador de video usando early returns."""
    
    def __init__(self) -> Any:
        self.loaded_models = set()
        self.processing = False
    
    def process_video_pipeline(self, video_path: str, model_name: str, batch_size: int) -> Dict[str, Any]:
        """
        Pipeline completo de procesamiento usando early returns.
        
        Patrón: Cada validación al inicio, early return si falla.
        """
        # Early return: validar video_path
        if video_path is None:
            return {"error": "video_path is required"}
        
        if not Path(video_path).exists():
            return {"error": f"Video file not found: {video_path}"}
        
        # Early return: validar formato de video
        valid_formats = {'.mp4', '.avi', '.mov', '.mkv'}
        if Path(video_path).suffix.lower() not in valid_formats:
            return {"error": f"Unsupported video format: {Path(video_path).suffix}"}
        
        # Early return: validar model_name
        if model_name is None:
            return {"error": "model_name is required"}
        
        if model_name not in self.loaded_models:
            return {"error": f"Model not loaded: {model_name}"}
        
        # Early return: validar batch_size
        if batch_size <= 0 or batch_size > 32:
            return {"error": f"Invalid batch_size: {batch_size}"}
        
        # Early return: verificar si ya está procesando
        if self.processing:
            return {"error": "Already processing another video"}
        
        # Early return: verificar memoria
        if EarlyReturnConditions.insufficient_memory(2048.0):
            return {"error": "Insufficient memory for processing"}
        
        # Early return: verificar sistema
        if EarlyReturnConditions.system_overloaded(85.0):
            return {"error": "System overloaded"}
        
        # Si llegamos aquí, todas las validaciones pasaron
        self.processing = True
        
        try:
            print(f"🚀 Iniciando pipeline para: {video_path}")
            print(f"📹 Modelo: {model_name}")
            print(f"⚙️ Batch size: {batch_size}")
            
            # Simular procesamiento
            time.sleep(3)
            
            return {
                "success": True,
                "video_path": video_path,
                "model_name": model_name,
                "batch_size": batch_size,
                "processed": True
            }
        
        finally:
            self.processing = False
    
    def load_model(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """
        Cargar modelo usando early returns.
        """
        # Early return: validar model_path
        if model_path is None:
            return {"error": "model_path is required"}
        
        if not Path(model_path).exists():
            return {"error": f"Model file not found: {model_path}"}
        
        # Early return: validar model_name
        if model_name is None:
            return {"error": "model_name is required"}
        
        if model_name in self.loaded_models:
            return {"error": f"Model already loaded: {model_name}"}
        
        # Early return: verificar memoria
        if EarlyReturnConditions.insufficient_memory(1024.0):
            return {"error": "Insufficient memory for model loading"}
        
        # Si llegamos aquí, todas las validaciones pasaron
        print(f"📦 Cargando modelo: {model_name}")
        print(f"📁 Archivo: {model_path}")
        
        # Simular carga
        time.sleep(2)
        
        self.loaded_models.add(model_name)
        
        return {
            "success": True,
            "model_name": model_name,
            "model_path": model_path,
            "loaded": True
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_early_returns(func: Callable) -> Callable:
    """Aplicar early returns a función existente."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Aplicar validaciones básicas
        result = EarlyReturnPatterns.validate_inputs(*args, **kwargs)
        if result is not None:
            return result
        
        return func(*args, **kwargs)
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        # Aplicar validaciones básicas
        result = EarlyReturnPatterns.validate_inputs(*args, **kwargs)
        if result is not None:
            return result
        
        return await func(*args, **kwargs)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

def create_early_return_validator(conditions: List[Callable], return_value: Any = None) -> Callable:
    """Crear validador de early returns personalizado."""
    def validator(*args, **kwargs) -> Optional[Any]:
        for condition in conditions:
            try:
                if condition(*args, **kwargs):
                    return return_value
            except Exception as e:
                logging.error(f"Error in early return condition: {e}")
                return return_value
        return None
    
    return validator

# =============================================================================
# INITIALIZATION
# =============================================================================

def setup_early_returns():
    """Configurar sistema de early returns."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Sistema de early returns inicializado")
    
    return {
        "conditions": EarlyReturnConditions,
        "patterns": EarlyReturnPatterns,
        "helpers": {
            "return_if_none": return_if_none,
            "return_if_empty": return_if_empty,
            "return_if_file_not_exists": return_if_file_not_exists,
            "return_if_invalid_batch_size": return_if_invalid_batch_size,
            "return_if_insufficient_memory": return_if_insufficient_memory,
            "return_if_system_overloaded": return_if_system_overloaded,
            "return_if_invalid_quality": return_if_invalid_quality,
            "return_if_data_corrupted": return_if_data_corrupted
        }
    }

# Configuración automática al importar
early_returns_system = setup_early_returns() 