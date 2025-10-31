from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

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
import inspect
import functools
from collections import defaultdict
import json
import re
from .error_handling import (
from typing import Any, List, Dict, Optional
"""
⚡ EARLY VALIDATION - FAIL FAST PRINCIPLE
========================================

Sistema de validación temprana que verifica todos los inputs y parámetros
al inicio de las funciones antes de comenzar cualquier procesamiento.
Implementa el principio "fail fast" para detectar errores lo antes posible.
"""

    Any, Optional, Union, Dict, List, Tuple, Callable, 
    TypeVar, Generic, Protocol, runtime_checkable, get_type_hints
)

    AIVideoError, ErrorCategory, ErrorSeverity, ErrorContext,
    ValidationError, ConfigurationError, DataValidationError
)

# =============================================================================
# VALIDATION TYPES
# =============================================================================

class ValidationType(Enum):
    """Tipos de validación."""
    TYPE = auto()
    RANGE = auto()
    FORMAT = auto()
    EXISTENCE = auto()
    SIZE = auto()
    CONTENT = auto()
    RELATIONSHIP = auto()
    CUSTOM = auto()

class ValidationLevel(Enum):
    """Niveles de validación."""
    STRICT = auto()
    NORMAL = auto()
    LENIENT = auto()

@dataclass
class ValidationRule:
    """Regla de validación."""
    name: str
    validation_type: ValidationType
    validator: Callable
    error_message: str
    level: ValidationLevel = ValidationLevel.NORMAL
    required: bool = True
    default_value: Any = None

@dataclass
class ValidationResult:
    """Resultado de validación."""
    valid: bool
    message: str
    field_name: str
    value: Any
    rule: ValidationRule
    details: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# TYPE VALIDATORS
# =============================================================================

class TypeValidators:
    """Validadores de tipos."""
    
    @staticmethod
    def is_string(value: Any) -> bool:
        """Validar que sea string."""
        return isinstance(value, str)
    
    @staticmethod
    def is_integer(value: Any) -> bool:
        """Validar que sea entero."""
        return isinstance(value, int) and not isinstance(value, bool)
    
    @staticmethod
    def is_float(value: Any) -> bool:
        """Validar que sea float."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    
    @staticmethod
    def is_boolean(value: Any) -> bool:
        """Validar que sea booleano."""
        return isinstance(value, bool)
    
    @staticmethod
    def is_list(value: Any) -> bool:
        """Validar que sea lista."""
        return isinstance(value, list)
    
    @staticmethod
    def is_dict(value: Any) -> bool:
        """Validar que sea diccionario."""
        return isinstance(value, dict)
    
    @staticmethod
    def is_numpy_array(value: Any) -> bool:
        """Validar que sea array de NumPy."""
        return isinstance(value, np.ndarray)
    
    @staticmethod
    def is_path(value: Any) -> bool:
        """Validar que sea Path o string convertible a Path."""
        return isinstance(value, (str, Path)) or hasattr(value, '__fspath__')
    
    @staticmethod
    def is_callable(value: Any) -> bool:
        """Validar que sea callable."""
        return callable(value)
    
    @staticmethod
    def is_async_callable(value: Any) -> bool:
        """Validar que sea async callable."""
        return asyncio.iscoroutinefunction(value)

# =============================================================================
# RANGE VALIDATORS
# =============================================================================

class RangeValidators:
    """Validadores de rangos."""
    
    @staticmethod
    def in_range(value: Union[int, float], min_val: Union[int, float], 
                 max_val: Union[int, float]) -> bool:
        """Validar que valor esté en rango."""
        return min_val <= value <= max_val
    
    @staticmethod
    def positive(value: Union[int, float]) -> bool:
        """Validar que valor sea positivo."""
        return value > 0
    
    @staticmethod
    def non_negative(value: Union[int, float]) -> bool:
        """Validar que valor sea no negativo."""
        return value >= 0
    
    @staticmethod
    def between_zero_one(value: float) -> bool:
        """Validar que valor esté entre 0 y 1."""
        return 0.0 <= value <= 1.0
    
    @staticmethod
    def power_of_two(value: int) -> bool:
        """Validar que valor sea potencia de 2."""
        return value > 0 and (value & (value - 1)) == 0
    
    @staticmethod
    def even(value: int) -> bool:
        """Validar que valor sea par."""
        return value % 2 == 0
    
    @staticmethod
    def odd(value: int) -> bool:
        """Validar que valor sea impar."""
        return value % 2 == 1

# =============================================================================
# FORMAT VALIDATORS
# =============================================================================

class FormatValidators:
    """Validadores de formato."""
    
    @staticmethod
    def is_email(value: str) -> bool:
        """Validar formato de email."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))
    
    @staticmethod
    def is_url(value: str) -> bool:
        """Validar formato de URL."""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, value))
    
    @staticmethod
    def is_filename(value: str) -> bool:
        """Validar formato de nombre de archivo."""
        pattern = r'^[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, value))
    
    @staticmethod
    def is_video_format(value: str) -> bool:
        """Validar formato de video."""
        valid_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        return value.lower() in valid_formats
    
    @staticmethod
    def is_image_format(value: str) -> bool:
        """Validar formato de imagen."""
        valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return value.lower() in valid_formats
    
    @staticmethod
    def is_json_string(value: str) -> bool:
        """Validar que string sea JSON válido."""
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def is_hex_color(value: str) -> bool:
        """Validar formato de color hexadecimal."""
        pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
        return bool(re.match(pattern, value))

# =============================================================================
# EXISTENCE VALIDATORS
# =============================================================================

class ExistenceValidators:
    """Validadores de existencia."""
    
    @staticmethod
    def file_exists(value: Union[str, Path]) -> bool:
        """Validar que archivo existe."""
        return Path(value).exists()
    
    @staticmethod
    def directory_exists(value: Union[str, Path]) -> bool:
        """Validar que directorio existe."""
        return Path(value).is_dir()
    
    @staticmethod
    def file_readable(value: Union[str, Path]) -> bool:
        """Validar que archivo es legible."""
        path = Path(value)
        return path.exists() and path.is_file() and os.access(path, os.R_OK)
    
    @staticmethod
    def file_writable(value: Union[str, Path]) -> bool:
        """Validar que archivo es escribible."""
        path = Path(value)
        return path.exists() and path.is_file() and os.access(path, os.W_OK)
    
    @staticmethod
    def directory_writable(value: Union[str, Path]) -> bool:
        """Validar que directorio es escribible."""
        path = Path(value)
        return path.exists() and path.is_dir() and os.access(path, os.W_OK)
    
    @staticmethod
    def not_empty(value: Any) -> bool:
        """Validar que valor no esté vacío."""
        if value is None:
            return False
        
        if isinstance(value, (str, list, tuple, dict, set)):
            return len(value) > 0
        
        if isinstance(value, np.ndarray):
            return value.size > 0
        
        return True
    
    @staticmethod
    def not_none(value: Any) -> bool:
        """Validar que valor no sea None."""
        return value is not None

# =============================================================================
# SIZE VALIDATORS
# =============================================================================

class SizeValidators:
    """Validadores de tamaño."""
    
    @staticmethod
    def string_length(value: str, min_len: int = 0, max_len: int = 1000) -> bool:
        """Validar longitud de string."""
        return min_len <= len(value) <= max_len
    
    @staticmethod
    def list_length(value: list, min_len: int = 0, max_len: int = 1000) -> bool:
        """Validar longitud de lista."""
        return min_len <= len(value) <= max_len
    
    @staticmethod
    def array_shape(value: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
        """Validar forma de array."""
        return value.shape == expected_shape
    
    @staticmethod
    def array_size(value: np.ndarray, min_size: int = 0, max_size: int = 1000000) -> bool:
        """Validar tamaño de array."""
        return min_size <= value.size <= max_size
    
    @staticmethod
    def file_size(value: Union[str, Path], max_size_mb: float) -> bool:
        """Validar tamaño de archivo."""
        try:
            file_size_mb = Path(value).stat().st_size / (1024 * 1024)
            return file_size_mb <= max_size_mb
        except:
            return False
    
    @staticmethod
    def memory_usage(value: np.ndarray, max_mb: float) -> bool:
        """Validar uso de memoria de array."""
        memory_mb = value.nbytes / (1024 * 1024)
        return memory_mb <= max_mb

# =============================================================================
# CONTENT VALIDATORS
# =============================================================================

class ContentValidators:
    """Validadores de contenido."""
    
    @staticmethod
    def no_nan_values(value: np.ndarray) -> bool:
        """Validar que no hay valores NaN."""
        return not np.isnan(value).any()
    
    @staticmethod
    def no_inf_values(value: np.ndarray) -> bool:
        """Validar que no hay valores infinitos."""
        return not np.isinf(value).any()
    
    @staticmethod
    def in_range_values(value: np.ndarray, min_val: float, max_val: float) -> bool:
        """Validar que todos los valores estén en rango."""
        return np.all((value >= min_val) & (value <= max_val))
    
    @staticmethod
    def positive_values(value: np.ndarray) -> bool:
        """Validar que todos los valores sean positivos."""
        return np.all(value > 0)
    
    @staticmethod
    def normalized_values(value: np.ndarray) -> bool:
        """Validar que valores estén normalizados (0-1)."""
        return np.all((value >= 0) & (value <= 1))
    
    @staticmethod
    def unique_values(value: list) -> bool:
        """Validar que valores sean únicos."""
        return len(value) == len(set(value))
    
    @staticmethod
    def valid_keys(value: dict, valid_keys: set) -> bool:
        """Validar que claves sean válidas."""
        return all(key in valid_keys for key in value.keys())
    
    @staticmethod
    def required_keys(value: dict, required_keys: set) -> bool:
        """Validar que claves requeridas estén presentes."""
        return all(key in value for key in required_keys)

# =============================================================================
# RELATIONSHIP VALIDATORS
# =============================================================================

class RelationshipValidators:
    """Validadores de relaciones."""
    
    @staticmethod
    def dimensions_match(array1: np.ndarray, array2: np.ndarray, axis: int = 0) -> bool:
        """Validar que dimensiones coincidan."""
        return array1.shape[axis] == array2.shape[axis]
    
    @staticmethod
    def shapes_compatible(array1: np.ndarray, array2: np.ndarray) -> bool:
        """Validar que formas sean compatibles para operaciones."""
        return array1.shape == array2.shape
    
    @staticmethod
    def batch_size_consistent(arrays: List[np.ndarray]) -> bool:
        """Validar que tamaños de batch sean consistentes."""
        if not arrays:
            return True
        first_batch_size = arrays[0].shape[0]
        return all(arr.shape[0] == first_batch_size for arr in arrays)
    
    @staticmethod
    def sequential_indices(indices: List[int]) -> bool:
        """Validar que índices sean secuenciales."""
        return indices == list(range(len(indices)))
    
    @staticmethod
    def no_overlap(ranges: List[Tuple[int, int]]) -> bool:
        """Validar que rangos no se superpongan."""
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        for i in range(len(sorted_ranges) - 1):
            if sorted_ranges[i][1] > sorted_ranges[i + 1][0]:
                return False
        return True

# =============================================================================
# EARLY VALIDATION DECORATOR
# =============================================================================

def early_validate(
    rules: Dict[str, ValidationRule],
    level: ValidationLevel = ValidationLevel.NORMAL,
    error_category: ErrorCategory = ErrorCategory.VALIDATION
):
    """Decorador para validación temprana."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Obtener nombres de parámetros
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validar cada regla
            for field_name, rule in rules.items():
                if field_name in bound_args.arguments:
                    value = bound_args.arguments[field_name]
                    
                    # Aplicar validación
                    result = _apply_validation_rule(field_name, value, rule)
                    
                    if not result.valid:
                        if rule.level == ValidationLevel.STRICT or level == ValidationLevel.STRICT:
                            raise ValidationError(
                                result.message,
                                category=error_category,
                                severity=ErrorSeverity.ERROR,
                                details=result.details
                            )
                        else:
                            logging.warning(f"Validación falló: {result.message}")
            
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Obtener nombres de parámetros
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validar cada regla
            for field_name, rule in rules.items():
                if field_name in bound_args.arguments:
                    value = bound_args.arguments[field_name]
                    
                    # Aplicar validación
                    result = _apply_validation_rule(field_name, value, rule)
                    
                    if not result.valid:
                        if rule.level == ValidationLevel.STRICT or level == ValidationLevel.STRICT:
                            raise ValidationError(
                                result.message,
                                category=error_category,
                                severity=ErrorSeverity.ERROR,
                                details=result.details
                            )
                        else:
                            logging.warning(f"Validación falló: {result.message}")
            
            return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator

def _apply_validation_rule(field_name: str, value: Any, rule: ValidationRule) -> ValidationResult:
    """Aplicar regla de validación."""
    try:
        # Verificar si valor es requerido
        if rule.required and value is None:
            return ValidationResult(
                valid=False,
                message=f"Campo requerido '{field_name}' es None",
                field_name=field_name,
                value=value,
                rule=rule
            )
        
        # Aplicar valor por defecto si es necesario
        if value is None and rule.default_value is not None:
            value = rule.default_value
        
        # Ejecutar validador
        if rule.validator(value):
            return ValidationResult(
                valid=True,
                message=f"Validación exitosa para '{field_name}'"f",
                field_name=field_name,
                value=value,
                rule=rule
            )
        else:
            return ValidationResult(
                valid=False,
                message=rule.error_message",
                field_name=field_name,
                value=value,
                rule=rule
            )
    
    except Exception as e:
        return ValidationResult(
            valid=False,
            message=f"Error en validación de '{field_name}': {e}",
            field_name=field_name,
            value=value,
            rule=rule,
            details={"exception": str(e)}
        )

# =============================================================================
# VALIDATION SCHEMAS
# =============================================================================

class ValidationSchema:
    """Esquema de validación para funciones."""
    
    def __init__(self, name: str):
        
    """__init__ function."""
self.name = name
        self.rules: Dict[str, ValidationRule] = {}
        self.level = ValidationLevel.NORMAL
    
    def add_rule(self, field_name: str, rule: ValidationRule):
        """Agregar regla de validación."""
        self.rules[field_name] = rule
        return self
    
    def set_level(self, level: ValidationLevel):
        """Establecer nivel de validación."""
        self.level = level
        return self
    
    def validate(self, **kwargs) -> List[ValidationResult]:
        """Validar argumentos."""
        results = []
        for field_name, rule in self.rules.items():
            if field_name in kwargs:
                result = _apply_validation_rule(field_name, kwargs[field_name], rule)
                results.append(result)
        return results
    
    def to_decorator(self, error_category: ErrorCategory = ErrorCategory.VALIDATION):
        """Convertir esquema a decorador."""
        return early_validate(self.rules, self.level, error_category)

# =============================================================================
# COMMON VALIDATION SCHEMAS
# =============================================================================

def create_video_validation_schema() -> ValidationSchema:
    """Crear esquema de validación para video."""
    schema = ValidationSchema("video_validation")
    
    schema.add_rule("video_path", ValidationRule(
        name="video_path_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=ExistenceValidators.file_exists,
        error_message="Archivo de video '{field}' no existe",
        required=True
    ))
    
    schema.add_rule("video_path", ValidationRule(
        name="video_format",
        validation_type=ValidationType.FORMAT,
        validator=lambda x: FormatValidators.is_video_format(Path(x).suffix),
        error_message="Formato de video '{field}' no soportado",
        required=True
    ))
    
    schema.add_rule("batch_size", ValidationRule(
        name="batch_size_range",
        validation_type=ValidationType.RANGE,
        validator=lambda x: RangeValidators.in_range(x, 1, 32),
        error_message="Tamaño de batch '{field}' debe estar entre 1 y 32",
        required=True
    ))
    
    return schema

def create_model_validation_schema() -> ValidationSchema:
    """Crear esquema de validación para modelo."""
    schema = ValidationSchema("model_validation")
    
    schema.add_rule("model_path", ValidationRule(
        name="model_path_exists",
        validation_type=ValidationType.EXISTENCE,
        validator=ExistenceValidators.file_exists,
        error_message="Archivo de modelo '{field}' no existe",
        required=True
    ))
    
    schema.add_rule("model_config", ValidationRule(
        name="model_config_dict",
        validation_type=ValidationType.TYPE,
        validator=TypeValidators.is_dict,
        error_message="Configuración de modelo '{field}' debe ser diccionario",
        required=True
    ))
    
    schema.add_rule("model_config", ValidationRule(
        name="model_config_required_keys",
        validation_type=ValidationType.CONTENT,
        validator=lambda x: ContentValidators.required_keys(x, {"model_type", "model_path"}),
        error_message="Configuración de modelo '{field}' debe contener 'model_type' y 'model_path'",
        required=True
    ))
    
    return schema

def create_data_validation_schema() -> ValidationSchema:
    """Crear esquema de validación para datos."""
    schema = ValidationSchema("data_validation")
    
    schema.add_rule("data", ValidationRule(
        name="data_not_none",
        validation_type=ValidationType.EXISTENCE,
        validator=ExistenceValidators.not_none,
        error_message="Datos '{field}' no pueden ser None",
        required=True
    ))
    
    schema.add_rule("data", ValidationRule(
        name="data_not_empty",
        validation_type=ValidationType.EXISTENCE,
        validator=ExistenceValidators.not_empty,
        error_message="Datos '{field}' no pueden estar vacíos",
        required=True
    ))
    
    schema.add_rule("data", ValidationRule(
        name="data_numpy_array",
        validation_type=ValidationType.TYPE,
        validator=TypeValidators.is_numpy_array,
        error_message="Datos '{field}' deben ser array de NumPy",
        required=True
    ))
    
    schema.add_rule("data", ValidationRule(
        name="data_no_nan",
        validation_type=ValidationType.CONTENT,
        validator=ContentValidators.no_nan_values,
        error_message="Datos '{field}' no pueden contener valores NaN",
        required=True
    ))
    
    return schema

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_all(*validators: Callable, **kwargs) -> bool:
    """Validar todos los validadores."""
    for validator in validators:
        if not validator(**kwargs):
            return False
    return True

def validate_any(*validators: Callable, **kwargs) -> bool:
    """Validar al menos un validador."""
    for validator in validators:
        if validator(**kwargs):
            return True
    return False

def create_custom_validator(validation_func: Callable, error_message: str) -> ValidationRule:
    """Crear validador personalizado."""
    return ValidationRule(
        name="custom_validator",
        validation_type=ValidationType.CUSTOM,
        validator=validation_func,
        error_message=error_message,
        required=True
    )

def validate_function_signature(func: Callable, **kwargs) -> List[ValidationResult]:
    """Validar firma de función."""
    sig = inspect.signature(func)
    results = []
    
    for param_name, param in sig.parameters.items():
        if param_name in kwargs:
            value = kwargs[param_name]
            
            # Validar tipo si hay anotación
            if param.annotation != inspect.Parameter.empty:
                if not isinstance(value, param.annotation):
                    results.append(ValidationResult(
                        valid=False,
                        message=f"Tipo incorrecto para '{param_name}': esperado {param.annotation}, obtenido {type(value)}",
                        field_name=param_name,
                        value=value,
                        rule=ValidationRule("type_check", ValidationType.TYPE, lambda x: True, "")
                    ))
    
    return results

# =============================================================================
# INITIALIZATION
# =============================================================================

def setup_early_validation():
    """Configurar sistema de validación temprana."""
    logger = logging.getLogger(__name__)
    logger.info("⚡ Sistema de validación temprana inicializado")
    
    # Crear esquemas comunes
    video_schema = create_video_validation_schema()
    model_schema = create_model_validation_schema()
    data_schema = create_data_validation_schema()
    
    return {
        "video_schema": video_schema,
        "model_schema": model_schema,
        "data_schema": data_schema
    }

# Configuración automática al importar
validation_schemas = setup_early_validation() 