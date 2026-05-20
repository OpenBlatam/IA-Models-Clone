from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import logging
from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import numpy as np
        import psutil
from typing import Any, List, Dict, Optional
import asyncio
"""
🎯 HAPPY PATH VALIDATORS - CORE MODULE
======================================

Módulo que contiene todas las funciones de validación utilizadas en el patrón
happy path last.
"""


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def _validate_inputs(*args, **kwargs) -> bool:
    """Validar inputs básicos."""
    try:
        # Validar que los argumentos no sean None
        for arg in args:
            if arg is None:
                return False
        
        # Validar que los kwargs requeridos no sean None
        required_kwargs = ['video_path', 'batch_size', 'quality']
        for key in required_kwargs:
            if key in kwargs and kwargs[key] is None:
                return False
        
        return True
    except Exception as e:
        logging.error(f"Input validation error: {e}")
        return False

def _validate_resources(*args, **kwargs) -> bool:
    """Validar recursos del sistema."""
    try:
        
        # Verificar memoria disponible
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        if available_memory < 1.0:  # Mínimo 1GB
            return False
        
        # Verificar CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 90.0:  # Máximo 90% CPU
            return False
        
        # Verificar espacio en disco
        disk_usage = psutil.disk_usage('/')
        free_space_gb = disk_usage.free / (1024 * 1024 * 1024)
        if free_space_gb < 5.0:  # Mínimo 5GB
            return False
        
        return True
    except Exception as e:
        logging.error(f"Resource validation error: {e}")
        return False

def _validate_state(*args, **kwargs) -> bool:
    """Validar estado del sistema."""
    try:
        # Aquí se pueden agregar validaciones de estado
        # Por ejemplo: verificar si el modelo está cargado, si hay procesos activos, etc.
        return True
    except Exception as e:
        logging.error(f"State validation error: {e}")
        return False

def validate_video_path(video_path: str) -> bool:
    """Validar ruta de video."""
    if video_path is None:
        return False
    
    if not isinstance(video_path, str):
        return False
    
    if not Path(video_path).exists():
        return False
    
    # Verificar extensión de video
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    if Path(video_path).suffix.lower() not in valid_extensions:
        return False
    
    return True

def validate_batch_size(batch_size: int) -> bool:
    """Validar tamaño de batch."""
    if batch_size is None:
        return False
    
    if not isinstance(batch_size, int):
        return False
    
    if batch_size <= 0 or batch_size > 32:
        return False
    
    return True

def validate_quality(quality: float) -> bool:
    """Validar calidad."""
    if quality is None:
        return False
    
    if not isinstance(quality, (int, float)):
        return False
    
    if quality < 0.0 or quality > 1.0:
        return False
    
    return True

def validate_model_config(model_config: Dict[str, Any]) -> bool:
    """Validar configuración de modelo."""
    if model_config is None:
        return False
    
    if not isinstance(model_config, dict):
        return False
    
    # Verificar claves requeridas
    required_keys = {"model_type", "batch_size", "learning_rate"}
    missing_keys = required_keys - set(model_config.keys())
    if missing_keys:
        return False
    
    # Validar batch_size en config
    batch_size = model_config.get("batch_size")
    if not validate_batch_size(batch_size):
        return False
    
    # Validar learning_rate
    lr = model_config.get("learning_rate")
    if lr <= 0.0 or lr > 1.0:
        return False
    
    return True

def validate_data_array(data: np.ndarray) -> bool:
    """Validar array de datos."""
    if data is None:
        return False
    
    if not isinstance(data, np.ndarray):
        return False
    
    if data.size == 0:
        return False
    
    # Verificar valores NaN o infinitos
    if np.isnan(data).any() or np.isinf(data).any():
        return False
    
    return True

def validate_operation(operation: str) -> bool:
    """Validar operación."""
    if operation is None:
        return False
    
    if not isinstance(operation, str):
        return False
    
    valid_operations = {"normalize", "scale", "filter", "transform"}
    if operation not in valid_operations:
        return False
    
    return True

# =============================================================================
# COMPOSITE VALIDATORS
# =============================================================================

def validate_video_processing_params(video_path: str, batch_size: int, quality: float) -> bool:
    """Validar parámetros de procesamiento de video."""
    return (
        validate_video_path(video_path) and
        validate_batch_size(batch_size) and
        validate_quality(quality)
    )

def validate_model_loading_params(model_path: str, model_config: Dict[str, Any]) -> bool:
    """Validar parámetros de carga de modelo."""
    if model_path is None or not Path(model_path).exists():
        return False
    
    return validate_model_config(model_config)

def validate_data_processing_params(data: np.ndarray, operation: str) -> bool:
    """Validar parámetros de procesamiento de datos."""
    return (
        validate_data_array(data) and
        validate_operation(operation)
    )

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def create_validation_chain(*validators) -> callable:
    """Crear cadena de validadores."""
    def validation_chain(*args, **kwargs) -> bool:
        for validator in validators:
            if not validator(*args, **kwargs):
                return False
        return True
    
    return validation_chain

def validate_with_context(validator: callable, context: Dict[str, Any]) -> callable:
    """Aplicar validador con contexto adicional."""
    def contextual_validator(*args, **kwargs) -> Any:
        # Agregar contexto a kwargs
        kwargs.update(context)
        return validator(*args, **kwargs)
    
    return contextual_validator 