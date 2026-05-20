"""
Validators Module for Deep Learning Generator

Pure validation functions following functional programming principles.
"""

from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_config_dict(config: Dict[str, Any]) -> None:
    """
    Valida que config sea un diccionario (función pura).
    
    Args:
        config: Configuración a validar
        
    Raises:
        TypeError: Si config no es un diccionario
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")


def validate_framework(framework: str, supported_frameworks: list[str]) -> None:
    """
    Valida que el framework sea soportado (función pura).
    
    Args:
        framework: Framework a validar
        supported_frameworks: Lista de frameworks soportados
        
    Raises:
        ValueError: Si el framework no es soportado
    """
    if framework not in supported_frameworks:
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Supported frameworks: {', '.join(supported_frameworks)}"
        )


def validate_model_type(model_type: str, supported_model_types: list[str]) -> None:
    """
    Valida que el tipo de modelo sea soportado (función pura).
    
    Args:
        model_type: Tipo de modelo a validar
        supported_model_types: Lista de tipos de modelos soportados
        
    Raises:
        ValueError: Si el tipo de modelo no es soportado
    """
    if model_type not in supported_model_types:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {', '.join(supported_model_types)}"
        )


def validate_generator_config(
    config: Dict[str, Any],
    supported_frameworks: list[str],
    supported_model_types: list[str]
) -> Tuple[bool, Optional[str]]:
    """
    Valida una configuración de generador completa.
    
    Args:
        config: Configuración a validar
        supported_frameworks: Lista de frameworks soportados
        supported_model_types: Lista de tipos de modelos soportados
        
    Returns:
        Tupla (is_valid, error_message)
    """
    try:
        validate_config_dict(config)
    except TypeError as e:
        return False, str(e)
    
    if "framework" in config:
        try:
            validate_framework(config["framework"], supported_frameworks)
        except ValueError as e:
            return False, str(e)
    
    if "model_type" in config:
        try:
            validate_model_type(config["model_type"], supported_model_types)
        except ValueError as e:
            return False, str(e)
    
    return True, None















