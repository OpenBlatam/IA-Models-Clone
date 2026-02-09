"""
Validators
==========

Validadores usando Pydantic schemas.
"""

import logging
from typing import Dict, Any, Optional

try:
    from pydantic import ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    ValidationError = Exception

from .schemas import (
    RouteRequestSchema,
    RouteResponseSchema,
    ModelConfigSchema,
    TrainingConfigSchema
)

logger = logging.getLogger(__name__)


class RouteRequestValidator:
    """Validador para requests de rutas."""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> RouteRequestSchema:
        """
        Validar request.
        
        Args:
            data: Datos a validar
            
        Returns:
            Schema validado
            
        Raises:
            ValidationError: Si la validación falla
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydantic no disponible, validación básica")
            return data
        
        try:
            return RouteRequestSchema(**data)
        except ValidationError as e:
            logger.error(f"Error de validación: {e}")
            raise


class RouteResponseValidator:
    """Validador para responses de rutas."""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> RouteResponseSchema:
        """
        Validar response.
        
        Args:
            data: Datos a validar
            
        Returns:
            Schema validado
            
        Raises:
            ValidationError: Si la validación falla
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydantic no disponible, validación básica")
            return data
        
        try:
            return RouteResponseSchema(**data)
        except ValidationError as e:
            logger.error(f"Error de validación: {e}")
            raise


class ModelConfigValidator:
    """Validador para configuración de modelos."""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> ModelConfigSchema:
        """
        Validar configuración de modelo.
        
        Args:
            data: Datos a validar
            
        Returns:
            Schema validado
            
        Raises:
            ValidationError: Si la validación falla
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydantic no disponible, validación básica")
            return data
        
        try:
            return ModelConfigSchema(**data)
        except ValidationError as e:
            logger.error(f"Error de validación: {e}")
            raise


class TrainingConfigValidator:
    """Validador para configuración de entrenamiento."""
    
    @staticmethod
    def validate(data: Dict[str, Any]) -> TrainingConfigSchema:
        """
        Validar configuración de entrenamiento.
        
        Args:
            data: Datos a validar
            
        Returns:
            Schema validado
            
        Raises:
            ValidationError: Si la validación falla
        """
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydantic no disponible, validación básica")
            return data
        
        try:
            return TrainingConfigSchema(**data)
        except ValidationError as e:
            logger.error(f"Error de validación: {e}")
            raise


# Funciones de conveniencia
def validate_route_request(data: Dict[str, Any]) -> RouteRequestSchema:
    """Validar request de ruta."""
    return RouteRequestValidator.validate(data)


def validate_route_response(data: Dict[str, Any]) -> RouteResponseSchema:
    """Validar response de ruta."""
    return RouteResponseValidator.validate(data)


def validate_model_config(data: Dict[str, Any]) -> ModelConfigSchema:
    """Validar configuración de modelo."""
    return ModelConfigValidator.validate(data)


def validate_training_config(data: Dict[str, Any]) -> TrainingConfigSchema:
    """Validar configuración de entrenamiento."""
    return TrainingConfigValidator.validate(data)

