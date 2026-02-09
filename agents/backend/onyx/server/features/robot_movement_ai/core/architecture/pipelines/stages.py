"""
Common Pipeline Stages (optimizado)

Etapas comunes y reutilizables para pipelines.
Incluye transformaciones, validaciones, filtros, y más.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar
from datetime import datetime

from .base import PipelineStage, PipelineContext

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class TransformStage(PipelineStage[T, R]):
    """
    Etapa de transformación (optimizado).
    
    Aplica una función de transformación a los datos.
    """
    
    def __init__(
        self,
        name: str,
        transform_func: Callable[[T], R],
        description: Optional[str] = None
    ):
        """
        Inicializar etapa de transformación (optimizado).
        
        Args:
            name: Nombre de la etapa
            transform_func: Función de transformación
            description: Descripción
        """
        super().__init__(name, description)
        self.transform_func = transform_func
    
    async def execute(self, input_data: T, context: PipelineContext) -> R:
        """
        Ejecutar transformación (optimizado).
        
        Args:
            input_data: Datos de entrada
            context: Contexto del pipeline
            
        Returns:
            Datos transformados
        """
        try:
            if asyncio.iscoroutinefunction(self.transform_func):
                return await self.transform_func(input_data)
            else:
                return self.transform_func(input_data)
        except Exception as e:
            logger.error(f"Transform stage '{self.name}' failed: {e}", exc_info=True)
            raise


class FilterStage(PipelineStage[T, T]):
    """
    Etapa de filtrado (optimizado).
    
    Filtra datos basado en una condición.
    """
    
    def __init__(
        self,
        name: str,
        filter_func: Callable[[T], bool],
        description: Optional[str] = None
    ):
        """
        Inicializar etapa de filtrado (optimizado).
        
        Args:
            name: Nombre de la etapa
            filter_func: Función de filtrado (retorna True para mantener)
            description: Descripción
        """
        super().__init__(name, description)
        self.filter_func = filter_func
    
    async def execute(self, input_data: T, context: PipelineContext) -> T:
        """
        Ejecutar filtrado (optimizado).
        
        Args:
            input_data: Datos de entrada
            context: Contexto del pipeline
            
        Returns:
            Datos filtrados o None si no pasan el filtro
        """
        try:
            if isinstance(input_data, (list, tuple)):
                # Filtrar lista
                if asyncio.iscoroutinefunction(self.filter_func):
                    results = []
                    for item in input_data:
                        if await self.filter_func(item):
                            results.append(item)
                    return type(input_data)(results)
                else:
                    return type(input_data)(filter(self.filter_func, input_data))
            else:
                # Filtrar item individual
                if asyncio.iscoroutinefunction(self.filter_func):
                    should_keep = await self.filter_func(input_data)
                else:
                    should_keep = self.filter_func(input_data)
                
                return input_data if should_keep else None
        except Exception as e:
            logger.error(f"Filter stage '{self.name}' failed: {e}", exc_info=True)
            raise


class ValidationStage(PipelineStage[T, T]):
    """
    Etapa de validación (optimizado).
    
    Valida datos y lanza excepción si no son válidos.
    """
    
    def __init__(
        self,
        name: str,
        validator: Callable[[T], bool],
        error_message: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Inicializar etapa de validación (optimizado).
        
        Args:
            name: Nombre de la etapa
            validator: Función validadora (retorna True si válido)
            error_message: Mensaje de error personalizado
            description: Descripción
        """
        super().__init__(name, description)
        self.validator = validator
        self.error_message = error_message or f"Validation failed in stage '{name}'"
    
    async def execute(self, input_data: T, context: PipelineContext) -> T:
        """
        Ejecutar validación (optimizado).
        
        Args:
            input_data: Datos de entrada
            context: Contexto del pipeline
            
        Returns:
            Datos si son válidos
            
        Raises:
            ValueError: Si la validación falla
        """
        try:
            if asyncio.iscoroutinefunction(self.validator):
                is_valid = await self.validator(input_data)
            else:
                is_valid = self.validator(input_data)
            
            if not is_valid:
                raise ValueError(self.error_message)
            
            return input_data
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Validation stage '{self.name}' failed: {e}", exc_info=True)
            raise ValueError(f"{self.error_message}: {str(e)}")


class LoggingStage(PipelineStage[T, T]):
    """
    Etapa de logging (optimizado).
    
    Registra información sobre los datos sin modificarlos.
    """
    
    def __init__(
        self,
        name: str,
        log_func: Optional[Callable[[T], str]] = None,
        log_level: int = logging.INFO,
        description: Optional[str] = None
    ):
        """
        Inicializar etapa de logging (optimizado).
        
        Args:
            name: Nombre de la etapa
            log_func: Función para generar mensaje de log
            log_level: Nivel de logging
            description: Descripción
        """
        super().__init__(name, description)
        self.log_func = log_func
        self.log_level = log_level
    
    async def execute(self, input_data: T, context: PipelineContext) -> T:
        """
        Ejecutar logging (optimizado).
        
        Args:
            input_data: Datos de entrada
            context: Contexto del pipeline
            
        Returns:
            Datos sin modificar
        """
        try:
            if self.log_func:
                if asyncio.iscoroutinefunction(self.log_func):
                    message = await self.log_func(input_data)
                else:
                    message = self.log_func(input_data)
            else:
                message = f"Stage '{self.name}' processed data: {type(input_data).__name__}"
            
            logger.log(self.log_level, message)
            return input_data
        except Exception as e:
            logger.warning(f"Logging stage '{self.name}' failed: {e}")
            return input_data  # No fallar el pipeline por logging


class ErrorHandlingStage(PipelineStage[T, T]):
    """
    Etapa de manejo de errores (optimizado).
    
    Captura errores y los maneja según una estrategia.
    """
    
    def __init__(
        self,
        name: str,
        error_handler: Optional[Callable[[Exception, T], Any]] = None,
        default_value: Optional[Any] = None,
        description: Optional[str] = None
    ):
        """
        Inicializar etapa de manejo de errores (optimizado).
        
        Args:
            name: Nombre de la etapa
            error_handler: Función para manejar errores
            default_value: Valor por defecto si hay error
            description: Descripción
        """
        super().__init__(name, description)
        self.error_handler = error_handler
        self.default_value = default_value
    
    async def execute(self, input_data: T, context: PipelineContext) -> T:
        """
        Ejecutar con manejo de errores (optimizado).
        
        Args:
            input_data: Datos de entrada
            context: Contexto del pipeline
            
        Returns:
            Datos procesados o valor por defecto si hay error
        """
        try:
            return input_data
        except Exception as e:
            logger.error(f"Error in stage '{self.name}': {e}", exc_info=True)
            
            if self.error_handler:
                if asyncio.iscoroutinefunction(self.error_handler):
                    return await self.error_handler(e, input_data)
                else:
                    return self.error_handler(e, input_data)
            
            if self.default_value is not None:
                return self.default_value
            
            raise


# Importar asyncio para verificar funciones async
import asyncio
