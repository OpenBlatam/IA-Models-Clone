"""
Template Service - Servicio de plantillas (optimizado)
=======================================================

Servicio especializado para gestionar y proporcionar plantillas de código.
Encapsula toda la lógica de plantillas en un servicio independiente.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.code_templates import (
    get_template,
    get_transformer_model_template,
    get_training_loop_template,
    get_dataloader_template,
    get_gradio_interface_template,
    get_config_yaml_template,
    TemplateType
)

logger = logging.getLogger(__name__)


class TemplateService:
    """
    Servicio de plantillas (optimizado).
    
    Gestiona y proporciona plantillas de código siguiendo mejores prácticas.
    Proporciona una interfaz limpia para acceder a plantillas.
    """
    
    def __init__(self):
        """Inicializar servicio de plantillas"""
        self.logger = logging.getLogger(f"{__name__}.TemplateService")
        self._template_cache: Dict[str, str] = {}
    
    def get_template(
        self,
        template_type: TemplateType,
        cache: bool = True,
        **kwargs
    ) -> str:
        """
        Obtener plantilla (optimizado).
        
        Args:
            template_type: Tipo de plantilla
            cache: Si cachear la plantilla
            **kwargs: Argumentos adicionales para la plantilla
            
        Returns:
            Código de la plantilla
        """
        # Crear clave de cache
        cache_key = f"{template_type.value}_{hash(frozenset(kwargs.items()))}"
        
        # Verificar cache
        if cache and cache_key in self._template_cache:
            self.logger.debug(f"Using cached template: {template_type.value}")
            return self._template_cache[cache_key]
        
        try:
            template = get_template(template_type, **kwargs)
            
            # Cachear si se solicita
            if cache:
                self._template_cache[cache_key] = template
            
            return template
        except Exception as e:
            self.logger.error(f"Error getting template {template_type.value}: {e}", exc_info=True)
            raise
    
    def get_transformer_model_template(
        self,
        framework: str = "pytorch",
        cache: bool = True
    ) -> str:
        """
        Obtener plantilla de modelo Transformer (optimizado).
        
        Args:
            framework: Framework a usar
            cache: Si cachear
            
        Returns:
            Código de plantilla
        """
        return self.get_template(
            TemplateType.TRANSFORMER_MODEL,
            framework=framework,
            cache=cache
        )
    
    def get_training_loop_template(
        self,
        mixed_precision: bool = True,
        gradient_accumulation: bool = True,
        multi_gpu: bool = False,
        cache: bool = True
    ) -> str:
        """
        Obtener plantilla de training loop (optimizado).
        
        Args:
            mixed_precision: Si usar mixed precision
            gradient_accumulation: Si usar gradient accumulation
            multi_gpu: Si usar multi-GPU
            cache: Si cachear
            
        Returns:
            Código de plantilla
        """
        return self.get_template(
            TemplateType.TRAINING_LOOP,
            mixed_precision=mixed_precision,
            gradient_accumulation=gradient_accumulation,
            multi_gpu=multi_gpu,
            cache=cache
        )
    
    def get_dataloader_template(
        self,
        pin_memory: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        cache: bool = True
    ) -> str:
        """
        Obtener plantilla de DataLoader (optimizado).
        
        Args:
            pin_memory: Si usar pin_memory
            num_workers: Número de workers
            prefetch_factor: Factor de prefetch
            cache: Si cachear
            
        Returns:
            Código de plantilla
        """
        return self.get_template(
            TemplateType.DATALOADER,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            cache=cache
        )
    
    def get_gradio_interface_template(self, cache: bool = True) -> str:
        """
        Obtener plantilla de interfaz Gradio (optimizado).
        
        Args:
            cache: Si cachear
            
        Returns:
            Código de plantilla
        """
        return self.get_template(TemplateType.GRADIO_INTERFACE, cache=cache)
    
    def get_config_yaml_template(
        self,
        features: Dict[str, Any],
        cache: bool = True
    ) -> str:
        """
        Obtener plantilla de configuración YAML (optimizado).
        
        Args:
            features: Características detectadas
            cache: Si cachear
            
        Returns:
            Código de plantilla YAML
        """
        return self.get_template(
            TemplateType.CONFIG_YAML,
            features=features,
            cache=cache
        )
    
    def clear_cache(self) -> None:
        """Limpiar cache de plantillas (optimizado)"""
        self._template_cache.clear()
        self.logger.debug("Template cache cleared")
    
    def get_cache_size(self) -> int:
        """
        Obtener tamaño del cache (optimizado).
        
        Returns:
            Número de plantillas en cache
        """
        return len(self._template_cache)

