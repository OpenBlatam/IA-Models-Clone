"""
Validation
==========

Validación robusta de inputs y outputs.
"""

import logging
from typing import Any, Optional, List, Dict
import torch
import numpy as np

logger = logging.getLogger(__name__)


class Validator:
    """Validador robusto."""
    
    @staticmethod
    def validate_tensor(
        tensor: Any,
        shape: Optional[tuple] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> bool:
        """
        Validar tensor.
        
        Args:
            tensor: Tensor a validar
            shape: Forma esperada
            dtype: Tipo esperado
            device: Dispositivo esperado
            allow_nan: Permitir NaN
            allow_inf: Permitir Inf
        
        Returns:
            True si válido
        """
        try:
            if not isinstance(tensor, torch.Tensor):
                logger.error(f"Input no es tensor: {type(tensor)}")
                return False
            
            if shape and tensor.shape != shape:
                logger.error(f"Forma incorrecta: esperado {shape}, obtenido {tensor.shape}")
                return False
            
            if dtype and tensor.dtype != dtype:
                logger.error(f"Tipo incorrecto: esperado {dtype}, obtenido {tensor.dtype}")
                return False
            
            if device and tensor.device != device:
                logger.error(f"Dispositivo incorrecto: esperado {device}, obtenido {tensor.device}")
                return False
            
            if not allow_nan and torch.isnan(tensor).any():
                logger.error("Tensor contiene NaN")
                return False
            
            if not allow_inf and torch.isinf(tensor).any():
                logger.error("Tensor contiene Inf")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validando tensor: {str(e)}")
            return False
    
    @staticmethod
    def validate_text(text: str, max_length: int = 10000, min_length: int = 1) -> bool:
        """
        Validar texto.
        
        Args:
            text: Texto a validar
            max_length: Longitud máxima
            min_length: Longitud mínima
        
        Returns:
            True si válido
        """
        if not isinstance(text, str):
            logger.error(f"Input no es string: {type(text)}")
            return False
        
        if len(text) < min_length:
            logger.error(f"Texto muy corto: {len(text)} < {min_length}")
            return False
        
        if len(text) > max_length:
            logger.error(f"Texto muy largo: {len(text)} > {max_length}")
            return False
        
        return True
    
    @staticmethod
    def validate_batch(
        batch: List[Any],
        min_size: int = 1,
        max_size: int = 1000
    ) -> bool:
        """
        Validar batch.
        
        Args:
            batch: Batch a validar
            min_size: Tamaño mínimo
            max_size: Tamaño máximo
        
        Returns:
            True si válido
        """
        if not isinstance(batch, (list, tuple)):
            logger.error(f"Batch no es lista: {type(batch)}")
            return False
        
        if len(batch) < min_size:
            logger.error(f"Batch muy pequeño: {len(batch)} < {min_size}")
            return False
        
        if len(batch) > max_size:
            logger.error(f"Batch muy grande: {len(batch)} > {max_size}")
            return False
        
        return True




