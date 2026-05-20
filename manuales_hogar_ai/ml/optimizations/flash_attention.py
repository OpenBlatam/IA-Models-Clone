"""
Flash Attention 2
==================

Optimización de atención con Flash Attention 2.
"""

import logging
import torch
from typing import Optional

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logging.warning("Flash Attention 2 no disponible. Instalar con: pip install flash-attn")

logger = logging.getLogger(__name__)


class FlashAttentionOptimizer:
    """Optimizador de atención con Flash Attention 2."""
    
    @staticmethod
    def enable_flash_attention(model: torch.nn.Module):
        """
        Habilitar Flash Attention 2 en modelo.
        
        Args:
            model: Modelo a optimizar
        """
        if not FLASH_ATTENTION_AVAILABLE:
            logger.warning("Flash Attention 2 no disponible")
            return model
        
        try:
            # Reemplazar atención en módulos
            for name, module in model.named_modules():
                if hasattr(module, 'config'):
                    module.config.use_flash_attention_2 = True
            
            logger.info("Flash Attention 2 habilitado")
            return model
        
        except Exception as e:
            logger.warning(f"No se pudo habilitar Flash Attention 2: {str(e)}")
            return model
    
    @staticmethod
    def apply_flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Aplicar Flash Attention directamente.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            dropout_p: Dropout probability
            softmax_scale: Escala de softmax
            causal: Causal attention
        
        Returns:
            Output tensor
        """
        if not FLASH_ATTENTION_AVAILABLE:
            raise ImportError("Flash Attention 2 no está disponible")
        
        try:
            return flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal
            )
        
        except Exception as e:
            logger.error(f"Error aplicando Flash Attention: {str(e)}")
            raise




