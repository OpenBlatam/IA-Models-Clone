"""
Flash Attention - Atención Flash Optimizada
============================================

Implementación de Flash Attention para máxima velocidad.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class FlashAttentionOptimizer:
    """Optimizador para usar Flash Attention"""
    
    @staticmethod
    def enable_flash_attention(model: nn.Module) -> nn.Module:
        """
        Habilitar Flash Attention en modelo
        
        Args:
            model: Modelo a optimizar
            
        Returns:
            Modelo optimizado
        """
        try:
            # Intentar usar xformers (más compatible)
            if hasattr(model, "enable_xformers_memory_efficient_attention"):
                model.enable_xformers_memory_efficient_attention()
                logger.info("Flash Attention habilitado via xformers")
                return model
            
            # Intentar usar flash_attn directamente
            try:
                import flash_attn
                # Aplicar a capas de atención
                for module in model.modules():
                    if hasattr(module, "use_flash_attention"):
                        module.use_flash_attention = True
                logger.info("Flash Attention habilitado")
            except ImportError:
                logger.warning("flash_attn no disponible")
            
            return model
        except Exception as e:
            logger.warning(f"Error habilitando Flash Attention: {e}")
            return model
    
    @staticmethod
    def optimize_attention_layers(model: nn.Module) -> nn.Module:
        """
        Optimizar todas las capas de atención
        
        Args:
            model: Modelo a optimizar
            
        Returns:
            Modelo optimizado
        """
        for name, module in model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                # Aplicar optimizaciones
                if hasattr(module, "enable_xformers_memory_efficient_attention"):
                    module.enable_xformers_memory_efficient_attention()
        
        return model




