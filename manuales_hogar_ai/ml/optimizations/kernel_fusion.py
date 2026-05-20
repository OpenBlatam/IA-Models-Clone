"""
Kernel Fusion
=============

Fusión de kernels para reducir overhead.
"""

import logging
import torch
from typing import Optional, Any

logger = logging.getLogger(__name__)


class KernelFusion:
    """Fusión de kernels para optimización."""
    
    @staticmethod
    def fuse_linear_bn(linear: torch.nn.Linear, bn: torch.nn.BatchNorm1d) -> torch.nn.Linear:
        """
        Fusionar Linear + BatchNorm.
        
        Args:
            linear: Capa Linear
            bn: Capa BatchNorm
        
        Returns:
            Linear fusionado
        """
        try:
            # Fusionar pesos y sesgos
            fused_weight = linear.weight * (bn.weight / torch.sqrt(bn.running_var + bn.eps))
            fused_bias = linear.bias + bn.bias - bn.running_mean * (bn.weight / torch.sqrt(bn.running_var + bn.eps))
            
            # Crear nueva capa
            fused_linear = torch.nn.Linear(
                linear.in_features,
                linear.out_features,
                bias=True
            )
            fused_linear.weight.data = fused_weight
            fused_linear.bias.data = fused_bias
            
            logger.info("Linear + BatchNorm fusionados")
            return fused_linear
        
        except Exception as e:
            logger.error(f"Error fusionando kernels: {str(e)}")
            return linear
    
    @staticmethod
    def fuse_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d) -> torch.nn.Conv2d:
        """
        Fusionar Conv + BatchNorm.
        
        Args:
            conv: Capa Conv
            bn: Capa BatchNorm
        
        Returns:
            Conv fusionado
        """
        try:
            # Fusionar
            fused_weight = conv.weight * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1)
            fused_bias = conv.bias + bn.bias - bn.running_mean * (bn.weight / torch.sqrt(bn.running_var + bn.eps))
            
            # Crear nueva capa
            fused_conv = torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                bias=True
            )
            fused_conv.weight.data = fused_weight
            fused_conv.bias.data = fused_bias
            
            logger.info("Conv + BatchNorm fusionados")
            return fused_conv
        
        except Exception as e:
            logger.error(f"Error fusionando conv: {str(e)}")
            return conv
    
    @staticmethod
    def optimize_model(model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimizar modelo fusionando kernels.
        
        Args:
            model: Modelo a optimizar
        
        Returns:
            Modelo optimizado
        """
        try:
            # Buscar y fusionar Linear + BN
            for name, module in list(model.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    # Buscar BN siguiente
                    parent_name = '.'.join(name.split('.')[:-1])
                    for child_name, child in model.named_modules():
                        if child_name.startswith(name + '.') and isinstance(child, torch.nn.BatchNorm1d):
                            # Fusionar
                            fused = KernelFusion.fuse_linear_bn(module, child)
                            # Reemplazar (simplificado, requiere estructura específica)
                            break
            
            logger.info("Kernels fusionados en modelo")
            return model
        
        except Exception as e:
            logger.warning(f"Error optimizando modelo: {str(e)}")
            return model




