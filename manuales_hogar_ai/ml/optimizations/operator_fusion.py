"""
Operator Fusion
===============

Fusión de operadores a nivel bajo.
"""

import logging
import torch
import torch.nn as nn
from typing import List, Optional

logger = logging.getLogger(__name__)


class OperatorFusion:
    """Fusión de operadores."""
    
    @staticmethod
    def fuse_conv_bn_relu(
        conv: nn.Conv2d,
        bn: Optional[nn.BatchNorm2d],
        relu: Optional[nn.ReLU]
    ) -> nn.Module:
        """
        Fusionar Conv + BN + ReLU.
        
        Args:
            conv: Capa convolucional
            bn: BatchNorm (opcional)
            relu: ReLU (opcional)
        
        Returns:
            Módulo fusionado
        """
        try:
            # Si hay BN, fusionar conv + bn
            if bn is not None:
                # Fusionar conv y bn
                fused_conv = OperatorFusion._fuse_conv_bn(conv, bn)
            else:
                fused_conv = conv
            
            # Si hay ReLU, agregar
            if relu is not None:
                return nn.Sequential(fused_conv, relu)
            
            return fused_conv
        
        except Exception as e:
            logger.error(f"Error fusionando conv+bn+relu: {str(e)}")
            return conv
    
    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """Fusionar Conv y BN."""
        # Fusionar pesos y sesgos
        fused_weight = conv.weight * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1)
        fused_bias = conv.bias + bn.bias - bn.running_mean * (bn.weight / torch.sqrt(bn.running_var + bn.eps))
        
        fused_conv = nn.Conv2d(
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
        
        return fused_conv
    
    @staticmethod
    def fuse_linear_activation(
        linear: nn.Linear,
        activation: Optional[nn.Module]
    ) -> nn.Module:
        """
        Fusionar Linear + Activation.
        
        Args:
            linear: Capa linear
            activation: Función de activación
        
        Returns:
            Módulo fusionado
        """
        if activation is None:
            return linear
        
        return nn.Sequential(linear, activation)
    
    @staticmethod
    def optimize_model_fusion(model: nn.Module) -> nn.Module:
        """
        Optimizar modelo fusionando operadores.
        
        Args:
            model: Modelo
        
        Returns:
            Modelo optimizado
        """
        try:
            # Buscar y fusionar patrones comunes
            for name, module in list(model.named_children()):
                if isinstance(module, nn.Sequential):
                    # Intentar fusionar secuencias
                    fused = OperatorFusion._fuse_sequence(module)
                    if fused != module:
                        setattr(model, name, fused)
            
            logger.info("Fusión de operadores aplicada")
            return model
        
        except Exception as e:
            logger.warning(f"Error en fusión de operadores: {str(e)}")
            return model
    
    @staticmethod
    def _fuse_sequence(seq: nn.Sequential) -> nn.Module:
        """Fusionar secuencia de módulos."""
        # Buscar patrones fusionables
        modules = list(seq)
        
        if len(modules) >= 2:
            # Intentar fusionar primeros dos
            if isinstance(modules[0], nn.Conv2d) and isinstance(modules[1], nn.BatchNorm2d):
                if len(modules) >= 3 and isinstance(modules[2], nn.ReLU):
                    fused = OperatorFusion.fuse_conv_bn_relu(modules[0], modules[1], modules[2])
                    return nn.Sequential(fused, *modules[3:])
                else:
                    fused = OperatorFusion.fuse_conv_bn_relu(modules[0], modules[1], None)
                    return nn.Sequential(fused, *modules[2:])
        
        return seq




