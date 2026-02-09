"""
Model Compression - Sistema de compresión de modelos
=====================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Métodos de compresión"""
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    LOW_RANK = "low_rank"


@dataclass
class CompressionConfig:
    """Configuración de compresión"""
    method: CompressionMethod
    target_sparsity: float = 0.5  # Para pruning
    quantization_bits: int = 8  # Para quantization
    prune_ratio: float = 0.3  # Ratio de pruning


class ModelCompressor:
    """Compresor de modelos"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    def prune_model(
        self,
        model: nn.Module,
        target_sparsity: Optional[float] = None
    ) -> nn.Module:
        """Aplica pruning a un modelo"""
        try:
            import torch.nn.utils.prune as prune
            
            sparsity = target_sparsity or self.config.target_sparsity
            
            # Pruning global
            parameters_to_prune = [
                (module, "weight")
                for module in model.modules()
                if isinstance(module, (nn.Linear, nn.Conv2d))
            ]
            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity
            )
            
            # Hacer pruning permanente
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            logger.info(f"Modelo pruned con sparsity {sparsity}")
            return model
        except Exception as e:
            logger.error(f"Error en pruning: {e}")
            return model
    
    def quantize_model(
        self,
        model: nn.Module,
        quantization_bits: Optional[int] = None
    ) -> nn.Module:
        """Cuantiza un modelo"""
        bits = quantization_bits or self.config.quantization_bits
        
        if bits == 8:
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Modelo cuantizado a 8-bit")
            return quantized_model
        elif bits == 16:
            # FP16
            model = model.half()
            logger.info("Modelo convertido a FP16")
            return model
        else:
            logger.warning(f"Cuantización a {bits}-bit no soportada")
            return model
    
    def compress_model(
        self,
        model: nn.Module,
        method: Optional[CompressionMethod] = None
    ) -> nn.Module:
        """Comprime un modelo"""
        method = method or self.config.method
        
        if method == CompressionMethod.PRUNING:
            return self.prune_model(model)
        elif method == CompressionMethod.QUANTIZATION:
            return self.quantize_model(model)
        else:
            logger.warning(f"Método {method} no implementado")
            return model
    
    def get_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """Obtiene el tamaño del modelo usando utilidades compartidas"""
        from .common_utils import calculate_model_size, count_parameters
        
        # Usar utilidades compartidas
        total_size_mb = calculate_model_size(model)
        param_info = count_parameters(model)
        
        # Calcular tamaño de buffers
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        buffer_size_mb = buffer_size / 1024**2
        
        return {
            "param_size_mb": total_size_mb - buffer_size_mb,
            "buffer_size_mb": buffer_size_mb,
            "total_size_mb": total_size_mb + buffer_size_mb,
            "num_parameters": param_info["total"]
        }

