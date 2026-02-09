"""
Model Compression Service - Compresión de modelos
==================================================

Sistema para comprimir modelos usando técnicas avanzadas.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import compression libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import torch_pruning
    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False
    logger.warning("torch_pruning not available")


@dataclass
class CompressionConfig:
    """Configuración de compresión"""
    method: str = "quantization"  # quantization, pruning, distillation
    quantization_type: str = "dynamic"  # dynamic, static, qat
    pruning_ratio: float = 0.3  # Percentage of parameters to prune
    pruning_method: str = "magnitude"  # magnitude, structured
    target_size_mb: Optional[float] = None


class ModelCompressionService:
    """Servicio de compresión de modelos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.compressed_models: Dict[str, Any] = {}
        logger.info("ModelCompressionService initialized")
    
    def quantize_model(
        self,
        model: nn.Module,
        quantization_type: str = "dynamic"
    ) -> nn.Module:
        """Cuantizar modelo"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        try:
            if quantization_type == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM, nn.Conv2d},
                    dtype=torch.qint8
                )
                return quantized_model
            elif quantization_type == "static":
                # Static quantization requires calibration
                logger.warning("Static quantization requires calibration dataset")
                return model
            elif quantization_type == "qat":
                # Quantization-Aware Training
                model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
                torch.quantization.prepare_qat(model, inplace=True)
                return model
            else:
                logger.warning(f"Unknown quantization type: {quantization_type}")
                return model
                
        except Exception as e:
            logger.error(f"Error quantizing model: {e}")
            return model
    
    def prune_model(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        method: str = "magnitude"
    ) -> nn.Module:
        """Podar modelo"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        try:
            if method == "magnitude":
                # Magnitude-based pruning
                parameters_to_prune = []
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        parameters_to_prune.append((module, 'weight'))
                
                if parameters_to_prune:
                    torch.nn.utils.prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=torch.nn.utils.prune.L1Unstructured,
                        amount=pruning_ratio,
                    )
                
                return model
            else:
                logger.warning(f"Unknown pruning method: {method}")
                return model
                
        except Exception as e:
            logger.error(f"Error pruning model: {e}")
            return model
    
    def compress_model(
        self,
        model_id: str,
        model: nn.Module,
        config: CompressionConfig
    ) -> Dict[str, Any]:
        """Comprimir modelo usando múltiples técnicas"""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        compressed_model = model
        
        # Apply compression methods
        if config.method == "quantization" or "quantization" in config.method:
            compressed_model = self.quantize_model(compressed_model, config.quantization_type)
        
        if config.method == "pruning" or "pruning" in config.method:
            compressed_model = self.prune_model(
                compressed_model,
                config.pruning_ratio,
                config.pruning_method
            )
        
        compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters())
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        self.compressed_models[model_id] = compressed_model
        
        return {
            "model_id": model_id,
            "original_size_mb": original_size / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "method": config.method,
        }
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Obtener tamaño del modelo"""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "size_bytes": size_bytes,
            "size_mb": size_bytes / (1024 * 1024),
            "size_gb": size_bytes / (1024 * 1024 * 1024),
        }




