"""
Modular Model Optimization
Quantization, pruning, and compression utilities
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.quantization as quant
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ModelQuantizer:
    """Model quantization utility"""
    
    def __init__(self, quantization_type: str = "int8"):
        self.quantization_type = quantization_type
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None
    ) -> nn.Module:
        """
        Quantize model
        
        Args:
            model: Model to quantize
            calibration_data: Optional calibration data
        
        Returns:
            Quantized model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if self.quantization_type == "int8":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Model quantized to INT8")
            return quantized_model
        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")


class ModelPruner:
    """Model pruning utility"""
    
    def __init__(self, pruning_type: str = "l1_unstructured"):
        self.pruning_type = pruning_type
    
    def prune(
        self,
        model: nn.Module,
        amount: float = 0.2,
        module_type: type = nn.Linear
    ) -> nn.Module:
        """
        Prune model
        
        Args:
            model: Model to prune
            amount: Fraction of parameters to prune
            module_type: Type of modules to prune
        
        Returns:
            Pruned model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if self.pruning_type == "l1_unstructured":
            for module in model.modules():
                if isinstance(module, module_type):
                    torch.nn.utils.prune.l1_unstructured(
                        module,
                        name='weight',
                        amount=amount
                    )
            logger.info(f"Pruned {amount*100}% of {module_type.__name__} parameters")
        else:
            raise ValueError(f"Unknown pruning type: {self.pruning_type}")
        
        return model


class ModelCompressor:
    """Model compression utility combining quantization and pruning"""
    
    def __init__(
        self,
        quantize: bool = True,
        prune: bool = True,
        quantization_type: str = "int8",
        pruning_amount: float = 0.2
    ):
        self.quantize = quantize
        self.prune = prune
        self.quantizer = ModelQuantizer(quantization_type) if quantize else None
        self.pruner = ModelPruner() if prune else None
        self.pruning_amount = pruning_amount
    
    def compress(self, model: nn.Module) -> nn.Module:
        """
        Compress model using quantization and/or pruning
        
        Args:
            model: Model to compress
        
        Returns:
            Compressed model
        """
        compressed_model = model
        
        if self.prune and self.pruner:
            compressed_model = self.pruner.prune(
                compressed_model,
                amount=self.pruning_amount
            )
        
        if self.quantize and self.quantizer:
            compressed_model = self.quantizer.quantize(compressed_model)
        
        return compressed_model



