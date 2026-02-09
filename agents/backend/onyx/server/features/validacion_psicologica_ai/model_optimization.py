"""
Model Optimization
==================
Quantization, pruning, and other optimization techniques
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import quantize_dynamic, quantize_static
import structlog

logger = structlog.get_logger()


class ModelQuantizer:
    """
    Model quantization for inference optimization
    """
    
    def __init__(self):
        """Initialize quantizer"""
        logger.info("ModelQuantizer initialized")
    
    def quantize_dynamic(
        self,
        model: nn.Module,
        qconfig_spec: Optional[Dict[type, Any]] = None
    ) -> nn.Module:
        """
        Dynamic quantization
        
        Args:
            model: Model to quantize
            qconfig_spec: Quantization config spec
            
        Returns:
            Quantized model
        """
        try:
            if qconfig_spec is None:
                qconfig_spec = {
                    nn.Linear: torch.quantization.default_dynamic_qconfig,
                    nn.LSTM: torch.quantization.default_dynamic_qconfig,
                    nn.GRU: torch.quantization.default_dynamic_qconfig
                }
            
            quantized_model = quantize_dynamic(model, qconfig_spec, dtype=torch.qint8)
            logger.info("Model quantized dynamically")
            return quantized_model
        except Exception as e:
            logger.error("Error in dynamic quantization", error=str(e))
            raise
    
    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: List[Dict[str, torch.Tensor]],
        qconfig: Optional[Any] = None
    ) -> nn.Module:
        """
        Static quantization
        
        Args:
            model: Model to quantize
            calibration_data: Data for calibration
            qconfig: Quantization config
            
        Returns:
            Quantized model
        """
        try:
            model.eval()
            model.qconfig = qconfig or torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare model
            prepared_model = torch.quantization.prepare(model)
            
            # Calibrate
            with torch.no_grad():
                for sample in calibration_data:
                    prepared_model(**sample)
            
            # Convert
            quantized_model = torch.quantization.convert(prepared_model)
            logger.info("Model quantized statically")
            return quantized_model
        except Exception as e:
            logger.error("Error in static quantization", error=str(e))
            raise
    
    def get_model_size_comparison(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module
    ) -> Dict[str, Any]:
        """
        Compare model sizes
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            
        Returns:
            Size comparison
        """
        def get_size(model):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 ** 2)  # MB
        
        original_size = get_size(original_model)
        quantized_size = get_size(quantized_model)
        
        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else 0,
            "size_reduction_percent": ((original_size - quantized_size) / original_size * 100) if original_size > 0 else 0
        }


class ModelPruner:
    """
    Model pruning for reducing model size
    """
    
    def __init__(self):
        """Initialize pruner"""
        logger.info("ModelPruner initialized")
    
    def prune_structured(
        self,
        model: nn.Module,
        pruning_config: Dict[str, float]
    ) -> nn.Module:
        """
        Structured pruning
        
        Args:
            model: Model to prune
            pruning_config: Dict of layer_name -> pruning_amount (0-1)
            
        Returns:
            Pruned model
        """
        try:
            for name, module in model.named_modules():
                if name in pruning_config:
                    amount = pruning_config[name]
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        torch.nn.utils.prune.ln_structured(
                            module,
                            name="weight",
                            amount=amount,
                            n=2,
                            dim=0
                        )
            
            logger.info("Model pruned (structured)")
            return model
        except Exception as e:
            logger.error("Error in structured pruning", error=str(e))
            raise
    
    def prune_unstructured(
        self,
        model: nn.Module,
        pruning_config: Dict[str, float]
    ) -> nn.Module:
        """
        Unstructured pruning
        
        Args:
            model: Model to prune
            pruning_config: Dict of layer_name -> pruning_amount (0-1)
            
        Returns:
            Pruned model
        """
        try:
            for name, module in model.named_modules():
                if name in pruning_config:
                    amount = pruning_config[name]
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        torch.nn.utils.prune.random_unstructured(
                            module,
                            name="weight",
                            amount=amount
                        )
            
            logger.info("Model pruned (unstructured)")
            return model
        except Exception as e:
            logger.error("Error in unstructured pruning", error=str(e))
            raise
    
    def get_pruning_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get pruning statistics
        
        Args:
            model: Model
            
        Returns:
            Pruning statistics
        """
        total_params = 0
        pruned_params = 0
        
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                total_params += module.weight.numel()
                if hasattr(module.weight, 'mask'):
                    pruned_params += (module.weight.mask == 0).sum().item()
        
        return {
            "total_parameters": total_params,
            "pruned_parameters": pruned_params,
            "remaining_parameters": total_params - pruned_params,
            "pruning_ratio": pruned_params / total_params if total_params > 0 else 0
        }


class ModelOptimizer:
    """
    Comprehensive model optimizer
    """
    
    def __init__(self):
        """Initialize optimizer"""
        self.quantizer = ModelQuantizer()
        self.pruner = ModelPruner()
        logger.info("ModelOptimizer initialized")
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        optimization_config: Dict[str, Any]
    ) -> nn.Module:
        """
        Optimize model for inference
        
        Args:
            model: Model to optimize
            optimization_config: Optimization configuration
            
        Returns:
            Optimized model
        """
        optimized_model = model
        
        # Pruning
        if optimization_config.get("pruning", {}).get("enabled", False):
            pruning_type = optimization_config["pruning"].get("type", "unstructured")
            pruning_config = optimization_config["pruning"].get("config", {})
            
            if pruning_type == "structured":
                optimized_model = self.pruner.prune_structured(optimized_model, pruning_config)
            else:
                optimized_model = self.pruner.prune_unstructured(optimized_model, pruning_config)
        
        # Quantization
        if optimization_config.get("quantization", {}).get("enabled", False):
            quant_type = optimization_config["quantization"].get("type", "dynamic")
            
            if quant_type == "dynamic":
                optimized_model = self.quantizer.quantize_dynamic(optimized_model)
            elif quant_type == "static":
                calibration_data = optimization_config["quantization"].get("calibration_data", [])
                optimized_model = self.quantizer.quantize_static(optimized_model, calibration_data)
        
        # JIT compilation
        if optimization_config.get("jit", {}).get("enabled", False):
            try:
                optimized_model = torch.jit.script(optimized_model)
                logger.info("Model compiled with JIT")
            except Exception as e:
                logger.warning("JIT compilation failed", error=str(e))
        
        return optimized_model


# Global instances
model_quantizer = ModelQuantizer()
model_pruner = ModelPruner()
model_optimizer = ModelOptimizer()




