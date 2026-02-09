"""
Model Optimization
Pruning, quantization, and compression
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ModelPruner:
    """
    Model pruning for model compression
    """
    
    @staticmethod
    def prune_weights(
        model: nn.Module,
        pruning_method: str = "l1_unstructured",
        amount: float = 0.2,
        modules: Optional[List[type]] = None
    ) -> nn.Module:
        """Prune model weights"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for pruning")
        
        if modules is None:
            modules = [nn.Linear, nn.Conv1d, nn.Conv2d]
        
        # Select pruning method
        if pruning_method == "l1_unstructured":
            prune_method = prune.L1Unstructured
        elif pruning_method == "l2_unstructured":
            prune_method = prune.L2Unstructured
        elif pruning_method == "random_unstructured":
            prune_method = prune.RandomUnstructured
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")
        
        # Prune modules
        for module in model.modules():
            if isinstance(module, tuple(modules)):
                prune_method.apply(module, name="weight", amount=amount)
        
        logger.info(f"Pruned model with {pruning_method}, amount={amount}")
        return model
    
    @staticmethod
    def prune_structured(
        model: nn.Module,
        amount: float = 0.2,
        dim: int = 0
    ) -> nn.Module:
        """Structured pruning"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for pruning")
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=amount,
                    n=2,
                    dim=dim
                )
        
        logger.info(f"Structured pruning applied, amount={amount}")
        return model
    
    @staticmethod
    def remove_pruning(model: nn.Module) -> nn.Module:
        """Remove pruning masks"""
        for module in model.modules():
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")
        
        return model


class ModelCompressor:
    """
    Model compression techniques
    """
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """Get model size in MB"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        
        return {
            "total_mb": size_all_mb,
            "parameters_mb": param_size / 1024**2,
            "buffers_mb": buffer_size / 1024**2,
            "num_parameters": sum(p.numel() for p in model.parameters())
        }
    
    @staticmethod
    def compress_model(
        model: nn.Module,
        method: str = "quantization",
        **kwargs
    ) -> nn.Module:
        """Compress model using various methods"""
        if method == "quantization":
            from .inference_optimizer import ModelQuantizer
            return ModelQuantizer.quantize_dynamic(model)
        elif method == "pruning":
            from .model_optimizer import ModelPruner
            amount = kwargs.get("amount", 0.2)
            return ModelPruner.prune_weights(model, amount=amount)
        else:
            raise ValueError(f"Unknown compression method: {method}")


class OptimizedModelManager:
    """
    Manage optimized models with caching and versioning
    """
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = cache_dir
        self.models: Dict[str, nn.Module] = {}
        self.optimization_stats: Dict[str, Dict[str, Any]] = {}
    
    def optimize_model(
        self,
        model: nn.Module,
        model_id: str,
        optimizations: List[str] = ["quantization"]
    ) -> nn.Module:
        """Apply optimizations to model"""
        original_size = ModelCompressor.get_model_size(model)
        
        optimized_model = model
        for opt in optimizations:
            if opt == "quantization":
                from .inference_optimizer import ModelQuantizer
                optimized_model = ModelQuantizer.quantize_dynamic(optimized_model)
            elif opt == "pruning":
                optimized_model = ModelPruner.prune_weights(optimized_model)
            elif opt == "torchscript":
                from .inference_optimizer import TorchScriptCompiler
                example_input = torch.randn(1, 169)
                optimized_model = TorchScriptCompiler.compile_trace(
                    optimized_model,
                    example_input
                )
        
        optimized_size = ModelCompressor.get_model_size(optimized_model)
        
        self.models[model_id] = optimized_model
        self.optimization_stats[model_id] = {
            "original_size_mb": original_size["total_mb"],
            "optimized_size_mb": optimized_size["total_mb"],
            "compression_ratio": original_size["total_mb"] / optimized_size["total_mb"],
            "optimizations": optimizations
        }
        
        logger.info(
            f"Model {model_id} optimized: "
            f"{original_size['total_mb']:.2f}MB -> {optimized_size['total_mb']:.2f}MB "
            f"({self.optimization_stats[model_id]['compression_ratio']:.2f}x)"
        )
        
        return optimized_model
    
    def get_model(self, model_id: str) -> Optional[nn.Module]:
        """Get optimized model"""
        return self.models.get(model_id)
    
    def get_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get optimization statistics"""
        return self.optimization_stats.get(model_id)

