"""
Model Utilities
===============

Advanced utilities for model management and optimization.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """
    Model analyzer for understanding model architecture and performance.
    
    Features:
    - Parameter counting
    - FLOPs estimation
    - Memory usage
    - Layer analysis
    """
    
    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            trainable_only: Count only trainable parameters
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return {
            "total": total_params,
            "trainable": trainable_params if trainable_only else trainable_params,
            "non_trainable": total_params - trainable_params
        }
    
    @staticmethod
    def estimate_flops(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: torch.device
    ) -> int:
        """
        Estimate FLOPs (Floating Point Operations).
        
        Args:
            model: PyTorch model
            input_shape: Input shape (batch_size, ...)
            device: Device
        
        Returns:
            Estimated FLOPs
        """
        # Simplified FLOPs estimation
        # In production, use tools like fvcore or thop
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(input_shape).to(device)
        
        flops = 0
        
        def count_flops_hook(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Linear):
                flops += input[0].shape[0] * module.weight.shape[0] * module.weight.shape[1]
            elif isinstance(module, nn.Conv2d):
                # Simplified conv2d FLOPs
                output_elements = output.numel()
                kernel_size = module.kernel_size[0] * module.kernel_size[1]
                flops += output_elements * kernel_size * module.in_channels
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(count_flops_hook)
                hooks.append(hook)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        for hook in hooks:
            hook.remove()
        
        return flops
    
    @staticmethod
    def analyze_layers(model: nn.Module) -> List[Dict[str, Any]]:
        """
        Analyze model layers.
        
        Args:
            model: PyTorch model
        
        Returns:
            List of layer information
        """
        layers = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters())
                }
                
                if isinstance(module, nn.Linear):
                    layer_info["in_features"] = module.in_features
                    layer_info["out_features"] = module.out_features
                elif isinstance(module, nn.Conv2d):
                    layer_info["in_channels"] = module.in_channels
                    layer_info["out_channels"] = module.out_channels
                    layer_info["kernel_size"] = module.kernel_size
                
                layers.append(layer_info)
        
        return layers


class ModelExporter:
    """Model exporter for different formats."""
    
    @staticmethod
    def export_onnx(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        device: torch.device,
        opset_version: int = 11
    ) -> None:
        """
        Export model to ONNX format.
        
        Args:
            model: PyTorch model
            input_shape: Input shape
            output_path: Output path
            device: Device
            opset_version: ONNX opset version
        """
        try:
            import torch.onnx
            
            model = model.to(device)
            model.eval()
            
            dummy_input = torch.randn(input_shape).to(device)
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
        except ImportError:
            logger.error("ONNX export requires torch.onnx")
            raise
    
    @staticmethod
    def export_torchscript(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        device: torch.device
    ) -> None:
        """
        Export model to TorchScript.
        
        Args:
            model: PyTorch model
            input_shape: Input shape
            output_path: Output path
            device: Device
        """
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(input_shape).to(device)
        
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
        
        logger.info(f"Model exported to TorchScript: {output_path}")


class ModelPruner:
    """Model pruning utilities."""
    
    @staticmethod
    def prune_magnitude(
        model: nn.Module,
        pruning_ratio: float = 0.2,
        layer_types: Optional[List[type]] = None
    ) -> nn.Module:
        """
        Prune model using magnitude-based pruning.
        
        Args:
            model: PyTorch model
            pruning_ratio: Ratio of weights to prune
            layer_types: Types of layers to prune (None = all)
        
        Returns:
            Pruned model
        """
        if layer_types is None:
            layer_types = [nn.Linear, nn.Conv2d]
        
        for name, module in model.named_modules():
            if any(isinstance(module, layer_type) for layer_type in layer_types):
                # Get weights
                if hasattr(module, 'weight') and module.weight is not None:
                    weights = module.weight.data
                    
                    # Calculate threshold
                    threshold = torch.quantile(
                        torch.abs(weights),
                        pruning_ratio
                    )
                    
                    # Create mask
                    mask = torch.abs(weights) > threshold
                    module.weight.data *= mask.float()
        
        return model
    
    @staticmethod
    def prune_structured(
        model: nn.Module,
        pruning_ratio: float = 0.2
    ) -> nn.Module:
        """
        Structured pruning (removes entire channels/filters).
        
        Args:
            model: PyTorch model
            pruning_ratio: Ratio to prune
        
        Returns:
            Pruned model
        """
        # This is a simplified version
        # In production, use torch.prune
        try:
            import torch.nn.utils.prune as prune
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                elif isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
        except ImportError:
            logger.warning("Structured pruning requires torch.prune")
        
        return model


class ModelQuantizer:
    """Model quantization utilities."""
    
    @staticmethod
    def quantize_dynamic(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Dynamic quantization.
        
        Args:
            model: PyTorch model
            dtype: Quantization dtype
        
        Returns:
            Quantized model
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=dtype
            )
            return quantized_model
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}")
            return model
    
    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Static quantization with calibration.
        
        Args:
            model: PyTorch model
            calibration_data: Calibration dataset
            dtype: Quantization dtype
        
        Returns:
            Quantized model
        """
        try:
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate
            with torch.no_grad():
                for data in calibration_data:
                    _ = model(data)
            
            torch.quantization.convert(model, inplace=True)
            return model
        except Exception as e:
            logger.error(f"Static quantization failed: {str(e)}")
            return model




