"""
Model Optimization Utilities

Provides utilities for model optimization:
- Quantization (INT8)
- Pruning
- ONNX export
- Model compression
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.quantization as quantization

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Quantize models to INT8 for faster inference
    """
    
    @staticmethod
    def quantize_dynamic(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Dynamic quantization (weights quantized, activations in FP32)
        
        Args:
            model: Model to quantize
            dtype: Quantization dtype (qint8, float16)
            
        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model dynamically to {dtype}")
        
        quantized_model = quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=dtype
        )
        
        logger.info("Model quantized successfully")
        return quantized_model
    
    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Static quantization (weights and activations quantized)
        
        Requires calibration data for better accuracy.
        
        Args:
            model: Model to quantize
            calibration_data: DataLoader for calibration
            dtype: Quantization dtype
            
        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model statically to {dtype}")
        
        # Set to eval mode
        model.eval()
        
        # Fuse modules for better quantization
        try:
            model_fused = quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
        except:
            model_fused = model
        
        # Set quantization config
        model_fused.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # Prepare
        model_prepared = quantization.prepare(model_fused)
        
        # Calibrate
        logger.info("Calibrating model...")
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, dict):
                    model_prepared(**batch)
                else:
                    model_prepared(batch)
        
        # Convert
        quantized_model = quantization.convert(model_prepared)
        
        logger.info("Model quantized and calibrated successfully")
        return quantized_model
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """
        Get model size in MB
        
        Args:
            model: Model to measure
            
        Returns:
            Dictionary with size information
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_size = (param_size + buffer_size) / (1024 * 1024)  # MB
        
        return {
            "total_size_mb": total_size,
            "param_size_mb": param_size / (1024 * 1024),
            "buffer_size_mb": buffer_size / (1024 * 1024),
            "num_parameters": sum(p.numel() for p in model.parameters())
        }


class ModelPruner:
    """
    Prune models to reduce size
    """
    
    @staticmethod
    def prune_magnitude(
        model: nn.Module,
        amount: float = 0.2,
        sparsity: float = 0.5
    ) -> nn.Module:
        """
        Magnitude-based pruning
        
        Args:
            model: Model to prune
            amount: Amount to prune (0.0 to 1.0)
            sparsity: Target sparsity
            
        Returns:
            Pruned model
        """
        from torch.nn.utils import prune
        
        logger.info(f"Pruning model with amount={amount}, sparsity={sparsity}")
        
        # Prune all linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        
        logger.info("Model pruned successfully")
        return model
    
    @staticmethod
    def get_sparsity(model: nn.Module) -> float:
        """
        Calculate model sparsity
        
        Args:
            model: Model to check
            
        Returns:
            Sparsity ratio (0.0 to 1.0)
        """
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0


class ONNXExporter:
    """
    Export models to ONNX format
    """
    
    @staticmethod
    def export(
        model: nn.Module,
        save_path: str,
        sample_input: Dict[str, torch.Tensor],
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        dynamic_axes: Optional[Dict[str, Any]] = None,
        opset_version: int = 14
    ) -> None:
        """
        Export model to ONNX
        
        Args:
            model: Model to export
            save_path: Path to save ONNX model
            sample_input: Sample input for tracing
            input_names: Names of input tensors
            output_names: Names of output tensors
            dynamic_axes: Dynamic axes for variable-length inputs
            opset_version: ONNX opset version
        """
        logger.info(f"Exporting model to ONNX: {save_path}")
        
        model.eval()
        
        try:
            torch.onnx.export(
                model,
                tuple(sample_input.values()) if isinstance(sample_input, dict) else sample_input,
                save_path,
                input_names=input_names or list(sample_input.keys()) if isinstance(sample_input, dict) else None,
                output_names=output_names or ["output"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True
            )
            logger.info(f"Model exported successfully to {save_path}")
        except Exception as e:
            logger.error(f"Error exporting model to ONNX: {e}", exc_info=True)
            raise


def compare_models(
    original_model: nn.Module,
    optimized_model: nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Compare original and optimized models
    
    Args:
        original_model: Original model
        optimized_model: Optimized model
        test_dataloader: Test data loader
        device: Device to run on
        
    Returns:
        Dictionary with comparison results
    """
    from .evaluation_utils import ModelEvaluator
    
    logger.info("Comparing models...")
    
    # Get sizes
    original_size = ModelQuantizer.get_model_size(original_model)
    optimized_size = ModelQuantizer.get_model_size(optimized_model)
    
    # Evaluate both
    original_evaluator = ModelEvaluator(original_model, device)
    optimized_evaluator = ModelEvaluator(optimized_model, device)
    
    original_results = original_evaluator.evaluate(test_dataloader)
    optimized_results = optimized_evaluator.evaluate(test_dataloader)
    
    # Compare
    comparison = {
        "size_reduction": {
            "original_mb": original_size["total_size_mb"],
            "optimized_mb": optimized_size["total_size_mb"],
            "reduction_percent": (1 - optimized_size["total_size_mb"] / original_size["total_size_mb"]) * 100
        },
        "accuracy": {
            "original": original_results.get("metrics", {}).get("accuracy", 0.0),
            "optimized": optimized_results.get("metrics", {}).get("accuracy", 0.0),
            "difference": optimized_results.get("metrics", {}).get("accuracy", 0.0) - 
                         original_results.get("metrics", {}).get("accuracy", 0.0)
        },
        "speed": {
            # Could add timing comparison here
        }
    }
    
    logger.info(f"Size reduction: {comparison['size_reduction']['reduction_percent']:.2f}%")
    logger.info(f"Accuracy difference: {comparison['accuracy']['difference']:.4f}")
    
    return comparison















