"""
Model Utilities
Helper functions for model operations
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


def get_model_size(model: nn.Module, unit: str = "MB") -> float:
    """
    Get model size
    
    Args:
        model: PyTorch model
        unit: Size unit (B, KB, MB, GB)
        
    Returns:
        Model size in specified unit
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return total_size / units.get(unit.upper(), 1)


def freeze_model(model: nn.Module, freeze_all: bool = True):
    """
    Freeze model parameters
    
    Args:
        model: PyTorch model
        freeze_all: Freeze all parameters
    """
    for param in model.parameters():
        param.requires_grad = not freeze_all


def unfreeze_model(model: nn.Module):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specific layers
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False


def get_layer_output_shape(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None
) -> Dict[str, torch.Size]:
    """
    Get output shape of each layer
    
    Args:
        model: PyTorch model
        input_shape: Input shape
        device: Device to use
        
    Returns:
        Dictionary mapping layer names to output shapes
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    shapes = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                shapes[name] = output.shape
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    dummy_input = torch.randn(input_shape).to(device)
    with torch.inference_mode():
        _ = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return shapes


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Compare two models
    
    Args:
        model1: First model
        model2: Second model
        input_shape: Input shape
        device: Device to use
        
    Returns:
        Comparison dictionary
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameter counts
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    # Model sizes
    size1 = get_model_size(model1)
    size2 = get_model_size(model2)
    
    # Forward pass timing
    dummy_input = torch.randn(input_shape).to(device)
    model1 = model1.to(device).eval()
    model2 = model2.to(device).eval()
    
    import time
    
    # Model 1
    with torch.inference_mode():
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()
        _ = model1(dummy_input)
        torch.cuda.synchronize() if device.type == "cuda" else None
        time1 = (time.perf_counter() - start) * 1000
    
    # Model 2
    with torch.inference_mode():
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()
        _ = model2(dummy_input)
        torch.cuda.synchronize() if device.type == "cuda" else None
        time2 = (time.perf_counter() - start) * 1000
    
    return {
        "model1": {
            "parameters": params1,
            "size_mb": size1,
            "inference_time_ms": time1
        },
        "model2": {
            "parameters": params2,
            "size_mb": size2,
            "inference_time_ms": time2
        },
        "differences": {
            "parameter_diff": params1["total"] - params2["total"],
            "size_diff_mb": size1 - size2,
            "time_diff_ms": time1 - time2
        }
    }


def export_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    format: str = "onnx",
    device: Optional[torch.device] = None
):
    """
    Export model to different formats
    
    Args:
        model: PyTorch model
        input_shape: Input shape
        output_path: Output file path
        format: Export format (onnx, torchscript)
        device: Device to use
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    dummy_input = torch.randn(input_shape).to(device)
    
    if format.lower() == "onnx":
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=14
            )
            logger.info(f"Model exported to ONNX: {output_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    elif format.lower() == "torchscript":
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(output_path)
            logger.info(f"Model exported to TorchScript: {output_path}")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model from checkpoint
    
    Args:
        model: Model to load into
        checkpoint_path: Path to checkpoint
        device: Device to use
        
    Returns:
        Checkpoint dictionary
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
    return checkpoint
