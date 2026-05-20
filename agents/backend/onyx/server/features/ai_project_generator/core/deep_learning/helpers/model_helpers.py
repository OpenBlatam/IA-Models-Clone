"""
Model Helpers - Utility Functions for Models
============================================

Helper functions for model operations:
- Parameter counting
- Model summary
- Layer freezing/unfreezing
- Model conversion
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


def count_parameters(
    model: nn.Module,
    trainable_only: bool = True,
    by_layer: bool = False
) -> Union[int, Dict[str, int]]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters
        by_layer: Return counts by layer
        
    Returns:
        Total count or dictionary by layer
    """
    if by_layer:
        counts = {}
        for name, param in model.named_parameters():
            if not trainable_only or param.requires_grad:
                counts[name] = param.numel()
        return counts
    else:
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module, input_size: tuple) -> str:
    """
    Get model summary (similar to torchsummary).
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, ...)
        
    Returns:
        Model summary string
    """
    try:
        from torchsummary import summary
        return str(summary(model, input_size))
    except ImportError:
        # Fallback summary
        total_params = count_parameters(model, trainable_only=False)
        trainable_params = count_parameters(model, trainable_only=True)
        
        summary = f"""
Model: {model.__class__.__name__}
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}
"""
        return summary


def freeze_layers(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    freeze_all: bool = False
) -> nn.Module:
    """
    Freeze model layers.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze (freezes all if None and freeze_all=True)
        freeze_all: Freeze all layers
        
    Returns:
        Model with frozen layers
    """
    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("All layers frozen")
    elif layer_names:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
        logger.info(f"Frozen layers: {layer_names}")
    
    return model


def unfreeze_layers(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    unfreeze_all: bool = False
) -> nn.Module:
    """
    Unfreeze model layers.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
        unfreeze_all: Unfreeze all layers
        
    Returns:
        Model with unfrozen layers
    """
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen")
    elif layer_names:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
        logger.info(f"Unfrozen layers: {layer_names}")
    
    return model


def save_model_onnx(
    model: nn.Module,
    save_path: Path,
    input_size: tuple,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path: Path to save ONNX model
        input_size: Input tensor size
        input_names: Input names
        output_names: Output names
        dynamic_axes: Dynamic axes configuration
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    # Default names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11
        )
        logger.info(f"Model exported to ONNX: {save_path}")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise


def load_model_onnx(onnx_path: Path) -> Any:
    """
    Load ONNX model (requires onnxruntime).
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        ONNX runtime inference session
    """
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))
        logger.info(f"ONNX model loaded: {onnx_path}")
        return session
    except ImportError:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
    except Exception as e:
        logger.error(f"ONNX load failed: {e}")
        raise



