"""
Export Utilities - Utilidades de exportación
============================================

Funciones para exportar modelos a diferentes formatos.
"""

import logging
from typing import Dict, Optional, Any, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 11
) -> bool:
    """
    Exportar modelo a ONNX.
    
    Args:
        model: Modelo a exportar
        input_shape: Forma de entrada (sin batch dimension)
        output_path: Ruta de salida
        input_names: Nombres de inputs (opcional)
        output_names: Nombres de outputs (opcional)
        dynamic_axes: Ejes dinámicos (opcional)
        opset_version: Versión de opset
    
    Returns:
        True si se exportó correctamente
    """
    try:
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names or ["input"],
            output_names=output_names or ["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to ONNX: {e}")
        return False


def export_to_torchscript(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    method: str = "trace"
) -> bool:
    """
    Exportar modelo a TorchScript.
    
    Args:
        model: Modelo a exportar
        input_shape: Forma de entrada (sin batch dimension)
        output_path: Ruta de salida
        method: Método ('trace' o 'script')
    
    Returns:
        True si se exportó correctamente
    """
    try:
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        if method == "trace":
            traced_model = torch.jit.trace(model, dummy_input)
        elif method == "script":
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        traced_model.save(output_path)
        logger.info(f"Model exported to TorchScript: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to TorchScript: {e}")
        return False


def export_model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str
) -> bool:
    """
    Exportar resumen del modelo a texto.
    
    Args:
        model: Modelo
        input_shape: Forma de entrada
        output_path: Ruta de salida
    
    Returns:
        True si se exportó correctamente
    """
    try:
        from core.utils.model_utils import count_parameters, get_model_size
        
        summary_lines = [
            "=" * 80,
            "MODEL SUMMARY",
            "=" * 80,
            "",
            f"Model: {type(model).__name__}",
            f"Input Shape: {input_shape}",
            "",
            "Parameters:",
            f"  Total: {count_parameters(model):,}",
            f"  Trainable: {count_parameters(model, trainable_only=True):,}",
            f"  Frozen: {count_parameters(model) - count_parameters(model, trainable_only=True):,}",
            "",
            f"Model Size: {get_model_size(model, unit='MB'):.2f} MB",
            "",
            "Architecture:",
            str(model),
            "",
            "=" * 80,
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Model summary exported to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting model summary: {e}")
        return False




