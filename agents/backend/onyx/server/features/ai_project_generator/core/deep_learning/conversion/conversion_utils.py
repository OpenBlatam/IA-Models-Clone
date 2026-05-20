"""
Conversion Utilities
====================

Format conversion utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import json
import yaml

logger = logging.getLogger(__name__)


def convert_model_format(
    model: nn.Module,
    input_path: Optional[Path] = None,
    output_path: Path,
    input_format: str = 'pytorch',
    output_format: str = 'onnx',
    input_shape: Optional[tuple] = None
) -> Path:
    """
    Convert model between formats.
    
    Args:
        model: PyTorch model (if input_format is 'pytorch')
        input_path: Input file path (if loading from file)
        output_path: Output file path
        input_format: Input format ('pytorch', 'onnx', 'torchscript')
        output_format: Output format ('pytorch', 'onnx', 'torchscript')
        input_shape: Input tensor shape (for ONNX/TorchScript)
        
    Returns:
        Path to converted model
    """
    from ..deployment import export_to_onnx, export_to_torchscript
    
    # Load model if needed
    if input_path and input_format != 'pytorch':
        if input_format == 'torchscript':
            model = torch.jit.load(str(input_path))
        elif input_format == 'onnx':
            # ONNX loading requires onnxruntime
            logger.warning("ONNX to PyTorch conversion not directly supported")
            return output_path
    
    # Convert to output format
    if output_format == 'onnx':
        if input_shape is None:
            raise ValueError("input_shape required for ONNX export")
        export_to_onnx(model, output_path, input_shape)
        
    elif output_format == 'torchscript':
        if input_shape is None:
            raise ValueError("input_shape required for TorchScript export")
        export_to_torchscript(model, output_path, input_shape)
        
    elif output_format == 'pytorch':
        torch.save(model.state_dict(), output_path)
        
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    
    logger.info(f"Model converted from {input_format} to {output_format}: {output_path}")
    return output_path


def convert_data_format(
    data: Any,
    input_format: str,
    output_format: str,
    output_path: Optional[Path] = None
) -> Any:
    """
    Convert data between formats.
    
    Args:
        data: Input data
        input_format: Input format ('json', 'yaml', 'pickle', 'numpy')
        output_format: Output format ('json', 'yaml', 'pickle', 'numpy')
        output_path: Output file path (optional)
        
    Returns:
        Converted data
    """
    import numpy as np
    import pickle
    
    # Load from input format
    if input_format == 'json' and isinstance(data, (str, Path)):
        with open(data, 'r') as f:
            data = json.load(f)
    elif input_format == 'yaml' and isinstance(data, (str, Path)):
        with open(data, 'r') as f:
            data = yaml.safe_load(f)
    elif input_format == 'pickle' and isinstance(data, (str, Path)):
        with open(data, 'rb') as f:
            data = pickle.load(f)
    elif input_format == 'numpy' and isinstance(data, (str, Path)):
        data = np.load(data)
    
    # Convert to output format
    if output_format == 'json':
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        return data
        
    elif output_format == 'yaml':
        if output_path:
            with open(output_path, 'w') as f:
                yaml.dump(data, f)
        return data
        
    elif output_format == 'pickle':
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        return data
        
    elif output_format == 'numpy':
        if isinstance(data, np.ndarray):
            if output_path:
                np.save(output_path, data)
            return data
        else:
            data = np.array(data)
            if output_path:
                np.save(output_path, data)
            return data
        
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def convert_config_format(
    config: Dict[str, Any],
    output_path: Path,
    output_format: str = 'yaml'
) -> Path:
    """
    Convert configuration between formats.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
        output_format: Output format ('yaml', 'json')
        
    Returns:
        Path to converted config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    elif output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    
    logger.info(f"Config converted to {output_format}: {output_path}")
    return output_path


class FormatConverter:
    """
    Comprehensive format converter.
    """
    
    def __init__(self):
        """Initialize format converter."""
        pass
    
    def convert(
        self,
        data: Any,
        input_format: str,
        output_format: str,
        output_path: Optional[Path] = None
    ) -> Any:
        """
        Convert data between formats.
        
        Args:
            data: Input data
            input_format: Input format
            output_format: Output format
            output_path: Output path
            
        Returns:
            Converted data
        """
        if input_format == output_format:
            return data
        
        return convert_data_format(data, input_format, output_format, output_path)



