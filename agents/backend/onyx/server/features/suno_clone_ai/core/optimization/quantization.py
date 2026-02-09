"""
Model Quantization

Utilities for model quantization.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class Quantizer:
    """Quantize models for inference."""
    
    @staticmethod
    def dynamic_quantization(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization.
        
        Args:
            model: Model to quantize
            dtype: Quantization dtype
            
        Returns:
            Quantized model
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},
                dtype=dtype
            )
            logger.info("Applied dynamic quantization")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
    @staticmethod
    def static_quantization(
        model: nn.Module,
        calibration_data: list,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply static quantization.
        
        Args:
            model: Model to quantize
            calibration_data: Calibration data
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
                    if isinstance(data, torch.Tensor):
                        _ = model(data)
                    else:
                        _ = model(*data)
            
            quantized_model = torch.quantization.convert(model, inplace=False)
            logger.info("Applied static quantization")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model


def quantize_model(
    model: nn.Module,
    method: str = "dynamic",
    **kwargs
) -> nn.Module:
    """
    Quantize model.
    
    Args:
        model: Model to quantize
        method: Quantization method ('dynamic', 'static')
        **kwargs: Additional arguments
        
    Returns:
        Quantized model
    """
    quantizer = Quantizer()
    
    if method == "dynamic":
        return quantizer.dynamic_quantization(model, **kwargs)
    elif method == "static":
        return quantizer.static_quantization(model, **kwargs)
    else:
        raise ValueError(f"Unknown quantization method: {method}")


def dynamic_quantization(
    model: nn.Module,
    **kwargs
) -> nn.Module:
    """Convenience function for dynamic quantization."""
    return Quantizer.dynamic_quantization(model, **kwargs)


def static_quantization(
    model: nn.Module,
    calibration_data: list,
    **kwargs
) -> nn.Module:
    """Convenience function for static quantization."""
    return Quantizer.static_quantization(model, calibration_data, **kwargs)



