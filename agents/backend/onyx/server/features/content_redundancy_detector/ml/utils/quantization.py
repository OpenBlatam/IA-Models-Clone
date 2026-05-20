"""
Quantization Utilities
Model quantization for deployment
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class QuantizationManager:
    """
    Manages model quantization for deployment
    """
    
    @staticmethod
    def quantize_dynamic(
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """
        Apply dynamic quantization
        
        Args:
            model: Model to quantize
            dtype: Quantization dtype
            
        Returns:
            Quantized model
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=dtype
            )
            logger.info(f"Applied dynamic quantization with dtype={dtype}")
            return quantized_model
        except Exception as e:
            logger.error(f"Error in dynamic quantization: {e}")
            raise
    
    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """
        Apply static quantization with calibration
        
        Args:
            model: Model to quantize
            calibration_data: DataLoader for calibration
            dtype: Quantization dtype
            
        Returns:
            Quantized model
        """
        try:
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate
            logger.info("Calibrating model...")
            with torch.no_grad():
                for inputs, _ in calibration_data:
                    _ = model(inputs)
            
            # Convert
            quantized_model = torch.quantization.convert(model, inplace=False)
            logger.info(f"Applied static quantization with dtype={dtype}")
            
            return quantized_model
        except Exception as e:
            logger.error(f"Error in static quantization: {e}")
            raise
    
    @staticmethod
    def get_model_size_comparison(
        original_model: nn.Module,
        quantized_model: nn.Module,
    ) -> Dict[str, Any]:
        """
        Compare model sizes
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            
        Returns:
            Dictionary with size comparison
        """
        def get_size(model):
            total_size = 0
            for param in model.parameters():
                total_size += param.numel() * param.element_size()
            return total_size
        
        original_size = get_size(original_model)
        quantized_size = get_size(quantized_model)
        
        return {
            'original_size_mb': original_size / (1024 ** 2),
            'quantized_size_mb': quantized_size / (1024 ** 2),
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0,
        }



