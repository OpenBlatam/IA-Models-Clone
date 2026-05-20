"""
Model Quantization
==================

Cuantización de modelos para inferencia más rápida.
"""

import logging
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.quantization as quantization
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    quantization = None

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Cuantizador de modelos.
    
    Reduce precisión para inferencia más rápida.
    """
    
    def __init__(self, quantization_type: str = "int8"):
        """
        Inicializar cuantizador.
        
        Args:
            quantization_type: Tipo (int8, dynamic, static)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            self.quantization_type = None
            return
        
        self.quantization_type = quantization_type
    
    def quantize_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Cuantización dinámica.
        
        Args:
            model: Modelo
            dtype: Tipo de datos
            
        Returns:
            Modelo cuantizado
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            quantized = quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=dtype
            )
            logger.info("Model quantized dynamically")
            return quantized
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {e}")
            return model
    
    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: Any,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Cuantización estática.
        
        Args:
            model: Modelo
            calibration_data: Datos de calibración
            dtype: Tipo de datos
            
        Returns:
            Modelo cuantizado
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            # Preparar modelo
            model.eval()
            model.qconfig = quantization.get_default_qconfig('fbgemm')
            
            # Preparar para cuantización
            prepared = quantization.prepare(model)
            
            # Calibrar
            with torch.no_grad():
                for data in calibration_data:
                    if isinstance(data, (list, tuple)):
                        prepared(data[0])
                    else:
                        prepared(data)
            
            # Convertir
            quantized = quantization.convert(prepared)
            logger.info("Model quantized statically")
            return quantized
        except Exception as e:
            logger.warning(f"Static quantization failed: {e}")
            return model
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None
    ) -> nn.Module:
        """
        Cuantizar modelo.
        
        Args:
            model: Modelo
            calibration_data: Datos de calibración (para static)
            
        Returns:
            Modelo cuantizado
        """
        if self.quantization_type == "dynamic":
            return self.quantize_dynamic(model)
        elif self.quantization_type == "static" and calibration_data:
            return self.quantize_static(model, calibration_data)
        else:
            return self.quantize_dynamic(model)


class FP16Inference:
    """
    Inferencia con FP16.
    
    Usa half precision para inferencia más rápida.
    """
    
    @staticmethod
    def convert_to_fp16(model: nn.Module) -> nn.Module:
        """
        Convertir modelo a FP16.
        
        Args:
            model: Modelo
            
        Returns:
            Modelo en FP16
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            model = model.half()
            logger.info("Model converted to FP16")
            return model
        except Exception as e:
            logger.warning(f"FP16 conversion failed: {e}")
            return model
    
    @staticmethod
    def inference_fp16(
        model: nn.Module,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Inferencia con FP16.
        
        Args:
            model: Modelo
            inputs: Inputs
            
        Returns:
            Outputs
        """
        if not TORCH_AVAILABLE:
            return inputs
        
        model.eval()
        model = model.half()
        inputs = inputs.half()
        
        with torch.no_grad():
            outputs = model(inputs)
        
        return outputs.float()  # Convertir de vuelta a FP32

