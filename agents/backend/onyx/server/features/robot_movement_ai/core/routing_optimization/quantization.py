"""
Model Quantization
==================

Cuantización de modelos para inferencia más rápida.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QuantizedModel(nn.Module):
    """
    Wrapper para modelo cuantizado.
    """
    
    def __init__(self, model: nn.Module, quantized_model: nn.Module):
        """
        Inicializar modelo cuantizado.
        
        Args:
            model: Modelo original
            quantized_model: Modelo cuantizado
        """
        super(QuantizedModel, self).__init__()
        self.original_model = model
        self.quantized_model = quantized_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.quantized_model(x)
    
    def get_size_reduction(self) -> float:
        """
        Obtener reducción de tamaño.
        
        Returns:
            Ratio de reducción
        """
        original_size = sum(p.numel() * 4 for p in self.original_model.parameters())  # float32 = 4 bytes
        quantized_size = sum(p.numel() * 1 for p in self.quantized_model.parameters())  # int8 = 1 byte
        
        return 1.0 - (quantized_size / original_size)


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """
    Cuantización dinámica (post-training).
    
    Args:
        model: Modelo a cuantizar
        
    Returns:
        Modelo cuantizado
    """
    model.eval()
    
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        logger.info("Modelo cuantizado dinámicamente")
        return quantized_model
    except Exception as e:
        logger.warning(f"Error en cuantización dinámica: {e}")
        return model


def static_quantize(model: nn.Module,
                   calibration_data,
                   device: str = "cpu") -> nn.Module:
    """
    Cuantización estática (requiere datos de calibración).
    
    Args:
        model: Modelo a cuantizar
        calibration_data: Datos para calibración
        device: Dispositivo
        
    Returns:
        Modelo cuantizado
    """
    model.eval()
    model.to(device)
    
    try:
        # Preparar modelo
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrar
        with torch.no_grad():
            for data in calibration_data[:100]:  # Primeros 100 samples
                if isinstance(data, tuple):
                    inputs = data[0]
                else:
                    inputs = data
                model(inputs.to(device))
        
        # Convertir
        quantized_model = torch.quantization.convert(model, inplace=False)
        logger.info("Modelo cuantizado estáticamente")
        return quantized_model
    except Exception as e:
        logger.warning(f"Error en cuantización estática: {e}")
        return model


def quantize_model(model: nn.Module,
                  method: str = "dynamic",
                  calibration_data=None) -> QuantizedModel:
    """
    Cuantizar modelo.
    
    Args:
        model: Modelo a cuantizar
        method: Método ("dynamic" o "static")
        calibration_data: Datos para calibración (solo static)
        
    Returns:
        Modelo cuantizado
    """
    if method == "dynamic":
        quantized = dynamic_quantize(model)
    elif method == "static":
        if calibration_data is None:
            raise ValueError("calibration_data requerido para cuantización estática")
        quantized = static_quantize(model, calibration_data)
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    return QuantizedModel(model, quantized)


def int8_quantization(model: nn.Module) -> nn.Module:
    """
    Cuantización INT8 usando torch.ao.quantization.
    
    Args:
        model: Modelo
        
    Returns:
        Modelo cuantizado
    """
    try:
        from torch.ao.quantization import quantize_dynamic
        quantized = quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        logger.info("Cuantización INT8 aplicada")
        return quantized
    except Exception as e:
        logger.warning(f"Error en cuantización INT8: {e}")
        return model

