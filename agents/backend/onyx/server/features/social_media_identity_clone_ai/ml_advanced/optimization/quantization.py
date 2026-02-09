"""
Model quantization para inferencia más rápida
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Cuantizador de modelos para optimización"""
    
    def __init__(self):
        pass
    
    def quantize_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Cuantización dinámica
        
        Args:
            model: Modelo a cuantizar
            dtype: Tipo de datos cuantizados
            
        Returns:
            Modelo cuantizado
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=dtype
            )
            logger.info("Modelo cuantizado dinámicamente")
            return quantized_model
        except Exception as e:
            logger.error(f"Error en cuantización dinámica: {e}")
            return model
    
    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: list,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Cuantización estática (requiere datos de calibración)
        
        Args:
            model: Modelo a cuantizar
            calibration_data: Datos para calibración
            dtype: Tipo de datos cuantizados
            
        Returns:
            Modelo cuantizado
        """
        try:
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrar
            with torch.no_grad():
                for data in calibration_data:
                    _ = model(data)
            
            # Convertir
            quantized_model = torch.quantization.convert(model, inplace=False)
            logger.info("Modelo cuantizado estáticamente")
            return quantized_model
        except Exception as e:
            logger.error(f"Error en cuantización estática: {e}")
            return model
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Obtiene tamaño del modelo"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimar tamaño en MB (asumiendo float32)
        size_mb = total_params * 4 / (1024 ** 2)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "size_mb": size_mb
        }




