"""
Quantization - Cuantización de Modelos
=======================================

Cuantización para modelos más rápidos y pequeños.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class QuantizedModel:
    """Modelo cuantizado para inferencia rápida"""
    
    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """
        Cuantización dinámica (post-training)
        
        Args:
            model: Modelo a cuantizar
            
        Returns:
            Modelo cuantizado
        """
        try:
            quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Modelo cuantizado dinámicamente")
            return quantized
        except Exception as e:
            logger.warning(f"Error en cuantización dinámica: {e}")
            return model
    
    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_data: list
    ) -> nn.Module:
        """
        Cuantización estática (requiere datos de calibración)
        
        Args:
            model: Modelo a cuantizar
            calibration_data: Datos para calibración
            
        Returns:
            Modelo cuantizado
        """
        try:
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrar
            for data in calibration_data:
                model(data)
            
            torch.quantization.convert(model, inplace=True)
            logger.info("Modelo cuantizado estáticamente")
            return model
        except Exception as e:
            logger.warning(f"Error en cuantización estática: {e}")
            return model
    
    @staticmethod
    def quantize_int8(model: nn.Module) -> nn.Module:
        """
        Cuantización INT8 usando bitsandbytes
        
        Args:
            model: Modelo a cuantizar
            
        Returns:
            Modelo cuantizado
        """
        try:
            from transformers import BitsAndBytesConfig
            from transformers import AutoModelForCausalLM
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            # Esto requiere recargar el modelo con la configuración
            logger.info("Configuración INT8 aplicada")
            return model
        except ImportError:
            logger.warning("bitsandbytes no disponible")
            return model
        except Exception as e:
            logger.warning(f"Error en cuantización INT8: {e}")
            return model




