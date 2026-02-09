"""
Model Quantization - Modular Quantization
=========================================

Quantization modular para optimización de modelos.
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Quantizer:
    """Clase base para quantizers."""
    
    def quantize(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantizar modelo."""
        raise NotImplementedError


class DynamicQuantizer(Quantizer):
    """Quantización dinámica."""
    
    def quantize(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Aplicar quantización dinámica.
        
        Args:
            model: Modelo a quantizar
            dtype: Tipo de datos quantizado
            
        Returns:
            Modelo quantizado
        """
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=dtype
            )
            logger.info("Model dynamically quantized")
            return quantized_model
        except Exception as e:
            logger.error(f"Error in dynamic quantization: {e}")
            raise


class StaticQuantizer(Quantizer):
    """Quantización estática."""
    
    def quantize(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Aplicar quantización estática.
        
        Args:
            model: Modelo a quantizar
            example_input: Ejemplo de entrada para calibration
            dtype: Tipo de datos quantizado
            
        Returns:
            Modelo quantizado
        """
        try:
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Preparar modelo
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrar con datos de ejemplo
            with torch.no_grad():
                _ = model(example_input)
            
            # Convertir
            quantized_model = torch.quantization.convert(model, inplace=False)
            
            logger.info("Model statically quantized")
            return quantized_model
        except Exception as e:
            logger.error(f"Error in static quantization: {e}")
            raise


class BitsAndBytesQuantizer(Quantizer):
    """Quantización usando bitsandbytes."""
    
    def quantize(
        self,
        model: nn.Module,
        quantization_type: str = '8bit',
        **kwargs
    ) -> nn.Module:
        """
        Aplicar quantización con bitsandbytes.
        
        Args:
            model: Modelo a quantizar
            quantization_type: Tipo ('8bit', '4bit')
            **kwargs: Argumentos adicionales
            
        Returns:
            Modelo quantizado
        """
        try:
            import bitsandbytes as bnb
            
            if quantization_type == '8bit':
                quantized_model = bnb.quantize(model, **kwargs)
            elif quantization_type == '4bit':
                quantized_model = bnb.quantize_4bit(model, **kwargs)
            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")
            
            logger.info(f"Model quantized with bitsandbytes ({quantization_type})")
            return quantized_model
        except ImportError:
            raise ImportError("bitsandbytes not available. Install with: pip install bitsandbytes")
        except Exception as e:
            logger.error(f"Error in bitsandbytes quantization: {e}")
            raise


class QuantizationFactory:
    """Factory para quantizers."""
    
    _quantizers = {
        'dynamic': DynamicQuantizer,
        'static': StaticQuantizer,
        'bitsandbytes': BitsAndBytesQuantizer,
        '8bit': BitsAndBytesQuantizer,
        '4bit': BitsAndBytesQuantizer
    }
    
    @classmethod
    def get_quantizer(cls, quantization_type: str) -> Quantizer:
        """
        Obtener quantizer por tipo.
        
        Args:
            quantization_type: Tipo de quantización
            
        Returns:
            Quantizer
        """
        if quantization_type not in cls._quantizers:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        return cls._quantizers[quantization_type]()
    
    @classmethod
    def register_quantizer(cls, quantization_type: str, quantizer_class: type):
        """Registrar nuevo quantizer."""
        cls._quantizers[quantization_type] = quantizer_class


def quantize_model(
    model: nn.Module,
    quantization_type: str = 'dynamic',
    **kwargs
) -> nn.Module:
    """
    Quantizar modelo.
    
    Args:
        model: Modelo a quantizar
        quantization_type: Tipo de quantización
        **kwargs: Argumentos adicionales
        
    Returns:
        Modelo quantizado
    """
    quantizer = QuantizationFactory.get_quantizer(quantization_type)
    
    if quantization_type in ['bitsandbytes', '8bit', '4bit']:
        qtype = '8bit' if quantization_type == '8bit' else '4bit'
        return quantizer.quantize(model, quantization_type=qtype, **kwargs)
    else:
        return quantizer.quantize(model, **kwargs)








