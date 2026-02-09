"""
Advanced Quantization - Cuantización avanzada de modelos
==========================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Tipos de cuantización"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    QAT = "qat"  # Quantization Aware Training
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class QuantizationConfig:
    """Configuración de cuantización"""
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    dtype: torch.dtype = torch.qint8
    per_channel: bool = False
    symmetric: bool = True


class AdvancedQuantizer:
    """Cuantizador avanzado"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.quantization_history: List[Dict[str, Any]] = []
    
    def quantize_model(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Cuantiza un modelo"""
        model.eval()
        
        if self.config.quantization_type == QuantizationType.DYNAMIC:
            return self._dynamic_quantize(model)
        elif self.config.quantization_type == QuantizationType.STATIC:
            if example_input is None:
                raise ValueError("Se requiere example_input para cuantización estática")
            return self._static_quantize(model, example_input)
        elif self.config.quantization_type == QuantizationType.QAT:
            return self._prepare_qat(model)
        else:
            return self._dynamic_quantize(model)
    
    def _dynamic_quantize(self, model: nn.Module) -> nn.Module:
        """Cuantización dinámica"""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU},
                dtype=self.config.dtype
            )
            logger.info("Modelo cuantizado dinámicamente")
            return quantized_model
        except Exception as e:
            logger.warning(f"Error en cuantización dinámica: {e}")
            return model
    
    def _static_quantize(
        self,
        model: nn.Module,
        example_input: torch.Tensor
    ) -> nn.Module:
        """Cuantización estática"""
        try:
            # Preparar modelo
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            prepared_model = torch.quantization.prepare(model, inplace=False)
            
            # Calibrar con datos de ejemplo
            with torch.no_grad():
                _ = prepared_model(example_input)
            
            # Convertir a cuantizado
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            logger.info("Modelo cuantizado estáticamente")
            return quantized_model
        except Exception as e:
            logger.warning(f"Error en cuantización estática: {e}")
            return model
    
    def _prepare_qat(self, model: nn.Module) -> nn.Module:
        """Prepara modelo para QAT"""
        try:
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            qat_model = torch.quantization.prepare_qat(model, inplace=False)
            logger.info("Modelo preparado para QAT")
            return qat_model
        except Exception as e:
            logger.warning(f"Error preparando QAT: {e}")
            return model
    
    def get_quantization_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Obtiene estadísticas de cuantización"""
        quantized_layers = 0
        total_layers = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                total_layers += 1
                if hasattr(module, 'weight') and isinstance(module.weight, torch.quantized.QTensor):
                    quantized_layers += 1
        
        return {
            "total_layers": total_layers,
            "quantized_layers": quantized_layers,
            "quantization_ratio": quantized_layers / total_layers if total_layers > 0 else 0
        }




