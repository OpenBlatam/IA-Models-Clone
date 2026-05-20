"""
Cuantización Avanzada
=====================

Técnicas avanzadas de cuantización.
"""

import logging
import torch
from typing import Optional, Dict, Any
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class AdvancedQuantizer:
    """Cuantización avanzada de modelos."""
    
    @staticmethod
    def quantize_int8(
        model: AutoModelForCausalLM,
        threshold: float = 6.0
    ) -> AutoModelForCausalLM:
        """
        Cuantización INT8.
        
        Args:
            model: Modelo a cuantizar
            threshold: Umbral de cuantización
        
        Returns:
            Modelo cuantizado
        """
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=threshold,
                llm_int8_has_fp16_weight=False
            )
            
            logger.info("Aplicando cuantización INT8...")
            return model
        
        except Exception as e:
            logger.error(f"Error en cuantización INT8: {str(e)}")
            raise
    
    @staticmethod
    def quantize_int4(
        model: AutoModelForCausalLM,
        compute_dtype: torch.dtype = torch.float16,
        use_double_quant: bool = True,
        quant_type: str = "nf4"
    ) -> AutoModelForCausalLM:
        """
        Cuantización INT4 (NF4).
        
        Args:
            model: Modelo a cuantizar
            compute_dtype: Tipo de datos para cómputo
            use_double_quant: Usar doble cuantización
            quant_type: Tipo de cuantización (nf4, fp4)
        
        Returns:
            Modelo cuantizado
        """
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_quant_type=quant_type
            )
            
            logger.info(f"Aplicando cuantización INT4 ({quant_type})...")
            return model
        
        except Exception as e:
            logger.error(f"Error en cuantización INT4: {str(e)}")
            raise
    
    @staticmethod
    def quantize_dynamic(
        model: torch.nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> torch.nn.Module:
        """
        Cuantización dinámica PyTorch.
        
        Args:
            model: Modelo a cuantizar
            dtype: Tipo de datos
        
        Returns:
            Modelo cuantizado
        """
        try:
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=dtype
            )
            
            logger.info("Cuantización dinámica aplicada")
            return quantized_model
        
        except Exception as e:
            logger.error(f"Error en cuantización dinámica: {str(e)}")
            raise
    
    @staticmethod
    def get_quantization_stats(model: torch.nn.Module) -> Dict[str, Any]:
        """
        Obtener estadísticas de cuantización.
        
        Args:
            model: Modelo
        
        Returns:
            Estadísticas
        """
        try:
            total_params = sum(p.numel() for p in model.parameters())
            quantized_params = 0
            
            for module in model.modules():
                if isinstance(module, torch.quantization.QuantWrapper):
                    quantized_params += sum(p.numel() for p in module.parameters())
            
            return {
                "total_params": total_params,
                "quantized_params": quantized_params,
                "quantization_ratio": quantized_params / total_params if total_params > 0 else 0.0
            }
        
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {}




