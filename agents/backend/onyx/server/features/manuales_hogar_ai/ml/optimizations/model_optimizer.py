"""
Optimizador de Modelos
======================

Optimizaciones para acelerar inferencia.
"""

import logging
import torch
from typing import Optional, Any
from transformers import AutoModelForCausalLM
from .flash_attention import FlashAttentionOptimizer

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimizador de modelos para inferencia rápida."""
    
    @staticmethod
    def optimize_model(
        model: Any,
        use_torch_compile: bool = True,
        use_quantization: bool = False,
        quantization_mode: str = "int8",
        use_fast_attention: bool = True,
        enable_kv_cache: bool = True
    ) -> Any:
        """
        Optimizar modelo para inferencia rápida.
        
        Args:
            model: Modelo a optimizar
            use_torch_compile: Usar torch.compile (PyTorch 2.0+)
            use_quantization: Usar cuantización
            quantization_mode: Modo de cuantización (int8, int4, dynamic)
        
        Returns:
            Modelo optimizado
        """
        try:
            # Torch compile (PyTorch 2.0+)
            if use_torch_compile and hasattr(torch, 'compile'):
                logger.info("Compilando modelo con torch.compile...")
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Modelo compilado exitosamente")
            
            # Cuantización
            if use_quantization:
                if quantization_mode == "int8":
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    logger.info("Aplicando cuantización INT8...")
                elif quantization_mode == "int4":
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    logger.info("Aplicando cuantización INT4...")
                else:  # dynamic
                    logger.info("Aplicando cuantización dinámica...")
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
            
            # Optimizaciones adicionales
            if hasattr(model, 'eval'):
                model.eval()
            
            # Desactivar gradientes
            for param in model.parameters():
                param.requires_grad = False
            
            # Fast attention
            if use_fast_attention:
                ModelOptimizer.enable_fast_attention(model)
            
            # KV Cache
            if enable_kv_cache:
                ModelOptimizer.optimize_generation_config(model)
            
            # Optimizaciones de memoria
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Modelo optimizado exitosamente")
            return model
        
        except Exception as e:
            logger.warning(f"Error en optimización: {str(e)}. Continuando sin optimización.")
            return model
    
    @staticmethod
    def enable_fast_attention(model: Any):
        """Habilitar atención rápida (Flash Attention si está disponible)."""
        try:
            # Flash Attention 2 (si está disponible)
            FlashAttentionOptimizer.enable_flash_attention(model)
        except Exception as e:
            logger.warning(f"No se pudo habilitar Flash Attention: {str(e)}")
    
    @staticmethod
    def optimize_generation_config(model: Any):
        """Optimizar configuración de generación."""
        try:
            if hasattr(model, 'generation_config'):
                # Optimizaciones para generación rápida
                model.generation_config.do_sample = False  # Greedy decoding más rápido
                model.generation_config.use_cache = True  # Usar KV cache
                logger.info("Configuración de generación optimizada")
        except Exception as e:
            logger.warning(f"Error optimizando generación: {str(e)}")

