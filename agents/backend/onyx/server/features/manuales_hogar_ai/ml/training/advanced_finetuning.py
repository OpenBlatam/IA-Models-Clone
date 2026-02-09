"""
Fine-tuning Avanzado
====================

Técnicas avanzadas de fine-tuning: AdaLoRA, P-tuning, etc.
"""

import logging
import torch
from typing import Optional, Dict, Any
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
    PromptTuningConfig,
    PromptTuningInit
)
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class AdvancedFineTuner:
    """Fine-tuning avanzado con múltiples técnicas."""
    
    @staticmethod
    def apply_adalora(
        model: AutoModelForCausalLM,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None
    ) -> AutoModelForCausalLM:
        """
        Aplicar AdaLoRA (Adaptive LoRA).
        
        Args:
            model: Modelo base
            r: Rank inicial
            lora_alpha: Alpha
            lora_dropout: Dropout
            target_modules: Módulos objetivo
        
        Returns:
            Modelo con AdaLoRA
        """
        try:
            if target_modules is None:
                target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
            
            adalora_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                init_r=12,
                target_r=8,
                beta1=0.85,
                beta2=0.85,
                tinit=200,
                tfinal=1000,
                deltaT=10,
                orth_reg_weight=0.5
            )
            
            model = get_peft_model(model, adalora_config)
            logger.info("AdaLoRA aplicado al modelo")
            return model
        
        except Exception as e:
            logger.error(f"Error aplicando AdaLoRA: {str(e)}")
            raise
    
    @staticmethod
    def apply_prompt_tuning(
        model: AutoModelForCausalLM,
        num_virtual_tokens: int = 20,
        prompt_tuning_init: PromptTuningInit = PromptTuningInit.RANDOM
    ) -> AutoModelForCausalLM:
        """
        Aplicar P-tuning (Prompt Tuning).
        
        Args:
            model: Modelo base
            num_virtual_tokens: Número de tokens virtuales
            prompt_tuning_init: Inicialización
        
        Returns:
            Modelo con P-tuning
        """
        try:
            prompt_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=num_virtual_tokens,
                prompt_tuning_init=prompt_tuning_init
            )
            
            model = get_peft_model(model, prompt_config)
            logger.info(f"P-tuning aplicado ({num_virtual_tokens} tokens virtuales)")
            return model
        
        except Exception as e:
            logger.error(f"Error aplicando P-tuning: {str(e)}")
            raise
    
    @staticmethod
    def apply_qlora(
        model: AutoModelForCausalLM,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        bits: int = 4
    ) -> AutoModelForCausalLM:
        """
        Aplicar QLoRA (Quantized LoRA).
        
        Args:
            model: Modelo base
            r: Rank
            lora_alpha: Alpha
            lora_dropout: Dropout
            bits: Bits de cuantización (4 u 8)
        
        Returns:
            Modelo con QLoRA
        """
        try:
            from transformers import BitsAndBytesConfig
            
            # Configurar cuantización
            if bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            
            # Aplicar LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
            )
            
            # Cargar modelo con cuantización
            model = AutoModelForCausalLM.from_pretrained(
                model.config.name_or_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            model = get_peft_model(model, lora_config)
            logger.info(f"QLoRA aplicado ({bits}-bit)")
            return model
        
        except Exception as e:
            logger.error(f"Error aplicando QLoRA: {str(e)}")
            raise




