"""
Modelo de Generación de Manuales con Transformers
==================================================

Modelo basado en transformers para generar manuales personalizados.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from ..optimizations.model_optimizer import ModelOptimizer

logger = logging.getLogger(__name__)


class ManualGeneratorModel(nn.Module):
    """
    Modelo de generación de manuales basado en transformers.
    
    Soporta fine-tuning con LoRA para eficiencia.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Inicializar modelo.
        
        Args:
            model_name: Nombre del modelo pre-entrenado
            use_lora: Usar LoRA para fine-tuning
            lora_r: Rank de LoRA
            lora_alpha: Alpha de LoRA
            lora_dropout: Dropout de LoRA
            device: Dispositivo (cuda/cpu)
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lora = use_lora
        
        logger.info(f"Inicializando modelo {model_name} en {self.device}")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Aplicar LoRA si se solicita
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA aplicado al modelo")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Optimizar modelo
        try:
            self.model = ModelOptimizer.optimize_model(
                self.model,
                use_torch_compile=True,
                use_quantization=False
            )
        except Exception as e:
            logger.warning(f"No se pudo optimizar modelo: {str(e)}")
        
        # Configuración de generación optimizada
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True  # KV cache para velocidad
        )
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generar texto desde prompt.
        
        Args:
            prompt: Prompt de entrada
            max_length: Longitud máxima
            temperature: Temperatura
            top_p: Top-p sampling
            **kwargs: Otros parámetros
        
        Returns:
            Texto generado
        """
        try:
            # Tokenizar
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generar
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
            
            # Decodificar
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remover prompt del resultado
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generando texto: {str(e)}")
            raise
    
    def generate_manual(
        self,
        problem_description: str,
        category: str = "general",
        **kwargs
    ) -> str:
        """
        Generar manual completo.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría del oficio
            **kwargs: Otros parámetros
        
        Returns:
            Manual generado
        """
        prompt = self._build_manual_prompt(problem_description, category)
        return self.generate(prompt, **kwargs)
    
    def _build_manual_prompt(
        self,
        problem_description: str,
        category: str
    ) -> str:
        """Construir prompt para generación."""
        category_names = {
            "plomeria": "Plomería",
            "techos": "Reparación de Techos",
            "carpinteria": "Carpintería",
            "electricidad": "Electricidad",
            "albanileria": "Albañilería",
            "pintura": "Pintura",
            "herreria": "Herrería",
            "jardineria": "Jardinería",
            "general": "Reparación General"
        }
        
        category_name = category_names.get(category, "Reparación General")
        
        prompt = f"""Genera un manual paso a paso tipo LEGO para {category_name}.

PROBLEMA:
{problem_description}

MANUAL:
"""
        return prompt
    
    def save_model(self, path: str):
        """Guardar modelo."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Modelo guardado en {path}")
    
    def load_model(self, path: str):
        """Cargar modelo."""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Modelo cargado desde {path}")

