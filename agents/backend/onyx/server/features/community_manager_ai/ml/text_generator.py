"""
Advanced Text Generator - Generador de Texto Avanzado
======================================================

Generación de texto usando modelos transformer grandes (GPT, Llama, etc.)
"""

import logging
import torch
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class AdvancedTextGenerator:
    """Generador de texto avanzado con transformers"""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        use_lora: bool = False,
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """
        Inicializar generador de texto
        
        Args:
            model_name: Nombre del modelo
            use_lora: Usar LoRA para fine-tuning eficiente
            device: Dispositivo
            load_in_8bit: Cargar modelo en 8-bit para ahorrar memoria
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_lora = use_lora
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configurar padding token si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Cargar modelo
            model_kwargs = {}
            if load_in_8bit and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    logger.warning("bitsandbytes no disponible, cargando modelo completo")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Aplicar LoRA si se solicita
            if use_lora:
                self._apply_lora()
            
            self.model.to(self.device)
            self.model.eval()
            
            # Optimizar para inferencia rápida
            if hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Modelo compilado con torch.compile")
                except Exception as e:
                    logger.warning(f"No se pudo compilar modelo: {e}")
            
            # Pipeline para generación
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Text Generator inicializado con {model_name} en {self.device}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.model = None
            self.tokenizer = None
            self.generator = None
    
    def _apply_lora(self):
        """Aplicar LoRA para fine-tuning eficiente"""
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA aplicado al modelo")
        except Exception as e:
            logger.warning(f"Error aplicando LoRA: {e}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generar texto
        
        Args:
            prompt: Prompt inicial
            max_length: Longitud máxima
            temperature: Temperatura (creatividad)
            top_p: Nucleus sampling
            num_return_sequences: Número de secuencias a generar
            
        Returns:
            Lista de textos generados
        """
        if not self.generator:
            return [f"[Modelo no disponible] {prompt}"]
        
        try:
            results = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_texts = []
            for result in results:
                generated_text = result["generated_text"]
                # Remover el prompt del resultado
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                generated_texts.append(generated_text)
            
            return generated_texts
        except Exception as e:
            logger.error(f"Error generando texto: {e}")
            return [f"[Error: {str(e)}]"]
    
    def generate_post(
        self,
        topic: str,
        platform: str,
        tone: str = "professional",
        length: str = "medium"
    ) -> str:
        """
        Generar post optimizado
        
        Args:
            topic: Tema del post
            platform: Plataforma objetivo
            tone: Tono deseado
            length: Longitud (short/medium/long)
            
        Returns:
            Post generado
        """
        length_map = {
            "short": 50,
            "medium": 100,
            "long": 200
        }
        max_length = length_map.get(length, 100)
        
        prompt = f"Write a {tone} social media post about {topic} for {platform}:"
        
        generated = self.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=0.8
        )
        
        return generated[0] if generated else f"Post about {topic}"

