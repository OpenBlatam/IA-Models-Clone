"""
Servicio de LLM Local
=====================

Servicio para usar modelos locales en lugar de OpenRouter.
"""

import logging
import torch
from typing import Optional, Dict, Any, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextGenerationPipeline
)

from ..models.manual_generator_model import ManualGeneratorModel

logger = logging.getLogger(__name__)


class LocalLLMService:
    """Servicio para modelos LLM locales."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        use_local_model: bool = False,
        device: Optional[str] = None
    ):
        """
        Inicializar servicio LLM local.
        
        Args:
            model_name: Nombre del modelo
            use_local_model: Usar modelo local en lugar de OpenRouter
            device: Dispositivo
        """
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if use_local_model:
            self._load_model()
    
    def _load_model(self):
        """Cargar modelo local."""
        try:
            logger.info(f"Cargando modelo local: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Crear pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Modelo local cargado exitosamente")
        
        except Exception as e:
            logger.error(f"Error cargando modelo local: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generar texto usando modelo local.
        
        Args:
            prompt: Prompt de entrada
            max_length: Longitud máxima
            temperature: Temperatura
            top_p: Top-p sampling
            **kwargs: Otros parámetros
        
        Returns:
            Texto generado
        """
        if not self.use_local_model or self.pipeline is None:
            raise ValueError("Modelo local no está cargado")
        
        try:
            result = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                **kwargs
            )
            
            generated_text = result[0]["generated_text"]
            
            # Remover prompt
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generando texto: {str(e)}")
            raise




