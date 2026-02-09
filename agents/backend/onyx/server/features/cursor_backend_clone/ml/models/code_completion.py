"""
Code Completion Model - Modelo para completar código
====================================================

Modelo basado en transformers para completar código.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel

logger = logging.getLogger(__name__)


class CodeCompletionModel(BaseModel):
    """Modelo para completar código usando transformers"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get("model_name", "gpt2")
        self.max_length = config.get("max_length", 512)
        self.pad_token_id = config.get("pad_token_id", None)
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def _initialize(self):
        """Inicializar modelo y tokenizer"""
        if self._initialized:
            return
        
        try:
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configurar pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.pad_token_id is None:
                    self.pad_token_id = self.tokenizer.eos_token_id
            
            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._initialized = True
            
            logger.info(f"CodeCompletionModel initialized with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass"""
        if not self._initialized:
            self._initialize()
        
        return self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            **kwargs
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generar código a partir de un prompt"""
        if not self._initialized:
            self._initialize()
        
        # Tokenizar
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generar
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.pad_token_id,
                **kwargs
            )
        
        # Decodificar
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover prompt original
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def complete_code(self, code: str, language: str = "python") -> str:
        """Completar código"""
        prompt = f"# {language}\n{code}\n"
        return self.generate(prompt, max_new_tokens=200, temperature=0.3)
    
    def save(self, path: str):
        """Guardar modelo"""
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """Cargar modelo"""
        config = {"model_name": path}
        model = cls(config)
        model._initialize()
        if device:
            model.to_device(device)
        return model


