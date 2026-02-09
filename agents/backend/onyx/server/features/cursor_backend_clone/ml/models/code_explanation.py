"""
Code Explanation Model - Modelo para explicar código
====================================================

Modelo para generar explicaciones de código en lenguaje natural.
"""

import logging
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import BaseModel

logger = logging.getLogger(__name__)


class CodeExplanationModel(BaseModel):
    """Modelo para explicar código usando seq2seq"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get("model_name", "t5-small")
        self.max_length = config.get("max_length", 512)
        self.max_target_length = config.get("max_target_length", 128)
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def _initialize(self):
        """Inicializar modelo y tokenizer"""
        if self._initialized:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._initialized = True
            
            logger.info(f"CodeExplanationModel initialized with {self.model_name}")
            
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
    
    def generate(self, code: str, **kwargs) -> str:
        """Generar explicación de código"""
        if not self._initialized:
            self._initialize()
        
        prompt = f"Explain this code: {code}"
        
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
                max_length=self.max_target_length,
                **kwargs
            )
        
        # Decodificar
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation
    
    def explain_code(self, code: str) -> str:
        """Explicar código"""
        return self.generate(code)


