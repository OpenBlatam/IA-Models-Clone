"""
Code Completion Model - Modelo para completar código
====================================================

Modelo basado en transformers para completar código usando modelos causales
como GPT-2, CodeGPT, etc.
"""

import logging
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel

logger = logging.getLogger(__name__)


class CodeCompletionModel(BaseModel):
    """
    Modelo para completar código usando modelos causales de transformers.
    
    Utiliza modelos como GPT-2, CodeGPT, o modelos similares para generar
    código a partir de un prompt o completar código existente.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Inicializar modelo de completado de código.
        
        Args:
            config: Diccionario de configuración con:
                - model_name: Nombre del modelo a usar (default: "gpt2")
                - max_length: Longitud máxima de entrada (default: 512)
                - pad_token_id: ID del token de padding (opcional)
        """
        super().__init__(config)
        
        self.model_name: str = config.get("model_name", "gpt2")
        self.max_length: int = config.get("max_length", 512)
        self.pad_token_id: Optional[int] = config.get("pad_token_id", None)
        
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._initialized: bool = False
    
    def _initialize(self) -> None:
        """
        Inicializar modelo y tokenizer.
        
        Raises:
            RuntimeError: Si hay error al cargar el modelo o tokenizer.
        """
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.pad_token_id is None:
                    self.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info(f"Loading model {self.model_name}...")
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if self.device.type == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._initialized = True
            
            logger.info(
                f"CodeCompletionModel initialized with {self.model_name} "
                f"on {self.device}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing CodeCompletionModel: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            self._initialized = False
            raise RuntimeError(f"Failed to initialize model: {e}") from e
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            input_ids: Tensor con IDs de tokens de entrada.
            attention_mask: Máscara de atención opcional.
            **kwargs: Argumentos adicionales para el modelo.
        
        Returns:
            Tensor con las salidas del modelo.
        
        Raises:
            RuntimeError: Si el modelo no está inicializado.
        """
        if not self._initialized:
            self._initialize()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        **kwargs: Any
    ) -> str:
        """
        Generar código a partir de un prompt.
        
        Args:
            prompt: Texto de entrada para generar código.
            max_new_tokens: Número máximo de tokens a generar (default: 100).
            temperature: Temperatura para sampling (default: 0.7).
            top_p: Nucleus sampling parameter (default: 0.9).
            top_k: Top-k sampling parameter (default: 50).
            do_sample: Si usar sampling o greedy decoding (default: True).
            **kwargs: Argumentos adicionales para generate().
        
        Returns:
            Código generado.
        
        Raises:
            RuntimeError: Si el modelo no está inicializado.
            ValueError: Si el prompt está vacío.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if not self._initialized:
            self._initialize()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
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
            
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating code: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate code: {e}") from e
    
    def complete_code(self, code: str, language: str = "python", **kwargs: Any) -> str:
        """
        Completar código existente.
        
        Args:
            code: Código a completar.
            language: Lenguaje de programación (default: "python").
            **kwargs: Argumentos adicionales para generate().
        
        Returns:
            Código completado.
        
        Raises:
            ValueError: Si el código está vacío.
        """
        if not code or not code.strip():
            raise ValueError("Code cannot be empty")
        
        prompt = f"# {language}\n{code}\n"
        return self.generate(
            prompt,
            max_new_tokens=200,
            temperature=0.3,
            **kwargs
        )
    
    def save(self, path: str) -> None:
        """
        Guardar modelo y tokenizer.
        
        Args:
            path: Ruta donde guardar el modelo.
        
        Raises:
            RuntimeError: Si el modelo no está inicializado.
        """
        if not self._initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized")
        
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"CodeCompletionModel saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save model: {e}") from e
    
    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[torch.device] = None
    ) -> "CodeCompletionModel":
        """
        Cargar modelo desde disco.
        
        Args:
            path: Ruta del modelo guardado.
            device: Dispositivo donde cargar el modelo (opcional).
        
        Returns:
            Instancia del modelo cargado.
        
        Raises:
            RuntimeError: Si hay error al cargar el modelo.
        """
        try:
            config = {"model_name": path}
            model = cls(config)
            if device:
                model.to_device(device)
            model._initialize()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {e}") from e



