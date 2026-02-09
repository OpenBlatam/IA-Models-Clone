"""
Transformers Backend Implementation

Standard HuggingFace Transformers backend.
"""

import logging
from typing import Dict, Any, Union, List, Optional
import torch

from .base import BaseBackend
from ..model_loader.types import ModelConfig, GenerationConfig, ModelType, QuantizationType

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoModelForVision2Seq,
        BitsAndBytesConfig,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None


class TransformersBackend(BaseBackend):
    """Transformers backend using HuggingFace."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device: Optional[str] = None
    
    def load(self, config: ModelConfig) -> Dict[str, Any]:
        """Load model using Transformers."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "Transformers is not available. Install with: pip install transformers"
            )
        
        logger.info(f"Loading model with Transformers: {config.model_name}")
        
        # Determine device
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Configure quantization
        quantization_config = None
        if config.quantization == QuantizationType.INT8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif config.quantization == QuantizationType.INT4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True
            )
        
        # Load model based on type
        if config.model_type == ModelType.CAUSAL_LM:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path or config.model_name,
                quantization_config=quantization_config,
                trust_remote_code=config.trust_remote_code,
                torch_dtype=self._get_torch_dtype(config.quantization),
                device_map="auto" if self.device == "cuda" else None,
                **config.extra_kwargs
            )
        elif config.model_type == ModelType.VISION_LM:
            self.model = AutoModelForVision2Seq.from_pretrained(
                config.model_path or config.model_name,
                trust_remote_code=config.trust_remote_code,
                **config.extra_kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path or config.model_name,
            trust_remote_code=config.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self._config = config
        self._loaded = True
        
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "backend": "transformers"
        }
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: GenerationConfig
    ) -> Union[str, List[str]]:
        """Generate text using Transformers."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.temperature > 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **config.extra_kwargs
            )
        
        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_texts[0] if len(generated_texts) == 1 else generated_texts
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        self._config = None
        self.device = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self.model is not None
    
    def _get_torch_dtype(self, quantization: QuantizationType) -> torch.dtype:
        """Get PyTorch dtype from quantization type."""
        dtype_map = {
            QuantizationType.FP32: torch.float32,
            QuantizationType.FP16: torch.float16,
            QuantizationType.BF16: torch.bfloat16,
        }
        return dtype_map.get(quantization, torch.float16)

