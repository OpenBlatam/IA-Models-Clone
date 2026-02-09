"""
vLLM Backend Implementation

High-performance inference using vLLM with PagedAttention.
"""

import logging
from typing import Dict, Any, Union, List, Optional

from .base import BaseBackend
from ..model_loader.types import ModelConfig, GenerationConfig, ModelType, QuantizationType

logger = logging.getLogger(__name__)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None


class VLLMBackend(BaseBackend):
    """vLLM backend for high-performance inference."""
    
    def __init__(self):
        super().__init__()
        self.engine: Optional[LLM] = None
        self.tokenizer: Optional[Any] = None
    
    def load(self, config: ModelConfig) -> Dict[str, Any]:
        """Load model using vLLM."""
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is not available. Install with: pip install vllm>=0.2.0"
            )
        
        if config.model_type != ModelType.CAUSAL_LM:
            raise ValueError(
                f"vLLM only supports CAUSAL_LM, got {config.model_type}"
            )
        
        logger.info(f"Loading model with vLLM: {config.model_name}")
        
        # Configure quantization
        quantization_config = None
        if config.quantization == QuantizationType.AWQ:
            quantization_config = "awq"
        elif config.quantization == QuantizationType.GPTQ:
            quantization_config = "gptq"
        
        # Get dtype
        dtype_map = {
            QuantizationType.FP32: "float32",
            QuantizationType.FP16: "float16",
            QuantizationType.BF16: "bfloat16",
        }
        dtype = dtype_map.get(config.quantization, "float16")
        
        # Create vLLM engine
        self.engine = LLM(
            model=config.model_path or config.model_name,
            trust_remote_code=config.trust_remote_code,
            quantization=quantization_config,
            dtype=dtype,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enable_prefix_caching=config.enable_prefix_caching,
            **config.extra_kwargs
        )
        
        # Load tokenizer
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_path or config.model_name,
                trust_remote_code=config.trust_remote_code
            )
        
        self._config = config
        self._loaded = True
        
        return {
            "model": self.engine,
            "tokenizer": self.tokenizer,
            "backend": "vllm"
        }
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: GenerationConfig
    ) -> Union[str, List[str]]:
        """Generate text using vLLM."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences if config.stop_sequences else None,
            repetition_penalty=config.repetition_penalty,
            **config.extra_kwargs
        )
        
        outputs = self.engine.generate(prompt, sampling_params)
        
        results = [output.outputs[0].text for output in outputs]
        
        return results[0] if len(results) == 1 else results
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.engine:
            # vLLM doesn't have explicit unload, but we can clear references
            del self.engine
            self.engine = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self._loaded = False
        self._config = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self.engine is not None

