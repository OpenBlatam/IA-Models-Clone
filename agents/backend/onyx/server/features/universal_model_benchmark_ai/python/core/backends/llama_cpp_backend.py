"""
llama.cpp Backend Implementation

CPU-optimized inference using llama.cpp.
"""

import logging
from typing import Dict, Any, Union, List, Optional

from .base import BaseBackend
from ..model_loader.types import ModelConfig, GenerationConfig, ModelType

logger = logging.getLogger(__name__)

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    llama_cpp = None


class LlamaCppBackend(BaseBackend):
    """llama.cpp backend for CPU-optimized inference."""
    
    def __init__(self):
        super().__init__()
        self.llm: Optional[Any] = None
    
    def load(self, config: ModelConfig) -> Dict[str, Any]:
        """Load model using llama.cpp."""
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is not available. Install with: pip install llama-cpp-python"
            )
        
        if config.model_type != ModelType.CAUSAL_LM:
            raise ValueError(
                f"llama.cpp only supports CAUSAL_LM, got {config.model_type}"
            )
        
        logger.info(f"Loading model with llama.cpp: {config.model_name}")
        
        model_path = config.model_path or config.model_name
        
        # Create llama.cpp model
        self.llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=config.max_model_len or 2048,
            n_threads=config.extra_kwargs.get("n_threads", 4),
            verbose=config.extra_kwargs.get("verbose", False),
        )
        
        self._config = config
        self._loaded = True
        
        return {
            "model": self.llm,
            "backend": "llama_cpp"
        }
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: GenerationConfig
    ) -> Union[str, List[str]]:
        """Generate text using llama.cpp."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        results = []
        for p in prompt:
            output = self.llm(
                p,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
                stop=config.stop_sequences if config.stop_sequences else None,
            )
            
            # Extract text from output
            if isinstance(output, dict):
                text = output.get("choices", [{}])[0].get("text", "")
            else:
                text = str(output)
            
            results.append(text)
        
        return results[0] if len(results) == 1 else results
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.llm:
            del self.llm
            self.llm = None
        
        self._loaded = False
        self._config = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded and self.llm is not None

