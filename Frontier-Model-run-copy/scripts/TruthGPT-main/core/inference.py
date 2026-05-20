"""
Inference Engine
Unified inference interface with optimization and caching
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for inference"""
    # Performance settings
    batch_size: int = 1
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Optimization settings
    use_cache: bool = True
    cache_size: int = 1000
    enable_optimization: bool = True
    
    # Generation settings
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    
    # Device settings
    device: str = "auto"
    precision: str = "float32"

class InferenceEngine:
    """Unified inference engine with optimization and caching"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # Caching
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.inference_times = []
        self.tokens_generated = 0
        
        # Set device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
            
        logger.info(f"Initialized InferenceEngine on {self.device}")
    
    def load_model(self, model: nn.Module, tokenizer: Optional[Any] = None) -> None:
        """Load model and tokenizer for inference"""
        self.model = model.to(self.device)
        self.model.eval()
        
        if tokenizer:
            self.tokenizer = tokenizer
            
        logger.info("Model loaded for inference")
    
    def generate(self, 
                 prompt: Union[str, List[int]], 
                 max_length: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None) -> Dict[str, Any]:
        """Generate text from prompt"""
        start_time = time.time()
        
        # Use config defaults if not provided
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        
        # Convert prompt to tokens if string
        if isinstance(prompt, str):
            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            else:
                raise ValueError("Tokenizer required for string prompts")
        else:
            input_ids = torch.tensor([prompt], dtype=torch.long)
        
        input_ids = input_ids.to(self.device)
        
        # Check cache
        cache_key = self._get_cache_key(input_ids, max_length, temperature, top_p, top_k)
        if self.config.use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Generate
        with torch.no_grad():
            if self.config.do_sample:
                output = self._sample_generate(
                    input_ids, max_length, temperature, top_p, top_k
                )
            else:
                output = self._beam_generate(input_ids, max_length)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        tokens_generated = output.size(1) - input_ids.size(1)
        
        # Update tracking
        self.inference_times.append(generation_time)
        self.tokens_generated += tokens_generated
        
        result = {
            'generated_ids': output.tolist(),
            'generated_text': self._decode_tokens(output),
            'generation_time': generation_time,
            'tokens_generated': tokens_generated,
            'tokens_per_second': tokens_generated / generation_time if generation_time > 0 else 0
        }
        
        # Cache result
        if self.config.use_cache:
            self._cache_result(cache_key, result)
        
        return result
    
    def _sample_generate(self, 
                        input_ids: torch.Tensor, 
                        max_length: int,
                        temperature: float,
                        top_p: float,
                        top_k: int) -> torch.Tensor:
        """Generate using sampling"""
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            # Get logits
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end token
            if next_token.item() == self.tokenizer.eos_token_id if self.tokenizer else 0:
                break
        
        return generated
    
    def _beam_generate(self, input_ids: torch.Tensor, max_length: int) -> torch.Tensor:
        """Generate using beam search"""
        # Simplified beam search implementation
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end token
                if next_token.item() == self.tokenizer.eos_token_id if self.tokenizer else 0:
                    break
        
        return generated
    
    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text"""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        else:
            # Simple character-based decoding
            return ''.join([chr(token_id) for token_id in token_ids[0] if token_id < 256])
    
    def _get_cache_key(self, 
                      input_ids: torch.Tensor, 
                      max_length: int,
                      temperature: float,
                      top_p: float,
                      top_k: int) -> str:
        """Generate cache key for input"""
        return f"{input_ids.tolist()}_{max_length}_{temperature}_{top_p}_{top_k}"
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache generation result"""
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        self.cache_misses += 1
    
    def batch_generate(self, 
                      prompts: List[Union[str, List[int]]],
                      **kwargs) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts"""
        results = []
        
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get inference performance metrics"""
        if not self.inference_times:
            return {}
        
        return {
            'total_inferences': len(self.inference_times),
            'total_tokens_generated': self.tokens_generated,
            'average_generation_time': np.mean(self.inference_times),
            'tokens_per_second': self.tokens_generated / sum(self.inference_times) if sum(self.inference_times) > 0 else 0,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the generation cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Inference cache cleared")
    
    def optimize_for_inference(self) -> None:
        """Optimize model for inference"""
        if self.model is None:
            logger.warning("No model loaded for optimization")
            return
        
        # Set to evaluation mode
        self.model.eval()
        
        # Apply optimizations
        if self.config.enable_optimization:
            # JIT compilation if supported
            try:
                self.model = torch.jit.script(self.model)
                logger.info("Model optimized with JIT compilation")
            except Exception as e:
                logger.warning(f"JIT optimization failed: {e}")
        
        logger.info("Model optimized for inference")

