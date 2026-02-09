"""
Base Model Classes
Object-oriented model architectures following PyTorch best practices.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseLLMModel(nn.Module, ABC):
    """
    Base class for Language Model architectures.
    Follows OOP principles for model development.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._build_model()
        self._initialize_weights()
    
    @abstractmethod
    def _build_model(self):
        """Build model architecture. Must be implemented by subclasses."""
        pass
    
    def _initialize_weights(self):
        """Initialize model weights using best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Labels for loss computation [batch_size, seq_length]
            
        Returns:
            Dictionary with logits and optionally loss
        """
        pass
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token IDs
        """
        self.eval()
        with torch.no_grad():
            return self._generate_impl(
                input_ids, max_length, temperature, top_p, top_k, do_sample, **kwargs
            )
    
    def _generate_impl(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        **kwargs
    ) -> torch.Tensor:
        """Implementation of generation logic."""
        # Default greedy generation
        # Subclasses should override for better implementations
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :] / temperature
            
            if do_sample:
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
    
    def to_device(self, device: Optional[str] = None):
        """Move model to specified device."""
        device = device or self.device
        self.to(device)
        self.device = device
        return self


class ModelManager:
    """
    Manager for model loading, caching, and lifecycle management.
    """
    
    def __init__(
        self,
        cache_size: int = 5,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        self.cache_size = cache_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self._cache: Dict[str, Any] = {}
        self._access_order: list = []
    
    def get_model(
        self,
        model_name: str,
        model_class: Optional[type] = None,
        **kwargs
    ) -> Any:
        """
        Get model from cache or load it.
        
        Args:
            model_name: Model identifier
            model_class: Model class to instantiate
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model
        """
        cache_key = f"{model_name}_{str(kwargs)}"
        
        if cache_key in self._cache:
            # Update access order
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            logger.info(f"Model {model_name} loaded from cache")
            return self._cache[cache_key]
        
        # Load new model
        if model_class:
            model = model_class(**kwargs)
        else:
            # Default: load from transformers
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                **kwargs
            )
        
        # Move to device if not using device_map
        if self.device != "cuda" or "device_map" not in kwargs:
            model = model.to(self.device)
        
        # Cache management
        if len(self._cache) >= self.cache_size:
            # Remove least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
            logger.info(f"Evicted model {lru_key} from cache")
        
        self._cache[cache_key] = model
        self._access_order.append(cache_key)
        
        logger.info(f"Model {model_name} loaded and cached")
        return model
    
    def clear_cache(self):
        """Clear model cache."""
        self._cache.clear()
        self._access_order.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")



