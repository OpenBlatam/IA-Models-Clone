"""
Model Configuration Module
==========================

Configuration classes for model setup.

Author: BUL System
Date: 2024
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """
    Configuration for model loading and setup.
    
    Attributes:
        model_name: Name or path of the model
        model_type: Type of model ("causal" or "seq2seq")
        tokenizer_name: Name of tokenizer (defaults to model_name)
        tokenizer_vocab_size: Vocabulary size for embeddings
        torch_dtype: Torch dtype for model
        device_map: Device mapping strategy
        low_cpu_mem_usage: Use low CPU memory mode
        trust_remote_code: Trust remote code
        
    Example:
        >>> config = ModelConfig(
        ...     model_name="gpt2",
        ...     model_type="causal"
        ... )
    """
    
    model_name: str
    model_type: str = "causal"
    tokenizer_name: Optional[str] = None
    tokenizer_vocab_size: Optional[int] = None
    torch_dtype: Optional[str] = None
    device_map: Optional[str] = None
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": self.trust_remote_code,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ModelConfig instance
        """
        return cls(**config_dict)

