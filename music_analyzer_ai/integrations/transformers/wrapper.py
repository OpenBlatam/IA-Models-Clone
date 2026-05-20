"""
Transformers Wrapper Module

Enhanced wrapper for HuggingFace Transformers models.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import torch
from torch.nn import Module

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        PreTrainedModel, PreTrainedTokenizer,
        TrainingArguments, Trainer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")


class EnhancedTransformerWrapper:
    """
    Enhanced wrapper for HuggingFace Transformers models.
    
    Features:
    - Efficient model loading
    - Mixed precision inference
    - Proper device management
    
    Args:
        model_name: HuggingFace model name or path.
        device: Device to use (auto-detect if None).
        use_mixed_precision: Enable mixed precision.
        torch_dtype: Data type for model (auto if None).
        cache_dir: Cache directory for models.
        trust_remote_code: Trust remote code in model.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
        use_mixed_precision: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required")
        
        self.model_name = model_name
        self.device = self._get_device(device)
        self.use_mixed_precision = (
            use_mixed_precision and 
            self.device.type == "cuda"
        )
        self.torch_dtype = torch_dtype or (
            torch.float16 if self.use_mixed_precision else torch.float32
        )
        
        # Load model and tokenizer
        self._load_model(cache_dir, trust_remote_code)
        self._load_tokenizer(cache_dir, trust_remote_code)
        
        logger.info(
            f"Transformer model loaded: {model_name} "
            f"on {self.device} with dtype={self.torch_dtype}"
        )
    
    def _get_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Get the appropriate device."""
        if device is not None:
            if isinstance(device, str):
                return torch.device(device)
            return device
        
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_model(self, cache_dir: Optional[str], trust_remote_code: bool):
        """Load transformer model."""
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code
            )
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def _load_tokenizer(self, cache_dir: Optional[str], trust_remote_code: bool):
        """Load tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Input text(s).
            **kwargs: Additional arguments for tokenizer.
        
        Returns:
            Embeddings tensor.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available")
        
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs
        ).to(self.device)
        
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)
        
        # Extract embeddings (pooler_output or mean of last_hidden_state)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state.mean(dim=1)



