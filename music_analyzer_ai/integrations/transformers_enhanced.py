"""
Enhanced Transformers Integration

Improved integration with HuggingFace Transformers following best practices:
- Proper model loading and caching
- Efficient fine-tuning with LoRA
- Mixed precision support
- Proper tokenization
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
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available for LoRA")


class EnhancedTransformerWrapper:
    """
    Enhanced wrapper for HuggingFace Transformers models.
    
    Features:
    - Efficient model loading
    - LoRA fine-tuning support
    - Mixed precision inference
    - Proper device management
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
        """
        Initialize transformer wrapper.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use (auto-detect if None)
            use_mixed_precision: Enable mixed precision
            torch_dtype: Data type for model (auto if None)
            cache_dir: Cache directory for models
            trust_remote_code: Trust remote code in model
        """
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
    
    def _load_model(
        self,
        cache_dir: Optional[str],
        trust_remote_code: bool
    ):
        """Load the transformer model."""
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Move to device if not using device_map
            if not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to eval mode by default
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_tokenizer(
        self,
        cache_dir: Optional[str],
        trust_remote_code: bool
    ):
        """Load the tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Encode texts using the tokenizer.
        
        Args:
            texts: Text or list of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Return format ("pt" for PyTorch)
            
        Returns:
            Dictionary with tokenized inputs
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        return encoded
    
    def forward(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            texts: Text or list of texts
            **kwargs: Additional arguments for tokenizer
            
        Returns:
            Model outputs
        """
        encoded = self.encode(texts, **kwargs)
        
        self.model.eval()
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)
        
        return outputs
    
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get embeddings for texts.
        
        Args:
            texts: Text or list of texts
            pooling: Pooling strategy ("mean", "cls", "max")
            
        Returns:
            Embeddings tensor
        """
        outputs = self.forward(texts)
        
        # Extract embeddings from last hidden state
        last_hidden = outputs.last_hidden_state
        
        if pooling == "mean":
            # Mean pooling
            attention_mask = self.encode(texts)["attention_mask"]
            embeddings = (
                last_hidden * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(1, keepdim=True)
        elif pooling == "cls":
            # CLS token
            embeddings = last_hidden[:, 0, :]
        elif pooling == "max":
            # Max pooling
            embeddings = last_hidden.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return embeddings
    
    def setup_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.1
    ):
        """
        Setup LoRA for efficient fine-tuning.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            target_modules: Target modules for LoRA (auto if None)
            lora_dropout: LoRA dropout
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for LoRA")
        
        if target_modules is None:
            # Auto-detect target modules
            target_modules = self._detect_target_modules()
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(
            f"LoRA configured: r={r}, alpha={lora_alpha}, "
            f"target_modules={target_modules}"
        )
    
    def _detect_target_modules(self) -> List[str]:
        """Auto-detect target modules for LoRA."""
        # Common patterns for attention modules
        patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value"]
        target_modules = []
        
        for name, module in self.model.named_modules():
            for pattern in patterns:
                if pattern in name.lower():
                    target_modules.append(name.split('.')[-1])
                    break
        
        if not target_modules:
            # Fallback to common names
            target_modules = ["q_proj", "v_proj"]
        
        return list(set(target_modules))
    
    def save_model(self, path: str):
        """Save the model."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model."""
        self.model = AutoModel.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded from {path}")


__all__ = [
    "EnhancedTransformerWrapper",
]



