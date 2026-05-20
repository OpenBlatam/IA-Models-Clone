"""
Transformer Utilities - Advanced Transformer Helpers
=====================================================

Utilities for working with Hugging Face Transformers.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import (
        AutoModel, AutoModelForSequenceClassification,
        AutoTokenizer, AutoConfig,
        PreTrainedModel, PreTrainedTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")


def load_pretrained_model(
    model_name: str,
    task: str = 'feature_extraction',
    num_labels: Optional[int] = None,
    cache_dir: Optional[Path] = None
) -> Any:
    """
    Load pre-trained model from Hugging Face.
    
    Args:
        model_name: Model name or path
        task: Task type ('feature_extraction', 'classification', 'generation')
        num_labels: Number of labels (for classification)
        cache_dir: Cache directory
        
    Returns:
        Loaded model
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library is required")
    
    if task == 'classification' and num_labels is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=str(cache_dir) if cache_dir else None
        )
    else:
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir) if cache_dir else None
        )
    
    logger.info(f"Loaded model: {model_name}")
    return model


def load_tokenizer(
    model_name: str,
    cache_dir: Optional[Path] = None
) -> PreTrainedTokenizer:
    """
    Load tokenizer from Hugging Face.
    
    Args:
        model_name: Model name or path
        cache_dir: Cache directory
        
    Returns:
        Loaded tokenizer
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library is required")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir) if cache_dir else None
    )
    
    logger.info(f"Loaded tokenizer: {model_name}")
    return tokenizer


def setup_lora(
    model: Any,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.1
) -> Any:
    """
    Setup LoRA for efficient fine-tuning.
    
    Args:
        model: Pre-trained model
        r: LoRA rank
        lora_alpha: LoRA alpha
        target_modules: Target modules for LoRA
        lora_dropout: LoRA dropout
        
    Returns:
        Model with LoRA applied
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        if target_modules is None:
            # Default target modules for common architectures
            target_modules = ['q_proj', 'v_proj', 'k_proj', 'out_proj']
        
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias='none',
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        model = get_peft_model(model, peft_config)
        logger.info(f"LoRA applied: r={r}, alpha={lora_alpha}")
        return model
        
    except ImportError:
        raise ImportError("PEFT library is required for LoRA. Install with: pip install peft")


class TokenizedDataset(torch.utils.data.Dataset):
    """
    Dataset for tokenized text data.
    
    Works seamlessly with Hugging Face tokenizers.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        padding: str = 'max_length',
        truncation: bool = True
    ):
        """
        Initialize tokenized dataset.
        
        Args:
            texts: List of texts
            labels: List of labels
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        if self.tokenizer is None:
            logger.warning("No tokenizer provided, using simple tokenization")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tokenized sample
        """
        text = self.texts[idx]
        
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors='pt'
            )
            # Remove batch dimension
            result = {k: v.squeeze(0) for k, v in encoded.items()}
        else:
            # Fallback: simple encoding
            result = {'input_ids': torch.tensor([hash(text) % 10000])}
        
        if self.labels is not None:
            result['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return result



