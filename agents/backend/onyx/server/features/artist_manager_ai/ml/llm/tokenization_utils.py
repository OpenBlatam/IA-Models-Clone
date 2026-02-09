"""
Tokenization Utilities
======================

Advanced tokenization utilities for transformers.
"""

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AdvancedTokenizer:
    """
    Advanced tokenizer with additional utilities.
    
    Features:
    - Smart truncation
    - Padding strategies
    - Special tokens management
    - Batch tokenization
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        padding_strategy: str = "max_length",
        truncation: bool = True
    ):
        """
        Initialize advanced tokenizer.
        
        Args:
            tokenizer_name: Name or path of tokenizer
            max_length: Maximum sequence length
            padding_strategy: Padding strategy
            truncation: Whether to truncate
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.padding_strategy = padding_strategy
        self.truncation = truncation
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._logger = logger
    
    def tokenize_batch(
        self,
        texts: List[str],
        return_tensors: str = "pt",
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize batch of texts.
        
        Args:
            texts: List of texts
            return_tensors: Return format
            add_special_tokens: Add special tokens
        
        Returns:
            Tokenized batch
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding_strategy,
            truncation=self.truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens
        )
    
    def smart_truncate(
        self,
        text: str,
        max_tokens: int,
        strategy: str = "head"
    ) -> str:
        """
        Smart truncation preserving important parts.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens
            strategy: Truncation strategy (head, tail, middle)
        
        Returns:
            Truncated text
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return text
        
        if strategy == "head":
            # Keep last max_tokens
            tokens = tokens[-max_tokens:]
        elif strategy == "tail":
            # Keep first max_tokens
            tokens = tokens[:max_tokens]
        elif strategy == "middle":
            # Keep middle part
            start = (len(tokens) - max_tokens) // 2
            tokens = tokens[start:start + max_tokens]
        
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def add_special_tokens(self, tokens: List[str]):
        """
        Add special tokens to tokenizer.
        
        Args:
            tokens: List of special tokens
        """
        self.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self._logger.info(f"Added {len(tokens)} special tokens")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def decode_batch(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode batch of token IDs.
        
        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Skip special tokens
        
        Returns:
            List of decoded texts
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        texts = []
        for ids in token_ids:
            text = self.tokenizer.decode(
                ids.tolist(),
                skip_special_tokens=skip_special_tokens
            )
            texts.append(text)
        
        return texts


class TokenizerManager:
    """
    Manager for multiple tokenizers.
    """
    
    def __init__(self):
        """Initialize tokenizer manager."""
        self.tokenizers: Dict[str, AdvancedTokenizer] = {}
        self._logger = logger
    
    def register_tokenizer(
        self,
        name: str,
        tokenizer: AdvancedTokenizer
    ):
        """
        Register a tokenizer.
        
        Args:
            name: Tokenizer name
            tokenizer: AdvancedTokenizer instance
        """
        self.tokenizers[name] = tokenizer
        self._logger.info(f"Registered tokenizer: {name}")
    
    def get_tokenizer(self, name: str) -> Optional[AdvancedTokenizer]:
        """
        Get tokenizer by name.
        
        Args:
            name: Tokenizer name
        
        Returns:
            Tokenizer or None
        """
        return self.tokenizers.get(name)
    
    def list_tokenizers(self) -> List[str]:
        """List all registered tokenizers."""
        return list(self.tokenizers.keys())




