"""
Tokenizer Utilities Module
==========================

Handles tokenizer setup, configuration, and tokenization functions.
Supports both causal and seq2seq model types.

Author: BUL System
Date: 2024
"""

import logging
from typing import Dict, Any, List, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TokenizerUtils:
    """
    Utilities for tokenizer setup and tokenization.
    
    Handles:
    - Tokenizer loading and configuration
    - Padding token setup
    - Tokenization functions for different model types
    
    Attributes:
        tokenizer: The configured tokenizer
        model_type: Type of model ("causal" or "seq2seq")
        max_length: Maximum sequence length
        
    Example:
        >>> utils = TokenizerUtils("gpt2", model_type="causal")
        >>> tokenizer = utils.get_tokenizer()
        >>> tokenized = utils.tokenize_examples(prompts, responses)
    """
    
    def __init__(
        self,
        tokenizer_name: str,
        model_type: str = "causal",
        max_length: int = 512
    ):
        """
        Initialize TokenizerUtils.
        
        Args:
            tokenizer_name: Name or path of the tokenizer
            model_type: Type of model ("causal" or "seq2seq")
            max_length: Maximum sequence length for tokenization
        """
        self.model_type = model_type
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer(tokenizer_name)
        self._setup_tokenizer()
    
    def _load_tokenizer(self, tokenizer_name: str) -> PreTrainedTokenizer:
        """
        Load pre-trained tokenizer.
        
        Args:
            tokenizer_name: Name or path of the tokenizer
            
        Returns:
            Loaded tokenizer
        """
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer
    
    def _setup_tokenizer(self) -> None:
        """Setup tokenizer with padding and special tokens."""
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Using EOS token as PAD token")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info("Added [PAD] token")
        
        # Set padding side based on model type
        if self.model_type == "causal":
            self.tokenizer.padding_side = "left"
        else:  # seq2seq
            self.tokenizer.padding_side = "right"
        
        logger.info(
            f"Tokenizer configured (vocab_size: {self.tokenizer.vocab_size}, "
            f"padding_side: {self.tokenizer.padding_side})"
        )
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the configured tokenizer."""
        return self.tokenizer
    
    def tokenize_examples(
        self,
        examples: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Tokenize examples for training.
        
        Args:
            examples: Dictionary with "prompt" and "response" lists
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        prompts = examples["prompt"]
        responses = examples["response"]
        
        if self.model_type == "causal":
            return self._tokenize_causal(prompts, responses)
        else:  # seq2seq
            return self._tokenize_seq2seq(prompts, responses)
    
    def _tokenize_causal(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> Dict[str, Any]:
        """
        Tokenize for causal language models.
        
        For causal models, concatenate prompt + response and tokenize.
        Labels are the same as input_ids.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            
        Returns:
            Tokenized dictionary with input_ids, attention_mask, and labels
        """
        # Concatenate prompt + response
        texts = [f"{p} {r}" for p, r in zip(prompts, responses)]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Labels are same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def _tokenize_seq2seq(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> Dict[str, Any]:
        """
        Tokenize for seq2seq models.
        
        For seq2seq models, tokenize prompts and responses separately.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            
        Returns:
            Tokenized dictionary with input_ids, attention_mask, and labels
        """
        # Tokenize prompts
        tokenized_prompts = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Tokenize responses
        tokenized_responses = self.tokenizer(
            responses,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokenized_prompts["input_ids"],
            "attention_mask": tokenized_prompts["attention_mask"],
            "labels": tokenized_responses["input_ids"],
        }
    
    def tokenize_for_inference(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = True
    ) -> Dict[str, Any]:
        """
        Tokenize text for inference.
        
        Args:
            text: Text to tokenize
            return_tensors: Format to return tensors in
            add_special_tokens: Whether to add special tokens
            
        Returns:
            Tokenized dictionary
        """
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            padding=False if return_tensors == "pt" else "max_length"
        )
    
    def get_token_count(self, text: str) -> int:
        """
        Get the number of tokens in a text without tokenizing.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens (uses max_length if None)
            
        Returns:
            Truncated text
        """
        if max_tokens is None:
            max_tokens = self.max_length
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False, max_length=max_tokens, truncation=True)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def resize_token_embeddings(self, model, new_size: Optional[int] = None) -> None:
        """
        Resize model token embeddings if tokenizer was modified.
        
        Args:
            model: The model to resize
            new_size: New vocabulary size (uses tokenizer vocab size if None)
        """
        if new_size is None:
            new_size = len(self.tokenizer)
        
        if new_size > model.config.vocab_size:
            model.resize_token_embeddings(new_size)
            logger.info(f"Resized token embeddings to {new_size}")

