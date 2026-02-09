"""
Text Tokenization Utilities

Implements proper tokenization and sequence handling for text data.
"""

import logging
from typing import List, Optional, Dict, Any, Union
import torch
import numpy as np

logger = logging.getLogger(__name__)


class TextTokenizer:
    """
    Text tokenizer wrapper for Transformers tokenizers.
    
    Handles proper tokenization and sequence handling.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ):
        """
        Initialize text tokenizer.
        
        Args:
            tokenizer: Hugging Face tokenizer instance
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text.
        
        Args:
            text: Text or list of texts to tokenize
            return_tensors: Format to return ('pt', 'np', None)
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Prepare tokenizer kwargs
        tokenizer_kwargs = {
            'padding': self.padding,
            'truncation': self.truncation,
            'return_tensors': return_tensors,
            **kwargs
        }
        
        if self.max_length:
            tokenizer_kwargs['max_length'] = self.max_length
        
        # Tokenize
        encoded = self.tokenizer(text, **tokenizer_kwargs)
        
        return encoded
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, np.ndarray, List[int]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            return_tensors: Format to return ('pt', 'np', None)
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text or list of texts
        """
        # Convert to list if tensor/array
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Handle batch vs single
        if isinstance(token_ids[0], list):
            return self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
        else:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad_token_id': self.tokenizer.pad_token_id,
            'bos_token_id': getattr(self.tokenizer, 'bos_token_id', None),
            'eos_token_id': getattr(self.tokenizer, 'eos_token_id', None),
            'unk_token_id': getattr(self.tokenizer, 'unk_token_id', None)
        }


def create_tokenizer(
    model_name: str,
    max_length: Optional[int] = None,
    **kwargs
) -> TextTokenizer:
    """
    Create tokenizer from model name.
    
    Args:
        model_name: Model name from Hugging Face
        max_length: Maximum sequence length
        **kwargs: Additional tokenizer arguments
        
    Returns:
        TextTokenizer instance
    """
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return TextTokenizer(
            tokenizer=tokenizer,
            max_length=max_length
        )
    except ImportError:
        raise ImportError("transformers library required for tokenization")



