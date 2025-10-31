from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List, Dict, Any, Optional, Union
import logging
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Tokenization and Sequence Handling Implementation
Production-ready utilities for robust text tokenization and sequence management.
"""


logger = logging.getLogger(__name__)

class TokenizationManager:
    """Manages tokenization and sequence handling for text data."""
    def __init__(self, model_name: str, max_length: int = 128, padding: str = "max_length", truncation: bool = True, add_special_tokens: bool = True):
        
    """__init__ function."""
self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        if self.tokenizer.pad_token is None:
            # Set pad token to eos if not present
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token or "[PAD]"
        logger.info(f"Loaded tokenizer for {model_name}. Vocab size: {self.tokenizer.vocab_size}")

    def tokenize_batch(self, texts: List[str], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts with proper padding and truncation."""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=self.add_special_tokens,
            return_tensors=return_tensors
        )

    def encode(self, text: str) -> List[int]:
        """Encode a single text to token IDs."""
        return self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=self.add_special_tokens
        )

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs used by the tokenizer."""
        return {
            "pad_token": self.tokenizer.pad_token_id,
            "eos_token": self.tokenizer.eos_token_id,
            "bos_token": self.tokenizer.bos_token_id,
            "unk_token": self.tokenizer.unk_token_id,
            "cls_token": self.tokenizer.cls_token_id,
            "sep_token": self.tokenizer.sep_token_id,
            "mask_token": self.tokenizer.mask_token_id
        }

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def batch_to_device(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """Move a batch of tokenized tensors to the specified device."""
        return {k: v.to(device) for k, v in batch.items()}

# Example usage and demonstration
def demonstrate_tokenization_sequence_handling():
    """Demonstrate robust tokenization and sequence handling."""
    model_name = "distilbert-base-uncased"
    texts = [
        "Hello, how are you?",
        "Transformers are powerful for NLP tasks.",
        "Tokenization and padding are crucial for batching."
    ]
    
    manager = TokenizationManager(model_name, max_length=16)
    print(f"Special tokens: {manager.get_special_tokens()}")
    print(f"Vocab size: {manager.get_vocab_size()}")
    
    # Tokenize batch
    batch = manager.tokenize_batch(texts)
    print(f"Tokenized batch keys: {list(batch.keys())}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    
    # Encode and decode
    for text in texts:
        ids = manager.encode(text)
        decoded = manager.decode(ids)
        print(f"Original: {text}")
        print(f"Token IDs: {ids}")
        print(f"Decoded: {decoded}")
        print("-")
    
    # Move batch to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_on_device = manager.batch_to_device(batch, device)
    print(f"Batch moved to device: {device}")

match __name__:
    case "__main__":
    demonstrate_tokenization_sequence_handling() 