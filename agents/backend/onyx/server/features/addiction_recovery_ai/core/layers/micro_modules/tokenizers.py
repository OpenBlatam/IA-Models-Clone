"""
Tokenizers - Ultra-Specific Tokenization Components
Each tokenizer in its own focused implementation
"""

from typing import Optional, Dict, List, Any
import torch
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TokenizerBase(ABC):
    """Base class for all tokenizers"""
    
    def __init__(self, name: str = "Tokenizer"):
        self.name = name
    
    @abstractmethod
    def tokenize(self, text: str) -> Any:
        """Tokenize text"""
        pass
    
    @abstractmethod
    def detokenize(self, tokens: Any) -> str:
        """Detokenize tokens back to text"""
        pass
    
    def tokenize_batch(self, texts: List[str]) -> List[Any]:
        """Tokenize batch of texts"""
        return [self.tokenize(text) for text in texts]


class SimpleTokenizer(TokenizerBase):
    """Simple word-based tokenizer"""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None, unk_token: int = 0):
        super().__init__("SimpleTokenizer")
        self.vocab = vocab or {}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
    
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """Build vocabulary from texts"""
        from collections import Counter
        
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        self.vocab = {
            word: idx + 1 for idx, (word, count) in enumerate(
                word_counts.items() if min_freq == 1 else 
                [(w, c) for w, c in word_counts.items() if c >= min_freq]
            )
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        logger.info(f"Vocabulary built with {len(self.vocab)} words")
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text"""
        words = text.lower().split()
        tokens = [self.vocab.get(word, self.unk_token) for word in words]
        return torch.tensor(tokens, dtype=torch.long)
    
    def detokenize(self, tokens: torch.Tensor) -> str:
        """Detokenize tokens"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        words = [self.reverse_vocab.get(token, "<UNK>") for token in tokens]
        return " ".join(words)


class CharacterTokenizer(TokenizerBase):
    """Character-level tokenizer"""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        super().__init__("CharacterTokenizer")
        if vocab is None:
            # Default ASCII characters
            self.vocab = {chr(i): i for i in range(32, 127)}
        else:
            self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text at character level"""
        tokens = [self.vocab.get(char, 0) for char in text]
        return torch.tensor(tokens, dtype=torch.long)
    
    def detokenize(self, tokens: torch.Tensor) -> str:
        """Detokenize tokens"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        chars = [self.reverse_vocab.get(token, "") for token in tokens]
        return "".join(chars)


class HuggingFaceTokenizer(TokenizerBase):
    """HuggingFace tokenizer wrapper"""
    
    def __init__(self, tokenizer: Any):
        super().__init__("HuggingFaceTokenizer")
        self.tokenizer = tokenizer
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using HuggingFace tokenizer"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
    
    def detokenize(self, tokens: Any) -> str:
        """Detokenize tokens"""
        if isinstance(tokens, dict):
            tokens = tokens.get('input_ids', tokens)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


class BPETokenizer(TokenizerBase):
    """Byte Pair Encoding tokenizer"""
    
    def __init__(self, vocab_size: int = 1000):
        super().__init__("BPETokenizer")
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: List[tuple] = []
    
    def train(self, texts: List[str]):
        """Train BPE tokenizer"""
        # Simplified BPE training
        # In practice, use sentencepiece or tokenizers library
        logger.info(f"BPE tokenizer training (vocab_size={self.vocab_size})")
        # Implementation would go here
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text with BPE"""
        # Simplified implementation
        # In practice, use trained BPE model
        return torch.tensor([ord(c) for c in text], dtype=torch.long)
    
    def detokenize(self, tokens: torch.Tensor) -> str:
        """Detokenize BPE tokens"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return "".join([chr(t) if t < 256 else "" for t in tokens])


# Factory for tokenizers
class TokenizerFactory:
    """Factory for creating tokenizers"""
    
    _registry = {
        'simple': SimpleTokenizer,
        'character': CharacterTokenizer,
        'bpe': BPETokenizer,
    }
    
    @classmethod
    def create(cls, tokenizer_type: str, **kwargs) -> TokenizerBase:
        """Create tokenizer"""
        tokenizer_type = tokenizer_type.lower()
        if tokenizer_type not in cls._registry:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        return cls._registry[tokenizer_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, tokenizer_class: type):
        """Register custom tokenizer"""
        cls._registry[name.lower()] = tokenizer_class


__all__ = [
    "TokenizerBase",
    "SimpleTokenizer",
    "CharacterTokenizer",
    "HuggingFaceTokenizer",
    "BPETokenizer",
    "TokenizerFactory",
]



