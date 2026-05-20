"""
Blatam AI - Advanced Tokenization and Sequence Handling Engine v6.0.0
Ultra-optimized PyTorch-based tokenization, sequence processing, and text handling
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import warnings
from collections import Counter, defaultdict
import json
import pickle
from dataclasses import dataclass

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# TOKENIZATION CONFIGURATION
# ============================================================================

@dataclass
class TokenizerConfig:
    """Configuration for tokenization and sequence handling."""
    
    # Basic tokenization
    vocab_size: int = 50000
    min_freq: int = 2
    max_length: int = 512
    padding_side: str = "right"  # "left" or "right"
    truncation_side: str = "right"  # "left" or "right"
    
    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    mask_token: str = "<mask>"
    sep_token: str = "<sep>"
    cls_token: str = "<cls>"
    
    # Tokenization strategy
    tokenization_type: str = "bpe"  # "basic", "bpe", "wordpiece", "sentencepiece"
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_numbers: bool = False
    
    # Sequence handling
    return_tensors: str = "pt"  # "pt", "np", "list"
    return_attention_mask: bool = True
    return_token_type_ids: bool = True
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    
    # Performance
    use_fast_tokenizer: bool = True
    cache_dir: Optional[str] = None
    use_auth_token: Optional[str] = None

# ============================================================================
# ADVANCED TOKENIZATION TECHNIQUES
# ============================================================================

class BasicTokenizer:
    """Basic tokenization with regex-based splitting."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab = {}
        self.reverse_vocab = {}
        self.word_counts = Counter()
        
        # Initialize special tokens
        self._init_special_tokens()
        
        # Compile regex patterns
        self._compile_patterns()
        
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        special_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.mask_token,
            self.config.sep_token,
            self.config.cls_token
        ]
        
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
            
    def _compile_patterns(self):
        """Compile regex patterns for tokenization."""
        # Word boundary pattern
        self.word_pattern = re.compile(r'\b\w+\b')
        
        # Punctuation pattern
        self.punct_pattern = re.compile(r'[^\w\s]')
        
        # Number pattern
        self.number_pattern = re.compile(r'\b\d+\b')
        
        # Whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        # Preprocess text
        if self.config.lowercase:
            text = text.lower()
            
        # Split into words
        words = self.word_pattern.findall(text)
        
        # Handle punctuation
        if not self.config.remove_punctuation:
            punct_tokens = self.punct_pattern.findall(text)
            words.extend(punct_tokens)
            
        # Handle numbers
        if not self.config.remove_numbers:
            number_tokens = self.number_pattern.findall(text)
            words.extend(number_tokens)
            
        # Clean and filter tokens
        tokens = [word.strip() for word in words if word.strip()]
        
        return tokens
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from text corpus."""
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)
            
        # Filter by minimum frequency
        filtered_words = [
            word for word, count in self.word_counts.items()
            if count >= self.config.min_freq
        ]
        
        # Add to vocabulary
        for i, word in enumerate(filtered_words):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                self.reverse_vocab[len(self.vocab) - 1] = word
                
        logger.info(f"Built vocabulary with {len(self.vocab)} tokens")
        
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        max_length = max_length or self.config.max_length
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.config.unk_token])
                
        # Truncate if necessary
        if len(token_ids) > max_length:
            if self.config.truncation_side == "right":
                token_ids = token_ids[:max_length]
            else:
                token_ids = token_ids[-max_length:]
                
        return token_ids
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if not skip_special_tokens or token not in [
                    self.config.pad_token, self.config.unk_token,
                    self.config.bos_token, self.config.eos_token
                ]:
                    tokens.append(token)
                    
        return " ".join(tokens)

class BPETokenizer:
    """Byte Pair Encoding (BPE) tokenizer implementation."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab = {}
        self.reverse_vocab = {}
        self.merges = {}
        self.word_freqs = Counter()
        
        # Initialize special tokens
        self._init_special_tokens()
        
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        special_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.mask_token,
            self.config.sep_token,
            self.config.cls_token
        ]
        
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
            
    def train(self, texts: List[str], vocab_size: Optional[int] = None):
        """Train BPE tokenizer on text corpus."""
        vocab_size = vocab_size or self.config.vocab_size
        
        # Initialize vocabulary with characters
        char_vocab = set()
        for text in texts:
            if self.config.lowercase:
                text = text.lower()
            char_vocab.update(text)
            
        # Add character tokens to vocabulary
        for char in char_vocab:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
                self.reverse_vocab[len(self.vocab) - 1] = char
                
        # Count word frequencies
        for text in texts:
            if self.config.lowercase:
                text = text.lower()
            words = text.split()
            self.word_freqs.update(words)
            
        # Perform BPE merges
        while len(self.vocab) < vocab_size:
            # Find most frequent pair
            pair_freqs = Counter()
            for word, freq in self.word_freqs.items():
                if len(word) < 2:
                    continue
                for i in range(len(word) - 1):
                    pair = word[i:i+2]
                    pair_freqs[pair] += freq
                    
            if not pair_freqs:
                break
                
            # Get most frequent pair
            most_frequent_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            
            # Add merged token to vocabulary
            merged_token = "".join(most_frequent_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.reverse_vocab[len(self.vocab) - 1] = merged_token
                self.merges[most_frequent_pair] = merged_token
                
        logger.info(f"Trained BPE tokenizer with {len(self.vocab)} tokens")
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using BPE."""
        if self.config.lowercase:
            text = text.lower()
            
        # Split into words
        words = text.split()
        tokens = []
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
            
        return tokens
        
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using BPE."""
        if word in self.vocab:
            return [word]
            
        # Apply BPE merges
        while len(word) > 1:
            # Find the longest matching pair
            longest_pair = None
            for pair in self.merges:
                if pair in word:
                    if longest_pair is None or len(pair) > len(longest_pair):
                        longest_pair = pair
                        
            if longest_pair is None:
                break
                
            # Split word at the pair
            parts = word.split(longest_pair)
            if len(parts) == 1:
                break
                
            # Apply merge
            word = self.merges[longest_pair].join(parts)
            
        # Split into individual characters if not in vocabulary
        if word not in self.vocab:
            return list(word)
        else:
            return [word]
            
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        max_length = max_length or self.config.max_length
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.config.unk_token])
                
        # Truncate if necessary
        if len(token_ids) > max_length:
            if self.config.truncation_side == "right":
                token_ids = token_ids[:max_length]
            else:
                token_ids = token_ids[-max_length:]
                
        return token_ids
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if not skip_special_tokens or token not in [
                    self.config.pad_token, self.config.unk_token,
                    self.config.bos_token, self.config.eos_token
                ]:
                    tokens.append(token)
                    
        return " ".join(tokens)

# ============================================================================
# SEQUENCE HANDLING AND PROCESSING
# ============================================================================

class SequenceProcessor:
    """Advanced sequence processing and handling."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        
    def pad_sequences(self, sequences: List[List[int]], 
                     max_length: Optional[int] = None,
                     padding: str = "post",
                     truncation: str = "post",
                     value: int = 0) -> torch.Tensor:
        """Pad sequences to uniform length."""
        max_length = max_length or self.config.max_length
        
        # Determine actual max length
        actual_max_length = max(len(seq) for seq in sequences)
        if max_length is not None:
            actual_max_length = min(actual_max_length, max_length)
            
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if len(seq) > actual_max_length:
                # Truncate
                if truncation == "post":
                    seq = seq[:actual_max_length]
                else:
                    seq = seq[-actual_max_length:]
            else:
                # Pad
                padding_length = actual_max_length - len(seq)
                if padding == "post":
                    seq = seq + [value] * padding_length
                else:
                    seq = [value] * padding_length + seq
                    
            padded_sequences.append(seq)
            
        return torch.tensor(padded_sequences, dtype=torch.long)
        
    def create_attention_mask(self, input_ids: torch.Tensor, 
                            pad_token_id: Optional[int] = None) -> torch.Tensor:
        """Create attention mask for padded sequences."""
        if pad_token_id is None:
            pad_token_id = 0
            
        attention_mask = (input_ids != pad_token_id).long()
        return attention_mask
        
    def create_token_type_ids(self, input_ids: torch.Tensor,
                             sep_token_id: Optional[int] = None) -> torch.Tensor:
        """Create token type IDs for sequence pairs."""
        if sep_token_id is None:
            sep_token_id = 0
            
        token_type_ids = torch.zeros_like(input_ids)
        
        # Find separator tokens
        for i, seq in enumerate(input_ids):
            sep_positions = (seq == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                # Mark tokens after separator as type 1
                sep_pos = sep_positions[0]
                token_type_ids[i, sep_pos+1:] = 1
                
        return token_type_ids
        
    def create_special_tokens_mask(self, input_ids: torch.Tensor,
                                  special_tokens: Optional[List[int]] = None) -> torch.Tensor:
        """Create mask for special tokens."""
        if special_tokens is None:
            special_tokens = [0, 1, 2, 3]  # Default special token IDs
            
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for i, seq in enumerate(input_ids):
            for j, token_id in enumerate(seq):
                if token_id in special_tokens:
                    special_tokens_mask[i, j] = True
                    
        return special_tokens_mask
        
    def create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create position IDs for sequences."""
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        return position_ids

class TextPreprocessor:
    """Advanced text preprocessing and cleaning."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters if configured
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
            
        # Remove numbers if configured
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        # Convert to lowercase if configured
        if self.config.lowercase:
            text = text.lower()
            
        # Normalize unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
        
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be enhanced with NLP libraries)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
        
    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

# ============================================================================
# ADVANCED TOKENIZATION MANAGER
# ============================================================================

class AdvancedTokenizationManager:
    """Manager for advanced tokenization and sequence handling."""
    
    def __init__(self, config: TokenizerConfig, device: str = "auto"):
        self.config = config
        self.device = self._get_device(device)
        
        # Tokenizers
        self.basic_tokenizer = BasicTokenizer(config)
        self.bpe_tokenizer = BPETokenizer(config)
        
        # Processors
        self.sequence_processor = SequenceProcessor(config)
        self.text_preprocessor = TextPreprocessor(config)
        
        # Current tokenizer
        self.current_tokenizer = None
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
            
    def train_tokenizer(self, texts: List[str], tokenizer_type: str = "bpe"):
        """Train tokenizer on text corpus."""
        if tokenizer_type == "basic":
            self.basic_tokenizer.build_vocab(texts)
            self.current_tokenizer = self.basic_tokenizer
        elif tokenizer_type == "bpe":
            self.bpe_tokenizer.train(texts)
            self.current_tokenizer = self.bpe_tokenizer
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
            
        logger.info(f"Trained {tokenizer_type} tokenizer")
        
    def encode_text(self, text: str, max_length: Optional[int] = None,
                   add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """Encode text with comprehensive output."""
        if self.current_tokenizer is None:
            raise ValueError("No tokenizer trained. Call train_tokenizer first.")
            
        # Preprocess text
        cleaned_text = self.text_preprocessor.clean_text(text)
        normalized_text = self.text_preprocessor.normalize_text(cleaned_text)
        
        # Encode to token IDs
        token_ids = self.current_tokenizer.encode(normalized_text, max_length)
        
        # Add special tokens if requested
        if add_special_tokens:
            if self.config.bos_token in self.current_tokenizer.vocab:
                token_ids = [self.current_tokenizer.vocab[self.config.bos_token]] + token_ids
            if self.config.eos_token in self.current_tokenizer.vocab:
                token_ids = token_ids + [self.current_tokenizer.vocab[self.config.eos_token]]
                
        # Convert to tensor
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        
        # Create additional outputs
        outputs = {'input_ids': input_ids}
        
        if self.config.return_attention_mask:
            attention_mask = self.sequence_processor.create_attention_mask(
                input_ids, self.current_tokenizer.vocab.get(self.config.pad_token, 0)
            )
            outputs['attention_mask'] = attention_mask
            
        if self.config.return_token_type_ids:
            token_type_ids = self.sequence_processor.create_token_type_ids(
                input_ids, self.current_tokenizer.vocab.get(self.config.sep_token, 0)
            )
            outputs['token_type_ids'] = token_type_ids
            
        if self.config.return_special_tokens_mask:
            special_tokens_mask = self.sequence_processor.create_special_tokens_mask(input_ids)
            outputs['special_tokens_mask'] = special_tokens_mask
            
        if self.config.return_position_ids:
            position_ids = self.sequence_processor.create_position_ids(input_ids)
            outputs['position_ids'] = position_ids
            
        return outputs
        
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None,
                     add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """Encode a batch of texts."""
        if self.current_tokenizer is None:
            raise ValueError("No tokenizer trained. Call train_tokenizer first.")
            
        # Encode each text
        all_token_ids = []
        for text in texts:
            # Preprocess text
            cleaned_text = self.text_preprocessor.clean_text(text)
            normalized_text = self.text_preprocessor.normalize_text(cleaned_text)
            
            # Encode to token IDs
            token_ids = self.current_tokenizer.encode(normalized_text, max_length)
            
            # Add special tokens if requested
            if add_special_tokens:
                if self.config.bos_token in self.current_tokenizer.vocab:
                    token_ids = [self.current_tokenizer.vocab[self.config.bos_token]] + token_ids
                if self.config.eos_token in self.current_tokenizer.vocab:
                    token_ids = token_ids + [self.current_tokenizer.vocab[self.config.eos_token]]
                    
            all_token_ids.append(token_ids)
            
        # Pad sequences
        input_ids = self.sequence_processor.pad_sequences(
            all_token_ids, max_length, 
            padding="post" if self.config.padding_side == "right" else "pre",
            truncation="post" if self.config.truncation_side == "right" else "pre",
            value=self.current_tokenizer.vocab.get(self.config.pad_token, 0)
        )
        
        # Create additional outputs
        outputs = {'input_ids': input_ids}
        
        if self.config.return_attention_mask:
            attention_mask = self.sequence_processor.create_attention_mask(
                input_ids, self.current_tokenizer.vocab.get(self.config.pad_token, 0)
            )
            outputs['attention_mask'] = attention_mask
            
        if self.config.return_token_type_ids:
            token_type_ids = self.sequence_processor.create_token_type_ids(
                input_ids, self.current_tokenizer.vocab.get(self.config.sep_token, 0)
            )
            outputs['token_type_ids'] = token_type_ids
            
        if self.config.return_special_tokens_mask:
            special_tokens_mask = self.sequence_processor.create_special_tokens_mask(input_ids)
            outputs['special_tokens_mask'] = special_tokens_mask
            
        if self.config.return_position_ids:
            position_ids = self.sequence_processor.create_position_ids(input_ids)
            outputs['position_ids'] = position_ids
            
        return outputs
        
    def decode_tokens(self, token_ids: Union[List[int], torch.Tensor],
                     skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        if self.current_tokenizer is None:
            raise ValueError("No tokenizer trained. Call train_tokenizer first.")
            
        return self.current_tokenizer.decode(token_ids, skip_special_tokens)
        
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        if self.current_tokenizer is None:
            return 0
        return len(self.current_tokenizer.vocab)
        
    def get_vocab(self) -> Dict[str, int]:
        """Get current vocabulary."""
        if self.current_tokenizer is None:
            return {}
        return self.current_tokenizer.vocab.copy()
        
    def save_tokenizer(self, save_path: str):
        """Save tokenizer to disk."""
        if self.current_tokenizer is None:
            raise ValueError("No tokenizer to save.")
            
        save_data = {
            'config': self.config,
            'vocab': self.current_tokenizer.vocab,
            'reverse_vocab': self.current_tokenizer.reverse_vocab,
            'tokenizer_type': 'bpe' if isinstance(self.current_tokenizer, BPETokenizer) else 'basic'
        }
        
        if isinstance(self.current_tokenizer, BPETokenizer):
            save_data['merges'] = self.current_tokenizer.merges
            
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        logger.info(f"Tokenizer saved to {save_path}")
        
    def load_tokenizer(self, load_path: str):
        """Load tokenizer from disk."""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
            
        # Update config
        self.config = save_data['config']
        
        # Reconstruct tokenizer
        if save_data['tokenizer_type'] == 'bpe':
            self.bpe_tokenizer.vocab = save_data['vocab']
            self.bpe_tokenizer.reverse_vocab = save_data['reverse_vocab']
            self.bpe_tokenizer.merges = save_data['merges']
            self.current_tokenizer = self.bpe_tokenizer
        else:
            self.basic_tokenizer.vocab = save_data['vocab']
            self.basic_tokenizer.reverse_vocab = save_data['reverse_vocab']
            self.current_tokenizer = self.basic_tokenizer
            
        logger.info(f"Tokenizer loaded from {load_path}")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def main():
    """Main examples for tokenization and sequence handling."""
    # Create configuration
    config = TokenizerConfig(
        vocab_size=10000,
        min_freq=2,
        max_length=512,
        tokenization_type="bpe",
        lowercase=True
    )
    
    # Initialize tokenization manager
    manager = AdvancedTokenizationManager(config)
    
    # Sample texts for training
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "Transformers have revolutionized natural language processing."
    ]
    
    # Train tokenizer
    manager.train_tokenizer(sample_texts, "bpe")
    
    # Encode single text
    single_output = manager.encode_text("Hello world, how are you?")
    logger.info(f"Single text encoding: {single_output}")
    
    # Encode batch of texts
    batch_texts = [
        "First sentence for processing.",
        "Second sentence with different content.",
        "Third sentence to complete the batch."
    ]
    
    batch_output = manager.encode_batch(batch_texts)
    logger.info(f"Batch encoding shapes: {batch_output}")
    
    # Decode tokens
    decoded_text = manager.decode_tokens(single_output['input_ids'][0])
    logger.info(f"Decoded text: {decoded_text}")
    
    # Get vocabulary info
    vocab_size = manager.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Save and load tokenizer
    manager.save_tokenizer("./tokenizer.pkl")
    
    # Create new manager and load
    new_manager = AdvancedTokenizationManager(config)
    new_manager.load_tokenizer("./tokenizer.pkl")
    
    print("Tokenization and sequence handling engine ready!")

if __name__ == "__main__":
    main()

