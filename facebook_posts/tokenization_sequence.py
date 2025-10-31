from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast,
    BatchEncoding, Encoding, TokenizerFast
)
import numpy as np
import json
import logging
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import time
from enum import Enum
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Tokenization and Sequence Handling
Comprehensive tokenization and sequence handling for text data with advanced features.
"""

    AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast,
    BatchEncoding, Encoding, TokenizerFast
)


class TokenizationType(Enum):
    """Types of tokenization approaches."""
    WORD_LEVEL = "word_level"
    SUBWORD_LEVEL = "subword_level"
    CHARACTER_LEVEL = "character_level"
    BYTE_LEVEL = "byte_level"
    SENTENCE_PIECE = "sentence_piece"
    BPE = "bpe"
    UNIGRAM = "unigram"


class SequenceHandlingType(Enum):
    """Types of sequence handling approaches."""
    PADDING = "padding"
    TRUNCATION = "truncation"
    SLIDING_WINDOW = "sliding_window"
    STRIDE = "stride"
    OVERLAP = "overlap"


@dataclass
class TokenizationConfig:
    """Configuration for tokenization and sequence handling."""
    # Tokenization parameters
    tokenization_type: TokenizationType = TokenizationType.SUBWORD_LEVEL
    model_name: str = "gpt2"
    vocab_size: int = 50000
    max_length: int = 512
    min_length: int = 1
    
    # Sequence handling parameters
    sequence_handling_type: SequenceHandlingType = SequenceHandlingType.PADDING
    padding_side: str = "right"
    truncation_side: str = "right"
    stride: int = 128
    overlap: int = 64
    
    # Advanced features
    use_fast_tokenizer: bool = True
    use_special_tokens: bool = True
    return_overflowing_tokens: bool = False
    return_length: bool = True
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    
    # Preprocessing parameters
    do_lower_case: bool = True
    remove_accents: bool = True
    strip_whitespace: bool = True
    normalize_unicode: bool = True


class AdvancedTokenizer:
    """Advanced tokenizer with comprehensive features."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizer = self._load_tokenizer()
        
        # Setup logging
        self._setup_logging()
        
        # Tokenization statistics
        self.tokenization_stats = {
            'total_tokens': 0,
            'total_sequences': 0,
            'avg_sequence_length': 0.0,
            'max_sequence_length': 0,
            'min_sequence_length': float('inf')
        }
    
    def _setup_logging(self) -> Any:
        """Setup logging for tokenization."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tokenization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer based on configuration."""
        self.logger.info(f"Loading tokenizer: {self.config.model_name}")
        
        try:
            if self.config.use_fast_tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    use_fast=True,
                    padding_side=self.config.padding_side,
                    truncation_side=self.config.truncation_side
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    use_fast=False,
                    padding_side=self.config.padding_side,
                    truncation_side=self.config.truncation_side
                )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configure tokenizer
            tokenizer.do_lower_case = self.config.do_lower_case
            tokenizer.remove_accents = self.config.remove_accents
            tokenizer.strip_whitespace = self.config.strip_whitespace
            tokenizer.normalize_unicode = self.config.normalize_unicode
            
            self.logger.info(f"Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def tokenize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Tokenize single text with advanced features."""
        self.logger.debug(f"Tokenizing text: {text[:50]}...")
        
        # Update tokenization parameters
        tokenization_kwargs = {
            'max_length': self.config.max_length,
            'min_length': self.config.min_length,
            'padding': 'max_length' if self.config.sequence_handling_type == SequenceHandlingType.PADDING else False,
            'truncation': True if self.config.sequence_handling_type == SequenceHandlingType.TRUNCATION else False,
            'return_attention_mask': self.config.return_attention_mask,
            'return_token_type_ids': self.config.return_token_type_ids,
            'return_length': self.config.return_length,
            'return_overflowing_tokens': self.config.return_overflowing_tokens,
            'add_special_tokens': self.config.use_special_tokens
        }
        tokenization_kwargs.update(kwargs)
        
        try:
            # Tokenize text
            encoding = self.tokenizer(
                text,
                **tokenization_kwargs
            )
            
            # Update statistics
            self._update_statistics(encoding)
            
            return encoding
            
        except Exception as e:
            self.logger.error(f"Error tokenizing text: {e}")
            raise
    
    def tokenize_batch(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Tokenize batch of texts with advanced features."""
        self.logger.info(f"Tokenizing batch of {len(texts)} texts")
        
        # Update tokenization parameters
        tokenization_kwargs = {
            'max_length': self.config.max_length,
            'min_length': self.config.min_length,
            'padding': 'max_length' if self.config.sequence_handling_type == SequenceHandlingType.PADDING else True,
            'truncation': True if self.config.sequence_handling_type == SequenceHandlingType.TRUNCATION else False,
            'return_attention_mask': self.config.return_attention_mask,
            'return_token_type_ids': self.config.return_token_type_ids,
            'return_length': self.config.return_length,
            'return_overflowing_tokens': self.config.return_overflowing_tokens,
            'add_special_tokens': self.config.use_special_tokens
        }
        tokenization_kwargs.update(kwargs)
        
        try:
            # Tokenize batch
            encoding = self.tokenizer(
                texts,
                **tokenization_kwargs
            )
            
            # Update statistics
            self._update_statistics(encoding)
            
            return encoding
            
        except Exception as e:
            self.logger.error(f"Error tokenizing batch: {e}")
            raise
    
    def _update_statistics(self, encoding: Dict[str, Any]):
        """Update tokenization statistics."""
        if 'input_ids' in encoding:
            input_ids = encoding['input_ids']
            if isinstance(input_ids, list):
                for seq in input_ids:
                    seq_len = len(seq)
                    self.tokenization_stats['total_tokens'] += seq_len
                    self.tokenization_stats['total_sequences'] += 1
                    self.tokenization_stats['max_sequence_length'] = max(
                        self.tokenization_stats['max_sequence_length'], seq_len
                    )
                    self.tokenization_stats['min_sequence_length'] = min(
                        self.tokenization_stats['min_sequence_length'], seq_len
                    )
            
            # Update average
            if self.tokenization_stats['total_sequences'] > 0:
                self.tokenization_stats['avg_sequence_length'] = (
                    self.tokenization_stats['total_tokens'] / self.tokenization_stats['total_sequences']
                )
    
    def create_sliding_windows(self, text: str, window_size: int = None, 
                              stride: int = None) -> List[Dict[str, Any]]:
        """Create sliding windows for long sequences."""
        if window_size is None:
            window_size = self.config.max_length
        if stride is None:
            stride = self.config.stride
        
        self.logger.info(f"Creating sliding windows with window_size={window_size}, stride={stride}")
        
        # Tokenize without padding/truncation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=False,
            truncation=False
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        windows = []
        for i in range(0, len(input_ids), stride):
            window_input_ids = input_ids[i:i + window_size]
            window_attention_mask = attention_mask[i:i + window_size]
            
            # Pad if necessary
            if len(window_input_ids) < window_size:
                padding_length = window_size - len(window_input_ids)
                window_input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
                window_attention_mask.extend([0] * padding_length)
            
            windows.append({
                'input_ids': window_input_ids,
                'attention_mask': window_attention_mask
            })
        
        return windows
    
    def create_overlapping_segments(self, text: str, segment_size: int = None, 
                                  overlap_size: int = None) -> List[Dict[str, Any]]:
        """Create overlapping segments for long sequences."""
        if segment_size is None:
            segment_size = self.config.max_length
        if overlap_size is None:
            overlap_size = self.config.overlap
        
        self.logger.info(f"Creating overlapping segments with segment_size={segment_size}, overlap_size={overlap_size}")
        
        # Tokenize without padding/truncation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            padding=False,
            truncation=False
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        segments = []
        start = 0
        
        while start < len(input_ids):
            end = min(start + segment_size, len(input_ids))
            
            segment_input_ids = input_ids[start:end]
            segment_attention_mask = attention_mask[start:end]
            
            # Pad if necessary
            if len(segment_input_ids) < segment_size:
                padding_length = segment_size - len(segment_input_ids)
                segment_input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
                segment_attention_mask.extend([0] * padding_length)
            
            segments.append({
                'input_ids': segment_input_ids,
                'attention_mask': segment_attention_mask
            })
            
            # Move start position with overlap
            start = end - overlap_size
        
        return segments
    
    def decode_tokens(self, input_ids: Union[List[int], torch.Tensor], 
                     skip_special_tokens: bool = True) -> str:
        """Decode tokens back to text."""
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        return self.tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(self, input_ids_batch: Union[List[List[int]], torch.Tensor], 
                    skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of tokens back to text."""
        if isinstance(input_ids_batch, torch.Tensor):
            input_ids_batch = input_ids_batch.tolist()
        
        return self.tokenizer.batch_decode(input_ids_batch, skip_special_tokens=skip_special_tokens)
    
    def get_vocabulary_info(self) -> Dict[str, Any]:
        """Get comprehensive vocabulary information."""
        vocab_info = {
            'vocab_size': self.tokenizer.vocab_size,
            'model_max_length': self.tokenizer.model_max_length,
            'pad_token': self.tokenizer.pad_token,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token': self.tokenizer.eos_token,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token': self.tokenizer.bos_token,
            'bos_token_id': self.tokenizer.bos_token_id,
            'unk_token': self.tokenizer.unk_token,
            'unk_token_id': self.tokenizer.unk_token_id,
            'cls_token': self.tokenizer.cls_token,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token': self.tokenizer.sep_token,
            'sep_token_id': self.tokenizer.sep_token_id,
            'mask_token': self.tokenizer.mask_token,
            'mask_token_id': self.tokenizer.mask_token_id,
            'special_tokens_map': self.tokenizer.special_tokens_map,
            'tokenizer_class': self.tokenizer.__class__.__name__
        }
        
        return vocab_info
    
    def get_tokenization_statistics(self) -> Dict[str, Any]:
        """Get tokenization statistics."""
        return self.tokenization_stats.copy()


class SequenceHandler:
    """Advanced sequence handling for tokenized data."""
    
    def __init__(self, config: TokenizationConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int = None, 
                     padding_side: str = None) -> Tuple[List[List[int]], List[List[int]]]:
        """Pad sequences to the same length."""
        if max_length is None:
            max_length = self.config.max_length
        if padding_side is None:
            padding_side = self.config.padding_side
        
        self.logger.info(f"Padding {len(sequences)} sequences to length {max_length}")
        
        padded_sequences = []
        attention_masks = []
        
        for sequence in sequences:
            seq_len = len(sequence)
            
            if seq_len > max_length:
                # Truncate sequence
                if padding_side == "right":
                    sequence = sequence[:max_length]
                else:
                    sequence = sequence[-max_length:]
                seq_len = max_length
            
            # Create attention mask
            attention_mask = [1] * seq_len
            
            # Pad sequence
            if seq_len < max_length:
                padding_length = max_length - seq_len
                if padding_side == "right":
                    sequence = sequence + [0] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                else:
                    sequence = [0] * padding_length + sequence
                    attention_mask = [0] * padding_length + attention_mask
            
            padded_sequences.append(sequence)
            attention_masks.append(attention_mask)
        
        return padded_sequences, attention_masks
    
    def truncate_sequences(self, sequences: List[List[int]], max_length: int = None, 
                          truncation_side: str = None) -> List[List[int]]:
        """Truncate sequences to maximum length."""
        if max_length is None:
            max_length = self.config.max_length
        if truncation_side is None:
            truncation_side = self.config.truncation_side
        
        self.logger.info(f"Truncating {len(sequences)} sequences to length {max_length}")
        
        truncated_sequences = []
        
        for sequence in sequences:
            if len(sequence) > max_length:
                if truncation_side == "right":
                    sequence = sequence[:max_length]
                else:
                    sequence = sequence[-max_length:]
            
            truncated_sequences.append(sequence)
        
        return truncated_sequences
    
    def create_attention_masks(self, sequences: List[List[int]], 
                              pad_token_id: int = 0) -> List[List[int]]:
        """Create attention masks for sequences."""
        self.logger.info(f"Creating attention masks for {len(sequences)} sequences")
        
        attention_masks = []
        
        for sequence in sequences:
            attention_mask = [1 if token_id != pad_token_id else 0 for token_id in sequence]
            attention_masks.append(attention_mask)
        
        return attention_masks
    
    def split_long_sequences(self, sequences: List[List[int]], max_length: int = None, 
                           overlap: int = None) -> List[List[List[int]]]:
        """Split long sequences into overlapping chunks."""
        if max_length is None:
            max_length = self.config.max_length
        if overlap is None:
            overlap = self.config.overlap
        
        self.logger.info(f"Splitting {len(sequences)} sequences with max_length={max_length}, overlap={overlap}")
        
        split_sequences = []
        
        for sequence in sequences:
            if len(sequence) <= max_length:
                split_sequences.append([sequence])
            else:
                chunks = []
                start = 0
                
                while start < len(sequence):
                    end = min(start + max_length, len(sequence))
                    chunk = sequence[start:end]
                    chunks.append(chunk)
                    start = end - overlap
                
                split_sequences.append(chunks)
        
        return split_sequences


class TokenizationAnalyzer:
    """Analyzer for tokenization and sequence handling."""
    
    def __init__(self) -> Any:
        self.analysis_results = {}
    
    def analyze_tokenization(self, tokenizer: AdvancedTokenizer, 
                           sample_texts: List[str]) -> Dict[str, Any]:
        """Analyze tokenization performance and characteristics."""
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Analyzing tokenization for {len(sample_texts)} sample texts")
        
        analysis = {
            'tokenization_stats': {},
            'sequence_lengths': [],
            'vocabulary_usage': {},
            'special_tokens_usage': {},
            'performance_metrics': {}
        }
        
        # Tokenize sample texts
        start_time = time.time()
        encodings = []
        
        for text in sample_texts:
            encoding = tokenizer.tokenize_text(text)
            encodings.append(encoding)
        
        tokenization_time = time.time() - start_time
        
        # Analyze sequence lengths
        sequence_lengths = []
        for encoding in encodings:
            if 'input_ids' in encoding:
                sequence_lengths.append(len(encoding['input_ids']))
        
        analysis['sequence_lengths'] = {
            'mean': np.mean(sequence_lengths),
            'std': np.std(sequence_lengths),
            'min': np.min(sequence_lengths),
            'max': np.max(sequence_lengths),
            'median': np.median(sequence_lengths)
        }
        
        # Analyze vocabulary usage
        all_tokens = []
        for encoding in encodings:
            if 'input_ids' in encoding:
                all_tokens.extend(encoding['input_ids'])
        
        unique_tokens, token_counts = np.unique(all_tokens, return_counts=True)
        analysis['vocabulary_usage'] = {
            'unique_tokens': len(unique_tokens),
            'total_tokens': len(all_tokens),
            'vocabulary_coverage': len(unique_tokens) / tokenizer.tokenizer.vocab_size,
            'most_common_tokens': list(zip(unique_tokens[:10], token_counts[:10]))
        }
        
        # Analyze special tokens usage
        special_tokens = {
            'pad': tokenizer.tokenizer.pad_token_id,
            'eos': tokenizer.tokenizer.eos_token_id,
            'bos': tokenizer.tokenizer.bos_token_id,
            'unk': tokenizer.tokenizer.unk_token_id,
            'cls': tokenizer.tokenizer.cls_token_id,
            'sep': tokenizer.tokenizer.sep_token_id,
            'mask': tokenizer.tokenizer.mask_token_id
        }
        
        special_token_counts = {}
        for token_name, token_id in special_tokens.items():
            if token_id is not None:
                special_token_counts[token_name] = all_tokens.count(token_id)
        
        analysis['special_tokens_usage'] = special_token_counts
        
        # Performance metrics
        analysis['performance_metrics'] = {
            'tokenization_time': tokenization_time,
            'tokens_per_second': len(all_tokens) / tokenization_time,
            'texts_per_second': len(sample_texts) / tokenization_time
        }
        
        # Get tokenization statistics
        analysis['tokenization_stats'] = tokenizer.get_tokenization_statistics()
        
        return analysis
    
    def benchmark_tokenization(self, tokenizer: AdvancedTokenizer, 
                             sample_texts: List[str], num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark tokenization performance."""
        self.logger.info(f"Benchmarking tokenization with {len(sample_texts)} texts, {num_runs} runs")
        
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Tokenize all texts
            for text in sample_texts:
                encoding = tokenizer.tokenize_text(text)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
        
        return {
            'tokenization_time': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            },
            'memory_usage_mb': {
                'mean': np.mean(memory_usage) if memory_usage else 0,
                'max': np.max(memory_usage) if memory_usage else 0
            },
            'texts_per_second': len(sample_texts) / np.mean(times)
        }


def demonstrate_tokenization_sequence():
    """Demonstrate tokenization and sequence handling capabilities."""
    print("Tokenization and Sequence Handling Demonstration")
    print("=" * 55)
    
    # Sample texts for testing
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning algorithms are transforming industries worldwide.",
        "Natural language processing enables human-computer interaction.",
        "Deep learning models can process vast amounts of data efficiently.",
        "Transformers have revolutionized natural language processing."
    ]
    
    # Test different configurations
    configs = [
        TokenizationConfig(
            model_name="gpt2",
            max_length=128,
            sequence_handling_type=SequenceHandlingType.PADDING
        ),
        TokenizationConfig(
            model_name="bert-base-uncased",
            max_length=256,
            sequence_handling_type=SequenceHandlingType.TRUNCATION
        ),
        TokenizationConfig(
            model_name="distilbert-base-uncased",
            max_length=512,
            sequence_handling_type=SequenceHandlingType.SLIDING_WINDOW
        )
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nTesting {config.model_name} tokenizer:")
        
        try:
            # Create tokenizer
            tokenizer = AdvancedTokenizer(config)
            
            # Get vocabulary info
            vocab_info = tokenizer.get_vocabulary_info()
            print(f"  Vocab size: {vocab_info['vocab_size']:,}")
            print(f"  Model max length: {vocab_info['model_max_length']}")
            
            # Test tokenization
            encodings = tokenizer.tokenize_batch(sample_texts)
            print(f"  Tokenized {len(sample_texts)} texts")
            print(f"  Input IDs shape: {len(encodings['input_ids'])} x {len(encodings['input_ids'][0])}")
            
            # Test decoding
            decoded_texts = tokenizer.decode_batch(encodings['input_ids'])
            print(f"  Decoded {len(decoded_texts)} texts")
            
            # Test sliding windows
            if config.sequence_handling_type == SequenceHandlingType.SLIDING_WINDOW:
                long_text = " ".join(sample_texts * 10)  # Create long text
                windows = tokenizer.create_sliding_windows(long_text, window_size=128, stride=64)
                print(f"  Created {len(windows)} sliding windows")
            
            # Analyze tokenization
            analyzer = TokenizationAnalyzer()
            analysis = analyzer.analyze_tokenization(tokenizer, sample_texts)
            print(f"  Average sequence length: {analysis['sequence_lengths']['mean']:.2f}")
            print(f"  Vocabulary coverage: {analysis['vocabulary_usage']['vocabulary_coverage']:.2%}")
            
            # Benchmark performance
            benchmark = analyzer.benchmark_tokenization(tokenizer, sample_texts)
            print(f"  Tokenization speed: {benchmark['texts_per_second']:.2f} texts/s")
            
            results[f"tokenizer_{i}"] = {
                'config': config,
                'vocab_info': vocab_info,
                'analysis': analysis,
                'benchmark': benchmark,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"tokenizer_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    return results


if __name__ == "__main__":
    # Demonstrate tokenization and sequence handling
    results = demonstrate_tokenization_sequence()
    print("\nTokenization and sequence handling demonstration completed!") 