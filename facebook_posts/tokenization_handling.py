from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling, DataCollatorWithPadding,
    BatchEncoding, Encoding
)
import numpy as np
import json
import logging
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import time
from enum import Enum
import re
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Tokenization and Sequence Handling
Comprehensive tokenization and sequence handling for text data with advanced features.
"""

    AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling, DataCollatorWithPadding,
    BatchEncoding, Encoding
)


class TokenizationType(Enum):
    """Types of tokenization approaches."""
    WORD_LEVEL = "word_level"
    SUBWORD = "subword"
    CHARACTER_LEVEL = "character_level"
    BYTE_PAIR = "byte_pair"
    SENTENCE_PIECE = "sentence_piece"
    WORD_PIECE = "word_piece"


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
    tokenization_type: TokenizationType = TokenizationType.SUBWORD
    model_name: str = "gpt2"
    vocab_size: int = 50000
    max_length: int = 512
    min_length: int = 1
    
    # Sequence handling
    sequence_handling_type: SequenceHandlingType = SequenceHandlingType.PADDING
    padding_side: str = "right"
    truncation_side: str = "right"
    stride: int = 128
    overlap: int = 64
    
    # Advanced features
    use_fast_tokenizer: bool = True
    return_overflowing_tokens: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = True
    
    # Special tokens
    pad_token: str = None
    unk_token: str = None
    bos_token: str = None
    eos_token: str = None
    mask_token: str = None


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
            'unique_tokens': set(),
            'sequence_lengths': [],
            'vocabulary_usage': {}
        }
    
    def _setup_logging(self) -> Any:
        """Setup comprehensive logging."""
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
        """Load pre-trained tokenizer."""
        self.logger.info(f"Loading tokenizer for {self.config.model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=self.config.use_fast_tokenizer,
                padding_side=self.config.padding_side,
                truncation_side=self.config.truncation_side
            )
            
            # Set special tokens if not present
            if tokenizer.pad_token is None and self.config.pad_token:
                tokenizer.pad_token = self.config.pad_token
            elif tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            if tokenizer.unk_token is None and self.config.unk_token:
                tokenizer.unk_token = self.config.unk_token
            
            if tokenizer.bos_token is None and self.config.bos_token:
                tokenizer.bos_token = self.config.bos_token
            
            if tokenizer.eos_token is None and self.config.eos_token:
                tokenizer.eos_token = self.config.eos_token
            
            if tokenizer.mask_token is None and self.config.mask_token:
                tokenizer.mask_token = self.config.mask_token
            
            self.logger.info(f"Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
            return tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def tokenize_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Tokenize text with advanced features."""
        self.logger.info(f"Tokenizing text: {text[:50]}...")
        
        # Update tokenization parameters
        tokenization_kwargs = {
            'max_length': self.config.max_length,
            'min_length': self.config.min_length,
            'padding': 'max_length' if self.config.sequence_handling_type == SequenceHandlingType.PADDING else False,
            'truncation': True if self.config.sequence_handling_type == SequenceHandlingType.TRUNCATION else False,
            'return_tensors': 'pt',
            'return_overflowing_tokens': self.config.return_overflowing_tokens,
            'return_offsets_mapping': self.config.return_offsets_mapping,
            'return_length': self.config.return_length,
            'stride': self.config.stride if self.config.sequence_handling_type == SequenceHandlingType.STRIDE else None
        }
        tokenization_kwargs.update(kwargs)
        
        try:
            # Tokenize text
            if self.config.sequence_handling_type == SequenceHandlingType.SLIDING_WINDOW:
                result = self._sliding_window_tokenization(text, **tokenization_kwargs)
            else:
                result = self.tokenizer(text, **tokenization_kwargs)
            
            # Update statistics
            self._update_tokenization_stats(result)
            
            self.logger.info(f"Tokenization completed. Input length: {len(text)}, Output tokens: {len(result['input_ids'][0])}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during tokenization: {e}")
            raise
    
    def _sliding_window_tokenization(self, text: str, **kwargs) -> Dict[str, Any]:
        """Tokenize text using sliding window approach."""
        # Split text into overlapping chunks
        chunks = self._create_sliding_windows(text)
        
        # Tokenize each chunk
        all_input_ids = []
        all_attention_masks = []
        
        for chunk in chunks:
            chunk_result = self.tokenizer(chunk, **kwargs)
            all_input_ids.append(chunk_result['input_ids'])
            all_attention_masks.append(chunk_result['attention_mask'])
        
        # Combine results
        result = {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0)
        }
        
        return result
    
    def _create_sliding_windows(self, text: str) -> List[str]:
        """Create sliding windows for text."""
        windows = []
        window_size = self.config.max_length
        stride = self.config.stride
        
        # Simple word-based sliding window
        words = text.split()
        
        for i in range(0, len(words), stride):
            window_words = words[i:i + window_size]
            window_text = ' '.join(window_words)
            windows.append(window_text)
        
        return windows
    
    def _update_tokenization_stats(self, result: Dict[str, Any]):
        """Update tokenization statistics."""
        input_ids = result['input_ids']
        
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.flatten().tolist()
        
        # Update statistics
        self.tokenization_stats['total_tokens'] += len(input_ids)
        self.tokenization_stats['unique_tokens'].update(input_ids)
        self.tokenization_stats['sequence_lengths'].append(len(input_ids))
        
        # Update vocabulary usage
        for token_id in input_ids:
            if token_id in self.tokenization_stats['vocabulary_usage']:
                self.tokenization_stats['vocabulary_usage'][token_id] += 1
            else:
                self.tokenization_stats['vocabulary_usage'][token_id] = 1
    
    def batch_tokenize(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Tokenize a batch of texts."""
        self.logger.info(f"Batch tokenizing {len(texts)} texts")
        
        try:
            # Batch tokenization
            result = self.tokenizer(
                texts,
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                padding=True,
                truncation=True,
                return_tensors='pt',
                return_length=self.config.return_length,
                **kwargs
            )
            
            # Update statistics
            for text in texts:
                self._update_tokenization_stats({'input_ids': result['input_ids']})
            
            self.logger.info(f"Batch tokenization completed. Batch size: {len(texts)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during batch tokenization: {e}")
            raise
    
    def decode_tokens(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode token IDs back to text."""
        try:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            decoded_text = self.tokenizer.decode(token_ids, **kwargs)
            
            self.logger.info(f"Decoded {len(token_ids)} tokens to text")
            return decoded_text
            
        except Exception as e:
            self.logger.error(f"Error during decoding: {e}")
            raise
    
    def get_tokenization_stats(self) -> Dict[str, Any]:
        """Get comprehensive tokenization statistics."""
        stats = {
            'total_tokens': self.tokenization_stats['total_tokens'],
            'unique_tokens': len(self.tokenization_stats['unique_tokens']),
            'vocabulary_size': self.tokenizer.vocab_size,
            'average_sequence_length': np.mean(self.tokenization_stats['sequence_lengths']) if self.tokenization_stats['sequence_lengths'] else 0,
            'max_sequence_length': max(self.tokenization_stats['sequence_lengths']) if self.tokenization_stats['sequence_lengths'] else 0,
            'min_sequence_length': min(self.tokenization_stats['sequence_lengths']) if self.tokenization_stats['sequence_lengths'] else 0,
            'vocabulary_usage': dict(sorted(self.tokenization_stats['vocabulary_usage'].items(), key=lambda x: x[1], reverse=True)[:100])
        }
        
        return stats
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity for tokenization."""
        analysis = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'character_count': len(text),
            'average_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'unique_words': len(set(text.split())),
            'vocabulary_diversity': len(set(text.split())) / len(text.split()) if text.split() else 0
        }
        
        # Tokenize and analyze
        tokenization_result = self.tokenize_text(text)
        token_ids = tokenization_result['input_ids'][0].tolist()
        
        analysis.update({
            'token_count': len(token_ids),
            'unique_tokens': len(set(token_ids)),
            'token_diversity': len(set(token_ids)) / len(token_ids) if token_ids else 0,
            'compression_ratio': len(token_ids) / len(text.split()) if text.split() else 0
        })
        
        return analysis


class SequenceHandler:
    """Advanced sequence handling for text data."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizer = None  # Will be set by AdvancedTokenizer
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> Any:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sequence_handling.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Set the tokenizer for sequence handling."""
        self.tokenizer = tokenizer
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for input IDs."""
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        return attention_mask
    
    def create_token_type_ids(self, input_ids: torch.Tensor, num_segments: int = 2) -> torch.Tensor:
        """Create token type IDs for multi-segment inputs."""
        batch_size, seq_len = input_ids.shape
        token_type_ids = torch.zeros_like(input_ids)
        
        for i in range(batch_size):
            # Simple segmentation: first half is segment 0, second half is segment 1
            mid_point = seq_len // 2
            token_type_ids[i, mid_point:] = 1
        
        return token_type_ids
    
    def create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create position IDs for input."""
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        return position_ids
    
    def handle_long_sequences(self, input_ids: torch.Tensor, max_length: int = None) -> List[torch.Tensor]:
        """Handle sequences longer than max_length."""
        if max_length is None:
            max_length = self.config.max_length
        
        if input_ids.size(-1) <= max_length:
            return [input_ids]
        
        # Split long sequences
        sequences = []
        for i in range(0, input_ids.size(-1), max_length):
            sequence = input_ids[..., i:i + max_length]
            sequences.append(sequence)
        
        return sequences
    
    def create_sliding_windows(self, input_ids: torch.Tensor, window_size: int = None, stride: int = None) -> List[torch.Tensor]:
        """Create sliding windows for sequence processing."""
        if window_size is None:
            window_size = self.config.max_length
        if stride is None:
            stride = self.config.stride
        
        seq_len = input_ids.size(-1)
        windows = []
        
        for i in range(0, seq_len - window_size + 1, stride):
            window = input_ids[..., i:i + window_size]
            windows.append(window)
        
        return windows
    
    def pad_sequences(self, sequences: List[torch.Tensor], padding_side: str = None) -> torch.Tensor:
        """Pad sequences to the same length."""
        if padding_side is None:
            padding_side = self.config.padding_side
        
        max_length = max(seq.size(-1) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            if padding_side == "right":
                padding = [0, max_length - seq.size(-1)]
            else:  # left
                padding = [max_length - seq.size(-1), 0]
            
            padded_seq = F.pad(seq, padding, value=self.tokenizer.pad_token_id)
            padded_sequences.append(padded_seq)
        
        return torch.stack(padded_sequences)
    
    def truncate_sequences(self, sequences: List[torch.Tensor], max_length: int = None, truncation_side: str = None) -> List[torch.Tensor]:
        """Truncate sequences to max_length."""
        if max_length is None:
            max_length = self.config.max_length
        if truncation_side is None:
            truncation_side = self.config.truncation_side
        
        truncated_sequences = []
        
        for seq in sequences:
            if seq.size(-1) > max_length:
                if truncation_side == "right":
                    truncated_seq = seq[..., :max_length]
                else:  # left
                    truncated_seq = seq[..., -max_length:]
                truncated_sequences.append(truncated_seq)
            else:
                truncated_sequences.append(seq)
        
        return truncated_sequences
    
    def create_data_collator(self, collator_type: str = "default") -> Callable:
        """Create data collator for batching."""
        if collator_type == "language_modeling":
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        elif collator_type == "padding":
            return DataCollatorWithPadding(tokenizer=self.tokenizer)
        else:
            # Default collator
            def default_collator(batch) -> Any:
                input_ids = [item['input_ids'] for item in batch]
                attention_masks = [item['attention_mask'] for item in batch]
                
                # Pad sequences
                input_ids = self.pad_sequences(input_ids)
                attention_masks = self.pad_sequences(attention_masks)
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_masks
                }
            
            return default_collator


class TokenizationAnalyzer:
    """Analyzer for tokenization and sequence handling."""
    
    def __init__(self) -> Any:
        self.analysis_results = {}
    
    def analyze_tokenization_efficiency(self, tokenizer: AdvancedTokenizer, texts: List[str]) -> Dict[str, Any]:
        """Analyze tokenization efficiency."""
        self.logger.info(f"Analyzing tokenization efficiency for {len(texts)} texts")
        
        results = {
            'text_lengths': [],
            'token_counts': [],
            'compression_ratios': [],
            'vocabulary_usage': {},
            'processing_times': []
        }
        
        for text in texts:
            start_time = time.time()
            
            # Analyze text complexity
            complexity = tokenizer.analyze_text_complexity(text)
            
            # Tokenize text
            tokenization_result = tokenizer.tokenize_text(text)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Collect results
            results['text_lengths'].append(complexity['text_length'])
            results['token_counts'].append(complexity['token_count'])
            results['compression_ratios'].append(complexity['compression_ratio'])
            results['processing_times'].append(processing_time)
            
            # Update vocabulary usage
            token_ids = tokenization_result['input_ids'][0].tolist()
            for token_id in token_ids:
                if token_id in results['vocabulary_usage']:
                    results['vocabulary_usage'][token_id] += 1
                else:
                    results['vocabulary_usage'][token_id] = 1
        
        # Calculate statistics
        results['avg_text_length'] = np.mean(results['text_lengths'])
        results['avg_token_count'] = np.mean(results['token_counts'])
        results['avg_compression_ratio'] = np.mean(results['compression_ratios'])
        results['avg_processing_time'] = np.mean(results['processing_times'])
        results['total_processing_time'] = sum(results['processing_times'])
        
        return results
    
    def benchmark_tokenizers(self, tokenizers: List[AdvancedTokenizer], texts: List[str]) -> Dict[str, Any]:
        """Benchmark different tokenizers."""
        self.logger.info(f"Benchmarking {len(tokenizers)} tokenizers with {len(texts)} texts")
        
        results = {}
        
        for i, tokenizer in enumerate(tokenizers):
            tokenizer_name = f"tokenizer_{i}"
            
            try:
                # Analyze efficiency
                efficiency_results = self.analyze_tokenization_efficiency(tokenizer, texts)
                
                # Get tokenization stats
                stats = tokenizer.get_tokenization_stats()
                
                results[tokenizer_name] = {
                    'efficiency': efficiency_results,
                    'stats': stats,
                    'success': True
                }
                
                self.logger.info(f"{tokenizer_name}: Avg processing time: {efficiency_results['avg_processing_time']:.4f}s")
                
            except Exception as e:
                self.logger.error(f"Error benchmarking {tokenizer_name}: {e}")
                results[tokenizer_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return results


def demonstrate_tokenization_handling():
    """Demonstrate tokenization and sequence handling capabilities."""
    print("Tokenization and Sequence Handling Demonstration")
    print("=" * 55)
    
    # Sample texts
    sample_texts = [
        "The future of artificial intelligence is bright and promising.",
        "Machine learning algorithms are transforming industries worldwide.",
        "Deep learning models can process vast amounts of data efficiently.",
        "Natural language processing enables human-computer interaction.",
        "Computer vision systems can recognize objects in images.",
        "Reinforcement learning agents learn through trial and error.",
        "Neural networks mimic the human brain's structure and function.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Transfer learning enables models to adapt to new tasks quickly."
    ]
    
    # Test different configurations
    configs = [
        TokenizationConfig(
            model_name="gpt2",
            tokenization_type=TokenizationType.SUBWORD,
            sequence_handling_type=SequenceHandlingType.PADDING,
            max_length=128
        ),
        TokenizationConfig(
            model_name="bert-base-uncased",
            tokenization_type=TokenizationType.SUBWORD,
            sequence_handling_type=SequenceHandlingType.TRUNCATION,
            max_length=256
        ),
        TokenizationConfig(
            model_name="gpt2",
            tokenization_type=TokenizationType.SUBWORD,
            sequence_handling_type=SequenceHandlingType.SLIDING_WINDOW,
            max_length=64,
            stride=32
        )
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nTesting {config.tokenization_type.value} with {config.sequence_handling_type.value}:")
        
        try:
            # Create tokenizer
            tokenizer = AdvancedTokenizer(config)
            
            # Create sequence handler
            sequence_handler = SequenceHandler(config)
            sequence_handler.set_tokenizer(tokenizer.tokenizer)
            
            # Test tokenization
            tokenization_results = []
            for text in sample_texts[:3]:  # Test with first 3 texts
                result = tokenizer.tokenize_text(text)
                tokenization_results.append(result)
            
            # Test batch tokenization
            batch_result = tokenizer.batch_tokenize(sample_texts[:3])
            
            # Test sequence handling
            input_ids = batch_result['input_ids']
            attention_masks = sequence_handler.create_attention_mask(input_ids)
            position_ids = sequence_handler.create_position_ids(input_ids)
            
            # Analyze tokenization
            analyzer = TokenizationAnalyzer()
            efficiency_results = analyzer.analyze_tokenization_efficiency(tokenizer, sample_texts[:3])
            
            # Get statistics
            stats = tokenizer.get_tokenization_stats()
            
            print(f"  Vocabulary size: {stats['vocabulary_size']:,}")
            print(f"  Average sequence length: {stats['average_sequence_length']:.1f}")
            print(f"  Average processing time: {efficiency_results['avg_processing_time']:.4f}s")
            print(f"  Batch shape: {batch_result['input_ids'].shape}")
            
            results[f"config_{i}"] = {
                'config': config,
                'stats': stats,
                'efficiency': efficiency_results,
                'batch_shape': batch_result['input_ids'].shape,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"config_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    return results


if __name__ == "__main__":
    # Demonstrate tokenization and sequence handling
    results = demonstrate_tokenization_handling()
    print("\nTokenization and sequence handling demonstration completed!") 