"""
Advanced Tokenization and Sequence Handling System for Text Data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast,
    AutoTokenizer, BertTokenizer, GPT2Tokenizer, T5Tokenizer,
    DataCollatorForLanguageModeling, DataCollatorForSeq2SeqLM,
    DataCollatorWithPadding, DataCollatorForTokenClassification
)
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
import numpy as np
import re
import json
import logging
from collections import defaultdict, Counter
import warnings


@dataclass
class TokenizationConfig:
    """Configuration for tokenization and sequence handling."""
    # Model and tokenizer settings
    model_name: str = "gpt2"
    tokenizer_type: str = "auto"  # auto, bert, gpt2, t5, custom
    use_fast_tokenizer: bool = True
    
    # Text processing
    max_length: int = 512
    truncation: bool = True
    padding: str = "longest"  # longest, max_length, do_not_pad
    return_tensors: str = "pt"  # pt, tf, np
    
    # Special tokens
    add_special_tokens: bool = True
    add_bos_token: bool = True
    add_eos_token: bool = True
    add_sep_token: bool = True
    add_cls_token: bool = True
    
    # Advanced features
    return_attention_mask: bool = True
    return_token_type_ids: bool = True
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    
    # Sequence handling
    stride: int = 0
    return_length: bool = False
    verbose: bool = True


@dataclass
class SequenceConfig:
    """Configuration for sequence processing and handling."""
    # Sequence properties
    max_sequence_length: int = 1024
    min_sequence_length: int = 1
    target_sequence_length: int = 512
    
    # Padding and truncation
    padding_strategy: str = "longest"  # longest, max_length, do_not_pad
    truncation_strategy: str = "longest_first"  # longest_first, only_first, only_second, do_not_truncate
    
    # Batching
    batch_size: int = 32
    dynamic_padding: bool = True
    pad_to_multiple_of: Optional[int] = None
    
    # Special handling
    handle_long_sequences: bool = True
    sliding_window: bool = False
    window_size: int = 512
    window_stride: int = 256
    
    # Data augmentation
    enable_augmentation: bool = False
    augmentation_methods: List[str] = None
    
    def __post_init__(self):
        if self.augmentation_methods is None:
            self.augmentation_methods = ["random_mask", "random_insert", "random_swap"]


class AdvancedTokenizer:
    """Advanced tokenizer with comprehensive text processing capabilities."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizer = self._load_tokenizer()
        self._setup_special_tokens()
        self._setup_logging()
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure tokenizer."""
        try:
            if self.config.tokenizer_type == "auto":
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    use_fast=self.config.use_fast_tokenizer
                )
            elif self.config.tokenizer_type == "bert":
                tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
            elif self.config.tokenizer_type == "gpt2":
                tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
            elif self.config.tokenizer_type == "t5":
                tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            else:
                raise ValueError(f"Unknown tokenizer type: {self.config.tokenizer_type}")
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
            
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise
    
    def _setup_special_tokens(self):
        """Setup special tokens for the tokenizer."""
        # Add missing special tokens
        special_tokens = {}
        
        if self.config.add_bos_token and self.tokenizer.bos_token is None:
            special_tokens['bos_token'] = '<s>'
        
        if self.config.add_eos_token and self.tokenizer.eos_token is None:
            special_tokens['eos_token'] = '</s>'
        
        if self.config.add_sep_token and self.tokenizer.sep_token is None:
            special_tokens['sep_token'] = '[SEP]'
        
        if self.config.add_cls_token and self.tokenizer.cls_token is None:
            special_tokens['cls_token'] = '[CLS]'
        
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
            print(f"✅ Added special tokens: {list(special_tokens.keys())}")
    
    def _setup_logging(self):
        """Setup logging for tokenization."""
        self.logger = logging.getLogger(__name__)
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)
    
    def tokenize_text(
        self, 
        text: Union[str, List[str]], 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with advanced options."""
        # Override config with kwargs
        tokenization_kwargs = {
            'max_length': self.config.max_length,
            'truncation': self.config.truncation,
            'padding': self.config.padding,
            'return_tensors': self.config.return_tensors,
            'add_special_tokens': self.config.add_special_tokens,
            'return_attention_mask': self.config.return_attention_mask,
            'return_token_type_ids': self.config.return_token_type_ids,
            'return_overflowing_tokens': self.config.return_overflowing_tokens,
            'return_special_tokens_mask': self.config.return_special_tokens_mask,
            'return_offsets_mapping': self.config.return_offsets_mapping,
            'stride': self.config.stride,
            'return_length': self.config.return_length,
            **kwargs
        }
        
        # Handle single text vs list of texts
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        try:
            tokenized = self.tokenizer(
                text,
                **tokenization_kwargs
            )
            
            if self.config.verbose:
                self.logger.info(f"Tokenized {len(text)} texts with max length {self.config.max_length}")
            
            return tokenized
            
        except Exception as e:
            self.logger.error(f"Tokenization error: {e}")
            raise
    
    def tokenize_with_metadata(
        self, 
        text: str, 
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """Tokenize text with additional metadata."""
        # Basic tokenization
        tokenized = self.tokenize_text(text)
        
        if return_metadata:
            # Add metadata
            metadata = {
                'original_text': text,
                'text_length': len(text),
                'token_count': len(tokenized['input_ids'][0]) if 'input_ids' in tokenized else 0,
                'vocabulary_size': self.tokenizer.vocab_size,
                'special_tokens': {
                    'pad_token': self.tokenizer.pad_token,
                    'bos_token': self.tokenizer.bos_token,
                    'eos_token': self.tokenizer.eos_token,
                    'unk_token': self.tokenizer.unk_token,
                    'sep_token': self.tokenizer.sep_token,
                    'cls_token': self.tokenizer.cls_token
                }
            }
            
            tokenized['metadata'] = metadata
        
        return tokenized
    
    def batch_tokenize(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """Tokenize texts in batches for memory efficiency."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_result = self.tokenize_text(batch_texts)
            results.append(batch_result)
            
            if self.config.verbose:
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return results
    
    def decode_tokens(
        self, 
        token_ids: Union[List[int], torch.Tensor], 
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def get_token_statistics(self, text: str) -> Dict[str, Any]:
        """Get comprehensive token statistics for text."""
        tokenized = self.tokenize_text(text)
        
        # Count tokens
        token_ids = tokenized['input_ids'][0] if 'input_ids' in tokenized else []
        token_counts = Counter(token_ids)
        
        # Calculate statistics
        stats = {
            'total_tokens': len(token_ids),
            'unique_tokens': len(token_counts),
            'vocabulary_coverage': len(token_counts) / self.tokenizer.vocab_size,
            'most_common_tokens': token_counts.most_common(10),
            'token_distribution': dict(token_counts),
            'average_token_frequency': np.mean(list(token_counts.values())),
            'token_frequency_std': np.std(list(token_counts.values()))
        }
        
        return stats


class SequenceProcessor:
    """Advanced sequence processing and handling."""
    
    def __init__(self, config: SequenceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_sequences(
        self, 
        sequences: List[torch.Tensor], 
        target_length: Optional[int] = None
    ) -> torch.Tensor:
        """Process sequences with padding and truncation."""
        if target_length is None:
            target_length = self.config.target_sequence_length
        
        processed_sequences = []
        
        for seq in sequences:
            # Truncate if too long
            if len(seq) > target_length:
                seq = self._truncate_sequence(seq, target_length)
            
            # Pad if too short
            if len(seq) < target_length:
                seq = self._pad_sequence(seq, target_length)
            
            processed_sequences.append(seq)
        
        return torch.stack(processed_sequences)
    
    def _truncate_sequence(
        self, 
        sequence: torch.Tensor, 
        target_length: int
    ) -> torch.Tensor:
        """Truncate sequence to target length."""
        if self.config.truncation_strategy == "longest_first":
            # Keep the beginning and end
            half_length = target_length // 2
            if target_length % 2 == 0:
                return torch.cat([sequence[:half_length], sequence[-half_length:]])
            else:
                return torch.cat([sequence[:half_length], sequence[-(half_length + 1):]])
        
        elif self.config.truncation_strategy == "only_first":
            return sequence[:target_length]
        
        elif self.config.truncation_strategy == "only_second":
            return sequence[-target_length:]
        
        else:  # do_not_truncate
            return sequence
    
    def _pad_sequence(
        self, 
        sequence: torch.Tensor, 
        target_length: int
    ) -> torch.Tensor:
        """Pad sequence to target length."""
        padding_length = target_length - len(sequence)
        
        if self.config.padding_strategy == "max_length":
            # Pad to exact length
            padding = torch.zeros(padding_length, dtype=sequence.dtype, device=sequence.device)
            return torch.cat([sequence, padding])
        
        elif self.config.padding_strategy == "longest":
            # Pad to longest sequence in batch
            padding = torch.zeros(padding_length, dtype=sequence.dtype, device=sequence.device)
            return torch.cat([sequence, padding])
        
        else:  # do_not_pad
            return sequence
    
    def create_sliding_windows(
        self, 
        sequence: torch.Tensor, 
        window_size: Optional[int] = None,
        stride: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Create sliding windows for long sequences."""
        if window_size is None:
            window_size = self.config.window_size
        if stride is None:
            stride = self.config.window_stride
        
        windows = []
        for i in range(0, len(sequence) - window_size + 1, stride):
            window = sequence[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def handle_long_sequences(
        self, 
        sequences: List[torch.Tensor], 
        max_length: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Handle sequences that exceed maximum length."""
        if max_length is None:
            max_length = self.config.max_sequence_length
        
        processed_sequences = []
        
        for seq in sequences:
            if len(seq) <= max_length:
                processed_sequences.append(seq)
            else:
                if self.config.sliding_window:
                    # Use sliding window approach
                    windows = self.create_sliding_windows(seq, max_length)
                    processed_sequences.extend(windows)
                else:
                    # Simple truncation
                    truncated = seq[:max_length]
                    processed_sequences.append(truncated)
        
        return processed_sequences
    
    def create_attention_masks(
        self, 
        sequences: torch.Tensor, 
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """Create attention masks for sequences."""
        attention_masks = (sequences != pad_token_id).long()
        return attention_masks
    
    def create_token_type_ids(
        self, 
        sequences: torch.Tensor, 
        sep_token_id: int
    ) -> torch.Tensor:
        """Create token type IDs for sequence pairs."""
        batch_size, seq_len = sequences.shape
        token_type_ids = torch.zeros_like(sequences)
        
        for i in range(batch_size):
            # Find separator token positions
            sep_positions = (sequences[i] == sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) > 0:
                # Mark tokens after first separator as type 1
                first_sep = sep_positions[0]
                token_type_ids[i, first_sep + 1:] = 1
        
        return token_type_ids


class TextPreprocessor:
    """Advanced text preprocessing utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_text(
        self, 
        text: str, 
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_phone_numbers: bool = True,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False
    ) -> str:
        """Clean and normalize text."""
        cleaned_text = text
        
        if remove_urls:
            cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
        
        if remove_emails:
            cleaned_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned_text)
        
        if remove_phone_numbers:
            cleaned_text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', cleaned_text)
        
        if normalize_whitespace:
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if remove_special_chars:
            cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        
        return cleaned_text
    
    def normalize_text(
        self, 
        text: str, 
        lowercase: bool = True,
        remove_accents: bool = False,
        normalize_unicode: bool = True
    ) -> str:
        """Normalize text for consistent processing."""
        normalized_text = text
        
        if lowercase:
            normalized_text = normalized_text.lower()
        
        if remove_accents:
            # Simple accent removal (can be enhanced with unidecode)
            import unicodedata
            normalized_text = ''.join(
                c for c in unicodedata.normalize('NFD', normalized_text)
                if not unicodedata.combining(c)
            )
        
        if normalize_unicode:
            normalized_text = unicodedata.normalize('NFC', normalized_text)
        
        return normalized_text
    
    def split_into_chunks(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 100,
        split_by: str = "sentences"  # sentences, words, characters
    ) -> List[str]:
        """Split long text into overlapping chunks."""
        if split_by == "sentences":
            # Split by sentence boundaries
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        elif split_by == "words":
            # Split by words
            sentences = text.split()
        else:  # characters
            sentences = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            return sentences
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if split_by == "words":
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            else:
                test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with overlap
                if split_by == "words":
                    words = current_chunk.split()
                    overlap_words = words[-overlap:] if len(words) >= overlap else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def create_text_augmentations(
        self, 
        text: str, 
        methods: List[str] = None
    ) -> List[str]:
        """Create text augmentations for data augmentation."""
        if methods is None:
            methods = ["random_mask", "random_insert", "random_swap"]
        
        augmented_texts = [text]
        
        for method in methods:
            if method == "random_mask":
                augmented = self._random_mask(text, mask_ratio=0.1)
                augmented_texts.append(augmented)
            
            elif method == "random_insert":
                augmented = self._random_insert(text, insert_ratio=0.1)
                augmented_texts.append(augmented)
            
            elif method == "random_swap":
                augmented = self._random_swap(text, swap_ratio=0.1)
                augmented_texts.append(augmented)
        
        return augmented_texts
    
    def _random_mask(self, text: str, mask_ratio: float = 0.1) -> str:
        """Randomly mask tokens in text."""
        words = text.split()
        num_masks = max(1, int(len(words) * mask_ratio))
        
        # Randomly select positions to mask
        mask_positions = np.random.choice(len(words), num_masks, replace=False)
        
        for pos in mask_positions:
            words[pos] = "[MASK]"
        
        return " ".join(words)
    
    def _random_insert(self, text: str, insert_ratio: float = 0.1) -> str:
        """Randomly insert tokens in text."""
        words = text.split()
        num_inserts = max(1, int(len(words) * insert_ratio))
        
        # Random insertion tokens
        insert_tokens = ["[INSERT]", "random", "token"]
        
        for _ in range(num_inserts):
            pos = np.random.randint(0, len(words) + 1)
            token = np.random.choice(insert_tokens)
            words.insert(pos, token)
        
        return " ".join(words)
    
    def _random_swap(self, text: str, swap_ratio: float = 0.1) -> str:
        """Randomly swap adjacent tokens in text."""
        words = text.split()
        num_swaps = max(1, int(len(words) * swap_ratio))
        
        for _ in range(num_swaps):
            if len(words) < 2:
                break
            
            pos = np.random.randint(0, len(words) - 1)
            words[pos], words[pos + 1] = words[pos + 1], words[pos]
        
        return " ".join(words)


class DataCollatorFactory:
    """Factory for creating appropriate data collators."""
    
    @staticmethod
    def create_collator(
        task_type: str = "language_modeling",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs
    ) -> Any:
        """Create appropriate data collator for the task."""
        if task_type == "language_modeling":
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Set to True for masked language modeling
                **kwargs
            )
        
        elif task_type == "sequence_to_sequence":
            return DataCollatorForSeq2SeqLM(
                tokenizer=tokenizer,
                **kwargs
            )
        
        elif task_type == "token_classification":
            return DataCollatorForTokenClassification(
                tokenizer=tokenizer,
                **kwargs
            )
        
        elif task_type == "sequence_classification":
            return DataCollatorWithPadding(
                tokenizer=tokenizer,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")


class AdvancedTextProcessor:
    """Main class combining all text processing capabilities."""
    
    def __init__(
        self,
        tokenization_config: TokenizationConfig,
        sequence_config: SequenceConfig
    ):
        self.tokenization_config = tokenization_config
        self.sequence_config = sequence_config
        
        # Initialize components
        self.tokenizer = AdvancedTokenizer(tokenization_config)
        self.sequence_processor = SequenceProcessor(sequence_config)
        self.text_preprocessor = TextPreprocessor()
        
        # Set config for text preprocessor
        self.text_preprocessor.config = sequence_config
    
    def process_text(
        self, 
        text: str, 
        clean_text: bool = True,
        normalize_text: bool = True,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """Complete text processing pipeline."""
        processed_text = text
        
        # Text preprocessing
        if clean_text:
            processed_text = self.text_preprocessor.clean_text(processed_text)
        
        if normalize_text:
            processed_text = self.text_preprocessor.normalize_text(processed_text)
        
        # Tokenization
        tokenized = self.tokenizer.tokenize_with_metadata(processed_text, return_metadata)
        
        # Sequence processing
        if 'input_ids' in tokenized:
            sequences = [torch.tensor(tokenized['input_ids'][0])]
            processed_sequences = self.sequence_processor.process_sequences(sequences)
            tokenized['processed_input_ids'] = processed_sequences
        
        # Add preprocessing info
        if return_metadata:
            tokenized['preprocessing'] = {
                'original_text': text,
                'cleaned_text': processed_text,
                'cleaning_applied': clean_text,
                'normalization_applied': normalize_text
            }
        
        return tokenized
    
    def process_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple texts in batches."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = [self.process_text(text, **kwargs) for text in batch_texts]
            results.extend(batch_results)
        
        return results
    
    def create_dataset_ready_batch(
        self, 
        texts: List[str], 
        task_type: str = "language_modeling"
    ) -> Dict[str, torch.Tensor]:
        """Create a batch ready for dataset creation."""
        # Process all texts
        processed = self.process_batch(texts)
        
        # Extract input IDs
        input_ids = [p['input_ids'][0] for p in processed]
        
        # Process sequences
        processed_sequences = self.sequence_processor.process_sequences(
            [torch.tensor(ids) for ids in input_ids]
        )
        
        # Create attention masks
        attention_masks = self.sequence_processor.create_attention_masks(processed_sequences)
        
        # Create batch
        batch = {
            'input_ids': processed_sequences,
            'attention_mask': attention_masks
        }
        
        # Add token type IDs if needed
        if self.tokenization_config.return_token_type_ids:
            token_type_ids = self.sequence_processor.create_token_type_ids(
                processed_sequences,
                self.tokenizer.tokenizer.sep_token_id or 0
            )
            batch['token_type_ids'] = token_type_ids
        
        return batch
    
    def get_processing_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Get comprehensive statistics about text processing."""
        stats = {
            'total_texts': len(texts),
            'total_characters': sum(len(text) for text in texts),
            'total_words': sum(len(text.split()) for text in texts),
            'average_text_length': np.mean([len(text) for text in texts]),
            'text_length_std': np.std([len(text) for text in texts]),
            'tokenization_stats': []
        }
        
        # Process each text and collect tokenization stats
        for text in texts:
            tokenized = self.tokenizer.tokenize_with_metadata(text)
            token_stats = self.tokenizer.get_token_statistics(text)
            stats['tokenization_stats'].append(token_stats)
        
        # Aggregate tokenization statistics
        if stats['tokenization_stats']:
            total_tokens = sum(s['total_tokens'] for s in stats['tokenization_stats'])
            unique_tokens = len(set().union(*[s['token_distribution'].keys() for s in stats['tokenization_stats']]))
            
            stats['aggregate_tokens'] = {
                'total_tokens': total_tokens,
                'unique_tokens': unique_tokens,
                'average_tokens_per_text': total_tokens / len(texts),
                'vocabulary_coverage': unique_tokens / self.tokenizer.tokenizer.vocab_size
            }
        
        return stats


# Usage example
def main():
    """Main function demonstrating the advanced tokenization system."""
    
    # Configuration
    tokenization_config = TokenizationConfig(
        model_name="gpt2",
        max_length=512,
        padding="longest",
        return_attention_mask=True,
        return_token_type_ids=True
    )
    
    sequence_config = SequenceConfig(
        max_sequence_length=1024,
        target_sequence_length=512,
        padding_strategy="longest",
        truncation_strategy="longest_first",
        handle_long_sequences=True,
        sliding_window=True
    )
    
    # Create processor
    processor = AdvancedTextProcessor(tokenization_config, sequence_config)
    
    # Sample texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample text for testing tokenization.",
        "Artificial intelligence is transforming the world. Machine learning models are becoming more sophisticated.",
        "Natural language processing enables computers to understand human language. It's a fascinating field."
    ]
    
    print("🚀 Advanced Tokenization & Sequence Handling System")
    print("=" * 60)
    
    # Process individual text
    print("\n📝 Processing individual text:")
    result = processor.process_text(sample_texts[0], return_metadata=True)
    print(f"   Original length: {len(sample_texts[0])} characters")
    print(f"   Tokenized length: {len(result['input_ids'][0])} tokens")
    print(f"   Vocabulary coverage: {result['metadata']['vocabulary_coverage']:.2%}")
    
    # Process batch
    print("\n📦 Processing batch:")
    batch_results = processor.process_batch(sample_texts, batch_size=2)
    print(f"   Processed {len(batch_results)} texts")
    
    # Create dataset-ready batch
    print("\n🎯 Creating dataset-ready batch:")
    dataset_batch = processor.create_dataset_ready_batch(sample_texts)
    print(f"   Batch shape: {dataset_batch['input_ids'].shape}")
    print(f"   Attention mask shape: {dataset_batch['attention_mask'].shape}")
    
    # Get statistics
    print("\n📊 Processing statistics:")
    stats = processor.get_processing_statistics(sample_texts)
    print(f"   Total texts: {stats['total_texts']}")
    print(f"   Total characters: {stats['total_characters']:,}")
    print(f"   Total tokens: {stats['aggregate_tokens']['total_tokens']:,}")
    print(f"   Average tokens per text: {stats['aggregate_tokens']['average_tokens_per_text']:.1f}")
    
    print("\n✅ Advanced Tokenization System Ready!")


if __name__ == "__main__":
    main()






