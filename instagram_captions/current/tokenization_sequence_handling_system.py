"""
Tokenization and Sequence Handling System
Comprehensive implementation of proper tokenization and sequence handling for text data
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, GPT2Tokenizer, BERTTokenizer, T5Tokenizer, RobertaTokenizer,
    PreTrainedTokenizer, PreTrainedTokenizerFast, TokenizerFast,
    DataCollatorForLanguageModeling, DataCollatorWithPadding,
    DataCollatorForTokenClassification, DataCollatorForSeq2SeqLM
)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import logging
import time
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.model_selection import train_test_split


@dataclass
class TokenizationConfig:
    """Configuration for tokenization and sequence handling."""
    
    # Tokenizer settings
    tokenizer_name: str = "gpt2"
    tokenizer_type: str = "auto"  # auto, gpt2, bert, t5, roberta, custom
    vocab_size: int = 50257
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"  # max_length, longest, do_not_pad
    return_tensors: str = "pt"  # pt, tf, np
    
    # Sequence handling
    stride: int = 128
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    
    # Special token handling
    add_special_tokens: bool = True
    add_prefix_space: bool = False
    clean_up_tokenization_spaces: bool = True
    
    # Custom tokenizer settings
    use_custom_tokenizer: bool = False
    custom_tokenizer_path: str = ""
    
    # Data processing
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True


class AdvancedTokenizer:
    """Advanced tokenizer with comprehensive features."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizer = None
        self.vocab_size = config.vocab_size
        self.max_length = config.max_length
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Setup special tokens
        self._setup_special_tokens()
        
        logging.info(f"Advanced tokenizer initialized with {config.tokenizer_name}")
    
    def _load_tokenizer(self):
        """Load the specified tokenizer."""
        try:
            if self.config.use_custom_tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.custom_tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token'):
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token_id
            
            logging.info(f"Successfully loaded {self.config.tokenizer_name}")
            
        except Exception as e:
            logging.error(f"Error loading tokenizer: {e}")
            raise
    
    def _setup_special_tokens(self):
        """Setup special tokens for the tokenizer."""
        # Ensure all necessary special tokens are set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.tokenizer.sep_token is None:
            self.tokenizer.sep_token = self.tokenizer.eos_token
        
        if self.tokenizer.cls_token is None:
            self.tokenizer.cls_token = self.tokenizer.bos_token
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text with comprehensive options."""
        return self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens,
            return_special_tokens_mask=self.config.return_special_tokens_mask,
            return_offsets_mapping=self.config.return_offsets_mapping,
            return_length=self.config.return_length,
            stride=self.config.stride,
            return_overflowing_tokens=self.config.return_overflowing_tokens
        )
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors,
            add_special_tokens=self.config.add_special_tokens,
            return_special_tokens_mask=self.config.return_special_tokens_mask,
            return_offsets_mapping=self.config.return_offsets_mapping,
            return_length=self.config.return_length
        )
    
    def decode_tokens(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(self, token_ids_batch: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token IDs back to texts."""
        return self.tokenizer.batch_decode(token_ids_batch, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, Any]:
        """Get special token information."""
        return {
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
            'mask_token_id': self.tokenizer.mask_token_id
        }
    
    def tokenize_with_analysis(self, text: str) -> Dict[str, Any]:
        """Tokenize text with detailed analysis."""
        # Basic tokenization
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        
        # Full encoding
        encoding = self.tokenize_text(text)
        
        # Analysis
        analysis = {
            'original_text': text,
            'tokens': tokens,
            'token_ids': token_ids,
            'num_tokens': len(tokens),
            'num_token_ids': len(token_ids),
            'encoding': encoding,
            'vocab_coverage': len(set(token_ids)) / self.get_vocab_size(),
            'special_tokens_used': sum(1 for tid in token_ids if tid in self.get_special_tokens().values())
        }
        
        return analysis


class CustomTokenizer:
    """Custom tokenizer implementation using HuggingFace tokenizers library."""
    
    def __init__(self, vocab_size: int = 30000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = None
        self._build_tokenizer()
    
    def _build_tokenizer(self):
        """Build a custom tokenizer."""
        # Initialize tokenizer
        self.tokenizer = Tokenizer(models.BPE())
        
        # Set pre-tokenizer
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Set decoder
        self.tokenizer.decoder = decoders.ByteLevel()
        
        # Set post-processor
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # Set trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )
        
        self.tokenizer.train_from_iterator = trainer
    
    def train_on_texts(self, texts: List[str]):
        """Train the tokenizer on a list of texts."""
        # Prepare training data
        def get_training_corpus():
            for text in texts:
                yield text
        
        # Train tokenizer
        self.tokenizer.train_from_iterator(get_training_corpus())
        
        # Save tokenizer
        self.tokenizer.save("custom_tokenizer.json")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)


class SequenceHandler:
    """Handle sequence processing and manipulation."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.max_length = config.max_length
        self.stride = config.stride
    
    def create_sequences(
        self,
        token_ids: List[int],
        max_length: int = None,
        stride: int = None
    ) -> List[List[int]]:
        """Create overlapping sequences from token IDs."""
        if max_length is None:
            max_length = self.max_length
        if stride is None:
            stride = self.stride
        
        sequences = []
        for i in range(0, len(token_ids), stride):
            sequence = token_ids[i:i + max_length]
            if len(sequence) >= max_length // 2:  # Only keep sequences that are at least half full
                sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(
        self,
        sequences: List[List[int]],
        max_length: int = None,
        padding: str = "max_length",
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """Pad sequences to the same length."""
        if max_length is None:
            max_length = self.max_length
        
        padded_sequences = []
        for sequence in sequences:
            if padding == "max_length":
                if len(sequence) > max_length:
                    sequence = sequence[:max_length]
                else:
                    sequence = sequence + [pad_token_id] * (max_length - len(sequence))
            elif padding == "longest":
                # Find the longest sequence length
                max_seq_len = max(len(seq) for seq in sequences)
                sequence = sequence + [pad_token_id] * (max_seq_len - len(sequence))
            
            padded_sequences.append(sequence)
        
        return torch.tensor(padded_sequences)
    
    def create_attention_masks(self, padded_sequences: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """Create attention masks for padded sequences."""
        attention_masks = (padded_sequences != pad_token_id).long()
        return attention_masks
    
    def truncate_sequences(
        self,
        sequences: List[List[int]],
        max_length: int = None,
        strategy: str = "longest_first"
    ) -> List[List[int]]:
        """Truncate sequences to maximum length."""
        if max_length is None:
            max_length = self.max_length
        
        truncated_sequences = []
        for sequence in sequences:
            if len(sequence) > max_length:
                if strategy == "longest_first":
                    # Remove tokens from the middle
                    half_length = max_length // 2
                    sequence = sequence[:half_length] + sequence[-half_length:]
                elif strategy == "truncate_end":
                    sequence = sequence[:max_length]
                elif strategy == "truncate_start":
                    sequence = sequence[-max_length:]
            
            truncated_sequences.append(sequence)
        
        return truncated_sequences
    
    def create_sliding_windows(
        self,
        token_ids: List[int],
        window_size: int = None,
        stride: int = None
    ) -> List[List[int]]:
        """Create sliding windows from token IDs."""
        if window_size is None:
            window_size = self.max_length
        if stride is None:
            stride = self.stride
        
        windows = []
        for i in range(0, len(token_ids) - window_size + 1, stride):
            window = token_ids[i:i + window_size]
            windows.append(window)
        
        return windows


class TokenizationDataset(Dataset):
    """Dataset for tokenized text data."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: AdvancedTokenizer,
        max_length: int = 512,
        task_type: str = "language_modeling"
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.sequence_handler = SequenceHandler(TokenizationConfig(max_length=max_length))
        
        # Tokenize all texts
        self.tokenized_data = self._tokenize_texts()
    
    def _tokenize_texts(self) -> List[Dict[str, Any]]:
        """Tokenize all texts and prepare data."""
        tokenized_data = []
        
        for text in self.texts:
            # Tokenize text
            tokenization = self.tokenizer.tokenize_with_analysis(text)
            
            # Create sequences if needed
            if self.task_type == "language_modeling":
                sequences = self.sequence_handler.create_sequences(
                    tokenization['token_ids'],
                    max_length=self.max_length
                )
                
                for sequence in sequences:
                    tokenized_data.append({
                        'input_ids': sequence,
                        'attention_mask': [1] * len(sequence),
                        'labels': sequence.copy()
                    })
            else:
                # For other tasks, use the full tokenization
                tokenized_data.append({
                    'input_ids': tokenization['token_ids'],
                    'attention_mask': [1] * len(tokenization['token_ids']),
                    'labels': tokenization['token_ids'].copy()
                })
        
        return tokenized_data
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        data = self.tokenized_data[idx]
        
        # Pad sequences
        padded_input_ids = self.sequence_handler.pad_sequences(
            [data['input_ids']], max_length=self.max_length
        )[0]
        
        padded_attention_mask = self.sequence_handler.pad_sequences(
            [data['attention_mask']], max_length=self.max_length
        )[0]
        
        padded_labels = self.sequence_handler.pad_sequences(
            [data['labels']], max_length=self.max_length
        )[0]
        
        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels
        }


class DataCollator:
    """Custom data collator for different tasks."""
    
    def __init__(self, tokenizer: AdvancedTokenizer, task_type: str = "language_modeling"):
        self.tokenizer = tokenizer
        self.task_type = task_type
        
        # Get pad token ID
        special_tokens = self.tokenizer.get_special_tokens()
        self.pad_token_id = special_tokens['pad_token_id']
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of data."""
        # Extract fields
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad sequences
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for i, (ids, mask, label) in enumerate(zip(input_ids, attention_masks, labels)):
            # Pad input_ids
            if len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            padded_input_ids.append(ids)
            
            # Pad attention_mask
            if len(mask) < max_length:
                mask = mask + [0] * (max_length - len(mask))
            padded_attention_masks.append(mask)
            
            # Pad labels
            if len(label) < max_length:
                label = label + [-100] * (max_length - len(label))  # -100 for ignored index
            padded_labels.append(label)
        
        return {
            'input_ids': torch.tensor(padded_input_ids),
            'attention_mask': torch.tensor(padded_attention_masks),
            'labels': torch.tensor(padded_labels)
        }


class TokenizationAnalyzer:
    """Analyze tokenization patterns and statistics."""
    
    def __init__(self, tokenizer: AdvancedTokenizer):
        self.tokenizer = tokenizer
    
    def analyze_text_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text statistics."""
        all_tokens = []
        all_token_ids = []
        token_lengths = []
        vocab_usage = Counter()
        
        for text in texts:
            analysis = self.tokenizer.tokenize_with_analysis(text)
            all_tokens.extend(analysis['tokens'])
            all_token_ids.extend(analysis['token_ids'])
            token_lengths.append(analysis['num_tokens'])
            vocab_usage.update(analysis['token_ids'])
        
        return {
            'total_texts': len(texts),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'vocab_coverage': len(vocab_usage) / self.tokenizer.get_vocab_size(),
            'avg_tokens_per_text': np.mean(token_lengths),
            'std_tokens_per_text': np.std(token_lengths),
            'min_tokens': min(token_lengths),
            'max_tokens': max(token_lengths),
            'most_common_tokens': vocab_usage.most_common(20),
            'token_length_distribution': token_lengths
        }
    
    def visualize_tokenization_stats(self, stats: Dict[str, Any]):
        """Visualize tokenization statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Token length distribution
        axes[0, 0].hist(stats['token_length_distribution'], bins=30, alpha=0.7)
        axes[0, 0].set_title('Token Length Distribution')
        axes[0, 0].set_xlabel('Number of Tokens')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Most common tokens
        tokens, counts = zip(*stats['most_common_tokens'][:10])
        axes[0, 1].bar(range(len(tokens)), counts)
        axes[0, 1].set_title('Most Common Tokens')
        axes[0, 1].set_xlabel('Token Index')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xticks(range(len(tokens)))
        axes[0, 1].set_xticklabels([f"Token {t}" for t in tokens], rotation=45)
        
        # 3. Vocabulary coverage
        axes[1, 0].pie([stats['vocab_coverage'], 1 - stats['vocab_coverage']], 
                      labels=['Used', 'Unused'], autopct='%1.1f%%')
        axes[1, 0].set_title('Vocabulary Coverage')
        
        # 4. Text length statistics
        categories = ['Min', 'Avg', 'Max']
        values = [stats['min_tokens'], stats['avg_tokens_per_text'], stats['max_tokens']]
        axes[1, 1].bar(categories, values)
        axes[1, 1].set_title('Token Length Statistics')
        axes[1, 1].set_ylabel('Number of Tokens')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_tokenization_quality(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze tokenization quality metrics."""
        quality_metrics = {
            'avg_tokens_per_word': [],
            'compression_ratio': [],
            'special_token_ratio': [],
            'unk_token_ratio': []
        }
        
        special_tokens = self.tokenizer.get_special_tokens()
        unk_token_id = special_tokens['unk_token_id']
        
        for text in texts:
            analysis = self.tokenizer.tokenize_with_analysis(text)
            
            # Calculate metrics
            word_count = len(text.split())
            token_count = analysis['num_tokens']
            
            quality_metrics['avg_tokens_per_word'].append(token_count / word_count if word_count > 0 else 0)
            quality_metrics['compression_ratio'].append(len(text) / token_count if token_count > 0 else 0)
            
            # Special token analysis
            special_token_count = sum(1 for tid in analysis['token_ids'] 
                                    if tid in special_tokens.values())
            quality_metrics['special_token_ratio'].append(special_token_count / token_count if token_count > 0 else 0)
            
            # UNK token analysis
            unk_count = analysis['token_ids'].count(unk_token_id)
            quality_metrics['unk_token_ratio'].append(unk_count / token_count if token_count > 0 else 0)
        
        return {
            'avg_tokens_per_word': np.mean(quality_metrics['avg_tokens_per_word']),
            'avg_compression_ratio': np.mean(quality_metrics['compression_ratio']),
            'avg_special_token_ratio': np.mean(quality_metrics['special_token_ratio']),
            'avg_unk_token_ratio': np.mean(quality_metrics['unk_token_ratio']),
            'detailed_metrics': quality_metrics
        }


class SequenceProcessor:
    """Process and manipulate sequences for different tasks."""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.sequence_handler = SequenceHandler(config)
    
    def create_language_modeling_data(
        self,
        texts: List[str],
        tokenizer: AdvancedTokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create data for language modeling tasks."""
        all_sequences = []
        
        for text in texts:
            # Tokenize text
            tokenization = tokenizer.tokenize_with_analysis(text)
            token_ids = tokenization['token_ids']
            
            # Create sequences
            sequences = self.sequence_handler.create_sequences(token_ids)
            all_sequences.extend(sequences)
        
        # Pad sequences
        padded_sequences = self.sequence_handler.pad_sequences(all_sequences)
        
        # Create attention masks
        attention_masks = self.sequence_handler.create_attention_masks(padded_sequences)
        
        # Create labels (shifted by 1 for next token prediction)
        labels = torch.roll(padded_sequences, shifts=-1, dims=1)
        labels[:, -1] = -100  # Ignore index for last position
        
        return padded_sequences, attention_masks, labels
    
    def create_classification_data(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AdvancedTokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create data for classification tasks."""
        tokenized_texts = []
        
        for text in texts:
            tokenization = tokenizer.tokenize_text(text)
            tokenized_texts.append(tokenization['input_ids'].squeeze())
        
        # Pad sequences
        padded_sequences = self.sequence_handler.pad_sequences(tokenized_texts)
        
        # Create attention masks
        attention_masks = self.sequence_handler.create_attention_masks(padded_sequences)
        
        # Convert labels to tensor
        label_tensor = torch.tensor(labels)
        
        return padded_sequences, attention_masks, label_tensor
    
    def create_sequence_to_sequence_data(
        self,
        source_texts: List[str],
        target_texts: List[str],
        tokenizer: AdvancedTokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create data for sequence-to-sequence tasks."""
        source_encodings = []
        target_encodings = []
        
        for source, target in zip(source_texts, target_texts):
            # Tokenize source and target
            source_encoding = tokenizer.tokenize_text(source)
            target_encoding = tokenizer.tokenize_text(target)
            
            source_encodings.append(source_encoding['input_ids'].squeeze())
            target_encodings.append(target_encoding['input_ids'].squeeze())
        
        # Pad sequences
        padded_source = self.sequence_handler.pad_sequences(source_encodings)
        padded_target = self.sequence_handler.pad_sequences(target_encodings)
        
        # Create attention masks
        source_masks = self.sequence_handler.create_attention_masks(padded_source)
        target_masks = self.sequence_handler.create_attention_masks(padded_target)
        
        return padded_source, source_masks, padded_target, target_masks


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Tokenization and Sequence Handling Demonstration ===\n")
    
    # Sample texts
    sample_texts = [
        "The transformer model revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Tokenization is a crucial step in text preprocessing for NLP tasks.",
        "Sequence handling ensures proper data formatting for deep learning models.",
        "Proper tokenization improves model performance and training efficiency."
    ]
    
    # 1. Test different tokenizers
    print("1. Testing Different Tokenizers...")
    
    tokenizer_configs = [
        TokenizationConfig(tokenizer_name="gpt2", tokenizer_type="gpt2"),
        TokenizationConfig(tokenizer_name="bert-base-uncased", tokenizer_type="bert"),
        TokenizationConfig(tokenizer_name="t5-base", tokenizer_type="t5")
    ]
    
    for config in tokenizer_configs:
        print(f"\nTesting {config.tokenizer_name}...")
        tokenizer = AdvancedTokenizer(config)
        
        # Test tokenization
        text = sample_texts[0]
        analysis = tokenizer.tokenize_with_analysis(text)
        
        print(f"Original text: {text}")
        print(f"Tokens: {analysis['tokens'][:10]}...")  # Show first 10 tokens
        print(f"Number of tokens: {analysis['num_tokens']}")
        print(f"Vocabulary coverage: {analysis['vocab_coverage']:.4f}")
    
    # 2. Test sequence handling
    print("\n2. Testing Sequence Handling...")
    
    config = TokenizationConfig(tokenizer_name="gpt2")
    tokenizer = AdvancedTokenizer(config)
    sequence_handler = SequenceHandler(config)
    
    # Tokenize a long text
    long_text = " ".join(sample_texts)
    tokenization = tokenizer.tokenize_with_analysis(long_text)
    
    # Create sequences
    sequences = sequence_handler.create_sequences(tokenization['token_ids'])
    print(f"Created {len(sequences)} sequences from long text")
    
    # Pad sequences
    padded_sequences = sequence_handler.pad_sequences(sequences)
    print(f"Padded sequences shape: {padded_sequences.shape}")
    
    # Create attention masks
    attention_masks = sequence_handler.create_attention_masks(padded_sequences)
    print(f"Attention masks shape: {attention_masks.shape}")
    
    # 3. Test custom dataset
    print("\n3. Testing Custom Dataset...")
    
    dataset = TokenizationDataset(sample_texts, tokenizer, max_length=128)
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    
    # 4. Test data collator
    print("\n4. Testing Data Collator...")
    
    data_collator = DataCollator(tokenizer, task_type="language_modeling")
    
    # Create a batch
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    collated_batch = data_collator.collate_fn(batch)
    
    print(f"Collated batch keys: {collated_batch.keys()}")
    print(f"Batch input IDs shape: {collated_batch['input_ids'].shape}")
    print(f"Batch attention masks shape: {collated_batch['attention_mask'].shape}")
    
    # 5. Test tokenization analysis
    print("\n5. Testing Tokenization Analysis...")
    
    analyzer = TokenizationAnalyzer(tokenizer)
    
    # Analyze statistics
    stats = analyzer.analyze_text_statistics(sample_texts)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average tokens per text: {stats['avg_tokens_per_text']:.2f}")
    print(f"Vocabulary coverage: {stats['vocab_coverage']:.4f}")
    
    # Analyze quality
    quality = analyzer.analyze_tokenization_quality(sample_texts)
    print(f"Average tokens per word: {quality['avg_tokens_per_word']:.2f}")
    print(f"Average compression ratio: {quality['avg_compression_ratio']:.2f}")
    print(f"Average UNK token ratio: {quality['avg_unk_token_ratio']:.4f}")
    
    # Visualize statistics
    analyzer.visualize_tokenization_stats(stats)
    
    # 6. Test sequence processor
    print("\n6. Testing Sequence Processor...")
    
    processor = SequenceProcessor(config)
    
    # Test language modeling data creation
    lm_inputs, lm_masks, lm_labels = processor.create_language_modeling_data(sample_texts, tokenizer)
    print(f"Language modeling data shapes:")
    print(f"  Inputs: {lm_inputs.shape}")
    print(f"  Masks: {lm_masks.shape}")
    print(f"  Labels: {lm_labels.shape}")
    
    # Test classification data creation
    fake_labels = [0, 1, 0, 1, 0]  # Binary classification labels
    cls_inputs, cls_masks, cls_labels = processor.create_classification_data(sample_texts, fake_labels, tokenizer)
    print(f"Classification data shapes:")
    print(f"  Inputs: {cls_inputs.shape}")
    print(f"  Masks: {cls_masks.shape}")
    print(f"  Labels: {cls_labels.shape}")
    
    # 7. Test custom tokenizer
    print("\n7. Testing Custom Tokenizer...")
    
    custom_tokenizer = CustomTokenizer(vocab_size=1000, min_frequency=1)
    
    # Train on sample texts
    custom_tokenizer.train_on_texts(sample_texts)
    
    # Test encoding/decoding
    test_text = "The transformer model is amazing."
    encoded = custom_tokenizer.encode(test_text)
    decoded = custom_tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    print("\n=== Demonstration Completed Successfully! ===")





