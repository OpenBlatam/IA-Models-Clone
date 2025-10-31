#!/usr/bin/env python3
"""
Advanced Tokenization and Sequence Handling for Blaze AI
Implements proper tokenization, sequence processing, and text data handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass
import warnings
import numpy as np
import re
import unicodedata
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
import hashlib
import time

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer"""
    vocab_size: int = 50000
    min_frequency: int = 2
    max_length: int = 512
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    mask_token: str = "<MASK>"
    sep_token: str = "<SEP>"
    cls_token: str = "<CLS>"
    do_lower_case: bool = True
    do_basic_tokenize: bool = True
    do_subword_tokenize: bool = True
    subword_type: str = "bpe"  # bpe, wordpiece, unigram
    max_subword_length: int = 100
    split_on_whitespace: bool = True
    strip_accents: bool = True
    remove_punctuation: bool = False


@dataclass
class SequenceConfig:
    """Configuration for sequence handling"""
    max_length: int = 512
    padding: str = "longest"  # longest, max_length, do_not_pad
    truncation: str = "longest_first"  # longest_first, only_first, only_second, do_not_truncate
    return_tensors: str = "pt"  # pt, tf, np
    return_attention_mask: bool = True
    return_token_type_ids: bool = True
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_length: bool = False
    return_offsets_mapping: bool = False
    add_special_tokens: bool = True
    stride: int = 0
    pad_to_multiple_of: Optional[int] = None


class BasicTokenizer:
    """Basic tokenizer for text preprocessing"""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.do_lower_case = config.do_lower_case
        self.strip_accents = config.strip_accents
        self.remove_punctuation = config.remove_punctuation
        self.split_on_whitespace = config.split_on_whitespace
    
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization"""
        if self.do_lower_case:
            text = text.lower()
        
        if self.strip_accents:
            text = self._strip_accents(text)
        
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        if self.split_on_whitespace:
            tokens = text.split()
        else:
            # Split on any whitespace character
            tokens = re.split(r'\s+', text.strip())
        
        return [token for token in tokens if token]
    
    def _strip_accents(self, text: str) -> str:
        """Remove accents from text"""
        return ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if not unicodedata.combining(char)
        )
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return re.sub(r'[^\w\s]', '', text)
    
    def clean_text(self, text: str) -> str:
        """Clean text for tokenization"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        return text


class SubwordTokenizer:
    """Subword tokenization using BPE, WordPiece, or Unigram"""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.subword_type = config.subword_type
        self.max_subword_length = config.max_subword_length
        self.vocab = set()
        self.subword_ranks = {}
        self.merges = []
        
        # Initialize based on subword type
        if self.subword_type == "bpe":
            self._init_bpe()
        elif self.subword_type == "wordpiece":
            self._init_wordpiece()
        elif self.subword_type == "unigram":
            self._init_unigram()
    
    def _init_bpe(self):
        """Initialize BPE tokenizer"""
        self.merges = []
        self.subword_ranks = {}
    
    def _init_wordpiece(self):
        """Initialize WordPiece tokenizer"""
        self.subword_ranks = {}
    
    def _init_unigram(self):
        """Initialize Unigram tokenizer"""
        self.subword_ranks = {}
    
    def train(self, texts: List[str], min_frequency: int = 2):
        """Train subword tokenizer on text corpus"""
        if self.subword_type == "bpe":
            self._train_bpe(texts, min_frequency)
        elif self.subword_type == "wordpiece":
            self._train_wordpiece(texts, min_frequency)
        elif self.subword_type == "unigram":
            self._train_unigram(texts, min_frequency)
    
    def _train_bpe(self, texts: List[str], min_frequency: int):
        """Train BPE tokenizer"""
        # Count character pairs
        pair_counts = Counter()
        word_counts = Counter()
        
        for text in texts:
            tokens = text.split()
            for token in tokens:
                word_counts[token] += 1
                pairs = self._get_pairs(token)
                for pair in pairs:
                    pair_counts[pair] += word_counts[token]
        
        # Build vocabulary
        self.vocab = set(word_counts.keys())
        
        # Build merges
        while len(self.merges) < self.config.vocab_size - len(self.vocab):
            if not pair_counts:
                break
            
            # Find most frequent pair
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            
            if pair_counts[best_pair] < min_frequency:
                break
            
            # Add merge
            self.merges.append(best_pair)
            self.subword_ranks[best_pair] = len(self.merges)
            
            # Update pair counts
            new_pair_counts = Counter()
            for word, count in word_counts.items():
                new_word = self._apply_merge(word, best_pair)
                if new_word != word:
                    new_pairs = self._get_pairs(new_word)
                    for pair in new_pairs:
                        new_pair_counts[pair] += count
                    word_counts[new_word] = count
                    del word_counts[word]
            
            pair_counts = new_pair_counts
    
    def _train_wordpiece(self, texts: List[str], min_frequency: int):
        """Train WordPiece tokenizer"""
        # Simplified WordPiece training
        # In practice, this would use a more sophisticated algorithm
        word_counts = Counter()
        
        for text in texts:
            tokens = text.split()
            for token in tokens:
                word_counts[token] += 1
        
        # Build vocabulary from most frequent words
        self.vocab = set(word_counts.keys())
        
        # Build subword vocabulary
        for word, count in word_counts.items():
            if count >= min_frequency:
                subwords = self._wordpiece_subwords(word)
                for subword in subwords:
                    self.subword_ranks[subword] = len(self.subword_ranks)
    
    def _train_unigram(self, texts: List[str], min_frequency: int):
        """Train Unigram tokenizer"""
        # Simplified Unigram training
        word_counts = Counter()
        
        for text in texts:
            tokens = text.split()
            for token in tokens:
                word_counts[token] += 1
        
        # Build vocabulary
        self.vocab = set(word_counts.keys())
        
        # Build subword vocabulary
        for word, count in word_counts.items():
            if count >= min_frequency:
                subwords = self._unigram_subwords(word)
                for subword in subwords:
                    self.subword_ranks[subword] = len(self.subword_ranks)
    
    def _get_pairs(self, word: str) -> List[Tuple[str, str]]:
        """Get character pairs from word"""
        pairs = []
        prev_char = word[0]
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        return pairs
    
    def _apply_merge(self, word: str, pair: Tuple[str, str]) -> str:
        """Apply BPE merge to word"""
        new_word = word
        while True:
            try:
                i = new_word.index(pair[0] + pair[1])
                new_word = new_word[:i] + pair[0] + pair[1] + new_word[i + 2:]
            except ValueError:
                break
        return new_word
    
    def _wordpiece_subwords(self, word: str) -> List[str]:
        """Get WordPiece subwords for a word"""
        subwords = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            while start < end:
                substr = word[start:end]
                if substr in self.subword_ranks:
                    cur_substr = substr
                    break
                end -= 1
            
            if cur_substr is None:
                cur_substr = word[start:start + 1]
            
            subwords.append(cur_substr)
            start += len(cur_substr)
        
        return subwords
    
    def _unigram_subwords(self, word: str) -> List[str]:
        """Get Unigram subwords for a word"""
        # Simplified: return character-level subwords
        return list(word)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using subword tokenization"""
        if self.subword_type == "bpe":
            return self._bpe_tokenize(text)
        elif self.subword_type == "wordpiece":
            return self._wordpiece_tokenize(text)
        elif self.subword_type == "unigram":
            return self._unigram_tokenize(text)
        else:
            return text.split()
    
    def _bpe_tokenize(self, text: str) -> List[str]:
        """BPE tokenization"""
        tokens = text.split()
        subword_tokens = []
        
        for token in tokens:
            if token in self.vocab:
                subword_tokens.append(token)
            else:
                # Apply merges
                word = list(token)
                for merge in self.merges:
                    word = self._apply_merge(''.join(word), merge)
                    word = list(word)
                
                subword_tokens.extend(word)
        
        return subword_tokens
    
    def _wordpiece_tokenize(self, text: str) -> List[str]:
        """WordPiece tokenization"""
        tokens = text.split()
        subword_tokens = []
        
        for token in tokens:
            subwords = self._wordpiece_subwords(token)
            subword_tokens.extend(subwords)
        
        return subword_tokens
    
    def _unigram_tokenize(self, text: str) -> List[str]:
        """Unigram tokenization"""
        tokens = text.split()
        subword_tokens = []
        
        for token in tokens:
            subwords = self._unigram_subwords(token)
            subword_tokens.extend(subwords)
        
        return subword_tokens


class CustomTokenizer:
    """Custom tokenizer with vocabulary management"""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.basic_tokenizer = BasicTokenizer(config)
        self.subword_tokenizer = SubwordTokenizer(config) if config.do_subword_tokenize else None
        
        # Special tokens
        self.special_tokens = {
            'pad_token': config.pad_token,
            'unk_token': config.unk_token,
            'bos_token': config.bos_token,
            'eos_token': config.eos_token,
            'mask_token': config.mask_token,
            'sep_token': config.sep_token,
            'cls_token': config.cls_token
        }
        
        # Vocabulary
        self.vocab = {}
        self.id_to_token = {}
        self.token_to_id = {}
        
        # Initialize vocabulary
        self._init_vocabulary()
    
    def _init_vocabulary(self):
        """Initialize vocabulary with special tokens"""
        token_id = 0
        
        # Add special tokens
        for token in self.special_tokens.values():
            self.vocab[token] = token_id
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id
            token_id += 1
    
    def train(self, texts: List[str]):
        """Train tokenizer on text corpus"""
        # Basic tokenization
        all_tokens = []
        for text in texts:
            tokens = self.basic_tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Filter by minimum frequency
        filtered_tokens = [
            token for token, count in token_counts.items()
            if count >= self.config.min_frequency
        ]
        
        # Add to vocabulary
        token_id = len(self.vocab)
        for token in filtered_tokens[:self.config.vocab_size - len(self.vocab)]:
            self.vocab[token] = token_id
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id
            token_id += 1
        
        # Train subword tokenizer if enabled
        if self.subword_tokenizer:
            self.subword_tokenizer.train(texts, self.config.min_frequency)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        # Clean text
        text = self.basic_tokenizer.clean_text(text)
        
        # Basic tokenization
        tokens = self.basic_tokenizer.tokenize(text)
        
        # Subword tokenization if enabled
        if self.subword_tokenizer:
            subword_tokens = []
            for token in tokens:
                if token in self.vocab:
                    subword_tokens.append(token)
                else:
                    subwords = self.subword_tokenizer.tokenize(token)
                    subword_tokens.extend(subwords)
            tokens = subword_tokens
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.special_tokens['bos_token']] + tokens + [self.special_tokens['eos_token']]
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id[self.special_tokens['unk_token']])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens.values():
                    continue
                tokens.append(token)
            else:
                tokens.append(self.special_tokens['unk_token'])
        
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def save(self, path: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            'config': self.config,
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens
        }
        
        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.config = tokenizer_data['config']
        self.vocab = tokenizer_data['vocab']
        self.id_to_token = tokenizer_data['id_to_token']
        self.token_to_id = tokenizer_data['token_to_id']
        self.special_tokens = tokenizer_data['special_tokens']


class SequenceProcessor:
    """Process sequences for model input"""
    
    def __init__(self, config: SequenceConfig):
        self.config = config
    
    def process_single_sequence(self, token_ids: List[int], 
                               tokenizer: CustomTokenizer) -> Dict[str, torch.Tensor]:
        """Process single sequence"""
        # Add special tokens if requested
        if self.config.add_special_tokens:
            if tokenizer.special_tokens['cls_token'] not in token_ids:
                token_ids = [tokenizer.token_to_id[tokenizer.special_tokens['cls_token']]] + token_ids
            if tokenizer.special_tokens['sep_token'] not in token_ids:
                token_ids = token_ids + [tokenizer.token_to_id[tokenizer.special_tokens['sep_token']]]
        
        # Truncation
        if self.config.truncation != "do_not_truncate":
            token_ids = self._truncate_sequence(token_ids)
        
        # Padding
        if self.config.padding != "do_not_pad":
            token_ids = self._pad_sequence(token_ids, tokenizer)
        
        # Convert to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        if self.config.padding != "do_not_pad":
            pad_id = tokenizer.token_to_id[tokenizer.special_tokens['pad_token']]
            attention_mask = (input_ids != pad_id).long()
        
        # Create token type IDs
        token_type_ids = torch.zeros_like(input_ids)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        
        return result
    
    def process_pair_sequences(self, token_ids_a: List[int], token_ids_b: List[int],
                              tokenizer: CustomTokenizer) -> Dict[str, torch.Tensor]:
        """Process pair of sequences (e.g., question-answer)"""
        # Add special tokens
        if self.config.add_special_tokens:
            cls_id = tokenizer.token_to_id[tokenizer.special_tokens['cls_token']]
            sep_id = tokenizer.token_to_id[tokenizer.special_tokens['sep_token']]
            
            token_ids_a = [cls_id] + token_ids_a + [sep_id]
            token_ids_b = token_ids_b + [sep_id]
        
        # Combine sequences
        combined_ids = token_ids_a + token_ids_b
        
        # Truncation
        if self.config.truncation != "do_not_truncate":
            combined_ids = self._truncate_pair_sequences(combined_ids, len(token_ids_a))
        
        # Padding
        if self.config.padding != "do_not_pad":
            combined_ids = self._pad_sequence(combined_ids, tokenizer)
        
        # Convert to tensor
        input_ids = torch.tensor(combined_ids, dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        if self.config.padding != "do_not_pad":
            pad_id = tokenizer.token_to_id[tokenizer.special_tokens['pad_token']]
            attention_mask = (input_ids != pad_id).long()
        
        # Create token type IDs
        token_type_ids = torch.zeros_like(input_ids)
        if self.config.return_token_type_ids:
            # Mark second sequence
            token_type_ids[len(token_ids_a):] = 1
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        
        return result
    
    def _truncate_sequence(self, token_ids: List[int]) -> List[int]:
        """Truncate sequence to max length"""
        if len(token_ids) <= self.config.max_length:
            return token_ids
        
        if self.config.truncation == "longest_first":
            return token_ids[:self.config.max_length]
        else:
            return token_ids[-self.config.max_length:]
    
    def _truncate_pair_sequences(self, combined_ids: List[int], 
                                first_seq_length: int) -> List[int]:
        """Truncate pair sequences"""
        if len(combined_ids) <= self.config.max_length:
            return combined_ids
        
        if self.config.truncation == "longest_first":
            # Truncate from the end of the first sequence
            excess = len(combined_ids) - self.config.max_length
            if excess <= first_seq_length:
                return combined_ids[:first_seq_length - excess] + combined_ids[first_seq_length:]
            else:
                return combined_ids[:self.config.max_length]
        elif self.config.truncation == "only_first":
            return combined_ids[:self.config.max_length]
        elif self.config.truncation == "only_second":
            return combined_ids[:first_seq_length] + combined_ids[-self.config.max_length + first_seq_length:]
        else:
            return combined_ids[:self.config.max_length]
    
    def _pad_sequence(self, token_ids: List[int], tokenizer: CustomTokenizer) -> List[int]:
        """Pad sequence to max length"""
        if len(token_ids) >= self.config.max_length:
            return token_ids
        
        pad_id = tokenizer.token_to_id[tokenizer.special_tokens['pad_token']]
        
        if self.config.padding == "longest":
            # Pad to the length of the longest sequence in the batch
            # This is handled at the batch level
            return token_ids
        elif self.config.padding == "max_length":
            # Pad to max_length
            padding_length = self.config.max_length - len(token_ids)
            return token_ids + [pad_id] * padding_length
        else:
            return token_ids
    
    def batch_process(self, sequences: List[List[int]], tokenizer: CustomTokenizer,
                     is_pair: bool = False) -> Dict[str, torch.Tensor]:
        """Process batch of sequences"""
        if is_pair:
            # Process pair sequences
            processed_sequences = []
            for seq_a, seq_b in sequences:
                processed = self.process_pair_sequences(seq_a, seq_b, tokenizer)
                processed_sequences.append(processed)
        else:
            # Process single sequences
            processed_sequences = []
            for seq in sequences:
                processed = self.process_single_sequence(seq, tokenizer)
                processed_sequences.append(processed)
        
        # Pad to longest sequence if needed
        if self.config.padding == "longest":
            max_length = max(len(seq['input_ids']) for seq in processed_sequences)
            
            for seq in processed_sequences:
                if len(seq['input_ids']) < max_length:
                    pad_length = max_length - len(seq['input_ids'])
                    pad_id = tokenizer.token_to_id[tokenizer.special_tokens['pad_token']]
                    
                    seq['input_ids'] = torch.cat([
                        seq['input_ids'],
                        torch.full((pad_length,), pad_id, dtype=torch.long)
                    ])
                    
                    seq['attention_mask'] = torch.cat([
                        seq['attention_mask'],
                        torch.zeros(pad_length, dtype=torch.long)
                    ])
                    
                    if self.config.return_token_type_ids:
                        seq['token_type_ids'] = torch.cat([
                            seq['token_type_ids'],
                            torch.zeros(pad_length, dtype=torch.long)
                        ])
        
        # Stack tensors
        result = {}
        for key in processed_sequences[0].keys():
            result[key] = torch.stack([seq[key] for seq in processed_sequences])
        
        return result


class TextDataset:
    """Dataset for text data with tokenization"""
    
    def __init__(self, texts: List[str], labels: Optional[List[Any]] = None,
                 tokenizer: Optional[CustomTokenizer] = None,
                 config: Optional[SequenceConfig] = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.config = config or SequenceConfig()
        self.processor = SequenceProcessor(self.config)
        
        # Tokenize texts
        if self.tokenizer:
            self.tokenized_texts = [self.tokenizer.encode(text) for text in self.texts]
        else:
            self.tokenized_texts = None
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.tokenized_texts is None:
            return {'text': self.texts[idx], 'label': self.labels[idx] if self.labels else None}
        
        # Process sequence
        processed = self.processor.process_single_sequence(
            self.tokenized_texts[idx], self.tokenizer
        )
        
        result = processed.copy()
        if self.labels is not None:
            result['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return result
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader"""
        if self.tokenized_texts is None:
            # Return raw text batch
            return {
                'texts': [item['text'] for item in batch],
                'labels': [item['label'] for item in batch] if 'label' in batch[0] else None
            }
        
        # Process batch
        sequences = [item['input_ids'].tolist() for item in batch]
        processed_batch = self.processor.batch_process(sequences, self.tokenizer)
        
        # Add labels if present
        if 'labels' in batch[0]:
            processed_batch['labels'] = torch.stack([item['labels'] for item in batch])
        
        return processed_batch


class TokenizationExperiments:
    """Collection of tokenization experiments"""
    
    @staticmethod
    def demonstrate_basic_tokenization():
        """Demonstrate basic tokenization"""
        
        logger.info("Demonstrating basic tokenization...")
        
        # Create tokenizer config
        config = TokenizerConfig(
            vocab_size=10000,
            min_frequency=2,
            do_lower_case=True,
            strip_accents=True,
            remove_punctuation=False
        )
        
        # Create basic tokenizer
        basic_tokenizer = BasicTokenizer(config)
        
        # Test text
        test_texts = [
            "Hello, world! This is a test.",
            "Machine learning is amazing.",
            "¿Cómo estás? I'm doing well."
        ]
        
        # Tokenize
        for text in test_texts:
            tokens = basic_tokenizer.tokenize(text)
            logger.info(f"Text: {text}")
            logger.info(f"Tokens: {tokens}")
            logger.info("---")
        
        return basic_tokenizer
    
    @staticmethod
    def demonstrate_subword_tokenization():
        """Demonstrate subword tokenization"""
        
        logger.info("Demonstrating subword tokenization...")
        
        # Create tokenizer config
        config = TokenizerConfig(
            vocab_size=1000,
            min_frequency=2,
            subword_type="bpe"
        )
        
        # Create subword tokenizer
        subword_tokenizer = SubwordTokenizer(config)
        
        # Training texts
        training_texts = [
            "machine learning is a subset of artificial intelligence",
            "deep learning uses neural networks with multiple layers",
            "natural language processing helps computers understand text",
            "computer vision enables machines to see and interpret images"
        ]
        
        # Train tokenizer
        subword_tokenizer.train(training_texts, min_frequency=1)
        
        # Test tokenization
        test_text = "machine learning neural networks"
        tokens = subword_tokenizer.tokenize(test_text)
        
        logger.info(f"Test text: {test_text}")
        logger.info(f"Subword tokens: {tokens}")
        
        return subword_tokenizer
    
    @staticmethod
    def demonstrate_custom_tokenizer():
        """Demonstrate custom tokenizer"""
        
        logger.info("Demonstrating custom tokenizer...")
        
        # Create tokenizer config
        config = TokenizerConfig(
            vocab_size=5000,
            min_frequency=2,
            do_lower_case=True,
            do_subword_tokenize=True,
            subword_type="bpe"
        )
        
        # Create custom tokenizer
        tokenizer = CustomTokenizer(config)
        
        # Training texts
        training_texts = [
            "the quick brown fox jumps over the lazy dog",
            "machine learning algorithms are powerful tools",
            "natural language processing is fascinating",
            "deep learning models can achieve amazing results"
        ]
        
        # Train tokenizer
        tokenizer.train(training_texts)
        
        # Test encoding and decoding
        test_text = "the quick brown fox"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        logger.info(f"Test text: {test_text}")
        logger.info(f"Encoded: {encoded}")
        logger.info(f"Decoded: {decoded}")
        logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        
        return tokenizer
    
    @staticmethod
    def demonstrate_sequence_processing():
        """Demonstrate sequence processing"""
        
        logger.info("Demonstrating sequence processing...")
        
        # Create sequence config
        config = SequenceConfig(
            max_length=20,
            padding="max_length",
            truncation="longest_first",
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        # Create processor
        processor = SequenceProcessor(config)
        
        # Create tokenizer
        tokenizer_config = TokenizerConfig(vocab_size=1000)
        tokenizer = CustomTokenizer(tokenizer_config)
        
        # Training texts
        training_texts = ["hello world", "machine learning", "deep learning"]
        tokenizer.train(training_texts)
        
        # Test sequences
        test_sequences = [
            "hello world",
            "machine learning is amazing",
            "deep learning neural networks"
        ]
        
        # Process sequences
        encoded_sequences = [tokenizer.encode(text) for text in test_sequences]
        processed_batch = processor.batch_process(encoded_sequences, tokenizer)
        
        logger.info("Processed batch:")
        for key, value in processed_batch.items():
            logger.info(f"{key}: {value.shape}")
            logger.info(f"{key}: {value}")
        
        return processor, tokenizer
    
    @staticmethod
    def demonstrate_text_dataset():
        """Demonstrate text dataset"""
        
        logger.info("Demonstrating text dataset...")
        
        # Create tokenizer
        tokenizer_config = TokenizerConfig(vocab_size=1000)
        tokenizer = CustomTokenizer(tokenizer_config)
        
        # Training texts
        training_texts = [
            "positive sentiment",
            "negative sentiment",
            "neutral sentiment",
            "very positive",
            "very negative"
        ]
        
        # Labels
        labels = [1, 0, 2, 1, 0]  # 1: positive, 0: negative, 2: neutral
        
        # Train tokenizer
        tokenizer.train(training_texts)
        
        # Create dataset
        dataset = TextDataset(training_texts, labels, tokenizer)
        
        # Test dataset
        for i in range(len(dataset)):
            item = dataset[i]
            logger.info(f"Item {i}:")
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {value.shape} - {value}")
                else:
                    logger.info(f"  {key}: {value}")
        
        return dataset


def main():
    """Main execution function"""
    logger.info("Starting Advanced Tokenization and Sequence Handling Demonstrations...")
    
    # Demonstrate basic tokenization
    logger.info("Testing basic tokenization...")
    basic_tokenizer = TokenizationExperiments.demonstrate_basic_tokenization()
    
    # Demonstrate subword tokenization
    logger.info("Testing subword tokenization...")
    subword_tokenizer = TokenizationExperiments.demonstrate_subword_tokenization()
    
    # Demonstrate custom tokenizer
    logger.info("Testing custom tokenizer...")
    custom_tokenizer = TokenizationExperiments.demonstrate_custom_tokenizer()
    
    # Demonstrate sequence processing
    logger.info("Testing sequence processing...")
    processor, tokenizer = TokenizationExperiments.demonstrate_sequence_processing()
    
    # Demonstrate text dataset
    logger.info("Testing text dataset...")
    dataset = TokenizationExperiments.demonstrate_text_dataset()
    
    # Create comprehensive tokenization system
    logger.info("Creating comprehensive tokenization system...")
    
    comprehensive_config = TokenizerConfig(
        vocab_size=50000,
        min_frequency=3,
        max_length=1024,
        do_lower_case=True,
        do_subword_tokenize=True,
        subword_type="bpe",
        strip_accents=True
    )
    
    comprehensive_tokenizer = CustomTokenizer(comprehensive_config)
    
    # Training corpus
    training_corpus = [
        "artificial intelligence is transforming the world",
        "machine learning algorithms learn from data",
        "deep learning uses neural networks with many layers",
        "natural language processing enables computers to understand text",
        "computer vision helps machines see and interpret images",
        "robotics combines hardware and software for automation",
        "data science extracts insights from large datasets",
        "cloud computing provides scalable computing resources"
    ]
    
    # Train comprehensive tokenizer
    comprehensive_tokenizer.train(training_corpus)
    
    # Test comprehensive system
    test_texts = [
        "artificial intelligence and machine learning",
        "deep learning neural networks for NLP",
        "computer vision and robotics applications"
    ]
    
    for text in test_texts:
        encoded = comprehensive_tokenizer.encode(text)
        decoded = comprehensive_tokenizer.decode(encoded)
        logger.info(f"Original: {text}")
        logger.info(f"Encoded: {encoded}")
        logger.info(f"Decoded: {decoded}")
        logger.info("---")
    
    # Summary
    logger.info("Tokenization and Sequence Handling Summary:")
    logger.info(f"Basic tokenization tested: ✓")
    logger.info(f"Subword tokenization tested: ✓")
    logger.info(f"Custom tokenizer tested: ✓")
    logger.info(f"Sequence processing tested: ✓")
    logger.info(f"Text dataset tested: ✓")
    logger.info(f"Comprehensive tokenization system created: ✓")
    logger.info(f"Vocabulary size: {comprehensive_tokenizer.get_vocab_size()}")
    
    logger.info("Advanced Tokenization and Sequence Handling demonstrations completed successfully!")


if __name__ == "__main__":
    main()
