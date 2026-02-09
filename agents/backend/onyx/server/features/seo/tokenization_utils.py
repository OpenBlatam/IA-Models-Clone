from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
from transformers.tokenization_utils_base import TruncationStrategy, PaddingStrategy
from transformers.data import DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Iterator, Generator
from dataclasses import dataclass, field
from functools import lru_cache, partial
import numpy as np
import logging
import json
import re
import os
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from tqdm import tqdm
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Tokenization and Sequence Handling for SEO Service
Comprehensive tokenization utilities with sequence management, caching, and optimization
"""

    AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast,
    BatchEncoding, TokenizerFast, Tokenizer
)

logger = logging.getLogger(__name__)

@dataclass
class TokenizationConfig:
    """Configuration for tokenization settings"""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    truncation: Union[bool, str, TruncationStrategy] = True
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    return_tensors: str = "pt"
    return_attention_mask: bool = True
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    add_special_tokens: bool = True
    return_overflowing_tokens: bool = False
    stride: int = 0
    is_split_into_words: bool = False
    pad_to_multiple_of: Optional[int] = None
    return_token_type_ids: bool = False
    verbose: bool = True

@dataclass
class SequenceConfig:
    """Configuration for sequence handling"""
    max_sequence_length: int = 512
    min_sequence_length: int = 1
    overlap_strategy: str = "sliding_window"  # "sliding_window", "sentence", "paragraph"
    overlap_size: int = 50
    chunk_strategy: str = "fixed_length"  # "fixed_length", "sentence", "paragraph", "semantic"
    preserve_boundaries: bool = True
    add_special_tokens: bool = True
    truncation_strategy: str = "longest_first"  # "longest_first", "only_first", "only_second"
    padding_strategy: str = "batch_longest"  # "batch_longest", "max_length", "do_not_pad"

@dataclass
class TokenizationStats:
    """Statistics for tokenization analysis"""
    total_tokens: int = 0
    unique_tokens: int = 0
    avg_sequence_length: float = 0.0
    max_sequence_length: int = 0
    min_sequence_length: int = 0
    vocabulary_size: int = 0
    oov_rate: float = 0.0
    padding_ratio: float = 0.0
    truncation_ratio: float = 0.0
    token_distribution: Dict[str, int] = field(default_factory=dict)
    sequence_length_distribution: Dict[int, int] = field(default_factory=dict)

class AdvancedTokenizer:
    """Advanced tokenizer with caching, optimization, and sequence handling"""
    
    def __init__(self, config: TokenizationConfig):
        
    """__init__ function."""
self.config = config
        self.tokenizer = None
        self.cache = {}
        self.stats = TokenizationStats()
        self._load_tokenizer()
    
    def _load_tokenizer(self) -> Any:
        """Load and configure tokenizer"""
        try:
            logger.info(f"Loading tokenizer: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = "[PAD]"
            
            # Update vocabulary size
            self.stats.vocabulary_size = self.tokenizer.vocab_size
            
            logger.info(f"Tokenizer loaded successfully. Vocab size: {self.stats.vocabulary_size}")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def _get_cache_key(self, text: str, **kwargs) -> str:
        """Generate cache key for tokenization"""
        # Create a hash of the text and parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        text_hash = hashlib.md5(f"{text}{params_str}".encode()).hexdigest()
        return text_hash
    
    def tokenize_text(self, text: str, use_cache: bool = True, **kwargs) -> BatchEncoding:
        """Tokenize single text with caching"""
        if use_cache:
            cache_key = self._get_cache_key(text, **kwargs)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Merge config with kwargs
        tokenization_kwargs = {
            'add_special_tokens': self.config.add_special_tokens,
            'return_attention_mask': self.config.return_attention_mask,
            'return_tensors': self.config.return_tensors,
            'padding': self.config.padding,
            'truncation': self.config.truncation,
            'max_length': self.config.max_length,
            'return_special_tokens_mask': self.config.return_special_tokens_mask,
            'return_offsets_mapping': self.config.return_offsets_mapping,
            'return_length': self.config.return_length,
            'return_overflowing_tokens': self.config.return_overflowing_tokens,
            'stride': self.config.stride,
            'is_split_into_words': self.config.is_split_into_words,
            'pad_to_multiple_of': self.config.pad_to_multiple_of,
            'return_token_type_ids': self.config.return_token_type_ids
        }
        tokenization_kwargs.update(kwargs)
        
        try:
            result = self.tokenizer(text, **tokenization_kwargs)
            
            if use_cache:
                cache_key = self._get_cache_key(text, **kwargs)
                self.cache[cache_key] = result
            
            # Update statistics
            self._update_stats(result, text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def tokenize_batch(self, texts: List[str], use_cache: bool = True, **kwargs) -> BatchEncoding:
        """Tokenize batch of texts with optimization"""
        if not texts:
            return BatchEncoding({})
        
        # Check cache for individual texts
        if use_cache:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, **kwargs)
                if cache_key in self.cache:
                    cached_results.append(self.cache[cache_key])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Tokenize uncached texts
            if uncached_texts:
                new_results = self._tokenize_batch_internal(uncached_texts, **kwargs)
                
                # Cache new results
                for i, text in enumerate(uncached_texts):
                    cache_key = self._get_cache_key(text, **kwargs)
                    self.cache[cache_key] = new_results[i]
                
                # Combine results
                all_results = []
                cached_idx = 0
                uncached_idx = 0
                
                for i in range(len(texts)):
                    if i in uncached_indices:
                        all_results.append(new_results[uncached_idx])
                        uncached_idx += 1
                    else:
                        all_results.append(cached_results[cached_idx])
                        cached_idx += 1
                
                return self._combine_batch_results(all_results)
            else:
                return self._combine_batch_results(cached_results)
        else:
            return self._tokenize_batch_internal(texts, **kwargs)
    
    def _tokenize_batch_internal(self, texts: List[str], **kwargs) -> List[BatchEncoding]:
        """Internal batch tokenization"""
        # Merge config with kwargs
        tokenization_kwargs = {
            'add_special_tokens': self.config.add_special_tokens,
            'return_attention_mask': self.config.return_attention_mask,
            'return_tensors': self.config.return_tensors,
            'padding': self.config.padding,
            'truncation': self.config.truncation,
            'max_length': self.config.max_length,
            'return_special_tokens_mask': self.config.return_special_tokens_mask,
            'return_offsets_mapping': self.config.return_offsets_mapping,
            'return_length': self.config.return_length,
            'return_overflowing_tokens': self.config.return_overflowing_tokens,
            'stride': self.config.stride,
            'is_split_into_words': self.config.is_split_into_words,
            'pad_to_multiple_of': self.config.pad_to_multiple_of,
            'return_token_type_ids': self.config.return_token_type_ids
        }
        tokenization_kwargs.update(kwargs)
        
        try:
            batch_result = self.tokenizer(texts, **tokenization_kwargs)
            
            # Split batch result into individual results
            individual_results = []
            batch_size = len(texts)
            
            for i in range(batch_size):
                individual_result = {}
                for key, value in batch_result.items():
                    if isinstance(value, torch.Tensor):
                        individual_result[key] = value[i:i+1]
                    elif isinstance(value, list):
                        individual_result[key] = [value[i]]
                    else:
                        individual_result[key] = value
                individual_results.append(BatchEncoding(individual_result))
            
            # Update statistics
            for i, text in enumerate(texts):
                self._update_stats(individual_results[i], text)
            
            return individual_results
            
        except Exception as e:
            logger.error(f"Error tokenizing batch: {e}")
            raise
    
    def _combine_batch_results(self, results: List[BatchEncoding]) -> BatchEncoding:
        """Combine individual results into batch result"""
        if not results:
            return BatchEncoding({})
        
        combined = {}
        for key in results[0].keys():
            if isinstance(results[0][key], torch.Tensor):
                combined[key] = torch.cat([r[key] for r in results], dim=0)
            elif isinstance(results[0][key], list):
                combined[key] = [item for r in results for item in r[key]]
            else:
                combined[key] = results[0][key]
        
        return BatchEncoding(combined)
    
    def _update_stats(self, result: BatchEncoding, text: str):
        """Update tokenization statistics"""
        if 'input_ids' in result:
            input_ids = result['input_ids']
            if isinstance(input_ids, torch.Tensor):
                sequence_length = input_ids.shape[-1]
                tokens = input_ids.flatten().tolist()
            else:
                sequence_length = len(input_ids)
                tokens = input_ids
            
            # Update basic stats
            self.stats.total_tokens += len(tokens)
            self.stats.max_sequence_length = max(self.stats.max_sequence_length, sequence_length)
            self.stats.min_sequence_length = min(self.stats.min_sequence_length, sequence_length) if self.stats.min_sequence_length > 0 else sequence_length
            
            # Update token distribution
            token_counter = Counter(tokens)
            for token, count in token_counter.items():
                token_str = str(token)
                self.stats.token_distribution[token_str] = self.stats.token_distribution.get(token_str, 0) + count
            
            # Update sequence length distribution
            self.stats.sequence_length_distribution[sequence_length] = self.stats.sequence_length_distribution.get(sequence_length, 0) + 1
    
    def get_stats(self) -> TokenizationStats:
        """Get current tokenization statistics"""
        if self.stats.total_tokens > 0:
            self.stats.avg_sequence_length = self.stats.total_tokens / self.stats.sequence_length_distribution.get(1, 1)
            self.stats.unique_tokens = len(self.stats.token_distribution)
        
        return self.stats
    
    def clear_cache(self) -> Any:
        """Clear tokenization cache"""
        self.cache.clear()
        logger.info("Tokenization cache cleared")
    
    def save_cache(self, file_path: str):
        """Save tokenization cache to file"""
        try:
            with open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(self.cache, f)
            logger.info(f"Tokenization cache saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def load_cache(self, file_path: str):
        """Load tokenization cache from file"""
        try:
            with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                self.cache = pickle.load(f)
            logger.info(f"Tokenization cache loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")

class SequenceHandler:
    """Advanced sequence handling for long texts"""
    
    def __init__(self, config: SequenceConfig):
        
    """__init__ function."""
self.config = config
    
    def split_text_into_chunks(self, text: str, tokenizer: PreTrainedTokenizer) -> List[str]:
        """Split text into manageable chunks based on strategy"""
        if len(text) <= self.config.max_sequence_length:
            return [text]
        
        if self.config.chunk_strategy == "fixed_length":
            return self._split_fixed_length(text)
        elif self.config.chunk_strategy == "sentence":
            return self._split_by_sentences(text, tokenizer)
        elif self.config.chunk_strategy == "paragraph":
            return self._split_by_paragraphs(text)
        elif self.config.chunk_strategy == "semantic":
            return self._split_semantic(text, tokenizer)
        else:
            return self._split_fixed_length(text)
    
    def _split_fixed_length(self, text: str) -> List[str]:
        """Split text into fixed-length chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.max_sequence_length
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > 0:
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end
        
        return chunks
    
    def _split_by_sentences(self, text: str, tokenizer: PreTrainedTokenizer) -> List[str]:
        """Split text by sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Estimate token count
            estimated_tokens = len(sentence.split()) * 1.3  # Rough estimate
            
            if len(current_chunk) + len(sentence) > self.config.max_sequence_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split it
                    sub_chunks = self._split_fixed_length(sentence)
                    chunks.extend(sub_chunks)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.config.max_sequence_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Single paragraph is too long, split it
                    sub_chunks = self._split_fixed_length(paragraph)
                    chunks.extend(sub_chunks)
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_semantic(self, text: str, tokenizer: PreTrainedTokenizer) -> List[str]:
        """Split text using semantic boundaries (simplified)"""
        # This is a simplified semantic splitting
        # In practice, you might use more sophisticated NLP techniques
        
        # First try sentence splitting
        chunks = self._split_by_sentences(text, tokenizer)
        
        # If chunks are still too long, use fixed length
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.config.max_sequence_length:
                sub_chunks = self._split_fixed_length(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def create_sliding_windows(self, text: str, tokenizer: PreTrainedTokenizer) -> List[str]:
        """Create sliding windows for overlapping chunks"""
        chunks = self.split_text_into_chunks(text, tokenizer)
        
        if len(chunks) <= 1:
            return chunks
        
        windows = []
        for i in range(len(chunks)):
            if i == 0:
                windows.append(chunks[i])
            else:
                # Create overlap with previous chunk
                overlap_text = chunks[i-1][-self.config.overlap_size:]
                current_text = chunks[i]
                
                # Combine with overlap
                combined = overlap_text + " " + current_text
                windows.append(combined)
        
        return windows

class TokenizedDataset(Dataset):
    """Custom dataset for tokenized data with caching and optimization"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer: AdvancedTokenizer = None, max_length: int = 512,
                 cache_dir: Optional[str] = None):
        
    """__init__ function."""
self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.cached_data = {}
        
        # Load cached data if available
        if cache_dir:
            self._load_cache()
    
    def _load_cache(self) -> Any:
        """Load cached tokenized data"""
        if not self.cache_dir:
            return
        
        cache_file = os.path.join(self.cache_dir, "tokenized_data.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    self.cached_data = pickle.load(f)
                logger.info(f"Loaded cached data for {len(self.cached_data)} samples")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
    
    def _save_cache(self) -> Any:
        """Save tokenized data to cache"""
        if not self.cache_dir:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "tokenized_data.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(self.cached_data, f)
            logger.info(f"Saved cached data for {len(self.cached_data)} samples")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache first
        if idx in self.cached_data:
            return self.cached_data[idx]
        
        text = self.texts[idx]
        
        # Tokenize text
        if self.tokenizer:
            encoding = self.tokenizer.tokenize_text(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
            
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            # Return raw text if no tokenizer
            item = {'text': text}
            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Cache the result
        self.cached_data[idx] = item
        
        return item
    
    def save_cache(self) -> Any:
        """Save current cache to disk"""
        self._save_cache()

class TokenizationPipeline:
    """Complete tokenization pipeline with preprocessing and postprocessing"""
    
    def __init__(self, tokenizer_config: TokenizationConfig, 
                 sequence_config: SequenceConfig = None):
        
    """__init__ function."""
self.tokenizer_config = tokenizer_config
        self.sequence_config = sequence_config or SequenceConfig()
        self.tokenizer = AdvancedTokenizer(tokenizer_config)
        self.sequence_handler = SequenceHandler(self.sequence_config)
    
    def process_text(self, text: str, **kwargs) -> BatchEncoding:
        """Process single text through the pipeline"""
        # Preprocessing (if needed)
        processed_text = self._preprocess_text(text)
        
        # Check if text needs chunking
        if len(processed_text) > self.tokenizer_config.max_length * 4:  # Rough estimate
            chunks = self.sequence_handler.split_text_into_chunks(
                processed_text, self.tokenizer.tokenizer
            )
            
            if len(chunks) == 1:
                return self.tokenizer.tokenize_text(processed_text, **kwargs)
            else:
                # For now, return the first chunk
                # In practice, you might want to handle multiple chunks differently
                return self.tokenizer.tokenize_text(chunks[0], **kwargs)
        else:
            return self.tokenizer.tokenize_text(processed_text, **kwargs)
    
    def process_batch(self, texts: List[str], **kwargs) -> BatchEncoding:
        """Process batch of texts through the pipeline"""
        processed_texts = [self._preprocess_text(text) for text in texts]
        return self.tokenizer.tokenize_batch(processed_texts, **kwargs)
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Basic cleaning
        text = text.replace('\n', ' ').replace('\t', ' ')
        
        return text
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tokenization statistics"""
        stats = self.tokenizer.get_stats()
        
        return {
            'tokenization_stats': stats,
            'config': {
                'tokenizer_config': self.tokenizer_config,
                'sequence_config': self.sequence_config
            },
            'cache_info': {
                'cache_size': len(self.tokenizer.cache),
                'cache_hit_ratio': 0.0  # Would need to track hits/misses
            }
        }

# Utility functions
def create_data_collator(tokenizer: PreTrainedTokenizer, 
                        task_type: str = "sequence_classification") -> Callable:
    """Create appropriate data collator for the task"""
    if task_type == "token_classification":
        return DataCollatorForTokenClassification(tokenizer)
    else:
        return DataCollatorWithPadding(tokenizer)

def analyze_tokenization_quality(tokenizer: PreTrainedTokenizer, 
                               texts: List[str]) -> Dict[str, Any]:
    """Analyze tokenization quality and provide insights"""
    analysis = {
        'total_texts': len(texts),
        'avg_text_length': np.mean([len(text) for text in texts]),
        'tokenization_stats': defaultdict(list),
        'vocabulary_coverage': {},
        'sequence_length_distribution': defaultdict(int)
    }
    
    for text in texts:
        # Tokenize
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        
        # Collect statistics
        analysis['tokenization_stats']['token_counts'].append(len(tokens))
        analysis['tokenization_stats']['input_id_counts'].append(len(input_ids))
        analysis['sequence_length_distribution'][len(input_ids)] += 1
        
        # Check vocabulary coverage
        for token in tokens:
            if token in tokenizer.get_vocab():
                analysis['vocabulary_coverage'][token] = analysis['vocabulary_coverage'].get(token, 0) + 1
    
    # Calculate averages
    analysis['avg_tokens_per_text'] = np.mean(analysis['tokenization_stats']['token_counts'])
    analysis['avg_input_ids_per_text'] = np.mean(analysis['tokenization_stats']['input_id_counts'])
    analysis['vocabulary_coverage_ratio'] = len(analysis['vocabulary_coverage']) / tokenizer.vocab_size
    
    return analysis

def optimize_tokenization_config(texts: List[str], 
                                base_config: TokenizationConfig) -> TokenizationConfig:
    """Optimize tokenization configuration based on data analysis"""
    analysis = analyze_tokenization_quality(
        AutoTokenizer.from_pretrained(base_config.model_name), texts
    )
    
    # Optimize max_length based on data distribution
    sequence_lengths = list(analysis['sequence_length_distribution'].keys())
    if sequence_lengths:
        # Use 95th percentile as max_length
        sorted_lengths = sorted(sequence_lengths)
        percentile_95 = sorted_lengths[int(len(sorted_lengths) * 0.95)]
        
        optimized_config = TokenizationConfig(
            model_name=base_config.model_name,
            max_length=min(percentile_95, 512),  # Cap at 512
            truncation=base_config.truncation,
            padding=base_config.padding,
            return_tensors=base_config.return_tensors,
            return_attention_mask=base_config.return_attention_mask,
            add_special_tokens=base_config.add_special_tokens
        )
        
        logger.info(f"Optimized max_length: {optimized_config.max_length}")
        return optimized_config
    
    return base_config 