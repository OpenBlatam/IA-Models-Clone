from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, Any, List, Optional, Union, Tuple, Iterator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
import numpy as np
from datasets import Dataset as HFDataset
import re
import json
import os
from datetime import datetime
import asyncio
from functools import lru_cache
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
import hashlib
from collections import defaultdict, Counter
import unicodedata
try:
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize, sent_tokenize as _nltk_sent_tokenize  # type: ignore
    from nltk.corpus import stopwords as _nltk_stopwords  # type: ignore
    import nltk  # type: ignore
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_config import settings
from onyx.server.features.ads.training_logger import TrainingLogger, TrainingPhase, AsyncTrainingLogger
from typing import Any, List, Dict, Optional
import logging
"""
Advanced tokenization and sequence handling service for ads generation.
Implements proper text preprocessing, tokenization strategies, and sequence management.
"""
    AutoTokenizer, 
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2SeqLM,
    DataCollatorWithPadding
)

def _safe_nltk_download() -> None:
    if not _NLTK_AVAILABLE:
        return
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass


logger = setup_logger()

class TextPreprocessor:
    """Advanced text preprocessing for ads content."""
    
    def __init__(self) -> Any:
        """Initialize text preprocessor."""
        self.stop_words = set()
        if _NLTK_AVAILABLE:
            _safe_nltk_download()
            try:
                self.stop_words = set(_nltk_stopwords.words('english'))
            except Exception:
                self.stop_words = set()
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.special_chars_pattern = re.compile(r'[^\w\s\-.,!?;:()]')
        
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing special characters and normalizing unicode."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        text = self.url_pattern.sub('[URL]', text)
        
        # Remove email addresses
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub('[PHONE]', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = self.special_chars_pattern.sub('', text)
        
        return text.strip()
    
    def clean_ads_text(self, text: str, remove_stopwords: bool = False) -> str:
        """Clean text specifically for ads content."""
        if not text:
            return ""
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            words = word_tokenize(text)
            words = [word for word in words if word.lower() not in self.stop_words]
            text = ' '.join(words)
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using frequency analysis."""
        if not text:
            return []
        
        # Clean text
        clean_text = self.normalize_text(text)
        
        # Tokenize and filter
        if _NLTK_AVAILABLE:
            try:
                words = _nltk_word_tokenize(clean_text)
            except Exception:
                words = clean_text.split()
        else:
            words = clean_text.split()
        words = [word for word in words if len(word) > 2 and word.lower() not in self.stop_words]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Return top keywords
        return [word for word, freq in word_freq.most_common(max_keywords)]
    
    def segment_text(self, text: str, max_segment_length: int = 512) -> List[str]:
        """Segment long text into smaller chunks."""
        if not text:
            return []
        
        # Split by sentences first
        if _NLTK_AVAILABLE:
            try:
                sentences = _nltk_sent_tokenize(text)
            except Exception:
                import re as _re
                sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        else:
            import re as _re
            sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= max_segment_length:
                current_segment += sentence + " "
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + " "
        
        # Add the last segment
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments

class AdvancedTokenizer:
    """Advanced tokenizer with caching and optimization."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize advanced tokenizer."""
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer()
        self.preprocessor = TextPreprocessor()
        self._cache = {}
        self._cache_size = 1000
        
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set special tokens
        special_tokens = {
            'additional_special_tokens': [
                '[URL]', '[EMAIL]', '[PHONE]', '[AD_START]', '[AD_END]',
                '[TARGET_AUDIENCE]', '[KEYWORDS]', '[BRAND]', '[CTA]'
            ]
        }
        
        # Add special tokens if they don't exist
        num_added = tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens to tokenizer")
        
        return tokenizer
    
    def _get_cache_key(self, text: str, max_length: int, truncation: bool) -> str:
        """Generate cache key for tokenization."""
        return hashlib.md5(f"{text}:{max_length}:{truncation}".encode()).hexdigest()
    
    def tokenize_text(
        self, 
        text: str, 
        max_length: int = 512, 
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: str = "pt",
        use_cache: bool = True
    ) -> BatchEncoding:
        """Tokenize text with caching and optimization."""
        if not text:
            return self.tokenizer(
                "",
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors=return_tensors
            )
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text, max_length, truncation)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Preprocess text
        processed_text = self.preprocessor.normalize_text(text)
        
        # Tokenize
        result = self.tokenizer(
            processed_text,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors
        )
        
        # Cache result
        if use_cache and len(self._cache) < self._cache_size:
            cache_key = self._get_cache_key(text, max_length, truncation)
            self._cache[cache_key] = result
        
        return result
    
    def tokenize_ads_prompt(
        self,
        prompt: str,
        target_audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        brand: Optional[str] = None,
        max_length: int = 512
    ) -> BatchEncoding:
        """Tokenize ads prompt with structured format."""
        # Build structured prompt
        structured_prompt = f"[AD_START] {prompt}"
        
        if target_audience:
            structured_prompt += f" [TARGET_AUDIENCE] {target_audience}"
        
        if keywords:
            structured_prompt += f" [KEYWORDS] {', '.join(keywords)}"
        
        if brand:
            structured_prompt += f" [BRAND] {brand}"
        
        structured_prompt += " [AD_END]"
        
        return self.tokenize_text(structured_prompt, max_length=max_length)
    
    def decode_tokens(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings."""
        return {
            'pad_token': self.tokenizer.pad_token_id,
            'eos_token': self.tokenizer.eos_token_id,
            'bos_token': self.tokenizer.bos_token_id,
            'unk_token': self.tokenizer.unk_token_id
        }

class SequenceManager:
    """Manages sequence handling for training and inference."""
    
    def __init__(self, tokenizer: AdvancedTokenizer):
        """Initialize sequence manager."""
        self.tokenizer = tokenizer
        self.max_sequence_length = 512
        self.min_sequence_length = 10
        
    def create_training_sequences(
        self,
        prompts: List[str],
        targets: List[str],
        max_length: int = 512
    ) -> List[Dict[str, torch.Tensor]]:
        """Create training sequences from prompts and targets."""
        sequences = []
        
        for prompt, target in zip(prompts, targets):
            if not prompt or not target:
                continue
            
            # Tokenize prompt and target
            prompt_tokens = self.tokenizer.tokenize_text(
                prompt, 
                max_length=max_length // 2,
                truncation=True,
                return_tensors="pt"
            )
            
            target_tokens = self.tokenizer.tokenize_text(
                target,
                max_length=max_length // 2,
                truncation=True,
                return_tensors="pt"
            )
            
            # Combine sequences
            input_ids = torch.cat([
                prompt_tokens['input_ids'],
                target_tokens['input_ids']
            ], dim=1)
            
            attention_mask = torch.cat([
                prompt_tokens['attention_mask'],
                target_tokens['attention_mask']
            ], dim=1)
            
            # Create labels (only for target part)
            labels = torch.cat([
                torch.full_like(prompt_tokens['input_ids'], -100),  # Ignore prompt in loss
                target_tokens['input_ids']
            ], dim=1)
            
            # Truncate if necessary
            if input_ids.shape[1] > max_length:
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
                labels = labels[:, :max_length]
            
            sequences.append({
                'input_ids': input_ids.squeeze(),
                'attention_mask': attention_mask.squeeze(),
                'labels': labels.squeeze()
            })
        
        return sequences
    
    def create_inference_sequence(
        self,
        prompt: str,
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Create sequence for inference."""
        tokens = self.tokenizer.tokenize_text(
            prompt,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
    
    def pad_sequences(
        self,
        sequences: List[Dict[str, torch.Tensor]],
        padding: str = "max_length",
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Pad sequences to same length."""
        if not sequences:
            return {}
        
        # Find max length
        if padding == "max_length":
            target_length = max_length
        else:
            target_length = max(seq['input_ids'].shape[0] for seq in sequences)
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for seq in sequences:
            current_length = seq['input_ids'].shape[0]
            padding_length = target_length - current_length
            
            # Pad input_ids
            padded_input = torch.cat([
                seq['input_ids'],
                torch.full((padding_length,), self.tokenizer.tokenizer.pad_token_id)
            ])
            padded_input_ids.append(padded_input)
            
            # Pad attention_mask
            padded_attention = torch.cat([
                seq['attention_mask'],
                torch.zeros(padding_length)
            ])
            padded_attention_mask.append(padded_attention)
            
            # Pad labels
            if 'labels' in seq:
                padded_label = torch.cat([
                    seq['labels'],
                    torch.full((padding_length,), -100)  # Ignore padding in loss
                ])
                padded_labels.append(padded_label)
        
        result = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask)
        }
        
        if padded_labels:
            result['labels'] = torch.stack(padded_labels)
        
        return result

class OptimizedAdsDataset(Dataset):
    """Optimized dataset for ads generation with advanced tokenization."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AdvancedTokenizer,
        max_length: int = 512,
        use_cache: bool = True
    ):
        """Initialize dataset."""
        self.data = data
        self.tokenizer = tokenizer
        self.sequence_manager = SequenceManager(tokenizer)
        self.max_length = max_length
        self.use_cache = use_cache
        self._cached_sequences = {}
        
    def __len__(self) -> Any:
        return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        """Get item with caching."""
        if self.use_cache and idx in self._cached_sequences:
            return self._cached_sequences[idx]
        
        item = self.data[idx]
        
        # Format input for ads generation
        if item.get("type") == "ads":
            prompt = f"Generate an ad for: {item['prompt']}"
            if item.get("target_audience"):
                prompt += f"\nTarget audience: {item['target_audience']}"
            if item.get("keywords"):
                prompt += f"\nKeywords: {', '.join(item['keywords'])}"
            
            target = item.get("content", {}).get("content", "")
        else:
            prompt = item.get("prompt", "")
            target = item.get("content", {}).get("content", "")
        
        # Create sequence
        sequences = self.sequence_manager.create_training_sequences(
            [prompt], [target], self.max_length
        )
        
        if sequences:
            result = sequences[0]
            
            # Cache result
            if self.use_cache:
                self._cached_sequences[idx] = result
            
            return result
        else:
            # Return empty sequence if processing failed
            return {
                'input_ids': torch.tensor([]),
                'attention_mask': torch.tensor([]),
                'labels': torch.tensor([])
            }

class TokenizationService:
    """Main tokenization service for ads generation."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize tokenization service."""
        self.tokenizer = AdvancedTokenizer(model_name)
        self.sequence_manager = SequenceManager(self.tokenizer)
        self.preprocessor = TextPreprocessor()
        self._redis_client = None
        self.training_logger = None
        
    @property
    async def redis_client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_client
    
    async def tokenize_ads_data(
        self,
        ads_data: List[Dict[str, Any]],
        max_length: int = 512,
        use_cache: bool = True,
        user_id: Optional[int] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Tokenize ads data for training."""
        # Initialize training logger if user_id provided
        if user_id and not self.training_logger:
            self.training_logger = AsyncTrainingLogger(
                user_id=user_id,
                model_name=self.model_name,
                log_dir=f"logs/tokenization/user_{user_id}"
            )
        
        if self.training_logger:
            self.training_logger.log_info("Starting ads data tokenization", TrainingPhase.DATA_PREPARATION)
            self.training_logger.log_info(f"Processing {len(ads_data)} items with max_length={max_length}")
        
        tokenized_data = []
        failed_items = 0
        
        for i, item in enumerate(ads_data):
            try:
                # Preprocess text
                prompt = self.preprocessor.clean_ads_text(item.get("prompt", ""))
                content = self.preprocessor.clean_ads_text(item.get("content", {}).get("content", ""))
                
                if not prompt or not content:
                    failed_items += 1
                    continue
                
                # Create training sequence
                sequences = self.sequence_manager.create_training_sequences(
                    [prompt], [content], max_length
                )
                
                if sequences:
                    tokenized_data.append(sequences[0])
                else:
                    failed_items += 1
                    
            except Exception as e:
                failed_items += 1
                if self.training_logger:
                    self.training_logger.log_warning(f"Failed to tokenize item {i}: {e}")
                logger.warning(f"Failed to tokenize item: {e}")
                continue
            
            # Log progress every 100 items
            if (i + 1) % 100 == 0 and self.training_logger:
                self.training_logger.log_info(f"Processed {i + 1}/{len(ads_data)} items")
        
        if self.training_logger:
            self.training_logger.log_info(f"Tokenization completed: {len(tokenized_data)} successful, {failed_items} failed")
        
        logger.info(f"Tokenized {len(tokenized_data)} items from {len(ads_data)} total")
        return tokenized_data
    
    async def create_training_dataset(
        self,
        ads_data: List[Dict[str, Any]],
        max_length: int = 512,
        batch_size: int = 8
    ) -> DataLoader:
        """Create training dataset with DataLoader."""
        # Tokenize data
        tokenized_data = await self.tokenize_ads_data(ads_data, max_length)
        
        # Create dataset
        dataset = OptimizedAdsDataset(tokenized_data, self.tokenizer, max_length)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer.tokenizer,
            mlm=False
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=2,
            pin_memory=True
        )
        
        return dataloader
    
    async def tokenize_for_inference(
        self,
        prompt: str,
        target_audience: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompt for inference."""
        # Preprocess prompt
        clean_prompt = self.preprocessor.clean_ads_text(prompt)
        
        # Create inference sequence
        sequence = self.sequence_manager.create_inference_sequence(
            clean_prompt, max_length
        )
        
        return sequence
    
    async def analyze_text_complexity(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Analyze text complexity for tokenization optimization."""
        if not text:
            return {}
        
        # Tokenize text
        tokens = self.tokenizer.tokenize_text(text)
        token_count = len(tokens['input_ids'][0])
        
        # Analyze text characteristics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        # Calculate metrics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        # Extract keywords
        keywords = self.preprocessor.extract_keywords(text)
        
        return {
            'token_count': token_count,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'unique_words': unique_words,
            'vocabulary_diversity': vocabulary_diversity,
            'keywords': keywords,
            'complexity_score': token_count / len(words) if words else 0
        }
    
    async def optimize_sequence_length(
        self,
        texts: List[str],
        target_token_count: int = 512
    ) -> List[str]:
        """Optimize sequence lengths for batch processing."""
        optimized_texts = []
        
        for text in texts:
            # Analyze text
            analysis = await self.analyze_text_complexity(text)
            
            if analysis['token_count'] <= target_token_count:
                optimized_texts.append(text)
            else:
                # Segment text if too long
                segments = self.preprocessor.segment_text(text, target_token_count)
                optimized_texts.extend(segments)
        
        return optimized_texts
    
    async def get_tokenization_stats(
        self,
        ads_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get tokenization statistics for dataset."""
        total_tokens = 0
        total_words = 0
        complexity_scores = []
        
        for item in ads_data:
            prompt = item.get("prompt", "")
            content = item.get("content", {}).get("content", "")
            
            # Analyze prompt
            if prompt:
                prompt_analysis = await self.analyze_text_complexity(prompt)
                total_tokens += prompt_analysis.get('token_count', 0)
                total_words += prompt_analysis.get('word_count', 0)
                complexity_scores.append(prompt_analysis.get('complexity_score', 0))
            
            # Analyze content
            if content:
                content_analysis = await self.analyze_text_complexity(content)
                total_tokens += content_analysis.get('token_count', 0)
                total_words += content_analysis.get('word_count', 0)
                complexity_scores.append(content_analysis.get('complexity_score', 0))
        
        return {
            'total_items': len(ads_data),
            'total_tokens': total_tokens,
            'total_words': total_words,
            'avg_tokens_per_item': total_tokens / len(ads_data) if ads_data else 0,
            'avg_words_per_item': total_words / len(ads_data) if ads_data else 0,
            'avg_complexity_score': np.mean(complexity_scores) if complexity_scores else 0,
            'vocabulary_size': self.tokenizer.get_vocab_size()
        }
    
    async def close(self) -> Any:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close() 