from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import re
from pathlib import Path
import structlog
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import hashlib
import os
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Data Loading and Preprocessing for Key Messages Feature - Modular Architecture
"""

logger = structlog.get_logger(__name__)

class MessageDataset(Dataset):
    """Custom dataset for key messages."""
    
    def __init__(self, data: pd.DataFrame, tokenizer=None, max_length: int = 512):
        
    """__init__ function."""
self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        
        # Encode labels if they exist
        if 'message_type' in self.data.columns:
            self.data['message_type_encoded'] = self.label_encoder.fit_transform(self.data['message_type'])
        
        logger.info("MessageDataset initialized", 
                   size=len(data), 
                   max_length=max_length,
                   columns=list(data.columns))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        row = self.data.iloc[idx]
        
        sample = {
            'id': row.get('message_id', idx),
            'original_message': row.get('original_message', ''),
            'message_type': row.get('message_type', 'informational'),
            'tone': row.get('tone', 'professional'),
            'target_audience': row.get('target_audience', ''),
            'industry': row.get('industry', ''),
            'keywords': row.get('keywords', []),
            'generated_response': row.get('generated_response', ''),
            'engagement_metrics': row.get('engagement_metrics', {}),
            'quality_score': row.get('quality_score', 0.0)
        }
        
        # Tokenize if tokenizer is available
        if self.tokenizer:
            sample.update(self._tokenize_text(sample['original_message']))
        
        return sample
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using the provided tokenizer."""
        try:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
        except Exception as e:
            logger.error("Tokenization failed", error=str(e), text=text[:100])
            # Return empty tensors as fallback
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }

class DataPreprocessor:
    """Data preprocessing pipeline for key messages."""
    
    def __init__(self, config: Dict[str, Any]):
        
    """__init__ function."""
self.config = config
        self.text_cleaner = TextCleaner()
        self.feature_extractor = FeatureExtractor()
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Complete data preprocessing pipeline."""
        logger.info("Starting data preprocessing", initial_size=len(data))
        
        # 1. Clean data
        data = self._clean_data(data)
        
        # 2. Extract features
        data = self._extract_features(data)
        
        # 3. Validate data
        data = self._validate_data(data)
        
        # 4. Encode categorical variables
        data = self._encode_categorical(data)
        
        logger.info("Data preprocessing completed", final_size=len(data))
        return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data."""
        # Remove duplicates
        initial_size = len(data)
        data = data.drop_duplicates()
        logger.info("Removed duplicates", removed=initial_size - len(data))
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Clean text fields
        if 'original_message' in data.columns:
            data['original_message'] = data['original_message'].apply(self.text_cleaner.clean_text)
        
        if 'generated_response' in data.columns:
            data['generated_response'] = data['generated_response'].apply(self.text_cleaner.clean_text)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill missing values with defaults
        fill_values = {
            'message_type': 'informational',
            'tone': 'professional',
            'target_audience': 'general',
            'industry': 'general',
            'keywords': [],
            'quality_score': 0.5
        }
        
        for column, default_value in fill_values.items():
            if column in data.columns:
                data[column] = data[column].fillna(default_value)
        
        # Remove rows with critical missing values
        critical_columns = ['original_message']
        data = data.dropna(subset=critical_columns)
        
        return data
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract additional features from the data."""
        # Text features
        if 'original_message' in data.columns:
            data['text_length'] = data['original_message'].str.len()
            data['word_count'] = data['original_message'].str.split().str.len()
            data['avg_word_length'] = data['original_message'].apply(self._calculate_avg_word_length)
        
        # Engagement features
        if 'engagement_metrics' in data.columns:
            data = self._extract_engagement_features(data)
        
        # Context features
        data = self._extract_context_features(data)
        
        return data
    
    def _calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        if not text or not isinstance(text, str):
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def _extract_engagement_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from engagement metrics."""
        def extract_metrics(metrics) -> Any:
            if isinstance(metrics, dict):
                return {
                    'clicks': metrics.get('clicks', 0),
                    'conversions': metrics.get('conversions', 0),
                    'shares': metrics.get('shares', 0),
                    'comments': metrics.get('comments', 0)
                }
            return {'clicks': 0, 'conversions': 0, 'shares': 0, 'comments': 0}
        
        engagement_df = data['engagement_metrics'].apply(extract_metrics).apply(pd.Series)
        data = pd.concat([data, engagement_df], axis=1)
        
        return data
    
    def _extract_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract contextual features."""
        # Audience size encoding
        if 'target_audience' in data.columns:
            data['audience_size'] = data['target_audience'].apply(self._encode_audience_size)
        
        # Industry encoding
        if 'industry' in data.columns:
            data['industry_encoded'] = data['industry'].apply(self._encode_industry)
        
        return data
    
    def _encode_audience_size(self, audience: str) -> str:
        """Encode audience size based on description."""
        audience_lower = audience.lower()
        if any(word in audience_lower for word in ['small', 'niche', 'specific']):
            return 'small'
        elif any(word in audience_lower for word in ['large', 'mass', 'general']):
            return 'large'
        else:
            return 'medium'
    
    def _encode_industry(self, industry: str) -> str:
        """Encode industry into categories."""
        industry_lower = industry.lower()
        industry_mapping = {
            'tech': ['technology', 'software', 'ai', 'machine learning'],
            'finance': ['finance', 'banking', 'investment'],
            'healthcare': ['healthcare', 'medical', 'pharmaceutical'],
            'education': ['education', 'learning', 'training'],
            'retail': ['retail', 'ecommerce', 'shopping']
        }
        
        for category, keywords in industry_mapping.items():
            if any(keyword in industry_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data according to business rules."""
        # Remove rows with invalid text length
        if 'text_length' in data.columns:
            data = data[(data['text_length'] >= 10) & (data['text_length'] <= 1000)]
        
        # Remove rows with invalid quality scores
        if 'quality_score' in data.columns:
            data = data[(data['quality_score'] >= 0.0) & (data['quality_score'] <= 1.0)]
        
        return data
    
    def _encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_columns = ['message_type', 'tone', 'audience_size', 'industry_encoded']
        
        for column in categorical_columns:
            if column in data.columns:
                encoder = LabelEncoder()
                data[f'{column}_encoded'] = encoder.fit_transform(data[column])
        
        return data

class TextCleaner:
    """Text cleaning utilities."""
    
    def __init__(self) -> Any:
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('[URL]', text)
        
        # Remove emails
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Remove phone numbers
        text = self.phone_pattern.sub('[PHONE]', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters (keep essential punctuation)
        text = re.sub(r'[^\w\s\.\!\?\,\:\;\-\(\)]', '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_special_chars(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters from text."""
        if keep_punctuation:
            return re.sub(r'[^\w\s\.\!\?\,\:\;\-\(\)]', '', text)
        else:
            return re.sub(r'[^\w\s]', '', text)

class FeatureExtractor:
    """Feature extraction utilities."""
    
    def __init__(self) -> Any:
        self.sentiment_analyzer = None  # Could be initialized with a sentiment analysis model
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive text features."""
        if not text:
            return self._empty_features()
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': self._calculate_avg_word_length(text),
            'unique_words': len(set(text.lower().split())),
            'hashtag_count': text.count('#'),
            'mention_count': text.count('@'),
            'url_count': text.count('http'),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        return features
    
    def _calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature set."""
        return {
            'length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'unique_words': 0,
            'hashtag_count': 0,
            'mention_count': 0,
            'url_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0.0
        }

class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create a DataLoader with optimal settings."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    @staticmethod
    def create_train_val_test_loaders(
        dataset: Dataset,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 32,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoaderFactory.create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoaderFactory.create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )
        test_loader = DataLoaderFactory.create_dataloader(
            test_dataset, batch_size=batch_size, shuffle=False, **kwargs
        )
        
        logger.info("Data loaders created", 
                   train_size=train_size,
                   val_size=val_size,
                   test_size=test_size,
                   batch_size=batch_size)
        
        return train_loader, val_loader, test_loader

class DataManager:
    """High-level data management class."""
    
    def __init__(self, config: Dict[str, Any]):
        
    """__init__ function."""
self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.cache_dir = Path(config.get('cache_dir', './cache'))
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_data(self, data_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load data from various formats with validation."""
        # Guard clauses for early validation
        if not data_path or not data_path.strip():
            raise ValueError("Data path cannot be empty")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        file_size = os.path.getsize(data_path)
        if file_size == 0:
            raise ValueError("Data file is empty")
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("Data file too large (max 100MB)")
        
        try:
            file_extension = os.path.splitext(data_path)[1].lower()
            
            if file_extension == '.json':
                return self._load_json_data(data_path)
            elif file_extension == '.csv':
                return self._load_csv_data(data_path)
            elif file_extension in ['.txt', '.md']:
                return self._load_text_data(data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error("Error loading data", error=str(e), data_path=data_path)
            raise

    def _validate_data_quality(self, data: List[Dict[str, Any]]) -> None:
        """Validate data quality with guard clauses."""
        if not data:
            raise ValueError("Data is empty")
        
        if len(data) < 10:
            raise ValueError("Insufficient data (minimum 10 samples required)")
        
        required_fields = ['text', 'label']
        for i, sample in enumerate(data):
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {i} is not a dictionary")
            
            for field in required_fields:
                if field not in sample:
                    raise ValueError(f"Sample {i} missing required field: {field}")
                
                if not sample[field] or not str(sample[field]).strip():
                    raise ValueError(f"Sample {i} has empty {field}")
            
            if len(str(sample['text'])) > 10000:
                raise ValueError(f"Sample {i} text too long (max 10000 characters)")
        
        logger.info("Data validation passed", sample_count=len(data))
    
    def _load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load raw data from various file formats."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _generate_cache_key(self, file_path: str) -> str:
        """Generate cache key based on file path and modification time."""
        file_path = Path(file_path)
        if not file_path.exists():
            return hashlib.md5(file_path.name.encode()).hexdigest()
        
        # Include file modification time in cache key
        mtime = file_path.stat().st_mtime
        key_string = f"{file_path.name}_{mtime}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def create_dataset(self, data: pd.DataFrame, tokenizer=None) -> MessageDataset:
        """Create a MessageDataset from preprocessed data."""
        return MessageDataset(data, tokenizer, self.config.get('max_length', 512))
    
    def get_data_loaders(
        self, 
        dataset: MessageDataset, 
        batch_size: int = 32,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders."""
        return DataLoaderFactory.create_train_val_test_loaders(
            dataset, batch_size=batch_size, **kwargs
        )

# Default configuration
DEFAULT_DATA_CONFIG = {
    'max_length': 512,
    'batch_size': 32,
    'num_workers': 4,
    'pin_memory': True,
    'cache_dir': './cache',
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
} 