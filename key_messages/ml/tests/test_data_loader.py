from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
from ..data_loader import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Tests for Data Loader Module
"""

    MessageDataset,
    DataPreprocessor,
    TextCleaner,
    FeatureExtractor,
    DataLoaderFactory,
    DataManager,
    DEFAULT_DATA_CONFIG
)

class TestMessageDataset:
    """Test MessageDataset class."""
    
    def test_dataset_initialization(self) -> Any:
        """Test MessageDataset initialization."""
        # Create sample data
        data = pd.DataFrame({
            'message_id': [1, 2, 3],
            'original_message': ['Hello world', 'Test message', 'Another test'],
            'message_type': ['informational', 'promotional', 'informational'],
            'tone': ['professional', 'casual', 'professional'],
            'target_audience': ['general', 'young', 'general'],
            'industry': ['tech', 'retail', 'tech'],
            'keywords': [['hello', 'world'], ['test'], ['another', 'test']],
            'generated_response': ['Response 1', 'Response 2', 'Response 3'],
            'engagement_metrics': [
                {'clicks': 10, 'conversions': 2},
                {'clicks': 5, 'conversions': 1},
                {'clicks': 15, 'conversions': 3}
            ],
            'quality_score': [0.8, 0.6, 0.9]
        })
        
        dataset = MessageDataset(data, max_length=512)
        
        assert len(dataset) == 3
        assert dataset.max_length == 512
        assert dataset.label_encoder is not None
    
    def test_dataset_getitem(self) -> Optional[Dict[str, Any]]:
        """Test MessageDataset __getitem__ method."""
        data = pd.DataFrame({
            'message_id': [1],
            'original_message': ['Hello world'],
            'message_type': ['informational'],
            'tone': ['professional'],
            'target_audience': ['general'],
            'industry': ['tech'],
            'keywords': [['hello', 'world']],
            'generated_response': ['Response 1'],
            'engagement_metrics': [{'clicks': 10, 'conversions': 2}],
            'quality_score': [0.8]
        })
        
        dataset = MessageDataset(data)
        sample = dataset[0]
        
        assert sample['id'] == 1
        assert sample['original_message'] == 'Hello world'
        assert sample['message_type'] == 'informational'
        assert sample['tone'] == 'professional'
        assert sample['target_audience'] == 'general'
        assert sample['industry'] == 'tech'
        assert sample['keywords'] == ['hello', 'world']
        assert sample['generated_response'] == 'Response 1'
        assert sample['quality_score'] == 0.8
    
    def test_dataset_with_tokenizer(self) -> Any:
        """Test MessageDataset with tokenizer."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        data = pd.DataFrame({
            'original_message': ['Hello world'],
            'message_type': ['informational']
        })
        
        dataset = MessageDataset(data, tokenizer=mock_tokenizer, max_length=10)
        sample = dataset[0]
        
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert sample['input_ids'].shape == (10,)
        assert sample['attention_mask'].shape == (10,)
    
    def test_dataset_tokenization_error_handling(self) -> Any:
        """Test MessageDataset tokenization error handling."""
        # Mock tokenizer that raises an exception
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = Exception("Tokenization failed")
        
        data = pd.DataFrame({
            'original_message': ['Hello world'],
            'message_type': ['informational']
        })
        
        dataset = MessageDataset(data, tokenizer=mock_tokenizer, max_length=10)
        sample = dataset[0]
        
        # Should return empty tensors as fallback
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert torch.all(sample['input_ids'] == 0)
        assert torch.all(sample['attention_mask'] == 0)

class TestTextCleaner:
    """Test TextCleaner class."""
    
    def test_text_cleaner_initialization(self) -> Any:
        """Test TextCleaner initialization."""
        cleaner = TextCleaner()
        
        assert cleaner.url_pattern is not None
        assert cleaner.email_pattern is not None
        assert cleaner.phone_pattern is not None
    
    def test_clean_text_basic(self) -> Any:
        """Test basic text cleaning."""
        cleaner = TextCleaner()
        
        text = "Hello World! This is a TEST message."
        cleaned = cleaner.clean_text(text)
        
        assert cleaned == "hello world! this is a test message."
    
    def test_clean_text_with_urls(self) -> Any:
        """Test text cleaning with URLs."""
        cleaner = TextCleaner()
        
        text = "Check out this link: https://example.com and also http://test.org"
        cleaned = cleaner.clean_text(text)
        
        assert "[URL]" in cleaned
        assert "https://example.com" not in cleaned
        assert "http://test.org" not in cleaned
    
    def test_clean_text_with_emails(self) -> Any:
        """Test text cleaning with emails."""
        cleaner = TextCleaner()
        
        text = "Contact us at test@example.com or support@company.org"
        cleaned = cleaner.clean_text(text)
        
        assert "[EMAIL]" in cleaned
        assert "test@example.com" not in cleaned
        assert "support@company.org" not in cleaned
    
    def test_clean_text_with_phone_numbers(self) -> Any:
        """Test text cleaning with phone numbers."""
        cleaner = TextCleaner()
        
        text = "Call us at 123-456-7890 or 987.654.3210"
        cleaned = cleaner.clean_text(text)
        
        assert "[PHONE]" in cleaned
        assert "123-456-7890" not in cleaned
        assert "987.654.3210" not in cleaned
    
    def test_clean_text_with_extra_whitespace(self) -> Any:
        """Test text cleaning with extra whitespace."""
        cleaner = TextCleaner()
        
        text = "Hello    world!   This   has   extra   spaces."
        cleaned = cleaner.clean_text(text)
        
        assert "    " not in cleaned
        assert cleaned == "hello world! this has extra spaces."
    
    def test_clean_text_with_special_characters(self) -> Any:
        """Test text cleaning with special characters."""
        cleaner = TextCleaner()
        
        text = "Hello @#$%^&*() world! This has special chars: <>[]{}|\\"
        cleaned = cleaner.clean_text(text)
        
        # Should keep essential punctuation but remove special chars
        assert "@#$%^&*()" not in cleaned
        assert "<>[]{}|\\" not in cleaned
        assert "hello world! this has special chars:" in cleaned
    
    def test_clean_text_empty_input(self) -> Any:
        """Test text cleaning with empty input."""
        cleaner = TextCleaner()
        
        assert cleaner.clean_text("") == ""
        assert cleaner.clean_text(None) == ""
        assert cleaner.clean_text(123) == ""
    
    def test_normalize_whitespace(self) -> Any:
        """Test whitespace normalization."""
        cleaner = TextCleaner()
        
        text = "Hello    world!   \n\nThis   has   \t\t\twhitespace."
        normalized = cleaner.normalize_whitespace(text)
        
        assert normalized == "Hello world! This has whitespace."
    
    def test_remove_special_chars(self) -> Any:
        """Test special character removal."""
        cleaner = TextCleaner()
        
        text = "Hello @#$%^&*() world! This has special chars: <>[]{}|\\"
        
        # Keep punctuation
        cleaned_with_punct = cleaner.remove_special_chars(text, keep_punctuation=True)
        assert "hello world! this has special chars:" in cleaned_with_punct.lower()
        assert "@#$%^&*()" not in cleaned_with_punct
        
        # Remove all special chars
        cleaned_no_punct = cleaner.remove_special_chars(text, keep_punctuation=False)
        assert "hello world this has special chars" in cleaned_no_punct.lower()
        assert "!" not in cleaned_no_punct

class TestFeatureExtractor:
    """Test FeatureExtractor class."""
    
    def test_feature_extractor_initialization(self) -> Any:
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        
        assert extractor.sentiment_analyzer is None
    
    def test_extract_text_features_basic(self) -> Any:
        """Test basic text feature extraction."""
        extractor = FeatureExtractor()
        
        text = "Hello world! This is a test message with some words."
        features = extractor.extract_text_features(text)
        
        assert features['length'] == len(text)
        assert features['word_count'] == 10
        assert features['sentence_count'] == 2
        assert features['avg_word_length'] > 0
        assert features['unique_words'] == 9  # "is" appears twice
        assert features['hashtag_count'] == 0
        assert features['mention_count'] == 0
        assert features['url_count'] == 0
        assert features['exclamation_count'] == 1
        assert features['question_count'] == 0
        assert features['uppercase_ratio'] >= 0
    
    def test_extract_text_features_with_hashtags_mentions(self) -> Any:
        """Test text feature extraction with hashtags and mentions."""
        extractor = FeatureExtractor()
        
        text = "Hello @user! Check out #awesome #content and follow @company"
        features = extractor.extract_text_features(text)
        
        assert features['hashtag_count'] == 2
        assert features['mention_count'] == 2
    
    def test_extract_text_features_with_urls(self) -> Any:
        """Test text feature extraction with URLs."""
        extractor = FeatureExtractor()
        
        text = "Visit https://example.com and http://test.org for more info"
        features = extractor.extract_text_features(text)
        
        assert features['url_count'] == 2
    
    def test_extract_text_features_with_questions_exclamations(self) -> Any:
        """Test text feature extraction with questions and exclamations."""
        extractor = FeatureExtractor()
        
        text = "Hello! How are you? This is amazing! What do you think?"
        features = extractor.extract_text_features(text)
        
        assert features['exclamation_count'] == 2
        assert features['question_count'] == 2
    
    def test_extract_text_features_empty_input(self) -> Any:
        """Test text feature extraction with empty input."""
        extractor = FeatureExtractor()
        
        features = extractor.extract_text_features("")
        
        assert features['length'] == 0
        assert features['word_count'] == 0
        assert features['sentence_count'] == 0
        assert features['avg_word_length'] == 0.0
        assert features['unique_words'] == 0
        assert features['uppercase_ratio'] == 0.0
    
    def test_calculate_avg_word_length(self) -> Any:
        """Test average word length calculation."""
        extractor = FeatureExtractor()
        
        assert extractor._calculate_avg_word_length("Hello world") == 5.0
        assert extractor._calculate_avg_word_length("A") == 1.0
        assert extractor._calculate_avg_word_length("") == 0.0
        assert extractor._calculate_avg_word_length(None) == 0.0

class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def test_preprocessor_initialization(self) -> Any:
        """Test DataPreprocessor initialization."""
        config = {'max_length': 512}
        preprocessor = DataPreprocessor(config)
        
        assert preprocessor.config == config
        assert preprocessor.text_cleaner is not None
        assert preprocessor.feature_extractor is not None
    
    def test_preprocess_data_basic(self) -> Any:
        """Test basic data preprocessing."""
        config = {'max_length': 512}
        preprocessor = DataPreprocessor(config)
        
        data = pd.DataFrame({
            'original_message': ['Hello world', 'Test message'],
            'message_type': ['informational', 'promotional'],
            'tone': ['professional', 'casual'],
            'target_audience': ['general', 'young'],
            'industry': ['tech', 'retail'],
            'keywords': [['hello', 'world'], ['test']],
            'generated_response': ['Response 1', 'Response 2'],
            'engagement_metrics': [
                {'clicks': 10, 'conversions': 2},
                {'clicks': 5, 'conversions': 1}
            ],
            'quality_score': [0.8, 0.6]
        })
        
        processed_data = preprocessor.preprocess_data(data)
        
        assert len(processed_data) == 2
        assert 'text_length' in processed_data.columns
        assert 'word_count' in processed_data.columns
        assert 'avg_word_length' in processed_data.columns
        assert 'clicks' in processed_data.columns
        assert 'conversions' in processed_data.columns
        assert 'audience_size' in processed_data.columns
        assert 'industry_encoded' in processed_data.columns
    
    def test_handle_missing_values(self) -> Any:
        """Test missing value handling."""
        config = {'max_length': 512}
        preprocessor = DataPreprocessor(config)
        
        data = pd.DataFrame({
            'original_message': ['Hello world', 'Test message', 'Another test'],
            'message_type': ['informational', None, 'promotional'],
            'tone': [None, 'casual', 'professional'],
            'target_audience': ['general', 'young', None],
            'industry': ['tech', None, 'retail'],
            'keywords': [['hello'], None, ['test']],
            'quality_score': [0.8, None, 0.6]
        })
        
        processed_data = preprocessor._handle_missing_values(data)
        
        # Check that missing values are filled
        assert processed_data['message_type'].isna().sum() == 0
        assert processed_data['tone'].isna().sum() == 0
        assert processed_data['target_audience'].isna().sum() == 0
        assert processed_data['industry'].isna().sum() == 0
        assert processed_data['quality_score'].isna().sum() == 0
    
    def test_extract_engagement_features(self) -> Any:
        """Test engagement feature extraction."""
        config = {'max_length': 512}
        preprocessor = DataPreprocessor(config)
        
        data = pd.DataFrame({
            'engagement_metrics': [
                {'clicks': 10, 'conversions': 2, 'shares': 5, 'comments': 3},
                {'clicks': 5, 'conversions': 1},
                {'shares': 10, 'comments': 7}
            ]
        })
        
        processed_data = preprocessor._extract_engagement_features(data)
        
        assert 'clicks' in processed_data.columns
        assert 'conversions' in processed_data.columns
        assert 'shares' in processed_data.columns
        assert 'comments' in processed_data.columns
        
        assert processed_data['clicks'].iloc[0] == 10
        assert processed_data['conversions'].iloc[0] == 2
        assert processed_data['shares'].iloc[0] == 5
        assert processed_data['comments'].iloc[0] == 3
    
    def test_encode_audience_size(self) -> Any:
        """Test audience size encoding."""
        config = {'max_length': 512}
        preprocessor = DataPreprocessor(config)
        
        assert preprocessor._encode_audience_size('small niche audience') == 'small'
        assert preprocessor._encode_audience_size('large mass market') == 'large'
        assert preprocessor._encode_audience_size('medium sized') == 'medium'
    
    def test_encode_industry(self) -> Any:
        """Test industry encoding."""
        config = {'max_length': 512}
        preprocessor = DataPreprocessor(config)
        
        assert preprocessor._encode_industry('technology software') == 'tech'
        assert preprocessor._encode_industry('finance banking') == 'finance'
        assert preprocessor._encode_industry('healthcare medical') == 'healthcare'
        assert preprocessor._encode_industry('education learning') == 'education'
        assert preprocessor._encode_industry('retail shopping') == 'retail'
        assert preprocessor._encode_industry('unknown industry') == 'general'

class TestDataLoaderFactory:
    """Test DataLoaderFactory class."""
    
    def test_create_dataloader(self) -> Any:
        """Test DataLoader creation."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        dataloader = DataLoaderFactory.create_dataloader(
            mock_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 32
        assert dataloader.shuffle is True
        assert dataloader.num_workers == 2
        assert dataloader.pin_memory is True
    
    def test_create_train_val_test_loaders(self) -> Any:
        """Test train/val/test data loader creation."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        
        train_loader, val_loader, test_loader = DataLoaderFactory.create_train_val_test_loaders(
            mock_dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=16
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        assert train_loader.batch_size == 16
        assert val_loader.batch_size == 16
        assert test_loader.batch_size == 16
        
        assert train_loader.shuffle is True
        assert val_loader.shuffle is False
        assert test_loader.shuffle is False

class TestDataManager:
    """Test DataManager class."""
    
    def test_data_manager_initialization(self) -> Any:
        """Test DataManager initialization."""
        config = {'max_length': 512, 'cache_dir': './test_cache'}
        manager = DataManager(config)
        
        assert manager.config == config
        assert manager.preprocessor is not None
        assert manager.cache_dir.exists()
    
    def test_load_data_csv(self) -> Any:
        """Test loading CSV data."""
        config = {'max_length': 512, 'cache_dir': './test_cache'}
        manager = DataManager(config)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("original_message,message_type,tone\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("Hello world,informational,professional\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("Test message,promotional,casual\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            temp_file = f.name
        
        try:
            data = manager.load_data(temp_file)
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 2
            assert 'original_message' in data.columns
            assert 'message_type' in data.columns
            assert 'tone' in data.columns
        finally:
            os.unlink(temp_file)
    
    def test_load_data_json(self) -> Any:
        """Test loading JSON data."""
        config = {'max_length': 512, 'cache_dir': './test_cache'}
        manager = DataManager(config)
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('''[
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                {"original_message": "Hello world", "message_type": "informational"},
                {"original_message": "Test message", "message_type": "promotional"}
            ]''')
            temp_file = f.name
        
        try:
            data = manager.load_data(temp_file)
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 2
            assert 'original_message' in data.columns
            assert 'message_type' in data.columns
        finally:
            os.unlink(temp_file)
    
    def test_load_data_unsupported_format(self) -> Any:
        """Test loading unsupported file format."""
        config = {'max_length': 512, 'cache_dir': './test_cache'}
        manager = DataManager(config)
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test data")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                manager.load_data(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_create_dataset(self) -> Any:
        """Test dataset creation."""
        config = {'max_length': 512, 'cache_dir': './test_cache'}
        manager = DataManager(config)
        
        data = pd.DataFrame({
            'original_message': ['Hello world', 'Test message'],
            'message_type': ['informational', 'promotional']
        })
        
        dataset = manager.create_dataset(data)
        
        assert isinstance(dataset, MessageDataset)
        assert len(dataset) == 2
    
    def test_get_data_loaders(self) -> Optional[Dict[str, Any]]:
        """Test data loader creation."""
        config = {'max_length': 512, 'cache_dir': './test_cache'}
        manager = DataManager(config)
        
        data = pd.DataFrame({
            'original_message': ['Hello world', 'Test message', 'Another test'],
            'message_type': ['informational', 'promotional', 'informational']
        })
        
        dataset = manager.create_dataset(data)
        train_loader, val_loader, test_loader = manager.get_data_loaders(dataset, batch_size=2)
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

class TestDefaultConfigurations:
    """Test default configurations."""
    
    def test_default_data_config(self) -> Any:
        """Test DEFAULT_DATA_CONFIG."""
        assert DEFAULT_DATA_CONFIG['max_length'] == 512
        assert DEFAULT_DATA_CONFIG['batch_size'] == 32
        assert DEFAULT_DATA_CONFIG['num_workers'] == 4
        assert DEFAULT_DATA_CONFIG['pin_memory'] is True
        assert DEFAULT_DATA_CONFIG['cache_dir'] == './cache'
        assert DEFAULT_DATA_CONFIG['train_ratio'] == 0.7
        assert DEFAULT_DATA_CONFIG['val_ratio'] == 0.15
        assert DEFAULT_DATA_CONFIG['test_ratio'] == 0.15

match __name__:
    case "__main__":
    pytest.main([__file__]) 