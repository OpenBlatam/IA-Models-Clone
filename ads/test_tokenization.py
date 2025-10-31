from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import asyncio
import torch
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from onyx.server.features.ads.tokenization_service import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive tests for the tokenization and sequence handling service.
"""

    TextPreprocessor,
    AdvancedTokenizer,
    SequenceManager,
    OptimizedAdsDataset,
    TokenizationService
)

class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_normalize_text(self) -> Any:
        """Test text normalization."""
        # Test basic normalization
        text = "Hello World! Visit https://example.com"
        normalized = self.preprocessor.normalize_text(text)
        assert "hello world" in normalized.lower()
        assert "[URL]" in normalized
        
        # Test email normalization
        text = "Contact us at info@example.com"
        normalized = self.preprocessor.normalize_text(text)
        assert "[EMAIL]" in normalized
        
        # Test phone normalization
        text = "Call us at 555-123-4567"
        normalized = self.preprocessor.normalize_text(text)
        assert "[PHONE]" in normalized
        
        # Test empty text
        assert self.preprocessor.normalize_text("") == ""
        assert self.preprocessor.normalize_text(None) == ""
    
    def test_clean_ads_text(self) -> Any:
        """Test ads-specific text cleaning."""
        text = "Amazing product for sale! Call 555-1234"
        cleaned = self.preprocessor.clean_ads_text(text, remove_stopwords=False)
        assert "amazing product" in cleaned.lower()
        assert "[PHONE]" in cleaned
        
        # Test with stopwords removal
        cleaned_no_stopwords = self.preprocessor.clean_ads_text(text, remove_stopwords=True)
        assert len(cleaned_no_stopwords.split()) < len(cleaned.split())
    
    def test_extract_keywords(self) -> Any:
        """Test keyword extraction."""
        text = "Our premium product offers amazing features and benefits"
        keywords = self.preprocessor.extract_keywords(text, max_keywords=5)
        
        assert len(keywords) <= 5
        assert all(isinstance(k, str) for k in keywords)
        assert "premium" in keywords or "product" in keywords
    
    def test_segment_text(self) -> Any:
        """Test text segmentation."""
        # Test short text
        short_text = "Short text."
        segments = self.preprocessor.segment_text(short_text, max_segment_length=100)
        assert len(segments) == 1
        assert segments[0] == short_text
        
        # Test long text
        long_text = "Sentence one. " * 50  # Create long text
        segments = self.preprocessor.segment_text(long_text, max_segment_length=100)
        assert len(segments) > 1
        assert all(len(seg) <= 100 for seg in segments)

class TestAdvancedTokenizer:
    """Test cases for AdvancedTokenizer class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = "<|endoftext|>"
            mock_tokenizer.return_value.vocab_size = 50257
            mock_tokenizer.return_value.add_special_tokens.return_value = 9
            
            self.tokenizer = AdvancedTokenizer("microsoft/DialoGPT-medium")
    
    def test_tokenize_text(self) -> Any:
        """Test basic text tokenization."""
        with patch.object(self.tokenizer.tokenizer, '__call__') as mock_call:
            mock_call.return_value = {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }
            
            result = self.tokenizer.tokenize_text("Test text", max_length=512)
            
            assert 'input_ids' in result
            assert 'attention_mask' in result
            mock_call.assert_called_once()
    
    def test_tokenize_ads_prompt(self) -> Any:
        """Test ads prompt tokenization."""
        with patch.object(self.tokenizer, 'tokenize_text') as mock_tokenize:
            mock_tokenize.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            result = self.tokenizer.tokenize_ads_prompt(
                prompt="Generate ad",
                target_audience="Young professionals",
                keywords=["premium", "quality"],
                brand="TestBrand"
            )
            
            assert result is not None
            mock_tokenize.assert_called_once()
    
    def test_decode_tokens(self) -> Any:
        """Test token decoding."""
        token_ids = [1, 2, 3, 4, 5]
        with patch.object(self.tokenizer.tokenizer, 'decode') as mock_decode:
            mock_decode.return_value = "Decoded text"
            
            result = self.tokenizer.decode_tokens(token_ids)
            
            assert result == "Decoded text"
            mock_decode.assert_called_once_with(token_ids, skip_special_tokens=True)
    
    def test_get_vocab_size(self) -> Optional[Dict[str, Any]]:
        """Test vocabulary size retrieval."""
        assert self.tokenizer.get_vocab_size() == 50257
    
    def test_get_special_tokens(self) -> Optional[Dict[str, Any]]:
        """Test special tokens retrieval."""
        with patch.object(self.tokenizer.tokenizer, 'pad_token_id', 0):
            with patch.object(self.tokenizer.tokenizer, 'eos_token_id', 1):
                with patch.object(self.tokenizer.tokenizer, 'bos_token_id', 2):
                    with patch.object(self.tokenizer.tokenizer, 'unk_token_id', 3):
                        result = self.tokenizer.get_special_tokens()
                        
                        assert result['pad_token'] == 0
                        assert result['eos_token'] == 1
                        assert result['bos_token'] == 2
                        assert result['unk_token'] == 3

class TestSequenceManager:
    """Test cases for SequenceManager class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        with patch('onyx.server.features.ads.tokenization_service.AdvancedTokenizer'):
            self.tokenizer = Mock()
            self.sequence_manager = SequenceManager(self.tokenizer)
    
    def test_create_training_sequences(self) -> Any:
        """Test training sequence creation."""
        prompts = ["Generate ad for product"]
        targets = ["Amazing product offers..."]
        
        with patch.object(self.tokenizer, 'tokenize_text') as mock_tokenize:
            mock_tokenize.side_effect = [
                {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])},
                {'input_ids': torch.tensor([[4, 5, 6]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
            ]
            
            sequences = self.sequence_manager.create_training_sequences(prompts, targets, max_length=512)
            
            assert len(sequences) == 1
            assert 'input_ids' in sequences[0]
            assert 'attention_mask' in sequences[0]
            assert 'labels' in sequences[0]
    
    def test_create_inference_sequence(self) -> Any:
        """Test inference sequence creation."""
        prompt = "Generate ad for product"
        
        with patch.object(self.tokenizer, 'tokenize_text') as mock_tokenize:
            mock_tokenize.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            sequence = self.sequence_manager.create_inference_sequence(prompt, max_length=512)
            
            assert 'input_ids' in sequence
            assert 'attention_mask' in sequence
            assert 'labels' not in sequence
    
    def test_pad_sequences(self) -> Any:
        """Test sequence padding."""
        sequences = [
            {'input_ids': torch.tensor([1, 2, 3]), 'attention_mask': torch.tensor([1, 1, 1])},
            {'input_ids': torch.tensor([4, 5]), 'attention_mask': torch.tensor([1, 1])}
        ]
        
        with patch.object(self.tokenizer.tokenizer, 'pad_token_id', 0):
            padded = self.sequence_manager.pad_sequences(sequences, padding="max_length", max_length=5)
            
            assert 'input_ids' in padded
            assert 'attention_mask' in padded
            assert padded['input_ids'].shape[1] == 5

class TestOptimizedAdsDataset:
    """Test cases for OptimizedAdsDataset class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.data = [
            {
                'prompt': 'Generate ad for product',
                'content': {'content': 'Amazing product offers...'},
                'type': 'ads'
            }
        ]
        
        with patch('onyx.server.features.ads.tokenization_service.AdvancedTokenizer'):
            with patch('onyx.server.features.ads.tokenization_service.SequenceManager'):
                self.tokenizer = Mock()
                self.dataset = OptimizedAdsDataset(self.data, self.tokenizer, max_length=512)
    
    def test_len(self) -> Any:
        """Test dataset length."""
        assert len(self.dataset) == 1
    
    def test_getitem(self) -> Optional[Dict[str, Any]]:
        """Test dataset item retrieval."""
        with patch.object(self.dataset.sequence_manager, 'create_training_sequences') as mock_create:
            mock_create.return_value = [{
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'labels': torch.tensor([4, 5, 6])
            }]
            
            item = self.dataset[0]
            
            assert 'input_ids' in item
            assert 'attention_mask' in item
            assert 'labels' in item

class TestTokenizationService:
    """Test cases for TokenizationService class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        with patch('onyx.server.features.ads.tokenization_service.AdvancedTokenizer'):
            with patch('onyx.server.features.ads.tokenization_service.SequenceManager'):
                with patch('onyx.server.features.ads.tokenization_service.OptimizedAdsDataset'):
                    self.service = TokenizationService()
    
    @pytest.mark.asyncio
    async def test_tokenize_ads_data(self) -> Any:
        """Test ads data tokenization."""
        ads_data = [
            {
                'prompt': 'Generate ad for product',
                'content': {'content': 'Amazing product offers...'}
            }
        ]
        
        with patch.object(self.service.sequence_manager, 'create_training_sequences') as mock_create:
            mock_create.return_value = [{
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'labels': torch.tensor([4, 5, 6])
            }]
            
            result = await self.service.tokenize_ads_data(ads_data, max_length=512)
            
            assert len(result) == 1
            assert 'input_ids' in result[0]
    
    @pytest.mark.asyncio
    async def test_create_training_dataset(self) -> Any:
        """Test training dataset creation."""
        ads_data = [
            {
                'prompt': 'Generate ad for product',
                'content': {'content': 'Amazing product offers...'}
            }
        ]
        
        with patch.object(self.service, 'tokenize_ads_data') as mock_tokenize:
            mock_tokenize.return_value = [{
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'labels': torch.tensor([4, 5, 6])
            }]
            
            with patch('torch.utils.data.DataLoader') as mock_dataloader:
                mock_dataloader.return_value = Mock()
                
                result = await self.service.create_training_dataset(ads_data, max_length=512, batch_size=8)
                
                assert result is not None
                mock_tokenize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_tokenize_for_inference(self) -> Any:
        """Test inference tokenization."""
        prompt = "Generate ad for product"
        
        with patch.object(self.service.sequence_manager, 'create_inference_sequence') as mock_create:
            mock_create.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            
            result = await self.service.tokenize_for_inference(prompt, max_length=512)
            
            assert 'input_ids' in result
            assert 'attention_mask' in result
    
    @pytest.mark.asyncio
    async def test_analyze_text_complexity(self) -> Any:
        """Test text complexity analysis."""
        text = "This is a test text with multiple words."
        
        with patch.object(self.service.tokenizer, 'tokenize_text') as mock_tokenize:
            mock_tokenize.return_value = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}
            
            with patch('nltk.tokenize.word_tokenize') as mock_word_tokenize:
                mock_word_tokenize.return_value = ['This', 'is', 'a', 'test', 'text']
                
                with patch('nltk.tokenize.sent_tokenize') as mock_sent_tokenize:
                    mock_sent_tokenize.return_value = ['This is a test text with multiple words.']
                    
                    result = await self.service.analyze_text_complexity(text)
                    
                    assert 'token_count' in result
                    assert 'word_count' in result
                    assert 'sentence_count' in result
                    assert 'complexity_score' in result
    
    @pytest.mark.asyncio
    async def test_optimize_sequence_length(self) -> Any:
        """Test sequence length optimization."""
        texts = ["Short text", "Very long text " * 100]
        
        with patch.object(self.service, 'analyze_text_complexity') as mock_analyze:
            mock_analyze.side_effect = [
                {'token_count': 5},  # Short text
                {'token_count': 1000}  # Long text
            ]
            
            with patch.object(self.service.preprocessor, 'segment_text') as mock_segment:
                mock_segment.return_value = ["Segment 1", "Segment 2"]
                
                result = await self.service.optimize_sequence_length(texts, target_token_count=512)
                
                assert len(result) >= len(texts)
    
    @pytest.mark.asyncio
    async def test_get_tokenization_stats(self) -> Optional[Dict[str, Any]]:
        """Test tokenization statistics."""
        ads_data = [
            {
                'prompt': 'Generate ad for product',
                'content': {'content': 'Amazing product offers...'}
            }
        ]
        
        with patch.object(self.service, 'analyze_text_complexity') as mock_analyze:
            mock_analyze.return_value = {
                'token_count': 10,
                'word_count': 8,
                'complexity_score': 1.25
            }
            
            result = await self.service.get_tokenization_stats(ads_data)
            
            assert 'total_items' in result
            assert 'total_tokens' in result
            assert 'total_words' in result
            assert 'avg_tokens_per_item' in result
            assert 'avg_words_per_item' in result
            assert 'avg_complexity_score' in result
            assert 'vocabulary_size' in result

class TestIntegration:
    """Integration tests for tokenization service."""
    
    @pytest.mark.asyncio
    async def test_full_tokenization_pipeline(self) -> Any:
        """Test complete tokenization pipeline."""
        # Create test data
        ads_data = [
            {
                'prompt': 'Generate an ad for our premium product',
                'content': {'content': 'Amazing premium product offers incredible features and benefits.'},
                'type': 'ads',
                'target_audience': 'Young professionals',
                'keywords': ['premium', 'quality', 'features']
            }
        ]
        
        # Initialize service
        with patch('onyx.server.features.ads.tokenization_service.AdvancedTokenizer'):
            with patch('onyx.server.features.ads.tokenization_service.SequenceManager'):
                with patch('onyx.server.features.ads.tokenization_service.OptimizedAdsDataset'):
                    service = TokenizationService()
        
        # Test tokenization
        with patch.object(service, 'tokenize_ads_data') as mock_tokenize:
            mock_tokenize.return_value = [{
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
                'labels': torch.tensor([6, 7, 8, 9, 10])
            }]
            
            result = await service.tokenize_ads_data(ads_data, max_length=512)
            
            assert len(result) == 1
            assert all(key in result[0] for key in ['input_ids', 'attention_mask', 'labels'])
    
    def test_text_preprocessing_pipeline(self) -> Any:
        """Test complete text preprocessing pipeline."""
        preprocessor = TextPreprocessor()
        
        # Test input text
        input_text = "Hello World! Visit https://example.com or call 555-123-4567"
        
        # Test normalization
        normalized = preprocessor.normalize_text(input_text)
        assert "[URL]" in normalized
        assert "[PHONE]" in normalized
        assert normalized.lower() == normalized
        
        # Test keyword extraction
        keywords = preprocessor.extract_keywords(normalized, max_keywords=5)
        assert len(keywords) <= 5
        assert all(isinstance(k, str) for k in keywords)
        
        # Test text segmentation
        long_text = "Sentence one. " * 100
        segments = preprocessor.segment_text(long_text, max_segment_length=200)
        assert len(segments) > 1
        assert all(len(seg) <= 200 for seg in segments)

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 