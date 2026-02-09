"""
Tests for Deep Learning Models
===============================
Comprehensive tests for model architectures
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from ..deep_learning_models import (
    PsychologicalEmbeddingModel,
    PersonalityClassifier,
    SentimentTransformerModel
)
from ..model_architecture import (
    MultiHeadAttention,
    TransformerBlock,
    ImprovedPersonalityModel
)


class TestPsychologicalEmbeddingModel:
    """Tests for embedding model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = PsychologicalEmbeddingModel()
        assert model is not None
        assert model.embedding_dim > 0
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = PsychologicalEmbeddingModel()
        texts = ["This is a test", "Another test"]
        
        embeddings = model(texts)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == model.embedding_dim
    
    def test_device_placement(self):
        """Test device placement"""
        model = PsychologicalEmbeddingModel()
        if torch.cuda.is_available():
            model = model.to("cuda")
            assert next(model.parameters()).is_cuda


class TestPersonalityClassifier:
    """Tests for personality classifier"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = PersonalityClassifier()
        assert model is not None
        assert model.num_traits == 5
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = PersonalityClassifier()
        texts = ["I love socializing", "I prefer quiet activities"]
        
        predictions = model(texts)
        assert len(predictions) == 5
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            assert trait in predictions
            assert predictions[trait].shape[0] == len(texts)


class TestMultiHeadAttention:
    """Tests for multi-head attention"""
    
    def test_attention_initialization(self):
        """Test attention initialization"""
        attn = MultiHeadAttention(embed_dim=768, num_heads=12)
        assert attn.embed_dim == 768
        assert attn.num_heads == 12
    
    def test_attention_forward(self):
        """Test attention forward pass"""
        attn = MultiHeadAttention(embed_dim=128, num_heads=4)
        batch_size, seq_len = 2, 10
        
        query = torch.randn(batch_size, seq_len, 128)
        key = torch.randn(batch_size, seq_len, 128)
        value = torch.randn(batch_size, seq_len, 128)
        
        output, weights = attn(query, key, value)
        assert output.shape == (batch_size, seq_len, 128)
        assert weights.shape == (batch_size, attn.num_heads, seq_len, seq_len)


class TestTransformerBlock:
    """Tests for transformer block"""
    
    def test_block_initialization(self):
        """Test block initialization"""
        block = TransformerBlock(embed_dim=256, num_heads=8)
        assert block is not None
    
    def test_block_forward(self):
        """Test block forward pass"""
        block = TransformerBlock(embed_dim=128, num_heads=4)
        batch_size, seq_len = 2, 10
        
        x = torch.randn(batch_size, seq_len, 128)
        output = block(x)
        assert output.shape == x.shape


class TestImprovedPersonalityModel:
    """Tests for improved personality model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = ImprovedPersonalityModel(
            vocab_size=1000,
            embed_dim=128,
            num_heads=4,
            num_layers=2
        )
        assert model is not None
        assert model.num_traits == 5
    
    def test_model_forward(self):
        """Test model forward pass"""
        model = ImprovedPersonalityModel(
            vocab_size=1000,
            embed_dim=128,
            num_heads=4,
            num_layers=2
        )
        
        batch_size, seq_len = 2, 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        predictions = model(input_ids, attention_mask)
        assert len(predictions) == 5
        for trait in predictions:
            assert predictions[trait].shape[0] == batch_size


@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "I love going to parties and meeting new people",
        "I prefer reading books alone at home",
        "I enjoy both social and solitary activities"
    ]


@pytest.fixture
def sample_batch():
    """Sample batch for testing"""
    return {
        "input_ids": torch.randint(0, 1000, (2, 20)),
        "attention_mask": torch.ones(2, 20),
        "labels": torch.randint(0, 3, (2,))
    }




