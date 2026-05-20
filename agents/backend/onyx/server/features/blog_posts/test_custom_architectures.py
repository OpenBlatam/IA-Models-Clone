from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import pytest
import numpy as np
from typing import Dict, Any, List
import tempfile
import os
from custom_model_architectures import (
        import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Comprehensive tests for custom PyTorch model architectures
Tests all custom nn.Module classes with proper shapes, gradients, and edge cases
"""


# Import custom architectures
    PositionalEncoding, MultiHeadAttentionWithRelativePosition, TransformerBlock,
    CustomTransformer, Conv1DBlock, CNNFeatureExtractor, LSTMWithAttention,
    DotProductAttention, GeneralAttention, ConcatAttention, CNNLSTMHybrid,
    TransformerCNN, MultiTaskModel, HierarchicalAttentionNetwork,
    ResidualBlock, DeepResidualCNN, ModelFactory
)


class TestPositionalEncoding:
    """Test PositionalEncoding module"""
    
    def test_positional_encoding_shape(self) -> Any:
        d_model = 512
        max_len = 1000
        pe = PositionalEncoding(d_model, max_len)
        
        x = torch.randn(50, d_model)  # (seq_len, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_positional_encoding_gradients(self) -> Any:
        d_model = 256
        pe = PositionalEncoding(d_model)
        
        x = torch.randn(20, d_model, requires_grad=True)
        output = pe(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_positional_encoding_device(self) -> Any:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            d_model = 128
            pe = PositionalEncoding(d_model).to(device)
            
            x = torch.randn(10, d_model, device=device)
            output = pe(x)
            
            assert output.device == device


class TestMultiHeadAttentionWithRelativePosition:
    """Test MultiHeadAttentionWithRelativePosition module"""
    
    def test_attention_shape(self) -> Any:
        d_model = 512
        n_heads = 8
        batch_size = 4
        seq_len = 20
        
        attention = MultiHeadAttentionWithRelativePosition(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_attention_with_mask(self) -> Any:
        d_model = 256
        n_heads = 4
        batch_size = 2
        seq_len = 15
        
        attention = MultiHeadAttentionWithRelativePosition(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, 10:] = 0  # Mask last 5 positions
        
        output = attention(x, mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_attention_gradients(self) -> Any:
        d_model = 128
        n_heads = 2
        attention = MultiHeadAttentionWithRelativePosition(d_model, n_heads)
        
        x = torch.randn(3, 10, d_model, requires_grad=True)
        output = attention(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestTransformerBlock:
    """Test TransformerBlock module"""
    
    def test_transformer_block_shape(self) -> Any:
        d_model = 512
        n_heads = 8
        d_ff = 2048
        
        block = TransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(4, 20, d_model)
        
        output = block(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_transformer_block_with_mask(self) -> Any:
        d_model = 256
        n_heads = 4
        d_ff = 1024
        
        block = TransformerBlock(d_model, n_heads, d_ff, use_relative_pos=False)
        x = torch.randn(2, 15, d_model)
        mask = torch.ones(2, 15, 15)
        mask[:, :, 10:] = 0
        
        output = block(x, mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_transformer_block_activations(self) -> Any:
        d_model = 128
        n_heads = 2
        d_ff = 512
        
        activations = ["gelu", "relu", "swish"]
        for activation in activations:
            block = TransformerBlock(d_model, n_heads, d_ff, activation=activation)
            x = torch.randn(3, 10, d_model)
            output = block(x)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()


class TestCustomTransformer:
    """Test CustomTransformer module"""
    
    def test_transformer_shape(self) -> Any:
        vocab_size = 10000
        d_model = 512
        n_layers = 4
        n_heads = 8
        d_ff = 2048
        
        transformer = CustomTransformer(vocab_size, d_model, n_layers, n_heads, d_ff)
        x = torch.randint(0, vocab_size, (4, 20))  # (batch_size, seq_len)
        
        output = transformer(x)
        
        assert output.shape == (4, 20, d_model)
        assert not torch.isnan(output).any()
    
    def test_transformer_with_mask(self) -> Any:
        vocab_size = 5000
        d_model = 256
        transformer = CustomTransformer(vocab_size, d_model, n_layers=2, n_heads=4, d_ff=1024)
        x = torch.randint(0, vocab_size, (2, 15))
        mask = torch.ones(2, 15, 15)
        mask[:, :, 10:] = 0
        
        output = transformer(x, mask)
        
        assert output.shape == (2, 15, d_model)
        assert not torch.isnan(output).any()
    
    def test_transformer_gradients(self) -> Any:
        vocab_size = 3000
        d_model = 128
        transformer = CustomTransformer(vocab_size, d_model, n_layers=2, n_heads=2, d_ff=512)
        
        x = torch.randint(0, vocab_size, (3, 10))
        output = transformer(x)
        
        loss = output.sum()
        loss.backward()
        
        # Check gradients for all parameters
        for name, param in transformer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestConv1DBlock:
    """Test Conv1DBlock module"""
    
    def test_conv1d_block_shape(self) -> Any:
        in_channels = 64
        out_channels = 128
        kernel_size = 3
        
        block = Conv1DBlock(in_channels, out_channels, kernel_size)
        x = torch.randn(4, in_channels, 20)  # (batch_size, channels, seq_len)
        
        output = block(x)
        
        assert output.shape == (4, out_channels, 20)
        assert not torch.isnan(output).any()
    
    def test_conv1d_block_activations(self) -> Any:
        in_channels = 32
        out_channels = 64
        kernel_size = 5
        
        activations = ["relu", "gelu", "swish", "leaky_relu"]
        for activation in activations:
            block = Conv1DBlock(in_channels, out_channels, kernel_size, activation=activation)
            x = torch.randn(2, in_channels, 15)
            output = block(x)
            
            assert output.shape == (2, out_channels, 15)
            assert not torch.isnan(output).any()
    
    def test_conv1d_block_no_batch_norm(self) -> Any:
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        
        block = Conv1DBlock(in_channels, out_channels, kernel_size, batch_norm=False)
        x = torch.randn(3, in_channels, 10)
        output = block(x)
        
        assert output.shape == (3, out_channels, 10)
        assert not torch.isnan(output).any()


class TestCNNFeatureExtractor:
    """Test CNNFeatureExtractor module"""
    
    def test_cnn_extractor_shape(self) -> Any:
        input_dim = 300
        hidden_dims = [128, 256, 512]
        kernel_sizes = [3, 4, 5]
        
        extractor = CNNFeatureExtractor(input_dim, hidden_dims, kernel_sizes)
        x = torch.randn(4, 20, input_dim)  # (batch_size, seq_len, input_dim)
        
        output = extractor(x)
        
        expected_dim = sum(hidden_dims)
        assert output.shape == (4, expected_dim)
        assert not torch.isnan(output).any()
    
    def test_cnn_extractor_pool_types(self) -> Any:
        input_dim = 200
        hidden_dims = [64, 128]
        kernel_sizes = [3, 5]
        
        pool_types = ["max", "avg"]
        for pool_type in pool_types:
            extractor = CNNFeatureExtractor(input_dim, hidden_dims, kernel_sizes, pool_type=pool_type)
            x = torch.randn(2, 15, input_dim)
            output = extractor(x)
            
            expected_dim = sum(hidden_dims)
            assert output.shape == (2, expected_dim)
            assert not torch.isnan(output).any()
    
    def test_cnn_extractor_multiple_kernels(self) -> Any:
        input_dim = 100
        hidden_dims = [64]
        kernel_sizes = [[3, 4, 5]]  # Multiple kernels for one layer
        
        extractor = CNNFeatureExtractor(input_dim, hidden_dims, kernel_sizes)
        x = torch.randn(3, 12, input_dim)
        output = extractor(x)
        
        expected_dim = hidden_dims[0]
        assert output.shape == (3, expected_dim)
        assert not torch.isnan(output).any()


class TestLSTMWithAttention:
    """Test LSTMWithAttention module"""
    
    def test_lstm_attention_shape(self) -> Any:
        input_size = 256
        hidden_size = 128
        num_layers = 2
        
        lstm = LSTMWithAttention(input_size, hidden_size, num_layers, bidirectional=True)
        x = torch.randn(4, 20, input_size)  # (batch_size, seq_len, input_size)
        
        output = lstm(x)
        
        expected_size = hidden_size * 2  # bidirectional
        assert output.shape == (4, 20, expected_size)
        assert not torch.isnan(output).any()
    
    def test_lstm_attention_with_lengths(self) -> Any:
        input_size = 128
        hidden_size = 64
        
        lstm = LSTMWithAttention(input_size, hidden_size, bidirectional=False)
        x = torch.randn(3, 15, input_size)
        lengths = torch.tensor([15, 10, 8])  # Variable lengths
        
        output = lstm(x, lengths)
        
        expected_size = hidden_size  # unidirectional
        assert output.shape == (3, 15, expected_size)
        assert not torch.isnan(output).any()
    
    def test_lstm_attention_types(self) -> Any:
        input_size = 100
        hidden_size = 50
        
        attention_types = ["dot", "general", "concat"]
        for attention_type in attention_types:
            lstm = LSTMWithAttention(input_size, hidden_size, attention_type=attention_type)
            x = torch.randn(2, 10, input_size)
            output = lstm(x)
            
            expected_size = hidden_size * 2  # bidirectional by default
            assert output.shape == (2, 10, expected_size)
            assert not torch.isnan(output).any()


class TestAttentionMechanisms:
    """Test different attention mechanisms"""
    
    def test_dot_product_attention(self) -> Any:
        hidden_size = 128
        attention = DotProductAttention(hidden_size)
        
        hidden_states = torch.randn(4, 20, hidden_size)
        output = attention(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    def test_general_attention(self) -> Any:
        hidden_size = 256
        attention = GeneralAttention(hidden_size)
        
        hidden_states = torch.randn(3, 15, hidden_size)
        output = attention(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
    
    def test_concat_attention(self) -> Any:
        hidden_size = 64
        attention = ConcatAttention(hidden_size)
        
        hidden_states = torch.randn(2, 10, hidden_size)
        output = attention(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()


class TestCNNLSTMHybrid:
    """Test CNNLSTMHybrid module"""
    
    def test_cnn_lstm_hybrid_shape(self) -> Any:
        vocab_size = 10000
        embed_dim = 300
        hidden_dims = [128, 256]
        kernel_sizes = [3, 4]
        lstm_hidden_size = 128
        num_classes = 5
        
        model = CNNLSTMHybrid(vocab_size, embed_dim, hidden_dims, kernel_sizes,
                             lstm_hidden_size, num_classes)
        x = torch.randint(0, vocab_size, (4, 20))  # (batch_size, seq_len)
        
        output = model(x)
        
        assert output.shape == (4, num_classes)
        assert not torch.isnan(output).any()
    
    def test_cnn_lstm_hybrid_with_lengths(self) -> Any:
        vocab_size = 5000
        embed_dim = 200
        hidden_dims = [64]
        kernel_sizes = [3]
        lstm_hidden_size = 64
        num_classes = 3
        
        model = CNNLSTMHybrid(vocab_size, embed_dim, hidden_dims, kernel_sizes,
                             lstm_hidden_size, num_classes)
        x = torch.randint(0, vocab_size, (3, 15))
        lengths = torch.tensor([15, 10, 8])
        
        output = model(x, lengths)
        
        assert output.shape == (3, num_classes)
        assert not torch.isnan(output).any()
    
    def test_cnn_lstm_hybrid_gradients(self) -> Any:
        vocab_size = 3000
        embed_dim = 100
        hidden_dims = [32]
        kernel_sizes = [3]
        lstm_hidden_size = 32
        num_classes = 2
        
        model = CNNLSTMHybrid(vocab_size, embed_dim, hidden_dims, kernel_sizes,
                             lstm_hidden_size, num_classes)
        x = torch.randint(0, vocab_size, (2, 10))
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestTransformerCNN:
    """Test TransformerCNN module"""
    
    def test_transformer_cnn_shape(self) -> Any:
        vocab_size = 8000
        d_model = 256
        n_layers = 3
        n_heads = 4
        d_ff = 1024
        cnn_hidden_dims = [128, 256]
        cnn_kernel_sizes = [3, 5]
        num_classes = 4
        
        model = TransformerCNN(vocab_size, d_model, n_layers, n_heads, d_ff,
                              cnn_hidden_dims, cnn_kernel_sizes, num_classes)
        x = torch.randint(0, vocab_size, (3, 18))
        
        output = model(x)
        
        assert output.shape == (3, num_classes)
        assert not torch.isnan(output).any()
    
    def test_transformer_cnn_with_mask(self) -> Any:
        vocab_size = 4000
        d_model = 128
        n_layers = 2
        n_heads = 2
        d_ff = 512
        cnn_hidden_dims = [64]
        cnn_kernel_sizes = [3]
        num_classes = 2
        
        model = TransformerCNN(vocab_size, d_model, n_layers, n_heads, d_ff,
                              cnn_hidden_dims, cnn_kernel_sizes, num_classes)
        x = torch.randint(0, vocab_size, (2, 12))
        mask = torch.ones(2, 12, 12)
        mask[:, :, 8:] = 0
        
        output = model(x, mask)
        
        assert output.shape == (2, num_classes)
        assert not torch.isnan(output).any()


class TestMultiTaskModel:
    """Test MultiTaskModel module"""
    
    def test_multi_task_model_shape(self) -> Any:
        vocab_size = 6000
        d_model = 256
        n_layers = 3
        n_heads = 4
        d_ff = 1024
        
        task_configs = {
            "sentiment": {"num_classes": 3, "type": "classification"},
            "topic": {"num_classes": 5, "type": "classification"},
            "quality": {"num_classes": 1, "type": "regression"}
        }
        
        model = MultiTaskModel(vocab_size, d_model, n_layers, n_heads, d_ff, task_configs)
        x = torch.randint(0, vocab_size, (4, 16))
        
        # Test each task
        for task in task_configs.keys():
            output = model(x, task)
            expected_classes = task_configs[task]["num_classes"]
            assert output.shape == (4, expected_classes)
            assert not torch.isnan(output).any()
    
    def test_multi_task_model_with_mask(self) -> Any:
        vocab_size = 3000
        d_model = 128
        n_layers = 2
        n_heads = 2
        d_ff = 512
        
        task_configs = {
            "classification": {"num_classes": 2, "type": "classification"}
        }
        
        model = MultiTaskModel(vocab_size, d_model, n_layers, n_heads, d_ff, task_configs)
        x = torch.randint(0, vocab_size, (3, 10))
        mask = torch.ones(3, 10, 10)
        mask[:, :, 7:] = 0
        
        output = model(x, "classification", mask)
        
        assert output.shape == (3, 2)
        assert not torch.isnan(output).any()
    
    def test_multi_task_model_invalid_task(self) -> Any:
        vocab_size = 2000
        d_model = 64
        n_layers = 1
        n_heads = 1
        d_ff = 256
        
        task_configs = {
            "valid_task": {"num_classes": 1, "type": "classification"}
        }
        
        model = MultiTaskModel(vocab_size, d_model, n_layers, n_heads, d_ff, task_configs)
        x = torch.randint(0, vocab_size, (2, 8))
        
        with pytest.raises(ValueError):
            model(x, "invalid_task")


class TestHierarchicalAttentionNetwork:
    """Test HierarchicalAttentionNetwork module"""
    
    def test_hierarchical_attention_shape(self) -> Any:
        vocab_size = 5000
        embed_dim = 200
        hidden_size = 128
        num_classes = 4
        num_sentences = 20
        
        model = HierarchicalAttentionNetwork(vocab_size, embed_dim, hidden_size,
                                           num_classes, num_sentences)
        x = torch.randint(0, vocab_size, (3, num_sentences, 15))  # (batch, sentences, words)
        
        output = model(x)
        
        assert output.shape == (3, num_classes)
        assert not torch.isnan(output).any()
    
    def test_hierarchical_attention_gradients(self) -> Any:
        vocab_size = 3000
        embed_dim = 100
        hidden_size = 64
        num_classes = 2
        num_sentences = 10
        
        model = HierarchicalAttentionNetwork(vocab_size, embed_dim, hidden_size,
                                           num_classes, num_sentences)
        x = torch.randint(0, vocab_size, (2, num_sentences, 12))
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestResidualBlock:
    """Test ResidualBlock module"""
    
    def test_residual_block_shape(self) -> Any:
        channels = 64
        kernel_size = 3
        
        block = ResidualBlock(channels, kernel_size)
        x = torch.randn(4, channels, 20)  # (batch_size, channels, seq_len)
        
        output = block(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_residual_block_different_kernel_sizes(self) -> Any:
        channels = 32
        kernel_sizes = [3, 5, 7]
        
        for kernel_size in kernel_sizes:
            block = ResidualBlock(channels, kernel_size)
            x = torch.randn(2, channels, 15)
            output = block(x)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()


class TestDeepResidualCNN:
    """Test DeepResidualCNN module"""
    
    def test_deep_residual_cnn_shape(self) -> Any:
        input_dim = 300
        hidden_dims = [64, 128, 256]
        num_classes = 5
        
        model = DeepResidualCNN(input_dim, hidden_dims, num_classes)
        x = torch.randn(4, 20, input_dim)  # (batch_size, seq_len, input_dim)
        
        output = model(x)
        
        assert output.shape == (4, num_classes)
        assert not torch.isnan(output).any()
    
    def test_deep_residual_cnn_different_blocks(self) -> Any:
        input_dim = 200
        hidden_dims = [32, 64]
        num_classes = 3
        
        num_residual_blocks_list = [1, 3, 5]
        for num_blocks in num_residual_blocks_list:
            model = DeepResidualCNN(input_dim, hidden_dims, num_classes, num_blocks)
            x = torch.randn(3, 15, input_dim)
            output = model(x)
            
            assert output.shape == (3, num_classes)
            assert not torch.isnan(output).any()
    
    def test_deep_residual_cnn_gradients(self) -> Any:
        input_dim = 100
        hidden_dims = [32]
        num_classes = 2
        
        model = DeepResidualCNN(input_dim, hidden_dims, num_classes)
        x = torch.randn(2, 10, input_dim)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestModelFactory:
    """Test ModelFactory class"""
    
    def test_model_factory_transformer(self) -> Any:
        config = {
            "vocab_size": 5000,
            "d_model": 256,
            "n_layers": 3,
            "n_heads": 4,
            "d_ff": 1024,
            "dropout": 0.1
        }
        
        model = ModelFactory.create_model("transformer", config)
        assert isinstance(model, CustomTransformer)
        
        x = torch.randint(0, config["vocab_size"], (2, 10))
        output = model(x)
        assert output.shape == (2, 10, config["d_model"])
    
    def test_model_factory_cnn_lstm(self) -> Any:
        config = {
            "vocab_size": 3000,
            "embed_dim": 200,
            "hidden_dims": [64, 128],
            "kernel_sizes": [3, 4],
            "lstm_hidden_size": 64,
            "num_classes": 3,
            "dropout": 0.1,
            "bidirectional": True
        }
        
        model = ModelFactory.create_model("cnn_lstm", config)
        assert isinstance(model, CNNLSTMHybrid)
        
        x = torch.randint(0, config["vocab_size"], (3, 12))
        output = model(x)
        assert output.shape == (3, config["num_classes"])
    
    def test_model_factory_invalid_type(self) -> Any:
        config = {"vocab_size": 1000}
        
        with pytest.raises(ValueError):
            ModelFactory.create_model("invalid_type", config)


class TestModelIntegration:
    """Integration tests for model combinations"""
    
    def test_model_serialization(self) -> Any:
        """Test model saving and loading"""
        vocab_size = 2000
        d_model = 128
        model = CustomTransformer(vocab_size, d_model, n_layers=2, n_heads=2, d_ff=512)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name
        
        try:
            # Load model
            new_model = CustomTransformer(vocab_size, d_model, n_layers=2, n_heads=2, d_ff=512)
            new_model.load_state_dict(torch.load(temp_path))
            
            # Test both models produce same output
            x = torch.randint(0, vocab_size, (2, 8))
            output1 = model(x)
            output2 = new_model(x)
            
            assert torch.allclose(output1, output2, atol=1e-6)
            
        finally:
            os.unlink(temp_path)
    
    def test_model_device_transfer(self) -> Any:
        """Test model transfer to different devices"""
        if torch.cuda.is_available():
            vocab_size = 1000
            d_model = 64
            model = CustomTransformer(vocab_size, d_model, n_layers=1, n_heads=1, d_ff=256)
            
            # Move to GPU
            model = model.cuda()
            x = torch.randint(0, vocab_size, (2, 6), device='cuda')
            output = model(x)
            
            assert output.device.type == 'cuda'
            assert not torch.isnan(output).any()
    
    def test_model_memory_efficiency(self) -> Any:
        """Test model memory usage"""
        vocab_size = 500
        d_model = 32
        model = CustomTransformer(vocab_size, d_model, n_layers=1, n_heads=1, d_ff=128)
        
        # Test with gradient checkpointing
        x = torch.randint(0, vocab_size, (1, 10), requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestModelPerformance:
    """Performance tests for models"""
    
    def test_model_inference_speed(self) -> Any:
        """Test model inference speed"""
        
        vocab_size = 1000
        d_model = 64
        model = CustomTransformer(vocab_size, d_model, n_layers=2, n_heads=2, d_ff=256)
        model.eval()
        
        x = torch.randint(0, vocab_size, (4, 20))
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.1  # Should be fast
    
    def test_model_memory_usage(self) -> Any:
        """Test model memory usage"""
        vocab_size = 500
        d_model = 32
        model = CustomTransformer(vocab_size, d_model, n_layers=1, n_heads=1, d_ff=128)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 