from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import MultiheadAttention, LayerNorm, Dropout
from typing import Optional, Tuple, List, Dict, Any
import math
import warnings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Custom PyTorch Model Architectures for Blog Post Analysis
Advanced neural network modules with modern optimizations
"""



class PositionalEncoding(nn.Module):
    """Advanced positional encoding with learnable parameters"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.dropout = Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Learnable positional encoding
        self.register_buffer('pe', pe)
        self.learnable_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        pe = self.pe[:seq_len] + self.learnable_pe[:seq_len]
        x = x + pe
        return self.dropout(x)


class MultiHeadAttentionWithRelativePosition(nn.Module):
    """Enhanced multi-head attention with relative positional encoding"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 max_relative_position: int = 32):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        
        # Relative positional encoding
        self.max_relative_position = max_relative_position
        self.relative_position_k = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        self.relative_position_v = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Relative positional encoding
        relative_positions = self._get_relative_positions(seq_len, x.device)
        relative_position_k = self.relative_position_k(relative_positions)
        relative_position_v = self.relative_position_v(relative_positions)
        
        # Attention with relative positions
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position scores
        relative_scores = torch.matmul(q, relative_position_k.transpose(-2, -1))
        scores = scores + relative_scores
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        
        # Add relative position values
        relative_context = torch.matmul(attention_weights, relative_position_v)
        context = context + relative_context
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return self.layer_norm(output + x)
    
    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        range_vec = torch.arange(seq_len, device=device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat


class TransformerBlock(nn.Module):
    """Enhanced transformer block with advanced features"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu", use_relative_pos: bool = True):
        
    """__init__ function."""
super().__init__()
        
        if use_relative_pos:
            self.attention = MultiHeadAttentionWithRelativePosition(d_model, n_heads, dropout)
        else:
            self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self._get_activation(activation),
            Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class CustomTransformer(nn.Module):
    """Custom transformer model with advanced features"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1, activation: str = "gelu",
                 use_relative_pos: bool = True):
        
    """__init__ function."""
super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation, use_relative_pos)
            for _ in range(n_layers)
        ])
        
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        return self.layer_norm(x)


class Conv1DBlock(nn.Module):
    """1D convolutional block with advanced features"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = True, activation: str = "relu",
                 dropout: float = 0.1, batch_norm: bool = True):
        
    """__init__ function."""
super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)
        
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_channels)
        else:
            self.batch_norm = None
            
        self.dropout = Dropout(dropout)
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CNNFeatureExtractor(nn.Module):
    """Advanced CNN feature extractor for text processing"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_sizes: List[int],
                 dropout: float = 0.1, activation: str = "relu", pool_type: str = "max"):
        
    """__init__ function."""
super().__init__()
        
        assert len(hidden_dims) == len(kernel_sizes), "Dimensions must match"
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = input_dim
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            # Multiple kernel sizes for each layer
            conv_block = nn.ModuleList([
                Conv1DBlock(in_channels, hidden_dim // len(kernel_sizes), k,
                           padding=k // 2, activation=activation, dropout=dropout)
                for k in kernel_size if isinstance(kernel_size, (list, tuple))
                else [kernel_size]
            ])
            self.conv_layers.append(conv_block)
            
            # Pooling layer
            if pool_type == "max":
                pool = nn.AdaptiveMaxPool1d(1)
            elif pool_type == "avg":
                pool = nn.AdaptiveAvgPool1d(1)
            else:
                pool = nn.AdaptiveMaxPool1d(1)
            self.pool_layers.append(pool)
            
            in_channels = hidden_dim
        
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        features = []
        for conv_block, pool in zip(self.conv_layers, self.pool_layers):
            # Apply multiple convolutions and concatenate
            conv_outputs = []
            for conv in conv_block:
                conv_outputs.append(conv(x))
            
            # Concatenate along channel dimension
            x = torch.cat(conv_outputs, dim=1)
            x = pool(x)  # Global pooling
            features.append(x.squeeze(-1))
        
        return torch.cat(features, dim=1)


class LSTMWithAttention(nn.Module):
    """LSTM with attention mechanism"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bidirectional: bool = True, dropout: float = 0.1,
                 attention_type: str = "dot"):
        
    """__init__ function."""
super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention_type = attention_type
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           bidirectional=bidirectional, dropout=dropout,
                           batch_first=True)
        
        # Attention mechanism
        match attention_type:
    case "dot":
            self.attention = DotProductAttention(hidden_size * (2 if bidirectional else 1))
        elmatch attention_type:
    case "general":
            self.attention = GeneralAttention(hidden_size * (2 if bidirectional else 1))
        elmatch attention_type:
    case "concat":
            self.attention = ConcatAttention(hidden_size * (2 if bidirectional else 1))
        else:
            self.attention = DotProductAttention(hidden_size * (2 if bidirectional else 1))
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pack sequence if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # Unpack sequence
        if lengths is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # Apply attention
        attended_output = self.attention(lstm_output)
        
        return attended_output


class DotProductAttention(nn.Module):
    """Dot product attention mechanism"""
    
    def __init__(self, hidden_size: int):
        
    """__init__ function."""
super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Calculate attention scores
        scores = torch.bmm(hidden_states, hidden_states.transpose(1, 2)) / math.sqrt(hidden_size)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        context = torch.bmm(attention_weights, hidden_states)
        
        return context


class GeneralAttention(nn.Module):
    """General attention mechanism with learnable parameters"""
    
    def __init__(self, hidden_size: int):
        
    """__init__ function."""
super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Calculate attention scores
        attention_scores = self.attention(hidden_states)  # (batch_size, seq_len, hidden_size)
        scores = torch.bmm(attention_scores, hidden_states.transpose(1, 2)) / math.sqrt(hidden_size)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        context = torch.bmm(attention_weights, hidden_states)
        
        return context


class ConcatAttention(nn.Module):
    """Concat attention mechanism"""
    
    def __init__(self, hidden_size: int):
        
    """__init__ function."""
super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Calculate attention scores
        hidden_states_expanded = hidden_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
        hidden_states_transposed = hidden_states.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        concat_features = torch.cat([hidden_states_expanded, hidden_states_transposed], dim=-1)
        attention_scores = torch.tanh(self.attention(concat_features))
        
        # Apply attention vector
        scores = torch.sum(attention_scores * self.v, dim=-1)  # (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        context = torch.bmm(attention_weights, hidden_states)
        
        return context


class CNNLSTMHybrid(nn.Module):
    """Hybrid CNN-LSTM model for text classification"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dims: List[int],
                 kernel_sizes: List[int], lstm_hidden_size: int, num_classes: int,
                 dropout: float = 0.1, bidirectional: bool = True):
        
    """__init__ function."""
super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            embed_dim, hidden_dims, kernel_sizes, dropout
        )
        
        # LSTM with attention
        self.lstm = LSTMWithAttention(
            self.cnn_extractor.output_dim, lstm_hidden_size,
            bidirectional=bidirectional, dropout=dropout
        )
        
        # Classification head
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # CNN feature extraction
        cnn_features = self.cnn_extractor(x)  # (batch_size, cnn_output_dim)
        
        # Reshape for LSTM (treat CNN features as sequence)
        cnn_features = cnn_features.unsqueeze(1)  # (batch_size, 1, cnn_output_dim)
        
        # LSTM with attention
        lstm_output = self.lstm(cnn_features)  # (batch_size, 1, lstm_output_dim)
        lstm_output = lstm_output.squeeze(1)  # (batch_size, lstm_output_dim)
        
        # Classification
        output = self.classifier(lstm_output)
        
        return output


class TransformerCNN(nn.Module):
    """Transformer-CNN hybrid model"""
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, cnn_hidden_dims: List[int], cnn_kernel_sizes: List[int],
                 num_classes: int, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        
        self.transformer = CustomTransformer(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout=dropout
        )
        
        self.cnn_extractor = CNNFeatureExtractor(
            d_model, cnn_hidden_dims, cnn_kernel_sizes, dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_extractor.output_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Transformer encoding
        transformer_output = self.transformer(x, mask)  # (batch_size, seq_len, d_model)
        
        # CNN feature extraction
        cnn_features = self.cnn_extractor(transformer_output)
        
        # Classification
        output = self.classifier(cnn_features)
        
        return output


class MultiTaskModel(nn.Module):
    """Multi-task learning model with shared encoder"""
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                 d_ff: int, task_configs: Dict[str, Dict[str, Any]], dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        
        # Shared encoder
        self.encoder = CustomTransformer(
            vocab_size, d_model, n_layers, n_heads, d_ff, dropout=dropout
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            num_classes = config.get('num_classes', 1)
            head_type = config.get('type', 'classification')
            
            if head_type == 'classification':
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, num_classes)
                )
            elif head_type == 'regression':
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )
            elif head_type == 'sequence':
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, num_classes)
                )
        
    def forward(self, x: torch.Tensor, task: str, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Shared encoding
        encoded = self.encoder(x, mask)  # (batch_size, seq_len, d_model)
        
        # Global average pooling
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            encoded = encoded * mask_expanded
            pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # Task-specific head
        if task not in self.task_heads:
            raise ValueError(f"Task '{task}' not found in task heads")
        
        output = self.task_heads[task](pooled)
        
        return output


class HierarchicalAttentionNetwork(nn.Module):
    """Hierarchical Attention Network for document classification"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_classes: int, num_sentences: int = 30, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Word-level attention
        self.word_encoder = nn.GRU(embed_dim, hidden_size, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(hidden_size * 2, 1)
        
        # Sentence-level attention
        self.sentence_encoder = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.sentence_attention = nn.Linear(hidden_size * 2, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.dropout = Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_sentences, sentence_length)
        batch_size, num_sentences, sentence_length = x.size()
        
        # Reshape for word-level processing
        x = x.view(batch_size * num_sentences, sentence_length)
        
        # Word embedding
        word_embeddings = self.embedding(x)  # (batch_size * num_sentences, sentence_length, embed_dim)
        
        # Word-level encoding and attention
        word_outputs, _ = self.word_encoder(word_embeddings)  # (batch_size * num_sentences, sentence_length, hidden_size * 2)
        word_attention_weights = F.softmax(self.word_attention(word_outputs), dim=1)
        sentence_vectors = torch.sum(word_attention_weights * word_outputs, dim=1)  # (batch_size * num_sentences, hidden_size * 2)
        
        # Reshape for sentence-level processing
        sentence_vectors = sentence_vectors.view(batch_size, num_sentences, -1)
        
        # Sentence-level encoding and attention
        sentence_outputs, _ = self.sentence_encoder(sentence_vectors)  # (batch_size, num_sentences, hidden_size * 2)
        sentence_attention_weights = F.softmax(self.sentence_attention(sentence_outputs), dim=1)
        document_vector = torch.sum(sentence_attention_weights * sentence_outputs, dim=1)  # (batch_size, hidden_size * 2)
        
        # Classification
        output = self.classifier(document_vector)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class DeepResidualCNN(nn.Module):
    """Deep residual CNN for text processing"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int,
                 num_residual_blocks: int = 3, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Initial convolution
        in_channels = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Conv1d(in_channels, hidden_dim, 3, padding=1))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            
            # Residual blocks
            for _ in range(num_residual_blocks):
                self.layers.append(ResidualBlock(hidden_dim, dropout=dropout))
            
            in_channels = hidden_dim
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)
        
        # Classification
        output = self.classifier(x)
        
        return output


# Model factory for easy instantiation
class ModelFactory:
    """Factory class for creating different model architectures"""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Create a model based on type and configuration"""
        
        if model_type == "transformer":
            return CustomTransformer(**config)
        elif model_type == "cnn_lstm":
            return CNNLSTMHybrid(**config)
        elif model_type == "transformer_cnn":
            return TransformerCNN(**config)
        elif model_type == "multi_task":
            return MultiTaskModel(**config)
        elif model_type == "hierarchical":
            return HierarchicalAttentionNetwork(**config)
        elif model_type == "residual_cnn":
            return DeepResidualCNN(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Example configurations
MODEL_CONFIGS = {
    "transformer": {
        "vocab_size": 30000,
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1
    },
    "cnn_lstm": {
        "vocab_size": 30000,
        "embed_dim": 300,
        "hidden_dims": [128, 256, 512],
        "kernel_sizes": [3, 4, 5],
        "lstm_hidden_size": 256,
        "num_classes": 10,
        "dropout": 0.1,
        "bidirectional": True
    },
    "transformer_cnn": {
        "vocab_size": 30000,
        "d_model": 512,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 2048,
        "cnn_hidden_dims": [256, 512],
        "cnn_kernel_sizes": [3, 5],
        "num_classes": 10,
        "dropout": 0.1
    }
} 