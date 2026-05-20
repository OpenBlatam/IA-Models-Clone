"""
Neural Network Architectures
Common model architectures used by the ModelManager
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class TransformerModel(nn.Module):
    """Standard Transformer model"""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(1024, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        device = x.device
        
        # Token and position embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
            
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

class TransformerLayer(nn.Module):
    """Single Transformer layer"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.ln1(x))
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.size()
        
        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_size, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(q, k, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.c_proj(attn_output)
    
    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Scaled dot-product attention"""
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply causal mask
        seq_len = scores.size(-1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

class MLP(nn.Module):
    """Multi-layer perceptron"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.c_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class ConvolutionalModel(nn.Module):
    """Convolutional Neural Network"""
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class RecurrentModel(nn.Module):
    """Recurrent Neural Network (LSTM)"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

class HybridModel(nn.Module):
    """Hybrid model combining different architectures"""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Convolutional component
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        
        # Recurrent component
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Attention component
        self.attention = MultiHeadAttention(hidden_size, 8)
        
        # Output
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(x)
        
        # Convolutional processing
        conv_input = embedded.transpose(1, 2)  # (batch, hidden, seq)
        conv_out = F.relu(self.conv1(conv_input))
        conv_out = F.relu(self.conv2(conv_out))
        conv_out = conv_out.transpose(1, 2)  # (batch, seq, hidden)
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_out)
        
        # Attention
        attn_out = self.attention(lstm_out)
        
        # Output projection
        output = self.fc(attn_out)
        
        return output

