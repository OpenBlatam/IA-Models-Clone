"""
Custom nn.Module Classes for Model Architectures
Comprehensive collection of custom PyTorch model architectures for various deep learning tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np

# =============================================================================
# BASE MODEL CLASSES
# =============================================================================

class BaseModel(nn.Module):
    """Base class for all custom models with common functionality"""
    
    def __init__(self):
        super().__init__()
        self.model_name = self.__class__.__name__
        self.is_training = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': self._get_architecture_info()
        }
    
    def _get_architecture_info(self) -> Dict[str, Any]:
        """Get architecture-specific information - override in subclasses"""
        return {}
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers by name"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                print(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers by name"""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")

# =============================================================================
# TRANSFORMER ARCHITECTURES
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with optional relative positioning"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 use_relative_position: bool = False, max_relative_position: int = 32):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_relative_position = use_relative_position
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Relative position embeddings
        if use_relative_position:
            self.relative_position_k = nn.Embedding(2 * max_relative_position + 1, self.d_k)
            self.relative_position_v = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(context)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(query + output)
        
        return output

class TransformerBlock(nn.Module):
    """Complete transformer block with attention and feed-forward layers"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 activation: str = 'gelu', norm_first: bool = True,
                 use_relative_position: bool = False):
        super().__init__()
        
        self.norm_first = norm_first
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout, use_relative_position
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        if activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'swish':
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_first:
            # Pre-norm architecture
            x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
            x = x + self.feed_forward(self.norm2(x))
        else:
            # Post-norm architecture
            x = self.norm1(x + self.attention(x, x, x, mask))
            x = self.norm2(x + self.feed_forward(x))
        
        return x

class CustomTransformerModel(BaseModel):
    """Custom transformer model with configurable architecture"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_seq_len: int, dropout: float = 0.1,
                 activation: str = 'gelu', norm_first: bool = True,
                 use_relative_position: bool = False, tie_weights: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout, activation, norm_first, use_relative_position
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Tie weights if specified
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using best practices"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def _get_architecture_info(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'tie_weights': self.tie_weights
        }

# =============================================================================
# CNN ARCHITECTURES
# =============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_batch_norm: bool = True,
                 activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        
        layers = []
        
        # Convolution layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        
        # Batch normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation function
        layers.append(self._get_activation(activation))
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.conv_block = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'swish':
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

class ResBlock(nn.Module):
    """Residual block for ResNet-style architectures"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_batch_norm: bool = True, activation: str = 'relu'):
        super().__init__()
        
        # Main path
        self.main_path = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=stride, use_batch_norm=use_batch_norm, activation=activation),
            ConvBlock(out_channels, out_channels, use_batch_norm=use_batch_norm, activation='none')
        )
        
        # Shortcut path
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
        
        # Final activation
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.main_path(x)
        out = out + residual
        return self.activation(out)

class CustomCNNModel(BaseModel):
    """Custom CNN model with configurable architecture"""
    
    def __init__(self, input_channels: int, num_classes: int, base_channels: int = 64,
                 num_layers: int = 5, use_batch_norm: bool = True, use_dropout: bool = True,
                 activation: str = 'relu', architecture: str = 'standard'):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.architecture = architecture
        
        # Feature extraction layers
        if architecture == 'resnet':
            self.features = self._build_resnet_features()
        else:
            self.features = self._build_standard_features()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(self._get_feature_size(), 512),
                self._get_activation(activation),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self._get_feature_size(), 512),
                self._get_activation(activation),
                nn.Linear(512, num_classes)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _build_standard_features(self) -> nn.Module:
        """Build standard CNN feature extractor"""
        layers = []
        in_channels = self.input_channels
        
        for i in range(self.num_layers):
            out_channels = self.base_channels * (2 ** i)
            layers.extend([
                ConvBlock(in_channels, out_channels, use_batch_norm=True, activation='relu'),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_resnet_features(self) -> nn.Module:
        """Build ResNet-style feature extractor"""
        layers = []
        in_channels = self.input_channels
        
        # Initial convolution
        layers.append(ConvBlock(in_channels, self.base_channels, kernel_size=7, stride=2, padding=3))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # ResNet blocks
        current_channels = self.base_channels
        for i in range(self.num_layers):
            out_channels = self.base_channels * (2 ** i)
            layers.append(ResBlock(current_channels, out_channels, stride=2 if i > 0 else 1))
            current_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _get_feature_size(self) -> int:
        """Calculate feature size after feature extraction"""
        if self.architecture == 'resnet':
            return self.base_channels * (2 ** (self.num_layers - 1))
        else:
            return self.base_channels * (2 ** (self.num_layers - 1))
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        if activation.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'swish':
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def _get_architecture_info(self) -> Dict[str, Any]:
        return {
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'base_channels': self.base_channels,
            'num_layers': self.num_layers,
            'architecture': self.architecture
        }

# =============================================================================
# RNN ARCHITECTURES
# =============================================================================

class LSTMCell(nn.Module):
    """Custom LSTM cell with configurable features"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0,
                 use_layer_norm: bool = False):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_layer_norm = use_layer_norm
        
        # LSTM gates
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        
        # Layer normalization (optional)
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(4)
            ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hx
        batch_size = x.size(0)
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute gates
        gates = self.gates(combined)
        
        # Split gates
        gates = gates.chunk(4, dim=1)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            gates = [norm(gate) for norm, gate in zip(self.layer_norms, gates)]
        
        # Extract gates
        i, f, g, o = gates
        
        # Apply activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # Update cell state
        c_next = f * c_prev + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        # Apply dropout
        h_next = self.dropout(h_next)
        
        return h_next, c_next

class CustomRNNModel(BaseModel):
    """Custom RNN model with configurable architecture"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.1, bidirectional: bool = False,
                 rnn_type: str = 'lstm', use_layer_norm: bool = False):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # Calculate hidden size for bidirectional
        self.total_hidden_size = hidden_size * (2 if bidirectional else 1)
        
        # RNN layers
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(self.total_hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'rnn' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Pack sequence if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Unpack sequence if it was packed
        if lengths is not None:
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        # Take the last output (or mean pooling for variable length sequences)
        if lengths is not None:
            # Variable length sequences - use mean pooling
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            rnn_out = (rnn_out * mask).sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            # Fixed length sequences - take last output
            rnn_out = rnn_out[:, -1, :]
        
        # Output projection
        output = self.output_projection(rnn_out)
        
        return output
    
    def _get_architecture_info(self) -> Dict[str, Any]:
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'bidirectional': self.bidirectional,
            'rnn_type': self.rnn_type,
            'total_hidden_size': self.total_hidden_size
        }

# =============================================================================
# HYBRID ARCHITECTURES
# =============================================================================

class CNNTransformerHybrid(BaseModel):
    """Hybrid model combining CNN and Transformer architectures"""
    
    def __init__(self, input_channels: int, num_classes: int, cnn_channels: int = 64,
                 cnn_layers: int = 4, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # CNN feature extractor
        self.cnn_features = self._build_cnn_features(input_channels, cnn_channels, cnn_layers)
        
        # Calculate CNN output size
        cnn_output_size = cnn_channels * (2 ** (cnn_layers - 1))
        
        # Projection from CNN to transformer
        self.cnn_to_transformer = nn.Linear(cnn_output_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _build_cnn_features(self, input_channels: int, base_channels: int, num_layers: int) -> nn.Module:
        """Build CNN feature extractor"""
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # CNN feature extraction
        cnn_out = self.cnn_features(x)
        
        # Global average pooling
        cnn_out = F.adaptive_avg_pool2d(cnn_out, (1, 1))
        cnn_out = cnn_out.view(batch_size, -1)
        
        # Project to transformer dimension
        transformer_input = self.cnn_to_transformer(cnn_out)
        transformer_input = transformer_input.unsqueeze(1)  # Add sequence dimension
        
        # Add positional encoding
        positions = torch.arange(1, device=x.device).unsqueeze(0)
        pos_enc = self.pos_encoding(positions)
        transformer_input = transformer_input + pos_enc
        
        # Apply transformer
        transformer_output = self.transformer(transformer_input)
        
        # Take the output and project to classes
        output = self.output_projection(transformer_output.squeeze(1))
        
        return output
    
    def _get_architecture_info(self) -> Dict[str, Any]:
        return {
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'cnn_channels': self.cnn_channels,
            'cnn_layers': self.cnn_layers,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_model_from_config(config: Dict[str, Any]) -> BaseModel:
    """Create a model from configuration dictionary"""
    model_type = config.get('type', 'transformer')
    
    if model_type == 'transformer':
        return CustomTransformerModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'gelu'),
            norm_first=config.get('norm_first', True),
            use_relative_position=config.get('use_relative_position', False),
            tie_weights=config.get('tie_weights', True)
        )
    
    elif model_type == 'cnn':
        return CustomCNNModel(
            input_channels=config['input_channels'],
            num_classes=config['num_classes'],
            base_channels=config['base_channels'],
            num_layers=config['num_layers'],
            use_batch_norm=config.get('use_batch_norm', True),
            use_dropout=config.get('use_dropout', True),
            activation=config.get('activation', 'relu'),
            architecture=config.get('architecture', 'standard')
        )
    
    elif model_type == 'rnn':
        return CustomRNNModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.1),
            bidirectional=config.get('bidirectional', False),
            rnn_type=config.get('rnn_type', 'lstm'),
            use_layer_norm=config.get('use_layer_norm', False)
        )
    
    elif model_type == 'hybrid':
        return CNNTransformerHybrid(
            input_channels=config['input_channels'],
            num_classes=config['num_classes'],
            cnn_channels=config['cnn_channels'],
            cnn_layers=config['cnn_layers'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config.get('dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_model_summary(model: BaseModel) -> str:
    """Get a formatted summary of the model"""
    info = model.get_model_info()
    
    summary = f"""
Model Summary: {info['model_name']}
{'=' * 50}
Total Parameters: {info['total_parameters']:,}
Trainable Parameters: {info['trainable_parameters']:,}
Model Size: {info['model_size_mb']:.2f} MB

Architecture Details:
"""
    
    for key, value in info['architecture'].items():
        summary += f"  {key}: {value}\n"
    
    return summary

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example of using custom model architectures"""
    
    print("=== Custom Model Architectures Example ===\n")
    
    # 1. Create Transformer model
    print("1. Creating Transformer Model...")
    transformer_config = {
        'type': 'transformer',
        'vocab_size': 30000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 512,
        'dropout': 0.1,
        'activation': 'gelu',
        'norm_first': True,
        'use_relative_position': False,
        'tie_weights': True
    }
    
    transformer_model = create_model_from_config(transformer_config)
    print(get_model_summary(transformer_model))
    
    # 2. Create CNN model
    print("\n2. Creating CNN Model...")
    cnn_config = {
        'type': 'cnn',
        'input_channels': 3,
        'num_classes': 1000,
        'base_channels': 64,
        'num_layers': 5,
        'use_batch_norm': True,
        'use_dropout': True,
        'activation': 'relu',
        'architecture': 'resnet'
    }
    
    cnn_model = create_model_from_config(cnn_config)
    print(get_model_summary(cnn_model))
    
    # 3. Create RNN model
    print("\n3. Creating RNN Model...")
    rnn_config = {
        'type': 'rnn',
        'input_size': 512,
        'hidden_size': 256,
        'num_layers': 3,
        'num_classes': 10,
        'dropout': 0.1,
        'bidirectional': True,
        'rnn_type': 'lstm',
        'use_layer_norm': True
    }
    
    rnn_model = create_model_from_config(rnn_config)
    print(get_model_summary(rnn_model))
    
    # 4. Test forward pass
    print("\n4. Testing Forward Pass...")
    
    # Test transformer
    transformer_input = torch.randint(0, 30000, (2, 100))
    transformer_output = transformer_model(transformer_input)
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # Test CNN
    cnn_input = torch.randn(2, 3, 224, 224)
    cnn_output = cnn_model(cnn_input)
    print(f"CNN output shape: {cnn_output.shape}")
    
    # Test RNN
    rnn_input = torch.randn(2, 50, 512)
    rnn_output = rnn_model(rnn_input)
    print(f"RNN output shape: {rnn_output.shape}")
    
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    example_usage()


