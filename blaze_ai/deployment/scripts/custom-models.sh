#!/usr/bin/env python3
"""
Custom nn.Module Classes for Blaze AI Model Architectures
Comprehensive collection of custom PyTorch model implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for transformer models"""
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    use_gelu: bool = True
    use_bias: bool = True


@dataclass
class CNNConfig:
    """Configuration for CNN models"""
    input_channels: int = 3
    num_classes: int = 1000
    base_channels: int = 64
    num_blocks: int = 4
    block_depth: int = 2
    use_batch_norm: bool = True
    use_residual: bool = True
    dropout_prob: float = 0.5


@dataclass
class RNNConfig:
    """Configuration for RNN models"""
    input_size: int = 100
    hidden_size: int = 256
    num_layers: int = 2
    num_classes: int = 10
    dropout_prob: float = 0.2
    bidirectional: bool = False
    use_lstm: bool = True


class PositionalEncoding(nn.Module):
    """Custom positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Custom multi-head attention mechanism"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, 
                 attention_dropout_prob: float = 0.1, use_bias: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear transformations
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=use_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=use_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=use_bias)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(attention_dropout_prob)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head attention"""
        
        # Linear transformations
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Transpose for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.output(context_layer)
        
        # Residual connection and layer normalization
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        return attention_output, attention_probs


class TransformerBlock(nn.Module):
    """Custom transformer block with attention and feed-forward layers"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout_prob,
            config.use_bias
        )
        
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size, 
                                    bias=config.use_bias)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size, 
                              bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Activation function
        if config.use_gelu:
            self.activation = F.gelu
        else:
            self.activation = F.relu
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block"""
        
        # Self-attention
        attention_output, _ = self.attention(hidden_states, attention_mask)
        
        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        
        # Output projection
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        
        # Residual connection and layer normalization
        layer_output = self.layer_norm(layer_output + attention_output)
        
        return layer_output


class CustomTransformer(nn.Module):
    """Custom transformer model with modular architecture"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = PositionalEncoding(config.hidden_size, 
                                                    config.max_position_embeddings)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer"""
        
        batch_size, sequence_length = input_ids.size()
        
        # Embeddings
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.position_embeddings(embeddings.transpose(0, 1)).transpose(0, 1)
        embeddings = self.dropout(embeddings)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, sequence_length, device=input_ids.device)
        
        # Convert attention mask to transformer format
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for block in self.blocks:
            hidden_states = block(hidden_states, extended_attention_mask)
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states


class ResidualBlock(nn.Module):
    """Custom residual block for CNN architectures"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 use_batch_norm: bool = True, use_residual: bool = True):
        super().__init__()
        
        self.use_residual = use_residual and (in_channels == out_channels)
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        # Shortcut connection
        if self.use_residual and stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                     stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        else:
            self.shortcut = None
            self.shortcut_bn = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block"""
        
        # Main path
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        if self.use_residual:
            if self.shortcut is not None:
                residual = self.shortcut_bn(self.shortcut(residual))
            out += residual
        
        out = F.relu(out)
        
        return out


class CustomCNN(nn.Module):
    """Custom CNN model with residual connections"""
    
    def __init__(self, config: CNNConfig):
        super().__init__()
        
        self.config = config
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(config.input_channels, config.base_channels, 
                                     kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(config.base_channels) if config.use_batch_norm else nn.Identity()
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        current_channels = config.base_channels
        
        for i in range(config.num_blocks):
            block_channels = current_channels * (2 ** i)
            block = nn.ModuleList([
                ResidualBlock(
                    current_channels if j == 0 else block_channels,
                    block_channels,
                    stride=2 if j == 0 else 1,
                    use_batch_norm=config.use_batch_norm,
                    use_residual=config.use_residual
                ) for j in range(config.block_depth)
            ])
            self.blocks.append(block)
            current_channels = block_channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_prob),
            nn.Linear(current_channels, config.num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN"""
        
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)
        x = self.initial_pool(x)
        
        # Residual blocks
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class LSTMCell(nn.Module):
    """Custom LSTM cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int, use_bias: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates: input, forget, cell, output
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=use_bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_tensor: torch.Tensor, 
                hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through LSTM cell"""
        
        h_prev, c_prev = hidden_state
        batch_size = input_tensor.size(0)
        
        # Concatenate input and previous hidden state
        combined = torch.cat([input_tensor, h_prev], dim=1)
        
        # Compute gates
        gates = self.gates(combined)
        
        # Split gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c_new = f * c_prev + i * g
        
        # Update hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class CustomRNN(nn.Module):
    """Custom RNN model with LSTM cells"""
    
    def __init__(self, config: RNNConfig):
        super().__init__()
        
        self.config = config
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            LSTMCell(
                config.input_size if i == 0 else config.hidden_size,
                config.hidden_size
            ) for i in range(config.num_layers)
        ])
        
        # Dropout between layers
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """Forward pass through RNN"""
        
        batch_size, sequence_length, input_size = x.size()
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            device = x.device
            hidden_states = [
                (torch.zeros(batch_size, self.config.hidden_size, device=device),
                 torch.zeros(batch_size, self.config.hidden_size, device=device))
                for _ in range(self.config.num_layers)
            ]
        
        # Process sequence
        for t in range(sequence_length):
            layer_input = x[:, t, :]
            
            # Pass through LSTM layers
            for layer_idx, lstm_layer in enumerate(self.lstm_layers):
                h, c = hidden_states[layer_idx]
                h_new, c_new = lstm_layer(layer_input, (h, c))
                
                # Apply dropout between layers (except last layer)
                if layer_idx < len(self.lstm_layers) - 1:
                    h_new = self.dropout(h_new)
                
                hidden_states[layer_idx] = (h_new, c_new)
                layer_input = h_new
        
        # Use final hidden state for classification
        final_hidden = hidden_states[-1][0]
        output = self.output_layer(final_hidden)
        
        return output


class HybridModel(nn.Module):
    """Hybrid model combining CNN and Transformer for vision-language tasks"""
    
    def __init__(self, vision_config: CNNConfig, language_config: TransformerConfig, 
                 fusion_size: int = 512, num_classes: int = 1000):
        super().__init__()
        
        # Vision encoder (CNN)
        self.vision_encoder = CustomCNN(vision_config)
        
        # Language encoder (Transformer)
        self.language_encoder = CustomTransformer(language_config)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(vision_config.num_classes + language_config.hidden_size, fusion_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_size, fusion_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output classifier
        self.classifier = nn.Linear(fusion_size // 2, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, vision_input: torch.Tensor, 
                language_input: torch.Tensor,
                language_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through hybrid model"""
        
        # Encode vision
        vision_features = self.vision_encoder(vision_input)
        
        # Encode language
        language_features = self.language_encoder(language_input, language_mask)
        # Use [CLS] token or mean pooling
        language_features = language_features.mean(dim=1)
        
        # Concatenate features
        combined_features = torch.cat([vision_features, language_features], dim=1)
        
        # Fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output


class ModelRegistry:
    """Registry for custom model architectures"""
    
    def __init__(self):
        self.models = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model architectures"""
        self.register_model('transformer', CustomTransformer, TransformerConfig)
        self.register_model('cnn', CustomCNN, CNNConfig)
        self.register_model('rnn', CustomRNN, RNNConfig)
        self.register_model('hybrid', HybridModel, (CNNConfig, TransformerConfig))
    
    def register_model(self, name: str, model_class: type, config_class: type):
        """Register a new model architecture"""
        self.models[name] = {
            'model_class': model_class,
            'config_class': config_class
        }
        logger.info(f"Registered model: {name}")
    
    def create_model(self, name: str, **config_kwargs):
        """Create a model instance"""
        if name not in self.models:
            raise ValueError(f"Unknown model: {name}")
        
        model_info = self.models[name]
        config_class = model_info['config_class']
        model_class = model_info['model_class']
        
        # Handle multiple config classes (e.g., hybrid models)
        if isinstance(config_class, tuple):
            configs = [cls(**config_kwargs.get(f"{cls.__name__.lower()}_config", {})) 
                      for cls in config_class]
            return model_class(*configs)
        else:
            config = config_class(**config_kwargs)
            return model_class(config)
    
    def list_models(self) -> List[str]:
        """List available model architectures"""
        return list(self.models.keys())


def main():
    """Main execution function"""
    logger.info("Starting Custom Model Architectures...")
    
    # Create model registry
    registry = ModelRegistry()
    logger.info(f"Available models: {registry.list_models()}")
    
    # Create transformer model
    transformer_config = TransformerConfig(
        vocab_size=10000,
        hidden_size=512,
        num_layers=6,
        num_attention_heads=8
    )
    transformer = CustomTransformer(transformer_config)
    logger.info(f"Transformer created: {transformer}")
    
    # Create CNN model
    cnn_config = CNNConfig(
        input_channels=3,
        num_classes=1000,
        base_channels=32,
        num_blocks=3
    )
    cnn = CustomCNN(cnn_config)
    logger.info(f"CNN created: {cnn}")
    
    # Create RNN model
    rnn_config = RNNConfig(
        input_size=100,
        hidden_size=128,
        num_layers=2,
        num_classes=10
    )
    rnn = CustomRNN(rnn_config)
    logger.info(f"RNN created: {rnn}")
    
    # Test forward pass
    batch_size = 2
    sequence_length = 10
    
    # Test transformer
    input_ids = torch.randint(0, transformer_config.vocab_size, (batch_size, sequence_length))
    transformer_output = transformer(input_ids)
    logger.info(f"Transformer output shape: {transformer_output.shape}")
    
    # Test CNN
    image_input = torch.randn(batch_size, 3, 224, 224)
    cnn_output = cnn(image_input)
    logger.info(f"CNN output shape: {cnn_output.shape}")
    
    # Test RNN
    rnn_input = torch.randn(batch_size, sequence_length, rnn_config.input_size)
    rnn_output = rnn(rnn_input)
    logger.info(f"RNN output shape: {rnn_output.shape}")
    
    logger.info("Custom model architectures created successfully!")


if __name__ == "__main__":
    main()
