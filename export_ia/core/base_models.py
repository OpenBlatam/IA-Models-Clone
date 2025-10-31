"""
Base Model Classes for Export IA
Refactored architecture with PyTorch best practices and modular design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import math
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Base configuration for all models"""
    model_name: str
    input_dim: int
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-6
    max_position_embeddings: int = 512
    vocab_size: int = 50000
    use_bias: bool = True
    weight_initialization: str = "xavier_uniform"
    gradient_checkpointing: bool = False
    mixed_precision: bool = False

class BaseModel(nn.Module, ABC):
    """Abstract base class for all Export IA models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_weights()
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass implementation"""
        pass
        
    def _initialize_weights(self):
        """Initialize model weights using specified strategy"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_initialization == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_initialization == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.config.weight_initialization == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif self.config.weight_initialization == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    
                if module.bias is not None and self.config.use_bias:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, epoch: int = 0):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str, optimizer=None, scheduler=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logger.info(f"Model checkpoint loaded from {path}")
        return checkpoint.get('epoch', 0)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with optimizations"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 use_flash_attention: bool = False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash_attention = use_flash_attention
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional flash attention"""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention
            attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            # Standard attention implementation
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
            
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + query)

class FeedForwardNetwork(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu", use_flash_attention: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, use_flash_attention)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(ff_output + attn_output)
        
        return output

class DocumentTransformer(BaseModel):
    """Advanced transformer model for document processing"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positional_encoding = PositionalEncoding(config.hidden_dim, config.max_position_embeddings)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim * 4,  # Standard FFN expansion
                config.dropout,
                config.activation,
                use_flash_attention=True
            )
            for _ in range(config.num_layers)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            for block in self.transformer_blocks:
                block.gradient_checkpointing = True
                
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.config.hidden_dim)
        
        # Positional encoding
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Create attention mask
        if attention_mask is not None:
            # Convert attention mask to 4D for multi-head attention
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            
        # Transformer blocks
        for block in self.transformer_blocks:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, attention_mask)
            else:
                x = block(x, attention_mask)
                
        # Global average pooling
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.squeeze(1).squeeze(1).unsqueeze(-1)
            x = x * mask_expanded
            pooled = x.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
            
        # Output projection
        output = self.output_projection(pooled)
        
        return output

class MultiModalFusionModel(BaseModel):
    """Multi-modal fusion model for text, image, and other modalities"""
    
    def __init__(self, config: ModelConfig, text_dim: int = 768, image_dim: int = 2048):
        super().__init__(config)
        
        # Modality encoders
        self.text_encoder = nn.Linear(text_dim, config.hidden_dim)
        self.image_encoder = nn.Linear(image_dim, config.hidden_dim)
        
        # Cross-modal attention
        self.cross_modal_attention = MultiHeadAttention(
            config.hidden_dim, config.num_heads, config.dropout
        )
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim * 4,
                config.dropout,
                config.activation
            )
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-modal fusion"""
        # Encode modalities
        text_encoded = self.text_encoder(text_features)
        image_encoded = self.image_encoder(image_features)
        
        # Cross-modal attention
        fused_features = self.cross_modal_attention(text_encoded, image_encoded, image_encoded)
        
        # Fusion layers
        for layer in self.fusion_layers:
            fused_features = layer(fused_features)
            
        # Global pooling and output
        pooled = fused_features.mean(dim=1)
        output = self.output_projection(pooled)
        
        return output

class DiffusionModel(BaseModel):
    """Base diffusion model for generative tasks"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
        # Main network (U-Net like architecture)
        self.down_blocks = nn.ModuleList([
            nn.Conv2d(config.input_dim, config.hidden_dim, 3, padding=1),
            nn.Conv2d(config.hidden_dim, config.hidden_dim * 2, 3, padding=1),
            nn.Conv2d(config.hidden_dim * 2, config.hidden_dim * 4, 3, padding=1)
        ])
        
        self.middle_block = nn.Sequential(
            nn.Conv2d(config.hidden_dim * 4, config.hidden_dim * 4, 3, padding=1),
            nn.GroupNorm(32, config.hidden_dim * 4),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim * 4, config.hidden_dim * 4, 3, padding=1)
        )
        
        self.up_blocks = nn.ModuleList([
            nn.ConvTranspose2d(config.hidden_dim * 4, config.hidden_dim * 2, 3, padding=1),
            nn.ConvTranspose2d(config.hidden_dim * 2, config.hidden_dim, 3, padding=1),
            nn.ConvTranspose2d(config.hidden_dim, config.output_dim, 3, padding=1)
        ])
        
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass for diffusion model"""
        # Time embedding
        t_emb = self.time_embedding(timestep)
        
        # Down sampling
        h = x
        down_outputs = []
        for down_block in self.down_blocks:
            h = F.relu(down_block(h))
            down_outputs.append(h)
            h = F.max_pool2d(h, 2)
            
        # Middle block
        h = self.middle_block(h)
        
        # Up sampling with skip connections
        for i, up_block in enumerate(self.up_blocks):
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
            if i < len(down_outputs):
                h = h + down_outputs[-(i+1)]  # Skip connection
            h = F.relu(up_block(h))
            
        return h

class ModelFactory:
    """Factory class for creating different model types"""
    
    @staticmethod
    def create_model(model_type: str, config: ModelConfig, **kwargs) -> BaseModel:
        """Create model instance based on type"""
        if model_type == "document_transformer":
            return DocumentTransformer(config)
        elif model_type == "multi_modal_fusion":
            return MultiModalFusionModel(config, **kwargs)
        elif model_type == "diffusion":
            return DiffusionModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    @staticmethod
    def create_config(model_name: str, **kwargs) -> ModelConfig:
        """Create model configuration with defaults"""
        defaults = {
            'model_name': model_name,
            'input_dim': 512,
            'output_dim': 256,
            'hidden_dim': 768,
            'num_layers': 6,
            'num_heads': 12,
            'dropout': 0.1,
            'activation': 'gelu',
            'max_position_embeddings': 512,
            'vocab_size': 50000,
            'gradient_checkpointing': True,
            'mixed_precision': True
        }
        
        # Update with provided kwargs
        defaults.update(kwargs)
        return ModelConfig(**defaults)

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test Document Transformer
    print("Testing Document Transformer...")
    config = ModelFactory.create_config("document_transformer", input_dim=512, output_dim=10)
    model = ModelFactory.create_model("document_transformer", config)
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        print(f"Document Transformer output shape: {output.shape}")
        
    # Print model statistics
    stats = model.get_model_size()
    print(f"Model statistics: {stats}")
    
    # Test Multi-Modal Fusion
    print("\nTesting Multi-Modal Fusion...")
    config = ModelFactory.create_config("multi_modal_fusion", input_dim=512, output_dim=10)
    model = ModelFactory.create_model("multi_modal_fusion", config, text_dim=768, image_dim=2048)
    
    # Test forward pass
    text_features = torch.randn(batch_size, seq_len, 768)
    image_features = torch.randn(batch_size, seq_len, 2048)
    
    with torch.no_grad():
        output = model(text_features, image_features)
        print(f"Multi-Modal Fusion output shape: {output.shape}")
        
    # Test Diffusion Model
    print("\nTesting Diffusion Model...")
    config = ModelFactory.create_config("diffusion", input_dim=3, output_dim=3)
    model = ModelFactory.create_model("diffusion", config)
    
    # Test forward pass
    x = torch.randn(batch_size, 3, 64, 64)
    timestep = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(x, timestep)
        print(f"Diffusion Model output shape: {output.shape}")
        
    print("\nBase models refactored successfully!")
























