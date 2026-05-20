"""
Vision Transformer Architecture for TruthGPT API
================================================

TensorFlow-like Vision Transformer implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from ..layers.attention import MultiHeadAttention


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer."""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        x = self.projection(x)  # (B, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Transformer block for Vision Transformer."""
    
    def __init__(self, 
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer architecture.
    
    Similar to tf.keras.applications.VisionTransformer, this class
    implements the Vision Transformer architecture.
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 name: Optional[str] = None):
        """
        Initialize Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP ratio
            dropout: Dropout rate
            name: Optional name for the model
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.name = name or f"vit_{embed_dim}_{depth}"
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Positional embedding
        num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout_layer(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification head
        cls_token_final = x[:, 0]  # (B, embed_dim)
        x = self.head(cls_token_final)
        
        return x
    
    def __repr__(self):
        return f"VisionTransformer(img_size={self.img_size}, patch_size={self.patch_size}, embed_dim={self.embed_dim}, depth={self.depth})"


def ViT_B16(num_classes: int = 1000, img_size: int = 224) -> VisionTransformer:
    """Vision Transformer Base-16 model."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        name="vit_b16"
    )


def ViT_B32(num_classes: int = 1000, img_size: int = 224) -> VisionTransformer:
    """Vision Transformer Base-32 model."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        name="vit_b32"
    )


def ViT_L16(num_classes: int = 1000, img_size: int = 224) -> VisionTransformer:
    """Vision Transformer Large-16 model."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
        name="vit_l16"
    )









