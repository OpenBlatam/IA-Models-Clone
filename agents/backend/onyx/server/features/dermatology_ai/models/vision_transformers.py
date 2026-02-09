"""
Vision Transformers for Dermatology AI
Implements Vision Transformer (ViT) models for skin analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import ViTModel, ViTConfig, ViTForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available. Vision Transformer features disabled.")


class PatchEmbedding(nn.Module):
    """
    Patch embedding module for Vision Transformer
    Splits image into patches and projects them to embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size, n_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(
            batch_size, n_patches, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(
            batch_size, n_patches, embed_dim
        )
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with attention and MLP
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections"""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for skin analysis
    Implements ViT architecture from scratch
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        num_classes: int = 10
    ):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.n_patches
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches+1, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Use class token for classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        
        # Classification head
        out = self.head(cls_token_final)
        
        return out


class ViTSkinAnalyzer(nn.Module):
    """
    Vision Transformer for multi-task skin analysis
    Uses Hugging Face transformers if available, otherwise custom implementation
    """
    
    def __init__(
        self,
        num_conditions: int = 6,
        num_metrics: int = 8,
        use_pretrained: bool = True,
        model_name: str = "google/vit-base-patch16-224"
    ):
        super(ViTSkinAnalyzer, self).__init__()
        
        if TRANSFORMERS_AVAILABLE and use_pretrained:
            try:
                # Use pre-trained ViT from Hugging Face
                self.vit = ViTModel.from_pretrained(model_name)
                embed_dim = self.vit.config.hidden_size
                logger.info(f"Loaded pre-trained ViT: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}. Using custom ViT.")
                self.vit = VisionTransformer(
                    img_size=224,
                    patch_size=16,
                    embed_dim=768,
                    depth=12,
                    num_heads=12
                )
                embed_dim = 768
        else:
            # Use custom implementation
            self.vit = VisionTransformer(
                img_size=224,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12
            )
            embed_dim = 768
        
        # Multi-task heads
        self.condition_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_conditions),
            nn.Sigmoid()
        )
        
        self.metric_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_metrics),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        if TRANSFORMERS_AVAILABLE and hasattr(self.vit, 'forward'):
            # Use Hugging Face ViT
            outputs = self.vit(pixel_values=x)
            features = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            # Use custom ViT
            features = self.vit(x)
        
        # Multi-task predictions
        conditions = self.condition_head(features)
        metrics = self.metric_head(features) * 100  # Scale to 0-100
        
        return {
            'conditions': conditions,
            'metrics': metrics
        }


class LoRAViT(nn.Module):
    """
    Vision Transformer with LoRA (Low-Rank Adaptation) for efficient fine-tuning
    """
    
    def __init__(
        self,
        base_model_name: str = "google/vit-base-patch16-224",
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        super(LoRAViT, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for LoRAViT")
        
        # Load base model
        self.vit = ViTModel.from_pretrained(base_model_name)
        self.embed_dim = self.vit.config.hidden_size
        
        # LoRA parameters
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Add LoRA adapters to attention layers
        self._add_lora_adapters()
    
    def _add_lora_adapters(self):
        """Add LoRA adapters to attention layers"""
        for layer in self.vit.encoder.layer:
            # Query projection LoRA
            layer.attention.attention.query.lora_A = nn.Linear(
                self.embed_dim, self.r, bias=False
            )
            layer.attention.attention.query.lora_B = nn.Linear(
                self.r, self.embed_dim, bias=False
            )
            nn.init.kaiming_uniform_(layer.attention.attention.query.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(layer.attention.attention.query.lora_B.weight)
            
            # Value projection LoRA
            layer.attention.attention.value.lora_A = nn.Linear(
                self.embed_dim, self.r, bias=False
            )
            layer.attention.attention.value.lora_B = nn.Linear(
                self.r, self.embed_dim, bias=False
            )
            nn.init.kaiming_uniform_(layer.attention.attention.value.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(layer.attention.attention.value.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA"""
        # This is a simplified version - full implementation would
        # modify the attention forward pass to include LoRA
        outputs = self.vit(pixel_values=x)
        return outputs.last_hidden_state[:, 0]  # CLS token













