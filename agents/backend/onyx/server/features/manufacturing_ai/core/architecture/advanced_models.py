"""
Advanced Model Architectures
============================

Arquitecturas avanzadas de modelos usando componentes reutilizables.
"""

import logging
from typing import Dict, Any, List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .model_builder import ModelBuilder, ArchitectureConfig
from .component_factory import ComponentFactory
from .attention_layers import MultiHeadAttention
from .residual_blocks import ResidualBlock

logger = logging.getLogger(__name__)


class AdvancedQualityPredictor(nn.Module):
    """
    Predictor de calidad avanzado.
    
    Usa atención y bloques residuales.
    """
    
    def __init__(
        self,
        image_input_size: int = 224,
        num_features: int = 10,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        """
        Inicializar modelo avanzado.
        
        Args:
            image_input_size: Tamaño de imagen
            num_features: Número de características
            embed_dim: Dimensión de embedding
            num_heads: Número de heads de atención
            num_layers: Número de capas
            use_attention: Usar atención
            use_residual: Usar conexiones residuales
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        # Image encoder (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Feature encoder
        self.feature_proj = nn.Linear(num_features, embed_dim)
        
        # Fusion
        self.fusion = nn.Linear(64 + embed_dim, embed_dim)
        
        # Transformer layers con atención
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_attention:
                attention = ComponentFactory.create_attention(
                    embed_dim, num_heads, "self"
                )
                if use_residual:
                    attention = ComponentFactory.create_residual_block(
                        embed_dim
                    )
                self.layers.append(attention)
            else:
                block = ComponentFactory.create_residual_block(embed_dim)
                self.layers.append(block)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, 3)  # pass, warning, fail
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Imágenes [batch, 3, H, W]
            features: Características [batch, num_features]
            
        Returns:
            Logits [batch, 3]
        """
        # Encoders
        image_features = self.image_encoder(images)
        feature_emb = self.feature_proj(features)
        
        # Fusion
        combined = torch.cat([image_features, feature_emb], dim=1)
        x = self.fusion(combined)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Classifier
        output = self.classifier(x)
        
        return output


class AdvancedProcessOptimizer(nn.Module):
    """
    Optimizador de procesos avanzado.
    
    Usa transformer con atención y feed-forward.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_hidden_dim: Optional[int] = None
    ):
        """
        Inicializar modelo.
        
        Args:
            input_dim: Dimensión de entrada
            output_dim: Dimensión de salida
            embed_dim: Dimensión de embedding
            num_heads: Número de heads
            num_layers: Número de capas
            ff_hidden_dim: Dimensión oculta FFN
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ComponentFactory.create_transformer_block(
                embed_dim,
                num_heads,
                ff_hidden_dim,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, input_dim]
            
        Returns:
            Output [batch, seq_len, input_dim]
        """
        # Project
        x = self.input_proj(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output
        x = self.output_proj(x)
        
        return x

