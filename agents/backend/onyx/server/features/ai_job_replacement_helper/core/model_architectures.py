"""
Model Architectures Service - Arquitecturas de modelos personalizadas
=======================================================================

Sistema para crear y gestionar arquitecturas de modelos personalizadas usando PyTorch.
Refactorizado siguiendo mejores prácticas de deep learning.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base_model_service import BaseModelService, DeviceConfig
from core.utils.model_utils import initialize_weights, get_model_size, count_parameters
from core.config.model_config import ModelArchitectureConfig, InitializationMethod

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class ModelConfig:
    """Configuración de modelo"""
    name: str
    architecture_type: str  # transformer, cnn, rnn, mlp
    input_size: int
    output_size: int
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.1
    use_batch_norm: bool = True


class SimpleMLP(nn.Module):
    """Multi-Layer Perceptron simple"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            if activation == "relu":
                self.layers.append(nn.ReLU())
            elif activation == "gelu":
                self.layers.append(nn.GELU())
            elif activation == "tanh":
                self.layers.append(nn.Tanh())
            
            # Dropout
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class TextClassifier(nn.Module):
    """Clasificador de texto usando embeddings"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use last hidden state
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        output = self.dropout(last_hidden)
        output = self.fc(output)
        return output


class AttentionLayer(nn.Module):
    """Capa de atención multi-head"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + self.dropout(attn_out))
        return x


class TransformerBlock(nn.Module):
    """Bloque Transformer completo"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = AttentionLayer(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        x = self.attention(x, mask)
        x = self.norm(x + self.ff(x))
        return x


class ModelArchitecturesService(BaseModelService):
    """Servicio de arquitecturas de modelos - Refactorizado con mejores prácticas"""
    
    def __init__(self, device_config: Optional[DeviceConfig] = None):
        """
        Inicializar servicio.
        
        Args:
            device_config: Configuración de dispositivo (opcional)
        """
        super().__init__(device_config=device_config)
        self.configs: Dict[str, ModelArchitectureConfig] = {}
        logger.info("ModelArchitecturesService initialized")
    
    def create_mlp(
        self,
        model_id: str,
        config: ModelConfig
    ) -> nn.Module:
        """Crear MLP"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        model = SimpleMLP(
            input_size=config.input_size,
            hidden_sizes=config.hidden_sizes,
            output_size=config.output_size,
            activation=config.activation,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm
        )
        
        self.models[model_id] = model
        self.configs[model_id] = config
        
        logger.info(f"MLP model {model_id} created")
        return model
    
    def create_text_classifier(
        self,
        model_id: str,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ) -> nn.Module:
        """Crear clasificador de texto"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        model = TextClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.models[model_id] = model
        
        logger.info(f"Text classifier {model_id} created")
        return model
    
    def create_transformer(
        self,
        model_id: str,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        max_seq_length: int,
        num_classes: Optional[int] = None,
        dropout: float = 0.1
    ) -> nn.Module:
        """Crear Transformer personalizado"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        class CustomTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.pos_encoding = nn.Parameter(
                    torch.randn(1, max_seq_length, embed_dim)
                )
                self.blocks = nn.ModuleList([
                    TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                    for _ in range(num_layers)
                ])
                self.norm = nn.LayerNorm(embed_dim)
                if num_classes:
                    self.classifier = nn.Linear(embed_dim, num_classes)
                else:
                    self.classifier = None
            
            def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
                
                for block in self.blocks:
                    x = block(x, mask)
                
                x = self.norm(x)
                
                if self.classifier:
                    # Use [CLS] token (first token)
                    x = x[:, 0, :]
                    x = self.classifier(x)
                
                return x
        
        model = CustomTransformer()
        self.models[model_id] = model
        
        logger.info(f"Transformer {model_id} created")
        return model
    
    def initialize_weights(
        self,
        model_id: str,
        method: str = "xavier_uniform",
        gain: float = 1.0
    ) -> bool:
        """
        Inicializar pesos del modelo usando utilidades refactorizadas.
        
        Args:
            model_id: ID del modelo
            method: Método de inicialización
            gain: Gain para inicialización
        
        Returns:
            True si se inicializó correctamente
        """
        if not TORCH_AVAILABLE:
            return False
        
        model = self.models.get(model_id)
        if not model:
            logger.warning(f"Model {model_id} not found")
            return False
        
        try:
            initialize_weights(model, method=method, gain=gain)
            logger.info(f"Weights initialized for {model_id} using {method}")
            return True
        except Exception as e:
            logger.error(f"Error initializing weights: {e}")
            return False
    
    def get_model_summary(self, model_id: str) -> Dict[str, Any]:
        """
        Obtener resumen del modelo usando utilidades refactorizadas.
        
        Args:
            model_id: ID del modelo
        
        Returns:
            Diccionario con información del modelo
        """
        model = self.models.get(model_id)
        if not model:
            return {"error": "Model not found"}
        
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        try:
            # Usar método de clase base
            info = self.get_model_info(model)
            info.update({
                "model_id": model_id,
                "model_type": type(model).__name__,
            })
            
            # Agregar información adicional usando utilidades
            info["model_size_mb"] = get_model_size(model, unit="MB")
            info["total_parameters"] = count_parameters(model, trainable_only=False)
            info["trainable_parameters"] = count_parameters(model, trainable_only=True)
            
            return info
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {"error": str(e)}
    
    def create_model(self, *args, **kwargs) -> nn.Module:
        """Implementación requerida por BaseModelService"""
        # Esta será implementada por métodos específicos como create_mlp
        raise NotImplementedError("Use specific methods like create_mlp()")

