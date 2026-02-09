"""
P-tuning
========

Implementación de P-tuning para fine-tuning eficiente de LLMs.
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


@dataclass
class PTuningConfig:
    """Configuración de P-tuning."""
    prefix_length: int = 10
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.1


class PromptEncoder(nn.Module):
    """
    Encoder de prompts para P-tuning.
    
    Genera embeddings de prompts aprendibles.
    """
    
    def __init__(
        self,
        prefix_length: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Inicializar encoder.
        
        Args:
            prefix_length: Longitud del prefijo
            hidden_size: Tamaño oculto
            num_layers: Número de capas
            dropout: Tasa de dropout
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for P-tuning")
        
        super().__init__()
        
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        
        # Embeddings de prefijo
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, hidden_size) * 0.02
        )
        
        # Encoder LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Proyección
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generar embeddings de prefijo.
        
        Args:
            batch_size: Tamaño del batch
            
        Returns:
            Embeddings de prefijo [batch_size, prefix_length, hidden_size]
        """
        # Expandir embeddings
        prefix_emb = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Pasar por LSTM
        lstm_out, _ = self.lstm(prefix_emb)
        
        # Proyección
        output = self.projection(lstm_out)
        output = self.dropout(output)
        
        return output


def apply_ptuning(
    model,
    config: PTuningConfig,
    tokenizer_vocab_size: Optional[int] = None
):
    """
    Aplicar P-tuning a modelo.
    
    Args:
        model: Modelo base
        config: Configuración de P-tuning
        tokenizer_vocab_size: Tamaño del vocabulario (opcional)
        
    Returns:
        Modelo con P-tuning aplicado
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for P-tuning")
    
    # Crear prompt encoder
    prompt_encoder = PromptEncoder(
        prefix_length=config.prefix_length,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    # Agregar prompt encoder al modelo
    if hasattr(model, 'prompt_encoder'):
        logger.warning("Model already has prompt_encoder")
    else:
        model.prompt_encoder = prompt_encoder
        logger.info("Applied P-tuning to model")
    
    return model

