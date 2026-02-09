"""
RNN Model - Recurrent Neural Network Architectures
===================================================

Implements RNN, LSTM, and GRU models following best practices:
- Proper initialization
- Dropout
- Bidirectional support
- Attention mechanisms (optional)
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Attention mechanism for RNN outputs."""
    
    def __init__(self, hidden_size: int):
        """
        Initialize attention layer.
        
        Args:
            hidden_size: Hidden size of RNN
        """
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, rnn_output: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to RNN outputs.
        
        Args:
            rnn_output: RNN output (batch_size, seq_len, hidden_size)
            
        Returns:
            Weighted sum (batch_size, hidden_size)
        """
        attention_weights = torch.softmax(self.attention(rnn_output), dim=1)
        weighted_output = torch.sum(attention_weights * rnn_output, dim=1)
        return weighted_output


class RNNModel(BaseModel):
    """
    Recurrent Neural Network model (RNN/LSTM/GRU).
    
    Supports bidirectional RNNs and optional attention mechanism.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_attention: bool = False,
        num_classes: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RNN model.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_size: Hidden size of RNN
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('rnn', 'lstm', 'gru')
            bidirectional: Use bidirectional RNN
            dropout: Dropout probability
            use_attention: Use attention mechanism
            num_classes: Number of output classes (for classification)
            config: Additional configuration
        """
        if config is None:
            config = {}
        config.update({
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'rnn_type': rnn_type,
            'bidirectional': bidirectional,
            'dropout': dropout,
            'use_attention': use_attention,
        })
        super().__init__(config)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # RNN layer
        rnn_class = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }.get(rnn_type.lower(), nn.LSTM)
        
        self.rnn = rnn_class(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(hidden_size * (2 if bidirectional else 1))
        
        # Output layer
        output_size = hidden_size * (2 if bidirectional else 1)
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(output_size, output_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_size // 2, num_classes)
            )
        else:
            self.classifier = None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            lengths: Sequence lengths for packed sequences (optional)
            
        Returns:
            Output tensor (batch_size, hidden_size) or (batch_size, num_classes)
        """
        # Embedding
        x = self.embedding(input_ids)
        x = self.embedding_dropout(x)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        
        # RNN
        rnn_output, hidden = self.rnn(x)
        
        # Unpack if packed
        if lengths is not None:
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )
        
        # Apply attention or use last hidden state
        if self.use_attention:
            output = self.attention(rnn_output)
        else:
            # Use last hidden state
            if isinstance(hidden, tuple):  # LSTM
                hidden = hidden[0]
            if self.config['bidirectional']:
                # Concatenate forward and backward
                output = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                output = hidden[-1]
        
        # Classifier if provided
        if self.classifier is not None:
            output = self.classifier(output)
        
        return output



