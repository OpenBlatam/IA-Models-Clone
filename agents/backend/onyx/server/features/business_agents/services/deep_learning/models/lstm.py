"""
LSTM Model - Long Short-Term Memory Network
===========================================

Bidirectional LSTM for text classification.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base_model import BaseModel


class LSTMTextClassifier(BaseModel):
    """
    Bidirectional LSTM for text classification.
    
    Architecture:
    - Embedding layer
    - Bidirectional LSTM
    - Dropout for regularization
    - Linear classifier
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            device: Target device
        """
        super().__init__(device)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # *2 for bidirectional
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
        self._initialize_weights("xavier_uniform")
        self.to(self.device)
        self._initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state from both directions
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        output = self.dropout(last_hidden)
        output = self.classifier(output)
        return output



