"""
Multi-Task Music Model Module

Implements multi-task learning model for music analysis.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class MultiTaskMusicModel(nn.Module):
    """
    Multi-task learning model for music analysis.
    Shared encoder with task-specific heads.
    
    Args:
        input_size: Input feature size.
        num_genres: Number of genre classes.
        num_moods: Number of mood classes.
        num_instruments: Number of instrument classes.
        shared_layers: List of shared layer sizes.
        dropout_rate: Dropout probability.
    """
    
    def __init__(
        self,
        input_size: int = 169,
        num_genres: int = 10,
        num_moods: int = 6,
        num_instruments: int = 15,
        shared_layers: List[int] = [512, 512, 256],
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.input_size = input_size
        self.num_genres = num_genres
        self.num_moods = num_moods
        self.num_instruments = num_instruments
        
        # Shared encoder
        self.shared_layers = nn.ModuleList()
        prev_size = input_size
        
        for layer_size in shared_layers:
            self.shared_layers.append(
                nn.Sequential(
                    nn.Linear(prev_size, layer_size),
                    nn.BatchNorm1d(layer_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            prev_size = layer_size
        
        # Task-specific heads
        self.genre_head = nn.Linear(prev_size, num_genres)
        self.mood_head = nn.Linear(prev_size, num_moods)
        self.instrument_head = nn.Linear(prev_size, num_instruments)
        
        # Initialize weights
        self._initialize_weights()
        logger.debug(f"Initialized MultiTaskMusicModel with input_size={input_size}")
    
    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_size]
        
        Returns:
            Dictionary with genre, mood, and instrument logits.
        """
        # Shared encoder
        for layer in self.shared_layers:
            x = layer(x)
        
        # Task-specific heads
        return {
            "genre": self.genre_head(x),
            "mood": self.mood_head(x),
            "instruments": self.instrument_head(x)
        }



