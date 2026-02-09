"""
Deep Mood Detector Module

Implements deep neural network for mood detection using CNN+LSTM.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DeepMoodDetector(nn.Module):
    """
    Deep Neural Network for Mood Detection.
    CNN + LSTM architecture for temporal modeling.
    
    Args:
        input_channels: Number of input channels.
        num_moods: Number of mood classes.
        cnn_channels: List of CNN channel sizes.
        lstm_hidden: LSTM hidden size.
        lstm_layers: Number of LSTM layers.
        dropout_rate: Dropout probability.
    """
    
    def __init__(
        self,
        input_channels: int = 13,
        num_moods: int = 6,
        cnn_channels: List[int] = [32, 64, 128],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        self.input_channels = input_channels
        self.num_moods = num_moods
        
        # CNN layers for feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in cnn_channels:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2)
                )
            )
            in_channels = out_channels
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layers
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_moods)
        
        # Initialize weights
        self._initialize_weights()
        self._initialize_lstm_weights()
        logger.debug(f"Initialized DeepMoodDetector with input_channels={input_channels}, num_moods={num_moods}")
    
    def _initialize_weights(self):
        """Initialize CNN and FC weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _initialize_lstm_weights(self):
        """Initialize LSTM weights with proper scaling."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: CNN -> LSTM -> FC.
        
        Args:
            x: Input tensor [batch_size, channels, seq_len]
        
        Returns:
            Mood logits [batch_size, num_moods]
        """
        # Validate input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("Input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # CNN feature extraction
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("NaN/Inf in CNN layer")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Reshape for LSTM (batch, seq_len, features)
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        
        # LSTM
        try:
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use last hidden state
            x = lstm_out[:, -1, :]  # (batch, hidden*2)
        except RuntimeError as e:
            logger.error(f"LSTM forward error: {str(e)}")
            # Fallback: use mean pooling
            x = x.mean(dim=1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        # Final validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("NaN/Inf in output")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x



