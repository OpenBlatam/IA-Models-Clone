"""
Advanced Time Series Analysis System for TruthGPT Optimization Core
Forecasting, anomaly detection, and temporal pattern recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

class TimeSeriesTask(Enum):
    """Time series analysis tasks"""
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    PATTERN_RECOGNITION = "pattern_recognition"
    TREND_ANALYSIS = "trend_analysis"

class ModelArchitecture(Enum):
    """Time series model architectures"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    ATTENTION_LSTM = "attention_lstm"
    WAVENET = "wavenet"

@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis"""
    # Task settings
    task: TimeSeriesTask = TimeSeriesTask.FORECASTING
    architecture: ModelArchitecture = ModelArchitecture.LSTM
    
    # Data settings
    sequence_length: int = 60
    prediction_horizon: int = 1
    num_features: int = 1
    num_classes: int = 2
    
    # Model settings
    hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.1
    
    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    
    # Advanced features
    enable_attention: bool = True
    enable_bidirectional: bool = True
    enable_residual: bool = True
    enable_normalization: bool = True
    
    def __post_init__(self):
        """Validate time series configuration"""
        if self.sequence_length < 1:
            raise ValueError("Sequence length must be at least 1")
        if self.prediction_horizon < 1:
            raise ValueError("Prediction horizon must be at least 1")

class TimeSeriesDataProcessor:
    """Time series data processing utilities"""
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info("✅ Time Series Data Processor initialized")
    
    def create_synthetic_data(self, length: int = 1000, trend: bool = True, 
                            seasonality: bool = True, noise: float = 0.1) -> np.ndarray:
        """Create synthetic time series data"""
        t = np.arange(length)
        
        # Base signal
        signal = np.zeros(length)
        
        # Add trend
        if trend:
            signal += 0.01 * t
        
        # Add seasonality
        if seasonality:
            signal += 2 * np.sin(2 * np.pi * t / 365)  # Annual seasonality
            signal += 0.5 * np.sin(2 * np.pi * t / 7)   # Weekly seasonality
        
        # Add noise
        signal += np.random.normal(0, noise, length)
        
        return signal
    
    def preprocess_data(self, data: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Preprocess time series data"""
        if fit_scaler and not self.is_fitted:
            data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
            self.is_fitted = True
        else:
            data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        return data_scaled
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        X, y = [], []
        
        for i in range(len(data) - self.config.sequence_length - self.config.prediction_horizon + 1):
            X.append(data[i:i + self.config.sequence_length])
            y.append(data[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data"""
        if self.is_fitted:
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        return data

class LSTMModel(nn.Module):
    """LSTM model for time series"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            bidirectional=config.enable_bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_size = config.hidden_size * (2 if config.enable_bidirectional else 1)
        
        if config.task == TimeSeriesTask.FORECASTING:
            self.output_layer = nn.Linear(lstm_output_size, config.prediction_horizon)
        elif config.task == TimeSeriesTask.CLASSIFICATION:
            self.output_layer = nn.Linear(lstm_output_size, config.num_classes)
        else:
            self.output_layer = nn.Linear(lstm_output_size, 1)
        
        # Attention mechanism
        if config.enable_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                dropout=config.dropout_rate,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
        logger.info(f"✅ LSTM Model initialized for {config.task.value}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.config.enable_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = attn_out
        
        # Use last output
        if self.config.task == TimeSeriesTask.FORECASTING:
            output = self.output_layer(lstm_out[:, -1, :])
        else:
            output = self.output_layer(lstm_out[:, -1, :])
        
        return output

class GRUModel(nn.Module):
    """GRU model for time series"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__()
        self.config = config
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            bidirectional=config.enable_bidirectional,
            batch_first=True
        )
        
        # Output layer
        gru_output_size = config.hidden_size * (2 if config.enable_bidirectional else 1)
        
        if config.task == TimeSeriesTask.FORECASTING:
            self.output_layer = nn.Linear(gru_output_size, config.prediction_horizon)
        elif config.task == TimeSeriesTask.CLASSIFICATION:
            self.output_layer = nn.Linear(gru_output_size, config.num_classes)
        else:
            self.output_layer = nn.Linear(gru_output_size, 1)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
        logger.info(f"✅ GRU Model initialized for {config.task.value}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Use last output
        output = self.output_layer(gru_out[:, -1, :])
        
        return output

class TransformerModel(nn.Module):
    """Transformer model for time series"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(config.num_features, config.hidden_size)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(config.sequence_length, config.hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Output layer
        if config.task == TimeSeriesTask.FORECASTING:
            self.output_layer = nn.Linear(config.hidden_size, config.prediction_horizon)
        elif config.task == TimeSeriesTask.CLASSIFICATION:
            self.output_layer = nn.Linear(config.hidden_size, config.num_classes)
        else:
            self.output_layer = nn.Linear(config.hidden_size, 1)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
        logger.info(f"✅ Transformer Model initialized for {config.task.value}")
    
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Use last output
        output = self.output_layer(transformer_out[:, -1, :])
        
        return output

class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for time series"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__()
        self.config = config
        
        # CNN layers
        self.conv1 = nn.Conv1d(config.num_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        if config.task == TimeSeriesTask.FORECASTING:
            self.output_layer = nn.Linear(config.hidden_size, config.prediction_horizon)
        elif config.task == TimeSeriesTask.CLASSIFICATION:
            self.output_layer = nn.Linear(config.hidden_size, config.num_classes)
        else:
            self.output_layer = nn.Linear(config.hidden_size, 1)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
        logger.info(f"✅ CNN-LSTM Model initialized for {config.task.value}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # CNN forward pass
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last output
        output = self.output_layer(lstm_out[:, -1, :])
        
        return output

class AnomalyDetector(nn.Module):
    """Anomaly detection model using autoencoder"""
    
    def __init__(self, config: TimeSeriesConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.sequence_length * config.num_features, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size // 4, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.sequence_length * config.num_features)
        )
        
        logger.info("✅ Anomaly Detector initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Flatten input
        x_flat = x.view(x.size(0), -1)
        
        # Encode
        encoded = self.encoder(x_flat)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Reshape to original shape
        output = decoded.view(x.size())
        
        return output
    
    def detect_anomalies(self, x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """Detect anomalies in time series"""
        self.eval()
        
        with torch.no_grad():
            reconstructed = self.forward(x)
            reconstruction_error = F.mse_loss(x, reconstructed, reduction='none')
            reconstruction_error = reconstruction_error.mean(dim=(1, 2))
            
            anomalies = reconstruction_error > threshold
        
        return anomalies

class TimeSeriesTrainer:
    """Time series model trainer"""
    
    def __init__(self, model: nn.Module, config: TimeSeriesConfig):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Loss function
        if config.task == TimeSeriesTask.FORECASTING:
            self.criterion = nn.MSELoss()
        elif config.task == TimeSeriesTask.CLASSIFICATION:
            self.criterion = nn.CrossEntropyLoss()
        elif config.task == TimeSeriesTask.ANOMALY_DETECTION:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
        
        # Training state
        self.training_history = []
        self.best_loss = float('inf')
        
        logger.info("✅ Time Series Trainer initialized")
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.config.task == TimeSeriesTask.ANOMALY_DETECTION:
                output = self.model(batch_x)
                loss = self.criterion(output, batch_x)
            else:
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            if self.config.task == TimeSeriesTask.CLASSIFICATION:
                pred = output.argmax(dim=1)
                correct += pred.eq(batch_y).sum().item()
                total += batch_y.size(0)
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                # Forward pass
                if self.config.task == TimeSeriesTask.ANOMALY_DETECTION:
                    output = self.model(batch_x)
                    loss = self.criterion(output, batch_x)
                else:
                    output = self.model(batch_x)
                    loss = self.criterion(output, batch_y)
                
                # Statistics
                total_loss += loss.item()
                if self.config.task == TimeSeriesTask.CLASSIFICATION:
                    pred = output.argmax(dim=1)
                    correct += pred.eq(batch_y).sum().item()
                    total += batch_y.size(0)
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader, num_epochs: int = None) -> Dict[str, Any]:
        """Train model"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        logger.info(f"🚀 Starting time series training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_stats = self.train_epoch(train_loader)
            
            # Validate
            val_stats = self.validate(val_loader)
            
            # Update best loss
            if val_stats['loss'] < self.best_loss:
                self.best_loss = val_stats['loss']
            
            # Record history
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'train_accuracy': train_stats['accuracy'],
                'val_loss': val_stats['loss'],
                'val_accuracy': val_stats['accuracy'],
                'best_loss': self.best_loss
            }
            self.training_history.append(epoch_stats)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_stats['loss']:.4f}, "
                          f"Val Loss = {val_stats['loss']:.4f}")
        
        final_stats = {
            'total_epochs': num_epochs,
            'best_loss': self.best_loss,
            'final_train_loss': self.training_history[-1]['train_loss'],
            'final_val_loss': self.training_history[-1]['val_loss'],
            'training_history': self.training_history
        }
        
        logger.info(f"✅ Time series training completed. Best loss: {self.best_loss:.4f}")
        return final_stats

# Factory functions
def create_timeseries_config(**kwargs) -> TimeSeriesConfig:
    """Create time series configuration"""
    return TimeSeriesConfig(**kwargs)

def create_lstm_model(config: TimeSeriesConfig) -> LSTMModel:
    """Create LSTM model"""
    return LSTMModel(config)

def create_gru_model(config: TimeSeriesConfig) -> GRUModel:
    """Create GRU model"""
    return GRUModel(config)

def create_transformer_model(config: TimeSeriesConfig) -> TransformerModel:
    """Create transformer model"""
    return TransformerModel(config)

def create_cnn_lstm_model(config: TimeSeriesConfig) -> CNNLSTMModel:
    """Create CNN-LSTM model"""
    return CNNLSTMModel(config)

def create_anomaly_detector(config: TimeSeriesConfig) -> AnomalyDetector:
    """Create anomaly detector"""
    return AnomalyDetector(config)

def create_timeseries_trainer(model: nn.Module, config: TimeSeriesConfig) -> TimeSeriesTrainer:
    """Create time series trainer"""
    return TimeSeriesTrainer(model, config)

# Example usage
def example_time_series_analysis():
    """Example of time series analysis"""
    # Create configuration
    config = create_timeseries_config(
        task=TimeSeriesTask.FORECASTING,
        architecture=ModelArchitecture.LSTM,
        sequence_length=60,
        prediction_horizon=1,
        num_features=1,
        hidden_size=128,
        num_layers=2,
        enable_attention=True,
        enable_bidirectional=True
    )
    
    # Create model
    model = create_lstm_model(config)
    
    # Create trainer
    trainer = create_timeseries_trainer(model, config)
    
    # Create synthetic data
    data_processor = TimeSeriesDataProcessor(config)
    synthetic_data = data_processor.create_synthetic_data(length=1000)
    processed_data = data_processor.preprocess_data(synthetic_data)
    
    # Create sequences
    X, y = data_processor.create_sequences(processed_data)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
    y_tensor = torch.FloatTensor(y)
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train model
    training_stats = trainer.train(dataloader, dataloader, num_epochs=20)
    
    # Test anomaly detection
    anomaly_config = create_timeseries_config(
        task=TimeSeriesTask.ANOMALY_DETECTION,
        sequence_length=60,
        num_features=1,
        hidden_size=64
    )
    
    anomaly_detector = create_anomaly_detector(anomaly_config)
    anomaly_trainer = create_timeseries_trainer(anomaly_detector, anomaly_config)
    anomaly_stats = anomaly_trainer.train(dataloader, dataloader, num_epochs=10)
    
    print(f"✅ Time Series Analysis Example Complete!")
    print(f"📈 Time Series Statistics:")
    print(f"   Task: {config.task.value}")
    print(f"   Architecture: {config.architecture.value}")
    print(f"   Sequence Length: {config.sequence_length}")
    print(f"   Prediction Horizon: {config.prediction_horizon}")
    print(f"   Best Loss: {training_stats['best_loss']:.4f}")
    print(f"   Final Train Loss: {training_stats['final_train_loss']:.4f}")
    print(f"   Final Val Loss: {training_stats['final_val_loss']:.4f}")
    print(f"🔍 Anomaly Detection Best Loss: {anomaly_stats['best_loss']:.4f}")
    
    return model

# Export utilities
__all__ = [
    'TimeSeriesTask',
    'ModelArchitecture',
    'TimeSeriesConfig',
    'TimeSeriesDataProcessor',
    'LSTMModel',
    'GRUModel',
    'TransformerModel',
    'CNNLSTMModel',
    'AnomalyDetector',
    'TimeSeriesTrainer',
    'create_timeseries_config',
    'create_lstm_model',
    'create_gru_model',
    'create_transformer_model',
    'create_cnn_lstm_model',
    'create_anomaly_detector',
    'create_timeseries_trainer',
    'example_time_series_analysis'
]

if __name__ == "__main__":
    example_time_series_analysis()
    print("✅ Time series analysis example completed successfully!")

