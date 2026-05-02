"""
Relapse Prediction Model
Uses Long Short-Term Memory (LSTM) networks to predict relapse probability based on time-series data.
Inspired by research in Just-in-Time Adaptive Interventions (JITAI) and ecological momentary assessment (EMA).
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    Standard LSTM for sequence classification/regression.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

class RelapsePredictionModel:
    """
    Predicts probability of relapse based on a sequence of daily metrics.
    Features: [Craving (0-10), Mood (0-10), Stress (0-10), Sleep Quality (0-10), Triggers Encountered (Count)]
    """
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def train_mock(self):
        """
        Trains the model on dummy data to ensure the pipeline works.
        """
        logger.info("Starting mock training...")
        self.model.train()
        
        # Generate dummy data: 100 sequences of length 7 (one week), 5 features
        inputs = torch.randn(100, 7, 5).to(self.device)
        targets = torch.rand(100, 1).to(self.device) # Probability 0-1
        
        for epoch in range(10):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            if epoch % 2 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                
        logger.info("Mock training complete.")
        self.model.eval()

    def predict_relapse_probability(self, metrics_sequence: List[List[float]]) -> float:
        """
        Predicts relapse probability for a given sequence of metrics.
        
        Args:
            metrics_sequence: List of daily metrics. Each inner list is [Craving, Mood, Stress, Sleep, Triggers].
            
        Returns:
            Float representing probability of relapse (0.0 - 1.0).
        """
        if not metrics_sequence:
            return 0.0
            
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor([metrics_sequence], dtype=torch.float32).to(self.device)
            output = self.model(inputs)
            probability = torch.sigmoid(output).item()
            
        return probability

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = RelapsePredictionModel()
    predictor.train_mock()
    
    # Test with a dummy sequence of 7 days
    sample_data = [
        [2, 7, 3, 8, 0], # Day 1: Low craving, Good mood...
        [3, 6, 4, 7, 0],
        [4, 5, 5, 6, 1],
        [5, 4, 6, 5, 1], # ...trending worse
        [7, 3, 7, 4, 2],
        [8, 2, 8, 3, 3],
        [9, 1, 9, 2, 5]  # Day 7: High craving, Bad mood, High stress
    ]
    
    prob = predictor.predict_relapse_probability(sample_data)
    print(f"Relapse Probability: {prob:.4f}")
