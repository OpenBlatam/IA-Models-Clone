"""Kalman filter implementation for training optimization."""
from collections import deque
from typing import Tuple

import numpy as np

DEFAULT_INITIAL_STATE = 0.0
DEFAULT_INITIAL_COVARIANCE = 1.0
DEFAULT_MOMENTUM = 0.9
DEFAULT_MEMORY_SIZE = 1000
DEFAULT_STATISTICS_MEAN = 0.0
DEFAULT_STATISTICS_STD = 1.0
WEIGHT_EXPONENT_START = -1
WEIGHT_EXPONENT_END = 0


class KalmanFilter:
    """Kalman filter for adaptive training optimization."""
    
    def __init__(self, process_noise: float, measurement_noise: float, memory_size: int = DEFAULT_MEMORY_SIZE):
        if process_noise < 0:
            raise ValueError("process_noise must be non-negative")
        if measurement_noise < 0:
            raise ValueError("measurement_noise must be non-negative")
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        
        self.Q = process_noise
        self.R = measurement_noise
        self.mu = DEFAULT_INITIAL_STATE
        self.P = DEFAULT_INITIAL_COVARIANCE
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.momentum = DEFAULT_MOMENTUM
        self.velocity = DEFAULT_INITIAL_STATE
        
    def update(self, measurement: float) -> float:
        """Update filter with new measurement."""
        if not isinstance(measurement, (int, float)):
            raise TypeError(f"measurement must be numeric, got {type(measurement)}")
        
        mu_pred = self.mu + self.momentum * self.velocity
        P_pred = self.P + self.Q
        
        K = P_pred / (P_pred + self.R)
        innovation = measurement - mu_pred
        self.mu = mu_pred + K * innovation
        self.P = (1 - K) * P_pred + self.Q
        
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * innovation
        
        self.memory.append(measurement)
            
        return self.mu
    
    def get_statistics(self) -> Tuple[float, float]:
        """Get mean and standard deviation of recent measurements with exponential weighting.
        
        Returns:
            Tuple[float, float]: (weighted_mean, weighted_std) of recent measurements
        """
        if not self.memory:
            return DEFAULT_STATISTICS_MEAN, DEFAULT_STATISTICS_STD
        
        memory_array = np.array(self.memory)
        weights = np.exp(np.linspace(WEIGHT_EXPONENT_START, WEIGHT_EXPONENT_END, len(memory_array)))
        weights /= weights.sum()
        
        weighted_mean = np.average(memory_array, weights=weights)
        weighted_std = np.sqrt(np.average((memory_array - weighted_mean) ** 2, weights=weights))
        
        return float(weighted_mean), float(weighted_std)


