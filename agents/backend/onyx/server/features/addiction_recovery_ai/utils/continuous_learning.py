"""
Continuous Learning and Online Learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OnlineLearner:
    """Online learning for continuous model updates"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        buffer_size: int = 1000,
        update_frequency: int = 10
    ):
        """
        Initialize online learner
        
        Args:
            model: Model to update
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device to use
            buffer_size: Buffer size for recent data
            update_frequency: Update frequency (every N samples)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        
        self.data_buffer = deque(maxlen=buffer_size)
        self.sample_count = 0
        self.update_count = 0
        
        logger.info("OnlineLearner initialized")
    
    def add_sample(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Add sample to buffer
        
        Args:
            inputs: Input tensor
            targets: Target tensor
        """
        self.data_buffer.append((inputs, targets))
        self.sample_count += 1
        
        # Update if frequency reached
        if self.sample_count % self.update_frequency == 0:
            self.update()
    
    def update(self):
        """Update model from buffer"""
        if len(self.data_buffer) == 0:
            return
        
        self.model.train()
        
        # Sample from buffer
        batch = list(self.data_buffer)
        inputs = torch.cat([x[0] for x in batch], dim=0).to(self.device)
        targets = torch.cat([x[1] for x in batch], dim=0).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        logger.debug(f"Online update {self.update_count}: loss={loss.item():.4f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "sample_count": self.sample_count,
            "update_count": self.update_count,
            "buffer_size": len(self.data_buffer),
            "update_frequency": self.update_frequency
        }


class IncrementalLearner:
    """Incremental learning for new data"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-4,
        memory_size: int = 500
    ):
        """
        Initialize incremental learner
        
        Args:
            model: Model to update
            device: Device to use
            learning_rate: Learning rate
            memory_size: Memory size for old samples
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        self.memory = deque(maxlen=memory_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        logger.info("IncrementalLearner initialized")
    
    def learn(
        self,
        new_data: List[tuple],
        epochs: int = 1,
        use_memory: bool = True
    ):
        """
        Learn from new data
        
        Args:
            new_data: List of (inputs, targets) tuples
            epochs: Number of epochs
            use_memory: Whether to use memory replay
        """
        # Combine new data with memory
        if use_memory:
            training_data = list(self.memory) + new_data
        else:
            training_data = new_data
        
        # Update memory
        for sample in new_data:
            self.memory.append(sample)
        
        # Train
        self.model.train()
        for epoch in range(epochs):
            for inputs, targets in training_data:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        
        logger.info(f"Incremental learning: {len(new_data)} new samples, {len(self.memory)} in memory")


class AdaptiveLearningRate:
    """Adaptive learning rate for continuous learning"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float = 1e-3,
        min_lr: float = 1e-6,
        decay_factor: float = 0.95,
        patience: int = 10
    ):
        """
        Initialize adaptive learning rate
        
        Args:
            optimizer: Optimizer
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            decay_factor: Decay factor
            patience: Patience for decay
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.current_lr = initial_lr
    
    def step(self, loss: float):
        """
        Update learning rate based on loss
        
        Args:
            loss: Current loss
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                # Decay learning rate
                self.current_lr = max(
                    self.min_lr,
                    self.current_lr * self.decay_factor
                )
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                
                self.patience_counter = 0
                logger.info(f"Learning rate decayed to {self.current_lr:.6f}")
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr

