"""
Base Model Class for TruthGPT API
=================================

Abstract base class for all TruthGPT models, providing TensorFlow-like interface.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class Model(ABC):
    """
    Abstract base class for TruthGPT models.
    
    This class provides the foundation for all TruthGPT models,
    implementing TensorFlow-like interface patterns.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the model.
        
        Args:
            name: Optional name for the model
        """
        self.name = name or self.__class__.__name__
        self._compiled = False
        self._optimizer = None
        self._loss = None
        self._metrics = []
        self._training_history = []
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        pass
    
    def __call__(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """Enable model to be called like a function."""
        return self.call(inputs, training)
    
    def compile(self, 
                optimizer: Any = None,
                loss: Any = None,
                metrics: List[str] = None,
                **kwargs):
        """
        Configure the model for training.
        
        Args:
            optimizer: Optimizer instance
            loss: Loss function
            metrics: List of metric names
            **kwargs: Additional compilation arguments
        """
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics or []
        self._compiled = True
        
        # Move model to device
        if hasattr(self, 'to'):
            self.to(self._device)
    
    def fit(self, 
            x: Union[np.ndarray, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            epochs: int = 1,
            batch_size: int = 32,
            validation_data: Optional[Tuple] = None,
            verbose: int = 1,
            **kwargs) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            x: Training data
            y: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            validation_data: Validation data tuple (x_val, y_val)
            verbose: Verbosity level
            **kwargs: Additional training arguments
            
        Returns:
            Training history dictionary
        """
        if not self._compiled:
            raise ValueError("Model must be compiled before training")
        
        # Convert to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        
        # Move to device
        x = x.to(self._device)
        y = y.to(self._device)
        
        history = {'loss': [], 'accuracy': []}
        if validation_data:
            history['val_loss'] = []
            history['val_accuracy'] = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            self.train()  # Set training mode
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                
                # Forward pass
                outputs = self(batch_x, training=True)
                
                # Calculate loss
                loss = self._loss(outputs, batch_y)
                
                # Backward pass
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == batch_y).float().mean()
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                num_batches += 1
            
            # Average metrics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_accuracy)
            
            # Validation
            if validation_data:
                val_loss, val_accuracy = self.evaluate(validation_data[0], validation_data[1], verbose=0)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
            
            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}")
        
        self._training_history = history
        return history
    
    def evaluate(self, 
                 x: Union[np.ndarray, torch.Tensor],
                 y: Union[np.ndarray, torch.Tensor],
                 batch_size: int = 32,
                 verbose: int = 1) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            x: Test data
            y: Test labels
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if not self._compiled:
            raise ValueError("Model must be compiled before evaluation")
        
        # Convert to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        
        # Move to device
        x = x.to(self._device)
        y = y.to(self._device)
        
        self.eval()  # Set evaluation mode
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                
                # Forward pass
                outputs = self(batch_x, training=False)
                
                # Calculate loss
                loss = self._loss(outputs, batch_y)
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == batch_y).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        if verbose > 0:
            print(f"Test loss: {avg_loss:.4f}, Test accuracy: {avg_accuracy:.4f}")
        
        return avg_loss, avg_accuracy
    
    def predict(self, 
                x: Union[np.ndarray, torch.Tensor],
                batch_size: int = 32,
                verbose: int = 1) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            x: Input data
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Predictions as numpy array
        """
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Move to device
        x = x.to(self._device)
        
        self.eval()  # Set evaluation mode
        
        predictions = []
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self._device)
                
                # Forward pass
                outputs = self(batch_x, training=False)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def train(self):
        """Set model to training mode."""
        if hasattr(self, 'train'):
            super().train()
    
    def eval(self):
        """Set model to evaluation mode."""
        if hasattr(self, 'eval'):
            super().eval()
    
    def to(self, device):
        """Move model to device."""
        self._device = device
        if hasattr(self, 'to'):
            return super().to(device)
        return self
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        self.load_state_dict(torch.load(filepath))
    
    def summary(self):
        """Print model summary."""
        print(f"Model: {self.name}")
        print(f"Device: {self._device}")
        print(f"Compiled: {self._compiled}")
        if self._compiled:
            print(f"Optimizer: {self._optimizer}")
            print(f"Loss: {self._loss}")
            print(f"Metrics: {self._metrics}")


