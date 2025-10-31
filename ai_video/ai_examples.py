from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
AI Examples - Concise Technical Implementations

Accurate Python examples for core AI concepts and patterns.
"""


# Example 1: Basic Neural Network
class SimpleNN(nn.Module):
    """Simple neural network with proper initialization."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        
    """__init__ function."""
super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
        # Proper weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Example 2: Training Loop with Best Practices
def train_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                epochs: int = 10) -> List[float]:
    """Efficient training loop with monitoring."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses

# Example 3: Model Evaluation
def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
    """Comprehensive model evaluation."""
    
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total
    }

# Example 4: Data Preprocessing Pipeline
class DataPreprocessor:
    """Efficient data preprocessing pipeline."""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        
    """__init__ function."""
self.mean = mean
        self.std = std
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to zero mean and unit variance."""
        return (data - self.mean) / self.std
    
    def augment(self, data: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation."""
        # Add noise
        noise = torch.randn_like(data) * 0.1
        return data + noise
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Complete preprocessing pipeline."""
        data = self.normalize(data)
        data = self.augment(data)
        return data

# Example 5: Model Saving and Loading
def save_model(model: nn.Module, path: str, metadata: Dict = None) -> None:
    """Save model with metadata."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.fc1.in_features,
            'hidden_size': model.fc1.out_features,
            'output_size': model.fc2.out_features
        },
        'metadata': metadata or {}
    }, path)

def load_model(path: str) -> Tuple[nn.Module, Dict]:
    """Load model with metadata."""
    checkpoint = torch.load(path)
    config = checkpoint['model_config']
    
    model = SimpleNN(config['input_size'], config['hidden_size'], config['output_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['metadata']

# Example 6: Batch Processing
def process_batch(batch: torch.Tensor, model: nn.Module, 
                  batch_size: int = 32) -> torch.Tensor:
    """Process data in batches to manage memory."""
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(batch), batch_size):
            batch_data = batch[i:i+batch_size]
            output = model(batch_data)
            results.append(output)
    
    return torch.cat(results, dim=0)

# Example 7: Model Performance Monitoring
class PerformanceMonitor:
    """Monitor model performance metrics."""
    
    def __init__(self) -> Any:
        self.metrics = {}
    
    def record_inference_time(self, model: nn.Module, input_data: torch.Tensor) -> float:
        """Record inference time."""
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(input_data)
        
        inference_time = time.time() - start_time
        self.metrics['inference_time'] = inference_time
        return inference_time
    
    def record_memory_usage(self) -> Dict[str, float]:
        """Record memory usage if CUDA is available."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            self.metrics['memory_allocated_mb'] = memory_allocated
            self.metrics['memory_reserved_mb'] = memory_reserved
            return {'allocated': memory_allocated, 'reserved': memory_reserved}
        return {}

# Example 8: Model Optimization
def optimize_for_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference."""
    
    # Set to evaluation mode
    model.eval()
    
    # Enable optimizations
    if torch.cuda.is_available():
        model = model.cuda()
        model = model.half()  # Use FP16
    
    # Fuse operations if possible
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    return model

# Example 9: Error Handling
def safe_model_inference(model: nn.Module, input_data: torch.Tensor) -> Optional[torch.Tensor]:
    """Safe model inference with error handling."""
    
    try:
        model.eval()
        with torch.no_grad():
            return model(input_data)
    except RuntimeError as e:
        print(f"Runtime error during inference: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example 10: Model Validation
def validate_model_output(output: torch.Tensor, expected_shape: Tuple[int, ...]) -> bool:
    """Validate model output."""
    
    # Check shape
    if output.shape != expected_shape:
        return False
    
    # Check for NaN values
    if torch.isnan(output).any():
        return False
    
    # Check for infinite values
    if torch.isinf(output).any():
        return False
    
    return True

# Usage Example
def main():
    """Demonstrate all examples."""
    
    # Create model
    model = SimpleNN(input_size=784, hidden_size=128, output_size=10)
    
    # Create dummy data
    dummy_data = torch.randn(100, 784)
    dummy_targets = torch.randint(0, 10, (100,))
    
    # Create dataloader
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train model
    losses = train_model(model, dataloader, epochs=5)
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader)
    print(f"Model accuracy: {metrics['accuracy']:.2f}%")
    
    # Monitor performance
    monitor = PerformanceMonitor()
    inference_time = monitor.record_inference_time(model, dummy_data[:16])
    print(f"Inference time: {inference_time:.4f}s")
    
    # Save and load model
    save_model(model, "model.pth", {"accuracy": metrics['accuracy']})
    loaded_model, metadata = load_model("model.pth")
    print(f"Loaded model metadata: {metadata}")

match __name__:
    case "__main__":
    main() 