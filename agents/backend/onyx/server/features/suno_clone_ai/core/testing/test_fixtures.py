"""
Test Fixtures

Utilities for creating test fixtures.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def create_test_model(
    input_dim: int = 128,
    hidden_dim: int = 256,
    output_dim: int = 128,
    num_layers: int = 2
) -> nn.Module:
    """
    Create a simple test model.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of layers
        
    Returns:
        Test model
    """
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    return nn.Sequential(*layers)


def create_test_data(
    num_samples: int = 100,
    input_dim: int = 128,
    output_dim: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create test data.
    
    Args:
        num_samples: Number of samples
        input_dim: Input dimension
        output_dim: Output dimension
        
    Returns:
        (inputs, targets) tuple
    """
    inputs = torch.randn(num_samples, input_dim)
    targets = torch.randn(num_samples, output_dim)
    
    return inputs, targets


class TestDataset(Dataset):
    """Simple test dataset."""
    
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Initialize test dataset.
        
        Args:
            inputs: Input tensors
            targets: Target tensors
        """
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'target': self.targets[idx]
        }


def create_test_dataloader(
    num_samples: int = 100,
    batch_size: int = 32,
    input_dim: int = 128,
    output_dim: int = 128
) -> DataLoader:
    """
    Create test data loader.
    
    Args:
        num_samples: Number of samples
        batch_size: Batch size
        input_dim: Input dimension
        output_dim: Output dimension
        
    Returns:
        DataLoader instance
    """
    inputs, targets = create_test_data(num_samples, input_dim, output_dim)
    dataset = TestDataset(inputs, targets)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



