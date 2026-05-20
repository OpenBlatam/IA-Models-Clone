"""
Advanced Test Data Generators
Sophisticated generators for creating test data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class DataDistribution(Enum):
    """Data distribution types"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    SPARSE = "sparse"
    DENSE = "dense"

@dataclass
class TestDataConfig:
    """Configuration for test data generation"""
    size: int = 100
    input_size: int = 10
    output_size: int = 5
    distribution: DataDistribution = DataDistribution.NORMAL
    seed: Optional[int] = None
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

class AdvancedTestDataGenerator:
    """Advanced test data generator"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        self.seed = seed
    
    def generate_tensor(
        self,
        shape: Tuple[int, ...],
        distribution: DataDistribution = DataDistribution.NORMAL,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Generate tensor with specified distribution"""
        if distribution == DataDistribution.UNIFORM:
            data = torch.rand(*shape, dtype=dtype, device=device)
        elif distribution == DataDistribution.NORMAL:
            data = torch.randn(*shape, dtype=dtype, device=device)
        elif distribution == DataDistribution.EXPONENTIAL:
            data = torch.empty(*shape, dtype=dtype, device=device)
            data.exponential_(1.0)
        elif distribution == DataDistribution.SPARSE:
            data = torch.zeros(*shape, dtype=dtype, device=device)
            # Make 10% non-zero
            num_nonzero = int(np.prod(shape) * 0.1)
            indices = torch.randperm(np.prod(shape))[:num_nonzero]
            data.view(-1)[indices] = torch.randn(num_nonzero, dtype=dtype)
        elif distribution == DataDistribution.DENSE:
            data = torch.ones(*shape, dtype=dtype, device=device) * 0.5
            data += torch.randn(*shape, dtype=dtype, device=device) * 0.1
        else:
            data = torch.randn(*shape, dtype=dtype, device=device)
        
        return data
    
    def generate_dataset(
        self,
        config: TestDataConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dataset with inputs and targets"""
        inputs = self.generate_tensor(
            (config.size, config.input_size),
            config.distribution,
            config.dtype,
            config.device
        )
        
        # Generate targets (could be based on inputs for more realistic data)
        targets = self.generate_tensor(
            (config.size, config.output_size),
            config.distribution,
            config.dtype,
            config.device
        )
        
        return inputs, targets
    
    def generate_model_weights(
        self,
        model: nn.Module,
        distribution: DataDistribution = DataDistribution.NORMAL,
        scale: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Generate random weights for a model"""
        weights = {}
        for name, param in model.named_parameters():
            if distribution == DataDistribution.NORMAL:
                weights[name] = torch.randn_like(param) * scale
            elif distribution == DataDistribution.UNIFORM:
                weights[name] = torch.rand_like(param) * scale - scale / 2
            else:
                weights[name] = torch.randn_like(param) * scale
        
        return weights
    
    def generate_sequence_data(
        self,
        seq_length: int,
        batch_size: int,
        vocab_size: int,
        distribution: DataDistribution = DataDistribution.UNIFORM
    ) -> torch.Tensor:
        """Generate sequence data for RNN/Transformer tests"""
        if distribution == DataDistribution.UNIFORM:
            return torch.randint(0, vocab_size, (batch_size, seq_length))
        else:
            # Normal distribution mapped to vocab
            data = torch.randn(batch_size, seq_length)
            data = ((data - data.min()) / (data.max() - data.min()) * vocab_size).long()
            return data.clamp(0, vocab_size - 1)
    
    def generate_image_data(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        distribution: DataDistribution = DataDistribution.NORMAL
    ) -> torch.Tensor:
        """Generate image-like data for CNN tests"""
        return self.generate_tensor(
            (batch_size, channels, height, width),
            distribution
        )
    
    def generate_edge_case_data(self, data_type: str) -> Any:
        """Generate edge case data"""
        if data_type == "empty":
            return torch.empty(0)
        elif data_type == "single_element":
            return torch.tensor([1.0])
        elif data_type == "all_zeros":
            return torch.zeros(10)
        elif data_type == "all_ones":
            return torch.ones(10)
        elif data_type == "very_large":
            return torch.randn(10000)
        elif data_type == "very_small":
            return torch.randn(1) * 1e-10
        elif data_type == "nan":
            return torch.tensor([float('nan')])
        elif data_type == "inf":
            return torch.tensor([float('inf')])
        else:
            return torch.randn(10)

class TestDataFactory:
    """Factory for creating test data generators"""
    
    @staticmethod
    def create_generator(seed: Optional[int] = None) -> AdvancedTestDataGenerator:
        """Create a test data generator"""
        return AdvancedTestDataGenerator(seed=seed)
    
    @staticmethod
    def create_config(**kwargs) -> TestDataConfig:
        """Create test data configuration"""
        return TestDataConfig(**kwargs)
    
    @staticmethod
    def create_standard_dataset(size: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create standard test dataset"""
        generator = AdvancedTestDataGenerator()
        config = TestDataConfig(size=size)
        return generator.generate_dataset(config)
    
    @staticmethod
    def create_edge_case_datasets() -> Dict[str, torch.Tensor]:
        """Create various edge case datasets"""
        generator = AdvancedTestDataGenerator()
        return {
            'empty': generator.generate_edge_case_data("empty"),
            'single': generator.generate_edge_case_data("single_element"),
            'zeros': generator.generate_edge_case_data("all_zeros"),
            'ones': generator.generate_edge_case_data("all_ones"),
            'large': generator.generate_edge_case_data("very_large"),
            'small': generator.generate_edge_case_data("very_small"),
            'nan': generator.generate_edge_case_data("nan"),
            'inf': generator.generate_edge_case_data("inf")
        }

def main():
    """Example usage"""
    factory = TestDataFactory()
    generator = factory.create_generator(seed=42)
    
    # Generate standard dataset
    inputs, targets = factory.create_standard_dataset(size=50)
    print(f"Generated dataset: inputs {inputs.shape}, targets {targets.shape}")
    
    # Generate edge cases
    edge_cases = factory.create_edge_case_datasets()
    print(f"\nGenerated {len(edge_cases)} edge case datasets")
    
    # Generate sequence data
    sequences = generator.generate_sequence_data(seq_length=20, batch_size=4, vocab_size=1000)
    print(f"\nGenerated sequences: {sequences.shape}")
    
    # Generate image data
    images = generator.generate_image_data(batch_size=2, channels=3, height=32, width=32)
    print(f"\nGenerated images: {images.shape}")

if __name__ == "__main__":
    main()







