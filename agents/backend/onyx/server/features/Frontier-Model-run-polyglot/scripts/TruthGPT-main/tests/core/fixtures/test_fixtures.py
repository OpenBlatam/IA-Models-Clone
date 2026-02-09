"""
Test Fixtures
Reusable test fixtures and factories
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass
from tests.test_utils import (
    create_test_model, create_test_dataset, create_test_tokenizer,
    SimpleModel, SimpleDataset, MockTokenizer
)

@dataclass
class TestFixture:
    """Container for test fixtures"""
    model: nn.Module
    dataset: Any
    tokenizer: Any
    config: Dict[str, Any]

class FixtureFactory:
    """Factory for creating test fixtures"""
    
    @staticmethod
    def create_basic_fixture(
        model_size: tuple = (10, 5),
        dataset_size: int = 100,
        vocab_size: int = 1000
    ) -> TestFixture:
        """Create basic test fixture"""
        model = create_test_model(
            input_size=model_size[0],
            output_size=model_size[1]
        )
        dataset = create_test_dataset(size=dataset_size)
        tokenizer = create_test_tokenizer(vocab_size=vocab_size)
        
        return TestFixture(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            config={
                'model_size': model_size,
                'dataset_size': dataset_size,
                'vocab_size': vocab_size
            }
        )
    
    @staticmethod
    def create_large_fixture() -> TestFixture:
        """Create large test fixture"""
        return FixtureFactory.create_basic_fixture(
            model_size=(100, 50),
            dataset_size=1000,
            vocab_size=5000
        )
    
    @staticmethod
    def create_small_fixture() -> TestFixture:
        """Create small test fixture"""
        return FixtureFactory.create_basic_fixture(
            model_size=(5, 3),
            dataset_size=10,
            vocab_size=100
        )
    
    @staticmethod
    def create_transformer_fixture(
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        vocab_size: int = 1000
    ) -> TestFixture:
        """Create transformer-specific fixture"""
        # Create a simple transformer-like model
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_size, num_heads),
                    num_layers
                )
                self.output = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x)
        
        model = SimpleTransformer()
        dataset = create_test_dataset(size=100, input_size=vocab_size, output_size=vocab_size)
        tokenizer = create_test_tokenizer(vocab_size=vocab_size)
        
        return TestFixture(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            config={
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'vocab_size': vocab_size
            }
        )

def get_fixture(name: str = "basic") -> TestFixture:
    """Get a test fixture by name"""
    factory = FixtureFactory()
    
    fixtures = {
        'basic': factory.create_basic_fixture,
        'small': factory.create_small_fixture,
        'large': factory.create_large_fixture,
        'transformer': factory.create_transformer_fixture,
    }
    
    if name not in fixtures:
        raise ValueError(f"Unknown fixture: {name}. Available: {list(fixtures.keys())}")
    
    return fixtures[name]()








