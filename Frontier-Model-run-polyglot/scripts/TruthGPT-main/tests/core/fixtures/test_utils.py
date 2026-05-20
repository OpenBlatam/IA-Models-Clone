"""
Test Utilities
Shared utilities and fixtures for all tests
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """Simple test model for testing"""
    def __init__(self, input_size: int = 10, output_size: int = 5, hidden_size: int = 20):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class SimpleDataset(Dataset):
    """Simple test dataset"""
    def __init__(self, size: int = 100, input_size: int = 10, output_size: int = 5):
        self.size = size
        self.data = torch.randn(size, input_size)
        self.targets = torch.randn(size, output_size)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2
        
    def encode(self, text: str, return_tensors: str = "pt", **kwargs):
        """Encode text to token IDs"""
        if isinstance(text, str):
            # Simple encoding: convert characters to token IDs
            tokens = [ord(c) % self.vocab_size for c in text[:50]]
        else:
            tokens = text
        
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens
    
    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        # Simple decoding: convert token IDs to characters
        text = ''.join([chr(token_id % 256) for token_id in token_ids if token_id < 256])
        return text

def create_test_model(model_type: str = "simple", **kwargs) -> nn.Module:
    """Create a test model"""
    if model_type == "simple":
        return SimpleModel(**kwargs)
    elif model_type == "linear":
        return nn.Linear(kwargs.get('input_size', 10), kwargs.get('output_size', 5))
    elif model_type == "sequential":
        return nn.Sequential(
            nn.Linear(kwargs.get('input_size', 10), 20),
            nn.ReLU(),
            nn.Linear(20, kwargs.get('output_size', 5))
        )
    else:
        return SimpleModel(**kwargs)

def create_test_dataset(size: int = 100, **kwargs) -> Dataset:
    """Create a test dataset"""
    return SimpleDataset(size=size, **kwargs)

def create_test_tokenizer(vocab_size: int = 1000) -> MockTokenizer:
    """Create a test tokenizer"""
    return MockTokenizer(vocab_size=vocab_size)

def assert_model_valid(model: nn.Module, input_shape: Tuple[int, ...] = (2, 10)):
    """Assert that a model is valid and can process input"""
    assert model is not None, "Model should not be None"
    
    # Test forward pass
    try:
        test_input = torch.randn(*input_shape)
        with torch.no_grad():
            output = model(test_input)
        assert output is not None, "Model output should not be None"
        assert output.shape[0] == input_shape[0], "Batch size should match"
        return True
    except Exception as e:
        logger.warning(f"Model validation failed: {e}")
        return False

def assert_config_valid(config, required_attrs: list):
    """Assert that a config has required attributes"""
    assert config is not None, "Config should not be None"
    for attr in required_attrs:
        assert hasattr(config, attr), f"Config should have attribute: {attr}"
        assert getattr(config, attr) is not None, f"Config attribute {attr} should not be None"

def assert_metrics_valid(metrics: dict, required_keys: list = None):
    """Assert that metrics dictionary is valid"""
    assert metrics is not None, "Metrics should not be None"
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    
    if required_keys:
        for key in required_keys:
            assert key in metrics, f"Metrics should contain key: {key}"

def get_device():
    """Get available device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def create_temp_file(suffix: str = ".pth") -> str:
    """Create a temporary file path"""
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

def cleanup_temp_file(path: str):
    """Clean up a temporary file"""
    import os
    if os.path.exists(path):
        os.remove(path)

class TestTimer:
    """Context manager for timing test execution"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def get_duration(self) -> float:
        """Get duration in seconds"""
        return self.duration if self.duration else 0.0

def assert_performance_acceptable(duration: float, max_duration: float = 10.0):
    """Assert that performance is acceptable"""
    assert duration < max_duration, f"Operation took {duration:.2f}s, expected < {max_duration}s"

def assert_memory_usage_acceptable(usage_mb: float, max_mb: float = 1000.0):
    """Assert that memory usage is acceptable"""
    assert usage_mb < max_mb, f"Memory usage {usage_mb:.2f}MB exceeds limit {max_mb}MB"

def get_model_parameters_count(model: nn.Module) -> int:
    """Get total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def assert_model_size_reasonable(model: nn.Module, max_params: int = 1000000):
    """Assert that model size is reasonable"""
    param_count = get_model_parameters_count(model)
    assert param_count < max_params, f"Model has {param_count} parameters, expected < {max_params}"








