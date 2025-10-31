from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .config_manager import (
from ml.config import load_config, get_config
from ml.config import (
from ml.config import ConfigManager
from ml.config import ConfigManager, ConfigValidationError
from ml.config import get_config
from ml.models import ModelFactory, ModelConfig
from ml.training import TrainingManager, TrainingConfig
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration Module for Key Messages ML Pipeline
Provides YAML-based configuration management with environment-specific overrides
"""

    ConfigManager,
    ConfigValidationError,
    load_config,
    get_model_config,
    get_training_config,
    get_data_config,
    get_evaluation_config
)

# Version information
__version__ = "1.0.0"

# Module exports
__all__ = [
    "ConfigManager",
    "ConfigValidationError", 
    "load_config",
    "get_model_config",
    "get_training_config",
    "get_data_config",
    "get_evaluation_config"
]

# Convenience function for quick configuration access
def get_config(environment: str = None) -> dict:
    """
    Quick access to configuration with environment support.
    
    Args:
        environment: Environment name (development, staging, production)
                    If None, uses ML_ENVIRONMENT env var or defaults to development
    
    Returns:
        Configuration dictionary
    """
    return load_config(environment=environment)

# Example usage
def example_usage():
    """
    Example usage of the configuration system.
    """
    print("""
# Configuration System Usage Examples

## 1. Basic Configuration Loading
```python

# Load configuration with default environment
config = load_config()

# Or specify environment
config = load_config(environment="production")

# Quick access
config = get_config("development")
```

## 2. Accessing Specific Configurations
```python
    get_model_config,
    get_training_config,
    get_data_config,
    get_evaluation_config
)

# Get model configuration
model_config = get_model_config("gpt2", environment="production")

# Get training configuration
training_config = get_training_config("production", environment="production")

# Get data configuration
data_config = get_data_config("high_performance", environment="production")

# Get evaluation configuration
eval_config = get_evaluation_config("comprehensive", environment="production")
```

## 3. Using ConfigManager for Advanced Usage
```python

# Initialize config manager
config_manager = ConfigManager(
    config_dir="config",
    environment="production"
)

# Load configuration
config = config_manager.load_config()

# Get specific configurations
model_config = config_manager.get_model_config("gpt2", config)
training_config = config_manager.get_training_config("production", config)

# Resolve device and dtype
device = config_manager.resolve_device("auto")  # Returns "cuda" or "cpu"
dtype = config_manager.resolve_torch_dtype("auto")  # Returns torch.float16 or torch.float32

# Update configuration
updates = {"training": {"default": {"batch_size": 64}}}
updated_config = config_manager.update_config(updates, config)

# Save configuration
config_manager.save_config(updated_config, "updated_config.yaml")

# Get configuration summary
summary = config_manager.get_config_summary(config)
print(summary)
```

## 4. Environment-Specific Configuration
```python
# Development environment (faster, less resources)
dev_config = load_config(environment="development")

# Production environment (optimized, full features)
prod_config = load_config(environment="production")

# The system automatically applies environment-specific overrides
```

## 5. Configuration Structure
```yaml
# Main config.yaml
app:
  name: "key_messages_ml_pipeline"
  version: "1.0.0"
  environment: "development"

models:
  gpt2:
    model_name: "gpt2"
    max_length: 512
    temperature: 0.7
    device: "auto"

training:
  default:
    batch_size: 16
    learning_rate: 1.0e-4
    num_epochs: 5
    use_mixed_precision: true

# Environment-specific overrides in environments/development.yaml
training:
  default:
    batch_size: 4  # Override for development
    num_epochs: 2
    use_mixed_precision: false
```

## 6. Configuration Validation
```python

try:
    config_manager = ConfigManager()
    config = config_manager.load_config()
    print("Configuration is valid!")
except ConfigValidationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Field: {e.field}")
    print(f"Value: {e.value}")
```

## 7. Integration with ML Pipeline
```python

# Load configuration
config = get_config("production")

# Create model with configuration
model_config_dict = config["models"]["gpt2"]
model_config = ModelConfig(**model_config_dict)
model = ModelFactory.create_model("gpt2", model_config)

# Create training manager with configuration
training_config_dict = config["training"]["production"]
training_config = TrainingConfig(**training_config_dict)
training_manager = TrainingManager(training_config)
```
""")

match __name__:
    case "__main__":
    example_usage() 