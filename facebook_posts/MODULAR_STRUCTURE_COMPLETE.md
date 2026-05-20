# Modular Structure Organization - Complete Guide
==============================================

## Overview

This document outlines how to organize the file structure into proper modules, integrating all patterns: type hints, Pydantic validation, async/sync patterns, RORO pattern, and named exports.

## Table of Contents

1. [Recommended Module Structure](#recommended-module-structure)
2. [Core Modules](#core-modules)
3. [Utils Modules](#utils-modules)
4. [Configs Modules](#configs-modules)
5. [Interfaces Modules](#interfaces-modules)
6. [Main Module Organization](#main-module-organization)
7. [Import/Export Patterns](#importexport-patterns)
8. [Best Practices](#best-practices)
9. [Integration with Existing Patterns](#integration-with-existing-patterns)
10. [Examples from Current Codebase](#examples-from-current-codebase)

## Recommended Module Structure

```
facebook_posts/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_networks.py
│   │   ├── transformers.py
│   │   └── diffusion_models.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── optimizers.py
│   │   ├── loss_functions.py
│   │   └── schedulers.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── validators.py
├── utils/
│   ├── __init__.py
│   ├── type_hints.py
│   ├── pydantic_models.py
│   ├── async_helpers.py
│   └── error_handlers.py
├── configs/
│   ├── __init__.py
│   ├── model_configs.py
│   ├── training_configs.py
│   └── data_configs.py
├── interfaces/
│   ├── __init__.py
│   ├── roro_pattern.py
│   └── named_exports.py
└── examples/
    ├── __init__.py
    ├── training_examples.py
    └── evaluation_examples.py
```

## Core Modules

### Core Models Module

**`core/models/neural_networks.py`:**
```python
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, confloat, constr
import torch.nn as nn

__all__ = [
    "create_feedforward_network",
    "create_convolutional_network", 
    "create_recurrent_network",
    "NeuralNetworkConfig",
    "NetworkArchitecture"
]

class NeuralNetworkConfig(BaseModel):
    """Pydantic model for neural network configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    input_dimension: conint(gt=0) = Field(description="Input dimension")
    output_dimension: conint(gt=0) = Field(description="Output dimension")
    hidden_layer_sizes: List[conint(gt=0)] = Field(description="Hidden layer sizes")
    activation_function: constr(strip_whitespace=True) = Field(
        default="relu",
        pattern=r"^(relu|sigmoid|tanh|softmax)$"
    )
    dropout_rate: confloat(ge=0.0, le=1.0) = Field(default=0.2)

def create_feedforward_network(
    config: NeuralNetworkConfig
) -> nn.Module:
    """Create feedforward neural network with type hints."""
    layers = []
    
    # Input layer
    layers.append(nn.Linear(config.input_dimension, config.hidden_layer_sizes[0]))
    layers.append(get_activation_function(config.activation_function))
    layers.append(nn.Dropout(config.dropout_rate))
    
    # Hidden layers
    for i in range(len(config.hidden_layer_sizes) - 1):
        layers.append(nn.Linear(config.hidden_layer_sizes[i], config.hidden_layer_sizes[i + 1]))
        layers.append(get_activation_function(config.activation_function))
        layers.append(nn.Dropout(config.dropout_rate))
    
    # Output layer
    layers.append(nn.Linear(config.hidden_layer_sizes[-1], config.output_dimension))
    
    return nn.Sequential(*layers)
```

### Core Training Module

**`core/training/optimizers.py`:**
```python
from typing import Callable
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import confloat, constr
import torch
import torch.nn as nn

__all__ = [
    "create_optimizer",
    "create_scheduler",
    "OptimizerConfig",
    "SchedulerConfig"
]

class OptimizerConfig(BaseModel):
    """Pydantic model for optimizer configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    optimizer_type: constr(strip_whitespace=True) = Field(
        pattern=r"^(adam|sgd|rmsprop|adamw)$"
    )
    learning_rate: confloat(gt=0.0, lt=1.0) = Field(default=0.001)
    weight_decay: confloat(ge=0.0) = Field(default=0.0)
    momentum: confloat(ge=0.0, le=1.0) = Field(default=0.9)

def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig
) -> torch.optim.Optimizer:
    """Create optimizer with type hints."""
    if config.optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")
```

### Core Data Module

**`core/data/loaders.py`:**
```python
import asyncio
import aiofiles
from typing import Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, constr
from pathlib import Path

__all__ = [
    "load_training_data_async",
    "load_validation_data_async",
    "DataLoaderConfig",
    "DataFormat"
]

class DataLoaderConfig(BaseModel):
    """Pydantic model for data loader configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    data_path: constr(strip_whitespace=True) = Field(description="Path to data")
    data_format: constr(strip_whitespace=True) = Field(
        default="json",
        pattern=r"^(json|csv|parquet|hdf5)$"
    )
    batch_size: conint(gt=0, le=10000) = Field(default=32)
    shuffle: bool = Field(default=True)
    num_workers: conint(ge=0, le=16) = Field(default=0)

async def load_training_data_async(
    config: DataLoaderConfig
) -> Dict[str, Any]:
    """Load training data asynchronously with type hints."""
    try:
        # Validate inputs
        if not Path(config.data_path).exists():
            raise ValueError(f"Data path does not exist: {config.data_path}")
        
        # Load data based on format
        async with aiofiles.open(config.data_path, 'r') as file:
            content = await file.read()
        
        if config.data_format == "json":
            import json
            data = json.loads(content)
        elif config.data_format == "csv":
            import csv
            from io import StringIO
            data = list(csv.DictReader(StringIO(content)))
        else:
            raise ValueError(f"Unsupported data format: {config.data_format}")
        
        return {
            "is_successful": True,
            "result": data,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

## Utils Modules

### Type Hints Module

**`utils/type_hints.py`:**
```python
from typing import Any, List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict

__all__ = [
    "ProcessingResult",
    "ValidationResult",
    "Result",
    "T",
    "U"
]

T = TypeVar('T')
U = TypeVar('U')

class ProcessingResult(BaseModel):
    """Pydantic model for processing results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether processing was successful")
    result: Optional[Any] = Field(default=None, description="Processing result")
    error: Optional[str] = Field(default=None, description="Error message")
    processing_time: Optional[float] = Field(default=None, description="Processing time")

class ValidationResult(BaseModel):
    """Pydantic model for validation results."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

class Result(Generic[T]):
    """Generic result type for type-safe operations."""
    
    def __init__(self, value: T, is_success: bool, error: Optional[str] = None):
        self.value = value
        self.is_success = is_success
        self.error = error
```

### Async Helpers Module

**`utils/async_helpers.py`:**
```python
import asyncio
from typing import Callable, Any
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import confloat, conint

__all__ = [
    "async_retry",
    "async_timeout",
    "AsyncConfig",
    "RetryConfig"
]

class AsyncConfig(BaseModel):
    """Pydantic model for async configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    timeout: confloat(gt=0.0) = Field(default=30.0, description="Timeout in seconds")
    max_retries: conint(ge=0, le=10) = Field(default=3, description="Maximum retries")
    retry_delay: confloat(gt=0.0) = Field(default=1.0, description="Delay between retries")

class RetryConfig(BaseModel):
    """Pydantic model for retry configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    max_attempts: conint(gt=0, le=10) = Field(default=3)
    backoff_factor: confloat(gt=0.0) = Field(default=2.0)
    initial_delay: confloat(gt=0.0) = Field(default=1.0)

async def async_retry(
    func: Callable,
    config: RetryConfig,
    *args,
    **kwargs
) -> Any:
    """Retry async function with exponential backoff."""
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            if attempt < config.max_attempts - 1:
                delay = config.initial_delay * (config.backoff_factor ** attempt)
                await asyncio.sleep(delay)
    
    raise last_exception

async def async_timeout(
    func: Callable,
    timeout: float,
    *args,
    **kwargs
) -> Any:
    """Execute async function with timeout."""
    try:
        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Function timed out after {timeout} seconds")
```

## Configs Modules

### Model Configs Module

**`configs/model_configs.py`:**
```python
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import conint, confloat, constr

__all__ = [
    "ModelConfiguration",
    "TrainingConfiguration",
    "DataConfiguration",
    "EvaluationConfiguration"
]

class ModelConfiguration(BaseModel):
    """Pydantic model for model configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    input_dimension: conint(gt=0) = Field(description="Input dimension")
    output_dimension: conint(gt=0) = Field(description="Output dimension")
    hidden_layer_sizes: List[conint(gt=0)] = Field(description="Hidden layer sizes")
    activation_function: constr(strip_whitespace=True) = Field(
        default="relu",
        pattern=r"^(relu|sigmoid|tanh|softmax)$"
    )
    dropout_rate: confloat(ge=0.0, le=1.0) = Field(default=0.2)
    learning_rate: confloat(gt=0.0, lt=1.0) = Field(default=0.001)

class TrainingConfiguration(BaseModel):
    """Pydantic model for training configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    max_epochs: conint(gt=0, le=10000) = Field(description="Maximum epochs")
    batch_size: conint(gt=0, le=10000) = Field(description="Batch size")
    learning_rate: confloat(gt=0.0, lt=1.0) = Field(default=0.001)
    early_stopping_patience: conint(ge=0, le=1000) = Field(default=10)
    checkpoint_path: Optional[constr(strip_whitespace=True)] = Field(default=None)
    validation_split: confloat(ge=0.0, lt=1.0) = Field(default=0.2)

class DataConfiguration(BaseModel):
    """Pydantic model for data configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    data_path: constr(strip_whitespace=True) = Field(description="Data path")
    data_format: constr(strip_whitespace=True) = Field(
        default="json",
        pattern=r"^(json|csv|parquet|hdf5)$"
    )
    normalization_type: constr(strip_whitespace=True) = Field(
        default="standard",
        pattern=r"^(standard|minmax|robust|none)$"
    )
    shuffle_data: bool = Field(default=True)
    random_seed: Optional[conint(ge=0)] = Field(default=None)

class EvaluationConfiguration(BaseModel):
    """Pydantic model for evaluation configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    metrics: List[constr(strip_whitespace=True)] = Field(
        default=["accuracy"],
        description="Evaluation metrics"
    )
    save_predictions: bool = Field(default=False)
    output_path: Optional[constr(strip_whitespace=True)] = Field(default=None)
```

## Interfaces Modules

### RORO Pattern Module

**`interfaces/roro_pattern.py`:**
```python
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

__all__ = [
    "roro_function",
    "RORORequest",
    "ROROResponse"
]

class RORORequest(BaseModel):
    """Pydantic model for RORO request."""
    
    model_config = ConfigDict(extra="forbid")
    
    data: Dict[str, Any] = Field(description="Request data")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")

class ROROResponse(BaseModel):
    """Pydantic model for RORO response."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether operation was successful")
    result: Optional[Any] = Field(default=None, description="Operation result")
    error: Optional[str] = Field(default=None, description="Error message")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Response metadata")

def roro_function(
    request: RORORequest
) -> ROROResponse:
    """Example RORO pattern function with type hints."""
    try:
        # Process request data
        data = request.data
        config = request.config or {}
        
        # Perform operation
        result = process_data(data, config)
        
        return ROROResponse(
            is_successful=True,
            result=result,
            error=None
        )
    except Exception as exc:
        return ROROResponse(
            is_successful=False,
            result=None,
            error=str(exc)
        )

def process_data(data: Dict[str, Any], config: Dict[str, Any]) -> Any:
    """Process data (placeholder implementation)."""
    return {"processed": data, "config": config}
```

### Named Exports Module

**`interfaces/named_exports.py`:**
```python
__all__ = [
    # Core modules exports
    "CORE_EXPORTS",
    "UTILS_EXPORTS", 
    "CONFIGS_EXPORTS",
    "INTERFACES_EXPORTS",
    "get_all_exports"
]

# Core modules exports
CORE_EXPORTS = [
    "NeuralNetworkModule",
    "OptimizerModule",
    "DataLoaderModule"
]

# Utils modules exports
UTILS_EXPORTS = [
    "TypeHintsModule",
    "AsyncHelpersModule"
]

# Configs modules exports
CONFIGS_EXPORTS = [
    "ModelConfigsModule"
]

# Interfaces modules exports
INTERFACES_EXPORTS = [
    "ROROPatternModule"
]

def get_all_exports() -> list:
    """Get all module exports."""
    return (
        CORE_EXPORTS +
        UTILS_EXPORTS +
        CONFIGS_EXPORTS +
        INTERFACES_EXPORTS
    )
```

## Main Module Organization

### Main Init File

**`__init__.py`:**
```python
# Import all modules
from .core.models.neural_networks import NeuralNetworkModule
from .core.training.optimizers import OptimizerModule
from .core.data.loaders import DataLoaderModule
from .utils.type_hints import TypeHintsModule
from .utils.async_helpers import AsyncHelpersModule
from .configs.model_configs import ModelConfigsModule
from .interfaces.roro_pattern import ROROPatternModule

# Define main exports
__all__ = [
    # Core modules
    "NeuralNetworkModule",
    "OptimizerModule", 
    "DataLoaderModule",
    
    # Utils modules
    "TypeHintsModule",
    "AsyncHelpersModule",
    
    # Configs modules
    "ModelConfigsModule",
    
    # Interfaces modules
    "ROROPatternModule",
    
    # Main functions
    "train_model",
    "evaluate_model",
    "load_data"
]

def train_model(
    model_config: ModelConfigsModule.ModelConfiguration,
    training_config: ModelConfigsModule.TrainingConfiguration,
    data_config: ModelConfigsModule.DataConfiguration
) -> TypeHintsModule.ProcessingResult:
    """Main training function with all patterns integrated."""
    # Implementation using all modules
    pass

def evaluate_model(
    model: nn.Module,
    data_config: ModelConfigsModule.DataConfiguration,
    eval_config: ModelConfigsModule.EvaluationConfiguration
) -> TypeHintsModule.ProcessingResult:
    """Main evaluation function with all patterns integrated."""
    # Implementation using all modules
    pass

async def load_data(
    config: ModelConfigsModule.DataConfiguration
) -> TypeHintsModule.ProcessingResult:
    """Main data loading function with all patterns integrated."""
    # Implementation using all modules
    pass
```

## Import/Export Patterns

### Proper Import Structure

**Module imports:**
```python
# Import specific functions/classes
from core.models.neural_networks import create_feedforward_network, NeuralNetworkConfig
from core.training.optimizers import create_optimizer, OptimizerConfig
from core.data.loaders import load_training_data_async, DataLoaderConfig

# Import entire modules
from utils.type_hints import ProcessingResult, ValidationResult
from utils.async_helpers import async_retry, async_timeout

# Import configurations
from configs.model_configs import ModelConfiguration, TrainingConfiguration

# Import interfaces
from interfaces.roro_pattern import roro_function, RORORequest, ROROResponse
```

### Named Exports Usage

**Using named exports:**
```python
# In each module
__all__ = [
    "function_name",
    "ClassName",
    "CONSTANT_NAME"
]

# When importing
from module_name import function_name, ClassName, CONSTANT_NAME
```

## Best Practices

### 1. Module Organization

**✅ Good:**
```python
# Clear module structure
core/
├── models/
│   ├── neural_networks.py
│   └── transformers.py
├── training/
│   ├── optimizers.py
│   └── loss_functions.py
└── data/
    ├── loaders.py
    └── preprocessing.py
```

**❌ Avoid:**
```python
# Flat structure
neural_networks.py
optimizers.py
loss_functions.py
data_loaders.py
```

### 2. Import Organization

**✅ Good:**
```python
# Standard library imports
import asyncio
import json
from pathlib import Path

# Third-party imports
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

# Local imports
from .utils.type_hints import ProcessingResult
from .configs.model_configs import ModelConfiguration
```

### 3. Export Organization

**✅ Good:**
```python
__all__ = [
    # Functions
    "create_model",
    "train_model",
    "evaluate_model",
    
    # Classes
    "ModelConfiguration",
    "TrainingConfiguration",
    
    # Constants
    "DEFAULT_CONFIG",
    "SUPPORTED_MODELS"
]
```

## Integration with Existing Patterns

### 1. Type Hints Integration

```python
# In each module
def module_function(
    param1: torch.Tensor,
    param2: ModelConfiguration,
    param3: Optional[str] = None
) -> ProcessingResult:
    """Function with comprehensive type hints."""
    pass
```

### 2. Pydantic Validation Integration

```python
# In configs modules
class ModuleConfig(BaseModel):
    """Pydantic model for module configuration."""
    
    model_config = ConfigDict(extra="forbid")
    
    required_param: conint(gt=0) = Field(description="Required parameter")
    optional_param: Optional[str] = Field(default=None, description="Optional parameter")
```

### 3. Async/Sync Pattern Integration

```python
# CPU-bound functions in core modules
def cpu_intensive_function(data: torch.Tensor) -> torch.Tensor:
    """CPU-bound function with type hints."""
    pass

# I/O-bound functions in core modules
async def io_intensive_function(config: DataLoaderConfig) -> ProcessingResult:
    """I/O-bound function with type hints."""
    pass
```

### 4. RORO Pattern Integration

```python
# In interfaces modules
def roro_module_function(request: RORORequest) -> ROROResponse:
    """RORO pattern function with type hints."""
    pass
```

## Examples from Current Codebase

### Files Following Good Patterns

1. **`evaluation_metrics.py`** - Core evaluation functionality
2. **`data_splitting_validation.py`** - Core data processing
3. **`training_evaluation.py`** - Core training functionality
4. **`efficient_data_loading.py`** - Core data loading

### Recommended Refactoring

**Current Pattern (Good):**
```python
# evaluation_metrics.py - Core functionality
def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return (predictions.argmax(dim=1) == targets).float().mean()

# data_splitting_validation.py - Core functionality
def split_data(data: List[Any], train_ratio: float = 0.7) -> Tuple[List[Any], List[Any]]:
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]
```

**Enhanced Pattern (Better):**
```python
# core/evaluation/metrics.py
def calculate_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> ProcessingResult:
    """Calculate accuracy with comprehensive type hints and validation."""
    try:
        # Validate inputs
        if not isinstance(predictions, torch.Tensor):
            raise ValueError("predictions must be a torch.Tensor")
        if not isinstance(targets, torch.Tensor):
            raise ValueError("targets must be a torch.Tensor")
        
        accuracy = (predictions.argmax(dim=1) == targets).float().mean()
        
        return ProcessingResult(
            is_successful=True,
            result=accuracy.item(),
            error=None
        )
    except Exception as exc:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=str(exc)
        )

# core/data/preprocessing.py
def split_data(
    data: List[Any],
    config: DataConfiguration
) -> ProcessingResult:
    """Split data with comprehensive type hints and validation."""
    try:
        # Validate inputs
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        if len(data) == 0:
            raise ValueError("data cannot be empty")
        
        split_index = int(len(data) * config.validation_split)
        train_data = data[:split_index]
        val_data = data[split_index:]
        
        return ProcessingResult(
            is_successful=True,
            result={"train": train_data, "validation": val_data},
            error=None
        )
    except Exception as exc:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=str(exc)
        )
```

## Summary

### Key Principles

1. **Organize by functionality:**
   - Core modules for main functionality
   - Utils modules for shared utilities
   - Configs modules for configurations
   - Interfaces modules for patterns

2. **Use proper imports/exports:**
   - Named exports for clear APIs
   - Organized import statements
   - Clear module dependencies

3. **Integrate all patterns:**
   - Type hints throughout all modules
   - Pydantic validation for configurations
   - Async/sync patterns for operations
   - RORO pattern for interfaces

4. **Follow best practices:**
   - Clear module structure
   - Consistent naming conventions
   - Proper error handling
   - Comprehensive documentation

### Benefits

- **Organization**: Clear separation of concerns
- **Maintainability**: Isolated changes and easy testing
- **Reusability**: Shared utilities and consistent interfaces
- **Type Safety**: Type hints throughout all modules
- **Validation**: Pydantic models for configurations
- **Scalability**: Modular architecture for growth

The existing codebase already demonstrates good patterns, and these guidelines help maintain consistency and improve organization across the entire system. 