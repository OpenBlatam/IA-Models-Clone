# Type Hints and Pydantic v2 Validation - Complete Guide
=======================================================

## Overview

This document outlines comprehensive type hints for all function signatures and Pydantic v2 validation for structured configurations, integrating with the RORO pattern and async/sync patterns.

## Table of Contents

1. [Type Hints Fundamentals](#type-hints-fundamentals)
2. [Pydantic v2 Models](#pydantic-v2-models)
3. [Function Signatures with Type Hints](#function-signatures-with-type-hints)
4. [Validation Patterns](#validation-patterns)
5. [Integration with Existing Patterns](#integration-with-existing-patterns)
6. [Best Practices](#best-practices)
7. [Performance Considerations](#performance-considerations)
8. [Error Handling](#error-handling)
9. [Examples from Current Codebase](#examples-from-current-codebase)

## Type Hints Fundamentals

### Basic Type Hints

**Simple Types:**
```python
def calculate_loss(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return nn.MSELoss()(predictions, targets)

def normalize_data(data: List[float], method: str = "standard") -> List_float:
    # Implementation
    pass

def validate_config(config: Dict[str, Any]) -> bool:
    # Implementation
    pass
```

**Complex Types:**
```python
from typing import Dict, List, Optional, Union, Tuple, Callable, TypeVar, Generic

def process_batch(
    data: List[torch.Tensor Burgess],
    processor: Callable[[torch.Tensor], torch.Tensor],
    config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, float]:
    # Implementation
    pass
```

### Generic Types

```python
T = TypeVar('T')
U = TypeVar('U')

class Result(Generic[T]):
    def __init__(self, value: T, is_success: bool, error: Optional[str] = None):
        self.value = value
        self.is_success = is_success
        self.error = error

def process_data_with_generics(
    data: List[T],
    processor: Callable[[T], U],
    validator: Optional[Callable[[T], bool]] = None
) -> Result[List[U]]:
    # Implementation
    pass
```

## Pydantic v2 Models

### Model Configuration

```python
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import conint, confloat, constr

class ModelConfiguration(BaseModel):
    """Pydantic model for model configuration validation."""
    
    model_config = ConfigDict(
        extra="forbid",  # Reject extra fields
        validate_assignment=True,  # Validate on assignment
        str_strip_whitespace=True,  # Strip whitespace from strings
    )
    
    # Required fields with validation
    input_dimension: conint(gt=0) = Field(
        description="Input dimension must be positive",
        gt=0
    )
    output_dimension: conint(gt=0) = Field(
        description="Output dimension must be positive", 
        gt=0
    )
    hidden_layer_sizes: List[conint(gt=0)] = Field(
        description="Hidden layer sizes must all be positive",
        min_items=1
    )
    
    # Optional fields with defaults and validation
    activation_function: constr(strip_whitespace=True) = Field(
        default="relu",
        description="Activation function name",
        pattern=r"^(relu|sigmoid|tanh|softmax)$"
    )
    dropout_rate: confloat(ge=0.0, le=1.0) = Field(
        default=0.2,
        description="Dropout rate between 0 and 1"
    )
    learning_rate: confloat(gt=0.0, lt=1.0) = Field(
        default=0.001,
        description="Learning rate between 0 and 1"
    )
    
    # Custom validators
    @validator('hidden_layer_sizes')
    def validate_hidden_layers(cls, v: List[int]) -> List[int]:
        """Validate hidden layer sizes are reasonable."""
        if any(size > 10000 for size in v):
            raise ValueError("Hidden layer sizes too large (>10000)")
        return v
    
    @validator('activation_function')
    def validate_activation_function(cls, v: str) -> str:
        """Validate activation function is supported."""
        supported_activations = {"relu", "sigmoid", "tanh", "softmax"}
        if v not in supported_activations:
            raise ValueError(f"Unsupported activation function: {v}")
        return v
```

### Training Configuration

```python
class TrainingConfiguration(BaseModel):
    """Pydantic model for training configuration validation."""
    
    model_config = ConfigDict(extra notice)
    
    # Required fields
    max_epochs: conint(gt=0, le=10000) = Field(
        description="Maximum training epochs",
        gt=0,
        le=10000
    )
    batch_size: conint(gt=0, le=10000) = Field(
        description="Training batch size",
        gt=0,
        le=10000
    )
    
    # Optional fields with validation
    learning_rate: confloat(gt=0.0, lt=1.0) = Field(
        default=0.001,
        description="Learning rate"
    )
    early_stopping_patience: conint(ge=0, le=1000) = Field(
        default=10,
        description="Early stopping patience"
    )
    checkpoint_path: Optional[constr(strip_whitespace=True)] = Field(
        default=None,
        description="Model checkpoint path"
    )
    validation_split: confloat(ge=0.0, lt=1.0) = Field(
        default=0.2,
        description="Validation data split ratio"
    )
    
    # Custom validators
    @validator('checkpoint_path')
    def validate_checkpoint_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate checkpoint path if provided."""
        if v is not None:
            path = Path(v)
            if not path.parent.exists():
                raise ValueError(f"Checkpoint directory does not exist: {path.parent}")
        return v
```

### Data Configuration

```python
class DataConfiguration(BaseModel):
    """Pydantic model for data configuration validation."""
    
    model_config = ConfigDict(extra="forbid")
    
    # Required fields
    data_path: constr(strip_whitespace=True) = Field(
        description="Path to training data"
    )
    data_format: constr(strip_whitespace=True) = Field(
        default="json",
        description="Data format",
        pattern=r"^(json|csv|parquet|hdf5)$"
    )
    
    # Optional fields
    normalization_type: constr(strip_whitespace=True) = Field(
        default="standard",
一秒 description="Data normalization type",
        pattern=r"^(standard|minmax|robust)$"
    )
    shuffle_data: bool = Field(
        default=True,
        description="Whether to shuffle data"
    )
    random_seed: Optional[conint(ge=0)] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    # Custom validators
    @validator('data_path')
    def validate_data_path(cls, v: str) -> str:
        """Validate data path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Data path does not exist: {v}")
        return str(path)
```

## Function Signatures with Type Hints

### CPU-Bound Functions

```python
def calculate_loss_function(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "mse"
) -> ProcessingResult:
    """Calculate loss function with comprehensive type hints."""
    try:
        # Validate inputs
        if not isinstance(predictions, torch.Tensor):
            raise ValueError("predictions must be a torch.Tensor")
        if not isinstance(targets, torch.Tensor):
            raise ValueError("targets must be a torch.Tensor")
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError("predictions and targets must have same batch size")
        
        # Calculate loss based on type
        if loss_type == "mse":
            loss = nn.MSELoss()(predictions, targets)
        elif loss_type == "cross_entropy":
            loss = nn.CrossEntropyLoss()(predictions, targets)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return ProcessingResult(
            is_successful=True,
            result=loss.item(),
            error=None
        )
    except Exception as exc:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=str(exc)
        )
```

### I/O-Bound Functions

```python
async def load_training_data_async(
    data_path: str,
    data_format: str = "json"
) -> ProcessingResult:
    """Load training data asynchronously with type hints."""
    try:
        # Validate inputs
        if not isinstance(data_path, str):
            raise ValueError("data_path must be a string")
        if not Path(data_path).exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Load data based on format
        async with aiofiles.open(data_path, 'r') as file:
            content = await file.read()
        
        if data_format == "json":
            import json
            data = json.loads(content)
        elif data_format == "csv":
            import csv
            from io import StringIO
            data = list(csv.DictReader(StringIO(content)))
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        return ProcessingResult(
            is_successful=True,
            result=data,
            error=None
        )
    except Exception as exc:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=str(exc)
        )
```

### Mixed Operations

```python
async def train_model_with_validation(
    model_config: ModelConfiguration,
    training_config: TrainingConfiguration,
    data_config: DataConfiguration
) -> ProcessingResult:
    """Train model with comprehensive validation and type hints."""
    try:
        # Validate all configurations using Pydantic
        model_config_validated = ModelConfiguration(**model_config.dict())
        training_config_validated = TrainingConfiguration(**training_config.dict())
        data_config_validated = DataConfiguration(**data_config.dict())
        
        # Load training data
        data_result = await load_training_data_async(
            data_path=data_config_validated.data_path,
            data_format=data_config_validated.data_format
        )
        
        if not data_result.is_successful:
            return ProcessingResult(
                is_successful=False,
                result=None,
                error=f"Failed to load data: {data_result.error}"
            )
        
        training_data = data_result.result
        
        # Normalize data if required
        if data_config_validated.normalization_type != "none":
            normalize_result = normalize_tensor_data(
                tensor_data=torch.tensor(training_data["features"]),
                normalization_type=data_config_validated.normalization_type
            )
            
            if not normalize_result.is_successful:
                return ProcessingResult(
                    is_successful=False,
                    result=None,
                    error=f"Failed to normalize data: {normalize_result.error}"
                )
            
            processed_features = normalize_result.result
        else:
            processed_features = torch.tensor(training_data["features"])
        
        # Training loop
        model = create_model_from_config(model_config_validated)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=training_config_validated.learning_rate
        )
        
        for epoch in range(training_config_validated.max_epochs):
            # Forward pass
            predictions = model(processed_features)
            targets = torch.tensor(training_data["targets"])
            
            # Calculate loss
            loss_result = calculate_loss_function(
                predictions=predictions,
                targets=targets,
                loss_type="mse"
            )
            
            if not loss_result.is_successful:
                return ProcessingResult(
                    is_successful=False,
                    result=None,
                    error=f"Failed to calculate loss: {loss_result.error}"
                )
            
            loss = loss_result.result
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save model checkpoint
        if training_config_validated.checkpoint_path:
            save_result = await save_model_checkpoint_async(
                model_state=model.state_dict(),
                checkpoint_path=training_config_validated.checkpoint_path
            )
            
            if not save_result.is_successful:
                return ProcessingResult(
                    is_successful=False,
                    result=None,
                    error=f"Failed to save checkpoint: {save_result.error}"
                )
        
        return ProcessingResult(
            is_successful=True,
            result={
                "model": model,
                "final_loss": loss.item(),
                "epochs_trained": training_config_validated.max_epochs
            },
            error=None
        )
        
    except Exception as exc:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=str(exc)
        )
```

## Validation Patterns

### Input Validation

```python
def validate_inputs(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str
) -> ValidationResult:
    """Validate function inputs."""
    errors = []
    warnings = []
    
    # Type validation
    if not isinstance(predictions, torch.Tensor):
        errors.append("predictions must be a torch.Tensor")
    
    if not isinstance(targets, torch.Tensor):
        errors.append("targets must be a torch.Tensor")
    
    # Shape validation
    if predictions.shape[0] != targets.shape[0]:
        errors.append("predictions and targets must have same batch size")
    
    # Value validation
    if torch.isnan(predictions).any():
        errors.append("predictions contain NaN values")
    
    if torch.isnan(targets).any():
        errors.append("targets contain NaN values")
    
    # Loss type validation
    supported_loss_types = {"mse", "cross_entropy", "binary_cross_entropy"}
    if loss_type not in supported_loss_types:
        errors.append(f"Unsupported loss type: {loss_type}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
```

### Configuration Validation

```python
def validate_model_configuration(config: Dict[str, Any]) -> ValidationResult:
    """Validate model configuration using Pydantic."""
    try:
        # Use Pydantic model for validation
        validated_config = ModelConfiguration(**config)
        
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[]
        )
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            errors=[str(exc)],
            warnings=[]
        )
```

## Integration with Existing Patterns

### RORO Pattern with Type Hints

```python
def process_data_roro(params: Dict[str, Any]) -> Dict[str, Any]:
    """Process data using RORO pattern with type hints."""
    try:
        # Extract and validate parameters
        data: List[torch.Tensor] = params["data"]
        processor: Callable[[torch.Tensor], torch.Tensor] = params["processor"]
        config: Optional[Dict[str, Any]] = params.get("config")
        
        # Validate inputs
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        
        if not callable(processor):
            raise ValueError("processor must be callable")
        
        # Process data
        processed_data = [processor(item) for item in data]
        
        return {
            "is_successful": True,
            "result": processed_data,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }
```

### Async/Sync Pattern with Type Hints

```python
# CPU-bound function
def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metrics: List[str]
) -> ProcessingResult:
    """Calculate metrics - CPU-bound with type hints."""
    try:
        results = {}
        
        if "accuracy" in metrics:
            correct = (predictions.argmax(dim=1) == targets).sum()
            accuracy = correct / len(targets)
            results["accuracy"] = accuracy.item()
        
        return ProcessingResult(
            is_successful=True,
            result=results,
            error=None
        )
    except Exception as exc:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=str(exc)
        )

# I/O-bound function
async def save_metrics_async(
    metrics: Dict[str, float],
    file_path: str
) -> ProcessingResult:
    """Save metrics - I/O-bound with type hints."""
    try:
        async with aiofiles.open(file_path, 'w') as file:
            import json
            await file.write(json.dumps(metrics))
        
        return ProcessingResult(
            is_successful=True,
            result=file_path,
            error=None
        )
    except Exception as exc:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=str(exc)
        )
```

## Best Practices

### 1. Comprehensive Type Hints

**✅ Good:**
```python
def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    callbacks: Optional[List[Callable]] = None
) -> Dict[str, List[float]]:
    """Train model with comprehensive type hints."""
    pass
```

**❌ Avoid:**
```python
def train_model(model, train_loader, optimizer, num_epochs, device, callbacks=None):
    """Train model without type hints."""
    pass
```

### 2. Pydantic Validation

**✅ Good:**
```python
class ModelConfig(BaseModel):
    input_dim: conint(gt=0) = Field(description="Input dimension")
    output_dim: conint(gt=0) = Field(description="Output dimension")
    hidden_dims: List[conint(gt=0)] = Field(description="Hidden dimensions")
    
    @validator('hidden_dims')
    def validate_hidden_dims(cls, v):
        if len(v) == 0:
            raise ValueError("Must have at least one hidden layer")
        return v
```

**❌ Avoid:**
```python
def validate_config(config):
    """Manual validation without Pydantic."""
    if config['input_dim'] <= 0:
        raise ValueError("Invalid input dimension")
    # ... more manual validation
```

### 3. Generic Types

**✅ Good:**
```python
T = TypeVar('T')
U = TypeVar('U')

def process_batch(
    batch: List[T],
    processor: Callable[[T], U]
) -> List[U]:
    """Process batch with generic types."""
    return [processor(item) for item in batch]
```

### 4. Union Types

**✅ Good:**
```python
def save_model(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
    path: str,
    format: Literal["pth", "onnx", "torchscript"] = "pth"
) -> bool:
    """Save model with union types."""
    pass
```

## Performance Considerations

### 1. Type Checking Overhead

- Type hints have minimal runtime overhead
- Use `from __future__ import annotations` for Python < 3.9
- Consider using `@typing.no_type_check` for performance-critical code

### 2. Pydantic Validation

- Pydantic v2 is significantly faster than v1
- Use `model_config = ConfigDict(validate_assignment=False)` for performance
- Consider lazy validation for large datasets

### 3. Memory Usage

- Type hints don't affect memory usage
- Pydantic models have minimal memory overhead
- Use `slots=True` for high-performance models

## Error Handling

### Consistent Error Patterns

```python
def function_with_validation(
    data: torch.Tensor,
    config: ModelConfiguration
) -> ProcessingResult:
    """Function with comprehensive error handling."""
    try:
        # Input validation
        if not isinstance(data, torch.Tensor):
            raise ValueError("data must be a torch.Tensor")
        
        # Configuration validation
        if not isinstance(config, ModelConfiguration):
            raise ValueError("config must be a ModelConfiguration")
        
        # Processing
        result = process_data(data, config)
        
        return ProcessingResult(
            is_successful=True,
            result=result,
            error=None
        )
    except ValueError as ve:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=f"Validation error: {ve}"
        )
    except Exception as exc:
        return ProcessingResult(
            is_successful=False,
            result=None,
            error=f"Processing error: {exc}"
        )
```

## Examples from Current Codebase

### Files Following Good Patterns

1. **`evaluation_metrics.py`** - CPU-bound calculations with type hints
2. **`data_splitting_validation.py`** - Validation logic with Pydantic
3. **`training_evaluation.py`** - Mixed operations with type hints
4. **`efficient_data_loading.py`** - I/O operations with type hints

### Recommended Refactoring

**Current Pattern (Good):**
```python
# evaluation_metrics.py - With type hints
def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return (predictions.argmax(dim=1) == targets).float().mean()

# data_splitting_validation.py - With Pydantic
class DataSplitConfig(BaseModel):
    train_ratio: confloat(gt=0.0, lt=1.0) = Field(default=0.7)
    val_ratio: confloat(gt=0.0, lt=1.0) = Field(default=0.15)
    test_ratio: confloat(gt=0.0, lt=1.0) = Field(default=0.15)
```

**Enhanced Pattern (Better):**
```python
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
```

## Summary

### Key Principles

1. **Use comprehensive type hints:**
   - All function parameters and return values
   - Generic types for reusable functions
   - Union types for flexible inputs

2. **Use Pydantic v2 for validation:**
   - Structured configuration models
   - Input validation with custom validators
   - Consistent error handling

3. **Integrate with existing patterns:**
   - RORO pattern with type hints
   - Async/sync patterns with validation
   - Named exports for type-safe modules

4. **Follow best practices:**
   - Comprehensive error handling
   - Performance considerations
   - Clear documentation

### Benefits

- **Type Safety**: Catch errors at development time
- **Documentation**: Self-documenting code
- **IDE Support**: Better autocomplete and refactoring
- **Validation**: Runtime input validation
- **Maintainability**: Clear interfaces and contracts
- **Reliability**: Consistent error handling

The existing codebase already demonstrates good patterns, and these guidelines help maintain consistency and improve code quality across the entire system. 