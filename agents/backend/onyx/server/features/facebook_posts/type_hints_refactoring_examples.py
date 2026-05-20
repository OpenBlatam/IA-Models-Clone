from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import aiohttp
import aiofiles
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import conint, confloat, constr
import numpy as np
from pathlib import Path
            import json
            import csv
            from io import StringIO
            import pandas as pd
            import h5py
            import json
from typing import Any, List, Dict, Optional
import logging
"""
Type Hints and Pydantic Refactoring Examples
===========================================

This file demonstrates how to refactor existing code to include:
- Comprehensive type hints for all function signatures
- Pydantic v2 validation for structured configurations
- Integration with RORO pattern and async/sync patterns
- Named exports for utilities
"""


# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # Refactored functions with type hints
    "calculate_loss_function_typed",
    "normalize_tensor_data_typed",
    "validate_model_configuration_typed",
    "load_training_data_async_typed",
    "save_model_checkpoint_async_typed",
    "train_model_with_validation_typed",
    
    # Pydantic models
    "ModelConfigurationTyped",
    "TrainingConfigurationTyped", 
    "DataConfigurationTyped",
    "ValidationResultTyped",
    "ProcessingResultTyped",
]

# ============================================================================
# BEFORE AND AFTER REFACTORING EXAMPLES
# ============================================================================

# ❌ BEFORE: No type hints, no validation
def calculate_loss_old(predictions, targets, loss_type="mse") -> Any:
    """Old implementation without type hints or validation."""
    if loss_type == "mse":
        loss = nn.MSELoss()(predictions, targets)
    elif loss_type == "cross_entropy":
        loss = nn.CrossEntropyLoss()(predictions, targets)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss.item()

def normalize_data_old(data, method="standard") -> Any:
    """Old implementation without type hints or validation."""
    if method == "standard":
        mean = torch.mean(data)
        std = torch.std(data)
        normalized = (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = torch.min(data)
        max_val = torch.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized

def validate_config_old(config) -> bool:
    """Old implementation without type hints or validation."""
    errors = []
    
    if config.get("input_dim", 0) <= 0:
        errors.append("Invalid input dimension")
    if config.get("output_dim", 0) <= 0:
        errors.append("Invalid output dimension")
    if not config.get("hidden_layers"):
        errors.append("Missing hidden layers")
    
    return len(errors) == 0, errors

# ✅ AFTER: Comprehensive type hints and Pydantic validation
class ProcessingResultTyped(BaseModel):
    """Pydantic model for processing results with type hints."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_successful: bool = Field(description="Whether processing was successful")
    result: Optional[Any] = Field(default=None, description="Processing result")
    error: Optional[str] = Field(default=None, description="Error message")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")

class ValidationResultTyped(BaseModel):
    """Pydantic model for validation results with type hints."""
    
    model_config = ConfigDict(extra="forbid")
    
    is_valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

class ModelConfigurationTyped(BaseModel):
    """Pydantic model for model configuration with type hints."""
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
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

class TrainingConfigurationTyped(BaseModel):
    """Pydantic model for training configuration with type hints."""
    
    model_config = ConfigDict(extra="forbid")
    
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

class DataConfigurationTyped(BaseModel):
    """Pydantic model for data configuration with type hints."""
    
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
        description="Data normalization type",
        pattern=r"^(standard|minmax|robust|none)$"
    )
    shuffle_data: bool = Field(
        default=True,
        description="Whether to shuffle data"
    )
    random_seed: Optional[conint(ge=0)] = Field(
        default=None,
        description="Random seed for reproducibility"
    )

def calculate_loss_function_typed(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: Literal["mse", "cross_entropy", "binary_cross_entropy"] = "mse"
) -> ProcessingResultTyped:
    """Calculate loss function with comprehensive type hints and validation."""
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
        elif loss_type == "binary_cross_entropy":
            loss = nn.BCELoss()(predictions, targets)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return ProcessingResultTyped(
            is_successful=True,
            result=loss.item(),
            error=None
        )
    except Exception as exc:
        return ProcessingResultTyped(
            is_successful=False,
            result=None,
            error=str(exc)
        )

def normalize_tensor_data_typed(
    tensor_data: torch.Tensor,
    normalization_type: Literal["standard", "minmax", "robust"] = "standard"
) -> ProcessingResultTyped:
    """Normalize tensor data with comprehensive type hints and validation."""
    try:
        # Validate inputs
        if not isinstance(tensor_data, torch.Tensor):
            raise ValueError("tensor_data must be a torch.Tensor")
        if tensor_data.numel() == 0:
            raise ValueError("tensor_data cannot be empty")
        
        # Perform normalization
        if normalization_type == "standard":
            mean = torch.mean(tensor_data)
            std = torch.std(tensor_data)
            normalized = (tensor_data - mean) / (std + 1e-8)
        elif normalization_type == "minmax":
            min_val = torch.min(tensor_data)
            max_val = torch.max(tensor_data)
            normalized = (tensor_data - min_val) / (max_val - min_val + 1e-8)
        elif normalization_type == "robust":
            median = torch.median(tensor_data)
            mad = torch.median(torch.abs(tensor_data - median))
            normalized = (tensor_data - median) / (mad + 1e-8)
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")
        
        return ProcessingResultTyped(
            is_successful=True,
            result=normalized,
            error=None
        )
    except Exception as exc:
        return ProcessingResultTyped(
            is_successful=False,
            result=None,
            error=str(exc)
        )

def validate_model_configuration_typed(
    config: Dict[str, Any]
) -> ValidationResultTyped:
    """Validate model configuration using Pydantic with type hints."""
    try:
        # Use Pydantic model for validation
        validated_config = ModelConfigurationTyped(**config)
        
        return ValidationResultTyped(
            is_valid=True,
            errors=[],
            warnings=[]
        )
    except Exception as exc:
        return ValidationResultTyped(
            is_valid=False,
            errors=[str(exc)],
            warnings=[]
        )

async def load_training_data_async_typed(
    data_path: str,
    data_format: Literal["json", "csv", "parquet", "hdf5"] = "json"
) -> ProcessingResultTyped:
    """Load training data asynchronously with comprehensive type hints."""
    try:
        # Validate inputs
        if not isinstance(data_path, str):
            raise ValueError("data_path must be a string")
        if not Path(data_path).exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Load data based on format
        async with aiofiles.open(data_path, 'r') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        if data_format == "json":
            data = json.loads(content)
        elif data_format == "csv":
            data = list(csv.DictReader(StringIO(content)))
        elif data_format == "parquet":
            data = pd.read_parquet(data_path).to_dict('records')
        elif data_format == "hdf5":
            with h5py.File(data_path, 'r') as f:
                data = {key: f[key][:] for key in f.keys()}
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        return ProcessingResultTyped(
            is_successful=True,
            result=data,
            error=None
        )
    except Exception as exc:
        return ProcessingResultTyped(
            is_successful=False,
            result=None,
            error=str(exc)
        )

async def save_model_checkpoint_async_typed(
    model_state: Dict[str, torch.Tensor],
    checkpoint_path: str
) -> ProcessingResultTyped:
    """Save model checkpoint asynchronously with comprehensive type hints."""
    try:
        # Validate inputs
        if not isinstance(model_state, dict):
            raise ValueError("model_state must be a dictionary")
        if not isinstance(checkpoint_path, str):
            raise ValueError("checkpoint_path must be a string")
        
        # Ensure checkpoint directory exists
        checkpoint_dir = Path(checkpoint_path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        async with aiofiles.open(checkpoint_path, 'w') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            # Convert tensors to lists for JSON serialization
            serializable_state = {
                key: value.cpu().numpy().tolist() 
                for key, value in model_state.items()
            }
            await file.write(json.dumps(serializable_state))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return ProcessingResultTyped(
            is_successful=True,
            result=checkpoint_path,
            error=None
        )
    except Exception as exc:
        return ProcessingResultTyped(
            is_successful=False,
            result=None,
            error=str(exc)
        )

# ============================================================================
# COMPLEX TYPE HINTS AND GENERICS
# ============================================================================

T = TypeVar('T')
U = TypeVar('U')

class ResultTyped(Generic[T]):
    """Generic result type for type-safe operations."""
    
    def __init__(self, value: T, is_success: bool, error: Optional[str] = None):
        
    """__init__ function."""
self.value = value
        self.is_success = is_success
        self.error = error

def process_data_with_generics_typed(
    data: List[T],
    processor: Callable[[T], U],
    validator: Optional[Callable[[T], bool]] = None
) -> ResultTyped[List[U]]:
    """Process data with generic type hints."""
    try:
        # Validate input data
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        
        # Apply validator if provided
        if validator is not None:
            filtered_data = [item for item in data if validator(item)]
        else:
            filtered_data = data
        
        # Process data
        processed_data = [processor(item) for item in filtered_data]
        
        return ResultTyped(
            value=processed_data,
            is_success=True
        )
    except Exception as exc:
        return ResultTyped(
            value=[],
            is_success=False,
            error=str(exc)
        )

# ============================================================================
# INTEGRATED TRAINING FUNCTION WITH VALIDATION
# ============================================================================

async def train_model_with_validation_typed(
    model_config: ModelConfigurationTyped,
    training_config: TrainingConfigurationTyped,
    data_config: DataConfigurationTyped
) -> ProcessingResultTyped:
    """Train model with comprehensive validation and type hints."""
    try:
        # Validate all configurations using Pydantic
        model_config_validated = ModelConfigurationTyped(**model_config.dict())
        training_config_validated = TrainingConfigurationTyped(**training_config.dict())
        data_config_validated = DataConfigurationTyped(**data_config.dict())
        
        # Load training data
        data_result = await load_training_data_async_typed(
            data_path=data_config_validated.data_path,
            data_format=data_config_validated.data_format
        )
        
        if not data_result.is_successful:
            return ProcessingResultTyped(
                is_successful=False,
                result=None,
                error=f"Failed to load data: {data_result.error}"
            )
        
        training_data = data_result.result
        
        # Normalize data if required
        if data_config_validated.normalization_type != "none":
            normalize_result = normalize_tensor_data_typed(
                tensor_data=torch.tensor(training_data["features"]),
                normalization_type=data_config_validated.normalization_type
            )
            
            if not normalize_result.is_successful:
                return ProcessingResultTyped(
                    is_successful=False,
                    result=None,
                    error=f"Failed to normalize data: {normalize_result.error}"
                )
            
            processed_features = normalize_result.result
        else:
            processed_features = torch.tensor(training_data["features"])
        
        # Training loop
        model = create_model_from_config_typed(model_config_validated)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=training_config_validated.learning_rate
        )
        
        for epoch in range(training_config_validated.max_epochs):
            # Forward pass
            predictions = model(processed_features)
            targets = torch.tensor(training_data["targets"])
            
            # Calculate loss
            loss_result = calculate_loss_function_typed(
                predictions=predictions,
                targets=targets,
                loss_type="mse"
            )
            
            if not loss_result.is_successful:
                return ProcessingResultTyped(
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
            save_result = await save_model_checkpoint_async_typed(
                model_state=model.state_dict(),
                checkpoint_path=training_config_validated.checkpoint_path
            )
            
            if not save_result.is_successful:
                return ProcessingResultTyped(
                    is_successful=False,
                    result=None,
                    error=f"Failed to save checkpoint: {save_result.error}"
                )
        
        return ProcessingResultTyped(
            is_successful=True,
            result={
                "model": model,
                "final_loss": loss,
                "epochs_trained": training_config_validated.max_epochs
            },
            error=None
        )
        
    except Exception as exc:
        return ProcessingResultTyped(
            is_successful=False,
            result=None,
            error=str(exc)
        )

def create_model_from_config_typed(config: ModelConfigurationTyped) -> nn.Module:
    """Create neural network model from validated configuration with type hints."""
    layers = []
    
    # Input layer
    layers.append(nn.Linear(config.input_dimension, config.hidden_layer_sizes[0]))
    layers.append(get_activation_function_typed(config.activation_function))
    layers.append(nn.Dropout(config.dropout_rate))
    
    # Hidden layers
    for i in range(len(config.hidden_layer_sizes) - 1):
        layers.append(nn.Linear(config.hidden_layer_sizes[i], config.hidden_layer_sizes[i + 1]))
        layers.append(get_activation_function_typed(config.activation_function))
        layers.append(nn.Dropout(config.dropout_rate))
    
    # Output layer
    layers.append(nn.Linear(config.hidden_layer_sizes[-1], config.output_dimension))
    
    return nn.Sequential(*layers)

def get_activation_function_typed(activation_name: str) -> nn.Module:
    """Get activation function by name with type hints."""
    activation_map: Dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1)
    }
    return activation_map.get(activation_name, nn.ReLU())

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_type_hints_refactoring():
    """Demonstrate type hints and Pydantic validation refactoring."""
    
    print("🔍 Demonstrating Type Hints and Pydantic Refactoring")
    print("=" * 60)
    
    # Example 1: Model configuration validation
    print("\n📋 Model Configuration Validation:")
    try:
        model_config = ModelConfigurationTyped(
            input_dimension=784,
            output_dimension=10,
            hidden_layer_sizes=[512, 256, 128],
            activation_function="relu",
            dropout_rate=0.2,
            learning_rate=0.001
        )
        print(f"✅ Valid model config: {model_config.input_dimension} → {model_config.output_dimension}")
    except Exception as e:
        print(f"❌ Invalid model config: {e}")
    
    # Example 2: Training configuration validation
    print("\n🏋️ Training Configuration Validation:")
    try:
        training_config = TrainingConfigurationTyped(
            max_epochs=100,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=10,
            checkpoint_path="models/checkpoint.pth"
        )
        print(f"✅ Valid training config: {training_config.max_epochs} epochs")
    except Exception as e:
        print(f"❌ Invalid training config: {e}")
    
    # Example 3: Function with type hints
    print("\n🔧 Function with Type Hints:")
    loss_result = calculate_loss_function_typed(
        predictions=torch.randn(10, 3),
        targets=torch.randint(0, 3, (10,)),
        loss_type="cross_entropy"
    )
    print(f"Loss calculation: {loss_result.is_successful}")
    if loss_result.is_successful:
        print(f"Loss value: {loss_result.result}")
    
    # Example 4: Generic function
    print("\n🔄 Generic Function with Type Hints:")
    data = [1, 2, 3, 4, 5]
    processor = lambda x: x * 2
    validator = lambda x: x > 0
    
    result = process_data_with_generics_typed(
        data=data,
        processor=processor,
        validator=validator
    )
    print(f"Generic processing: {result.is_success}")
    if result.is_success:
        print(f"Processed data: {result.value}")

def show_refactoring_benefits():
    """Show the benefits of adding type hints and Pydantic validation."""
    
    benefits = {
        "type_safety": [
            "Catch errors at development time",
            "Better IDE support with autocomplete",
            "Clearer function interfaces",
            "Easier refactoring"
        ],
        "validation": [
            "Runtime input validation",
            "Structured configuration management",
            "Consistent error handling",
            "Self-documenting code"
        ],
        "maintainability": [
            "Clear function signatures",
            "Explicit data contracts",
            "Better code documentation",
            "Easier testing"
        ],
        "performance": [
            "Minimal runtime overhead",
            "Pydantic v2 is significantly faster",
            "Type hints don't affect execution",
            "Better memory management"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate refactoring patterns
    benefits = show_refactoring_benefits()
    
    print("✅ Type hints and Pydantic refactoring examples created successfully!")
    
    print("\n🎯 Key Refactoring Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    # Run async demonstration
    asyncio.run(demonstrate_type_hints_refactoring()) 