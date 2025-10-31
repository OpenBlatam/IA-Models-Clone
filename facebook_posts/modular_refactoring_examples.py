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
                import json
                import csv
                from io import StringIO
from typing import Any, List, Dict, Optional
import logging
"""
Modular Refactoring Examples - Organized File Structure
=====================================================

This file demonstrates how to refactor existing code into proper modules:
- Converting flat structure to modular organization
- Integrating all patterns (type hints, Pydantic, async/sync, RORO)
- Named exports for utilities
- Proper import/export patterns
"""


# ============================================================================
# BEFORE: FLAT STRUCTURE (EXISTING CODE)
# ============================================================================

# ❌ BEFORE: All functions in one file
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

def load_data_old(data_path, data_format="json") -> Any:
    """Old implementation without type hints or validation."""
    with open(data_path, 'r') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        if data_format == "json":
            return json.load(file)
        elif data_format == "csv":
            return list(csv.DictReader(file))
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

def train_model_old(model, data, config) -> Any:
    """Old implementation without type hints or validation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
    
    for epoch in range(config.get("max_epochs", 100)):
        # Training loop
        predictions = model(data["features"])
        loss = nn.MSELoss()(predictions, data["targets"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

# ============================================================================
# AFTER: MODULAR STRUCTURE (REFACTORED CODE)
# ============================================================================

# ✅ AFTER: Organized into proper modules

# ============================================================================
# CORE MODULES - MODELS
# ============================================================================

# core/models/neural_networks.py
class NeuralNetworksModule:
    """Core neural networks module with proper exports."""
    
    __all__ = [
        "create_feedforward_network",
        "create_convolutional_network",
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
    
    def create_convolutional_network(
        config: NeuralNetworkConfig
    ) -> nn.Module:
        """Create convolutional neural network with type hints."""
        # Implementation for CNN
        pass

# ============================================================================
# CORE MODULES - TRAINING
# ============================================================================

# core/training/loss_functions.py
class LossFunctionsModule:
    """Core loss functions module with proper exports."""
    
    __all__ = [
        "calculate_loss_function",
        "LossConfig",
        "LossType"
    ]
    
    class LossConfig(BaseModel):
        """Pydantic model for loss configuration."""
        
        model_config = ConfigDict(extra="forbid")
        
        loss_type: constr(strip_whitespace=True) = Field(
            pattern=r"^(mse|cross_entropy|binary_cross_entropy|huber)$"
        )
        reduction: constr(strip_whitespace=True) = Field(
            default="mean",
            pattern=r"^(mean|sum|none)$"
        )
    
    class LossType(BaseModel):
        """Pydantic model for loss type validation."""
        
        model_config = ConfigDict(extra="forbid")
        
        type_name: constr(strip_whitespace=True) = Field(
            pattern=r"^(mse|cross_entropy|binary_cross_entropy|huber)$"
        )
        description: Optional[str] = Field(default=None)
    
    def calculate_loss_function(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        config: LossConfig
    ) -> Dict[str, Any]:
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
            if config.loss_type == "mse":
                loss = nn.MSELoss(reduction=config.reduction)(predictions, targets)
            elif config.loss_type == "cross_entropy":
                loss = nn.CrossEntropyLoss(reduction=config.reduction)(predictions, targets)
            elif config.loss_type == "binary_cross_entropy":
                loss = nn.BCELoss(reduction=config.reduction)(predictions, targets)
            elif config.loss_type == "huber":
                loss = nn.HuberLoss(reduction=config.reduction)(predictions, targets)
            else:
                raise ValueError(f"Unsupported loss type: {config.loss_type}")
            
            return {
                "is_successful": True,
                "result": loss.item(),
                "error": None
            }
        except Exception as exc:
            return {
                "is_successful": False,
                "result": None,
                "error": str(exc)
            }

# core/training/optimizers.py
class OptimizersModule:
    """Core optimizers module with proper exports."""
    
    __all__ = [
        "create_optimizer",
        "OptimizerConfig",
        "OptimizerType"
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
        elif config.optimizer_type == "rmsprop":
            return torch.optim.RMSprop(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

# ============================================================================
# CORE MODULES - DATA
# ============================================================================

# core/data/preprocessing.py
class PreprocessingModule:
    """Core preprocessing module with proper exports."""
    
    __all__ = [
        "normalize_tensor_data",
        "PreprocessingConfig",
        "NormalizationType"
    ]
    
    class PreprocessingConfig(BaseModel):
        """Pydantic model for preprocessing configuration."""
        
        model_config = ConfigDict(extra="forbid")
        
        normalization_type: constr(strip_whitespace=True) = Field(
            default="standard",
            pattern=r"^(standard|minmax|robust|none)$"
        )
        epsilon: confloat(gt=0.0) = Field(default=1e-8, description="Epsilon for numerical stability")
    
    class NormalizationType(BaseModel):
        """Pydantic model for normalization type validation."""
        
        model_config = ConfigDict(extra="forbid")
        
        type_name: constr(strip_whitespace=True) = Field(
            pattern=r"^(standard|minmax|robust|none)$"
        )
        description: Optional[str] = Field(default=None)
    
    def normalize_tensor_data(
        tensor_data: torch.Tensor,
        config: PreprocessingConfig
    ) -> Dict[str, Any]:
        """Normalize tensor data with comprehensive type hints and validation."""
        try:
            # Validate inputs
            if not isinstance(tensor_data, torch.Tensor):
                raise ValueError("tensor_data must be a torch.Tensor")
            if tensor_data.numel() == 0:
                raise ValueError("tensor_data cannot be empty")
            
            # Perform normalization
            if config.normalization_type == "standard":
                mean = torch.mean(tensor_data)
                std = torch.std(tensor_data)
                normalized = (tensor_data - mean) / (std + config.epsilon)
            elif config.normalization_type == "minmax":
                min_val = torch.min(tensor_data)
                max_val = torch.max(tensor_data)
                normalized = (tensor_data - min_val) / (max_val - min_val + config.epsilon)
            elif config.normalization_type == "robust":
                median = torch.median(tensor_data)
                mad = torch.median(torch.abs(tensor_data - median))
                normalized = (tensor_data - median) / (mad + config.epsilon)
            else:
                raise ValueError(f"Unsupported normalization type: {config.normalization_type}")
            
            return {
                "is_successful": True,
                "result": normalized,
                "error": None
            }
        except Exception as exc:
            return {
                "is_successful": False,
                "result": None,
                "error": str(exc)
            }

# core/data/loaders.py
class DataLoadersModule:
    """Core data loaders module with proper exports."""
    
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
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            if config.data_format == "json":
                data = json.loads(content)
            elif config.data_format == "csv":
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

# ============================================================================
# UTILS MODULES
# ============================================================================

# utils/type_hints.py
class TypeHintsModule:
    """Type hints utility module with proper exports."""
    
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
            
    """__init__ function."""
self.value = value
            self.is_success = is_success
            self.error = error

# utils/async_helpers.py
class AsyncHelpersModule:
    """Async helpers utility module with proper exports."""
    
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
    
    async def async_retry(
        func: Callable,
        config: "RetryConfig",
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

# ============================================================================
# CONFIGS MODULES
# ============================================================================

# configs/model_configs.py
class ModelConfigsModule:
    """Model configurations module with proper exports."""
    
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

# ============================================================================
# MAIN MODULE INTEGRATION
# ============================================================================

# __init__.py
class MainModule:
    """Main module with proper imports and exports."""
    
    # Import all modules
    neural_networks = NeuralNetworksModule()
    loss_functions = LossFunctionsModule()
    optimizers = OptimizersModule()
    preprocessing = PreprocessingModule()
    data_loaders = DataLoadersModule()
    type_hints = TypeHintsModule()
    async_helpers = AsyncHelpersModule()
    model_configs = ModelConfigsModule()
    
    # Define main exports
    __all__ = [
        # Core modules
        "NeuralNetworksModule",
        "LossFunctionsModule",
        "OptimizersModule",
        "PreprocessingModule",
        "DataLoadersModule",
        
        # Utils modules
        "TypeHintsModule",
        "AsyncHelpersModule",
        
        # Configs modules
        "ModelConfigsModule",
        
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
        try:
            # Create model
            model = neural_networks.create_feedforward_network(model_config)
            
            # Create optimizer
            optimizer_config = optimizers.OptimizerConfig(
                optimizer_type="adam",
                learning_rate=training_config.learning_rate
            )
            optimizer = optimizers.create_optimizer(model, optimizer_config)
            
            # Load data
            data_result = await data_loaders.load_training_data_async(data_config)
            if not data_result["is_successful"]:
                return TypeHintsModule.ProcessingResult(
                    is_successful=False,
                    result=None,
                    error=data_result["error"]
                )
            
            training_data = data_result["result"]
            
            # Preprocess data
            preprocessing_config = preprocessing.PreprocessingConfig(
                normalization_type=data_config.normalization_type
            )
            normalize_result = preprocessing.normalize_tensor_data(
                torch.tensor(training_data["features"]),
                preprocessing_config
            )
            
            if not normalize_result["is_successful"]:
                return TypeHintsModule.ProcessingResult(
                    is_successful=False,
                    result=None,
                    error=normalize_result["error"]
                )
            
            processed_features = normalize_result["result"]
            
            # Training loop
            for epoch in range(training_config.max_epochs):
                # Forward pass
                predictions = model(processed_features)
                targets = torch.tensor(training_data["targets"])
                
                # Calculate loss
                loss_config = loss_functions.LossConfig(loss_type="mse")
                loss_result = loss_functions.calculate_loss_function(
                    predictions, targets, loss_config
                )
                
                if not loss_result["is_successful"]:
                    return TypeHintsModule.ProcessingResult(
                        is_successful=False,
                        result=None,
                        error=loss_result["error"]
                    )
                
                loss = loss_result["result"]
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            return TypeHintsModule.ProcessingResult(
                is_successful=True,
                result={
                    "model": model,
                    "final_loss": loss,
                    "epochs_trained": training_config.max_epochs
                },
                error=None
            )
            
        except Exception as exc:
            return TypeHintsModule.ProcessingResult(
                is_successful=False,
                result=None,
                error=str(exc)
            )

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def demonstrate_modular_refactoring():
    """Demonstrate the modular refactoring with all patterns."""
    
    print("🏗️ Demonstrating Modular Refactoring with All Patterns")
    print("=" * 60)
    
    # Example 1: Using core modules
    print("\n📦 Core Modules:")
    neural_networks = NeuralNetworksModule()
    loss_functions = LossFunctionsModule()
    optimizers = OptimizersModule()
    preprocessing = PreprocessingModule()
    data_loaders = DataLoadersModule()
    
    # Example 2: Using utils modules
    print("\n🔧 Utils Modules:")
    type_hints = TypeHintsModule()
    async_helpers = AsyncHelpersModule()
    
    # Example 3: Using configs modules
    print("\n⚙️ Configs Modules:")
    model_configs = ModelConfigsModule()
    
    # Example 4: Using main module
    print("\n🎯 Main Module:")
    main_module = MainModule()
    
    print("\n✅ Modular refactoring demonstrated successfully!")
    print(f"📦 Core modules: 5 (NeuralNetworks, LossFunctions, Optimizers, Preprocessing, DataLoaders)")
    print(f"🔧 Utils modules: 2 (TypeHints, AsyncHelpers)")
    print(f"⚙️ Configs modules: 1 (ModelConfigs)")
    print(f"🎯 Main module: 1 (MainModule)")

def show_refactoring_benefits():
    """Show the benefits of modular refactoring."""
    
    benefits = {
        "organization": [
            "Clear separation of concerns",
            "Logical grouping of functionality",
            "Easy to navigate and understand",
            "Scalable architecture"
        ],
        "maintainability": [
            "Isolated changes don't affect other modules",
            "Easier to test individual components",
            "Clear dependencies between modules",
            "Reduced coupling"
        ],
        "reusability": [
            "Modules can be imported independently",
            "Shared utilities across modules",
            "Consistent interfaces",
            "Named exports for clear APIs"
        ],
        "type_safety": [
            "Type hints throughout all modules",
            "Pydantic validation for configurations",
            "Consistent error handling",
            "Clear function signatures"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate modular refactoring
    demonstrate_modular_refactoring()
    
    benefits = show_refactoring_benefits()
    
    print("\n🎯 Key Modular Refactoring Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    print("\n✅ Modular structure organization completed successfully!") 