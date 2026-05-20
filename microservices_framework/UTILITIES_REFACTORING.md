# Utilities Refactoring - Enhanced Modularity

This document describes the utilities module that enhances the framework's modularity.

## 🛠️ New Utility Modules

### 1. Decorators (`shared/ml/utils/decorators.py`)

**Purpose**: Reusable decorators for common operations

**Available Decorators**:
- `@timing_decorator`: Measure execution time
- `@gpu_memory_tracker`: Track GPU memory usage
- `@error_handler`: Handle errors gracefully
- `@validate_inputs`: Validate function inputs
- `@cache_result`: Cache function results
- `@retry`: Retry on failure
- `@torch_no_grad`: Disable gradient computation
- `@torch_eval_mode`: Set model to eval mode

**Usage**:
```python
from shared.ml import timing_decorator, gpu_memory_tracker, retry

@timing_decorator
@gpu_memory_tracker
@retry(max_attempts=3, delay=1.0)
def train_model(model, data):
    # Training logic
    pass
```

### 2. Validators (`shared/ml/utils/validators.py`)

**Purpose**: Input validation utilities

**Available Validators**:
- `TypeValidator`: Validate type
- `RangeValidator`: Validate numeric range
- `TensorShapeValidator`: Validate tensor shape
- `TensorDtypeValidator`: Validate tensor dtype
- `NotNoneValidator`: Check not None
- `NotEmptyValidator`: Check not empty
- `InListValidator`: Check value in list
- `CompositeValidator`: Combine validators

**Usage**:
```python
from shared.ml import (
    validate_model_input,
    validate_generation_params,
    RangeValidator,
)

# Validate model input
validate_model_input(
    input_tensor,
    expected_shape=(1, 512),
    expected_dtype=torch.long,
    check_finite=True,
)

# Validate generation params
validate_generation_params(
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
)

# Custom validator
validator = RangeValidator(min_value=0.0, max_value=1.0)
assert validator.validate(0.5) == True
```

### 3. Transformers (`shared/ml/utils/transformers.py`)

**Purpose**: Functional data transformations

**Available Transformers**:
- `NormalizeTransformer`: Normalize tensors
- `ToTensorTransformer`: Convert to tensor
- `PadTransformer`: Pad sequences
- `TruncateTransformer`: Truncate sequences
- `ComposeTransformer`: Compose multiple transformers
- `LambdaTransformer`: Apply lambda function

**Usage**:
```python
from shared.ml import (
    ComposeTransformer,
    NormalizeTransformer,
    ToTensorTransformer,
    create_text_transformer_pipeline,
)

# Compose transformers
pipeline = ComposeTransformer([
    ToTensorTransformer(),
    NormalizeTransformer(mean=0.0, std=1.0),
])

# Text transformation pipeline
text_pipeline = create_text_transformer_pipeline(
    tokenizer,
    max_length=512,
    padding=True,
    truncation=True,
)
```

### 4. Callbacks (`shared/ml/utils/callbacks.py`)

**Purpose**: Callback system for training and inference

**Available Callbacks**:
- `Callback`: Base callback class
- `EarlyStoppingCallback`: Early stopping
- `ModelCheckpointCallback`: Model checkpointing
- `LoggingCallback`: Logging
- `CallbackManager`: Manage multiple callbacks

**Usage**:
```python
from shared.ml import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LoggingCallback,
    CallbackManager,
)

# Create callbacks
callbacks = CallbackManager([
    EarlyStoppingCallback(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    ModelCheckpointCallback(
        save_dir="./checkpoints",
        monitor="val_loss",
        save_best=True,
    ),
    LoggingCallback(log_interval=10),
])

# Use in training
for epoch in range(num_epochs):
    callbacks.on_epoch_start(epoch)
    # Training loop
    metrics = {"val_loss": 0.5}
    callbacks.on_epoch_end(epoch, metrics, model=model)
```

## 📊 Complete Module Structure

```
shared/ml/
├── core/                    # Core interfaces and patterns
├── utils/                   # 🆕 Utility modules
│   ├── decorators.py       # Decorators
│   ├── validators.py       # Validators
│   ├── transformers.py     # Data transformers
│   └── callbacks.py        # Callback system
├── models/                  # Model architectures
├── data/                    # Data processing
├── training/                # Training operations
├── inference/               # Inference operations
├── optimization/            # Model optimization
├── evaluation/              # Evaluation operations
├── monitoring/              # Profiling and tracking
├── quantization/            # Quantization
├── registry/                # Model registry
├── schedulers/              # Learning rate scheduling
├── distributed/             # Distributed training
└── gradio/                  # Gradio utilities
```

## 🎯 Usage Examples

### Complete Training with Callbacks

```python
from shared.ml import (
    TrainingPipelineBuilder,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LoggingCallback,
    CallbackManager,
)

# Build training pipeline
trainer = (
    TrainingPipelineBuilder()
    .with_model(model)
    .with_data_loaders(train_loader, val_loader)
    .with_optimizer("adamw", learning_rate=5e-5)
    .build()
)

# Setup callbacks
callbacks = CallbackManager([
    EarlyStoppingCallback(monitor="val_loss", patience=5),
    ModelCheckpointCallback(save_dir="./checkpoints"),
    LoggingCallback(log_interval=10),
])

# Training with callbacks
callbacks.on_training_start()
for epoch in range(num_epochs):
    callbacks.on_epoch_start(epoch)
    # Training loop
    metrics = trainer.validate()
    callbacks.on_epoch_end(epoch, metrics, model=model)
callbacks.on_training_end()
```

### Decorated Functions

```python
from shared.ml import (
    timing_decorator,
    gpu_memory_tracker,
    error_handler,
    validate_inputs,
)

@timing_decorator
@gpu_memory_tracker
@error_handler(default_return=None)
@validate_inputs(
    max_length=lambda x: 1 <= x <= 2048,
    temperature=lambda x: 0.0 <= x <= 2.0,
)
def generate_text(prompt, max_length, temperature):
    # Generation logic
    return generated_text
```

### Data Transformation Pipeline

```python
from shared.ml import (
    ComposeTransformer,
    ToTensorTransformer,
    NormalizeTransformer,
    PadTransformer,
)

# Create transformation pipeline
transform = ComposeTransformer([
    ToTensorTransformer(dtype=torch.float32),
    PadTransformer(max_length=512, pad_value=0.0),
    NormalizeTransformer(mean=0.0, std=1.0),
])

# Apply transformation
transformed_data = transform(raw_data)
```

## ✨ Benefits

### 1. Code Reusability
- Decorators can be applied to any function
- Validators can be reused across different contexts
- Transformers can be composed in different ways

### 2. Clean Code
- Decorators add functionality without cluttering code
- Validators provide clear validation logic
- Callbacks separate concerns

### 3. Extensibility
- Easy to add new decorators
- Easy to create custom validators
- Easy to implement custom callbacks

### 4. Testability
- Each utility can be tested independently
- Decorators can be tested in isolation
- Validators have clear test cases

## 🎉 Summary

The utilities module provides:
- ✅ **Decorators**: Reusable function enhancements
- ✅ **Validators**: Input validation utilities
- ✅ **Transformers**: Functional data transformations
- ✅ **Callbacks**: Training and inference callbacks

This enhances the framework's modularity and makes it even more production-ready!

---

**The framework now has comprehensive utilities for all ML operations! 🚀**



