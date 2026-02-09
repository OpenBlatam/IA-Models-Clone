# Modular Architecture - Design Principles

This document describes the highly modular architecture of the ML framework.

## 🏗️ Architecture Principles

### 1. Separation of Concerns
Each module has a single, well-defined responsibility:
- **Core**: Interfaces, factories, builders
- **Models**: Model architectures and management
- **Data**: Data processing pipelines
- **Training**: Training operations
- **Inference**: Inference operations
- **Optimization**: Model optimization (LoRA, quantization)
- **Evaluation**: Model evaluation
- **Monitoring**: Profiling and tracking

### 2. Interface-Based Design
All major components implement interfaces:
- `IModelLoader`: Model loading operations
- `IInferenceEngine`: Inference operations
- `ITrainer`: Training operations
- `IEvaluator`: Evaluation operations
- `IQuantizer`: Quantization operations
- `IProfiler`: Profiling operations

### 3. Factory Pattern
Factories create components without exposing instantiation logic:
- `ModelLoaderFactory`: Model loaders
- `OptimizerFactory`: Optimizers
- `LossFunctionFactory`: Loss functions
- `DeviceFactory`: Device management
- `ComponentFactory`: All components

### 4. Builder Pattern
Builders construct complex objects step by step:
- `TrainingPipelineBuilder`: Training pipelines
- `InferencePipelineBuilder`: Inference pipelines
- `ModelOptimizationBuilder`: Model optimization

## 📦 Module Structure

```
shared/ml/
├── core/                    # Core interfaces and patterns
│   ├── interfaces.py       # Abstract interfaces
│   ├── factories.py        # Factory classes
│   ├── builders.py         # Builder classes
│   └── losses.py           # Custom loss functions
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

## 🔧 Design Patterns Used

### 1. Factory Pattern

**Purpose**: Create objects without specifying exact classes

**Example**:
```python
from shared.ml import OptimizerFactory, LossFunctionFactory

# Create optimizer
optimizer = OptimizerFactory.create(
    "adamw",
    model,
    learning_rate=5e-5,
    weight_decay=0.01
)

# Create loss function
loss_fn = LossFunctionFactory.create(
    "cross_entropy",
    ignore_index=-100
)
```

**Benefits**:
- Decouples object creation from usage
- Easy to add new types
- Centralized creation logic

### 2. Builder Pattern

**Purpose**: Construct complex objects step by step

**Example**:
```python
from shared.ml import TrainingPipelineBuilder

trainer = (
    TrainingPipelineBuilder()
    .with_model(model)
    .with_data_loaders(train_loader, val_loader)
    .with_optimizer("adamw", learning_rate=5e-5)
    .with_loss_function("cross_entropy")
    .with_scheduler("cosine", num_training_steps=1000)
    .with_trainer_config(use_amp=True)
    .build()
)
```

**Benefits**:
- Fluent, readable API
- Flexible construction
- Validation at build time

### 3. Interface Segregation

**Purpose**: Clients should not depend on interfaces they don't use

**Example**:
```python
from shared.ml.core import IInferenceEngine, ITrainer

# Inference only needs IInferenceEngine
class InferenceService:
    def __init__(self, engine: IInferenceEngine):
        self.engine = engine
    
    def generate(self, prompt: str):
        return self.engine.generate(prompt)

# Training only needs ITrainer
class TrainingService:
    def __init__(self, trainer: ITrainer):
        self.trainer = trainer
    
    def train(self, epochs: int):
        return self.trainer.train(epochs)
```

**Benefits**:
- Clear dependencies
- Easy to mock for testing
- Flexible implementations

### 4. Dependency Injection

**Purpose**: Dependencies are injected rather than created internally

**Example**:
```python
# Good: Dependencies injected
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

# Usage
optimizer = OptimizerFactory.create("adamw", model)
criterion = LossFunctionFactory.create("cross_entropy")
trainer = Trainer(model, optimizer, criterion)
```

**Benefits**:
- Testable (easy to mock)
- Flexible (can swap implementations)
- Clear dependencies

## 🎯 Modularity Benefits

### 1. Testability
Each module can be tested independently:
```python
# Test optimizer factory
def test_optimizer_factory():
    model = torch.nn.Linear(10, 1)
    optimizer = OptimizerFactory.create("adam", model)
    assert isinstance(optimizer, torch.optim.Adam)

# Test builder
def test_training_builder():
    builder = TrainingPipelineBuilder()
    builder.with_model(model)
    # ... test each step
```

### 2. Extensibility
Easy to add new components:
```python
# Add new optimizer
class CustomOptimizer(torch.optim.Optimizer):
    # ... implementation

OptimizerFactory.register("custom", CustomOptimizer)

# Add new loss
class CustomLoss(nn.Module):
    # ... implementation

LossFunctionFactory.register("custom", CustomLoss)
```

### 3. Reusability
Components can be reused across different contexts:
```python
# Same optimizer factory used in different places
optimizer1 = OptimizerFactory.create("adamw", model1)
optimizer2 = OptimizerFactory.create("adamw", model2)
```

### 4. Maintainability
Clear structure makes code easy to maintain:
- Each module has single responsibility
- Changes are localized
- Easy to understand and modify

## 📝 Usage Examples

### Complete Training Pipeline

```python
from shared.ml import (
    TrainingPipelineBuilder,
    ModelManager,
    create_data_pipeline,
)

# Load model
manager = ModelManager()
model = manager.get_model("gpt2")

# Prepare data
tokenizer = AutoTokenizer.from_pretrained("gpt2")
data_loaders = create_data_pipeline(texts, tokenizer)

# Build training pipeline
trainer = (
    TrainingPipelineBuilder()
    .with_model(model)
    .with_data_loaders(data_loaders["train"], data_loaders["val"])
    .with_optimizer("adamw", learning_rate=5e-5)
    .with_loss_function("cross_entropy")
    .with_scheduler("cosine", num_training_steps=1000)
    .with_trainer_config(use_amp=True)
    .build()
)

# Train
trainer.train(num_epochs=3)
```

### Inference Pipeline

```python
from shared.ml import InferencePipelineBuilder

# Build inference pipeline
engine = (
    InferencePipelineBuilder()
    .with_model(model)
    .with_tokenizer(tokenizer)
    .with_config(use_amp=True, max_batch_size=32)
    .build()
)

# Generate
text = engine.generate("The future of AI", max_length=100)
```

### Model Optimization

```python
from shared.ml import ModelOptimizationBuilder

# Build optimized model
optimized = (
    ModelOptimizationBuilder()
    .with_model(model)
    .add_lora(r=8, alpha=16)
    .add_quantization("int8")
    .build()
)
```

## 🎉 Summary

The modular architecture provides:
- ✅ **Clear separation of concerns**
- ✅ **Interface-based design**
- ✅ **Factory pattern for creation**
- ✅ **Builder pattern for construction**
- ✅ **Easy testing and mocking**
- ✅ **High extensibility**
- ✅ **Code reusability**
- ✅ **Maintainable structure**

This makes the framework:
- **Easy to understand**: Clear structure
- **Easy to test**: Isolated modules
- **Easy to extend**: Plugin-like architecture
- **Easy to maintain**: Localized changes
- **Production-ready**: Enterprise patterns

---

**The framework is now highly modular and follows industry best practices! 🚀**



