# Modular Architecture - Version 3.4.0

## 🏗️ Modular Design Principles

### 1. Base Classes (`core/base/`)

#### BaseModel (`base_model.py`)
- Abstract base class for all models
- Common functionality (compilation, device management)
- Interface definitions

**Usage**:
```python
from addiction_recovery_ai.core.base.base_model import BasePredictor

class MyPredictor(BasePredictor):
    def forward(self, x):
        return self.model(x)
    
    def predict(self, inputs, **kwargs):
        return self.forward(inputs)
    
    def _process_batch(self, batch, **kwargs):
        return [self.predict(item) for item in batch]
```

#### BaseTrainer (`base_trainer.py`)
- Abstract base class for trainers
- Common training functionality
- Interface for training loops

**Usage**:
```python
from addiction_recovery_ai.core.base.base_trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        # Implementation
        pass
    
    def validate(self, val_loader, criterion):
        # Implementation
        pass
    
    def train(self, train_loader, val_loader, optimizer, criterion, num_epochs, **kwargs):
        # Implementation
        pass
```

### 2. Factory Pattern (`core/factories/`)

#### ModelFactory (`model_factory.py`)
- Register and create models
- Type-safe model creation
- Configuration-based instantiation

**Usage**:
```python
from addiction_recovery_ai.core.factories.model_factory import ModelFactory, ModelBuilder

# Register model
ModelFactory.register("MyModel", MyModelClass)

# Create model
model = ModelFactory.create(
    "MyModel",
    config={"input_size": 10, "hidden_size": 128},
    device=torch.device("cuda")
)

# Or use builder
model = (ModelBuilder()
    .with_device(torch.device("cuda"))
    .with_config(input_size=10, hidden_size=128)
    .with_mixed_precision(True)
    .build("MyModel"))
```

#### TrainerFactory (`trainer_factory.py`)
- Register and create trainers
- Configuration-based trainer creation

**Usage**:
```python
from addiction_recovery_ai.core.factories.trainer_factory import TrainerFactory

# Register trainer
TrainerFactory.register("MyTrainer", MyTrainerClass)

# Create trainer
trainer = TrainerFactory.create(
    "MyTrainer",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config={"use_mixed_precision": True}
)
```

### 3. Configuration System (`config/`)

#### ConfigLoader (`core/config/config_loader.py`)
- YAML-based configuration
- Environment variable overrides
- Nested configuration access

**Usage**:
```python
from addiction_recovery_ai.core.config.config_loader import get_config

# Load config
config = get_config("config/model_config.yaml")

# Get values
device = config.get("models.sentiment_analyzer.device", "cpu")
batch_size = config.get("training.batch_size", 32)

# Get model config
model_config = config.get_model_config("sentiment_analyzer")

# Get training config
training_config = config.get_training_config()
```

#### YAML Configuration (`config/model_config.yaml`)
- Centralized configuration
- Model-specific settings
- Training parameters
- Data loader settings

**Example**:
```yaml
models:
  sentiment_analyzer:
    type: "RecoverySentimentAnalyzer"
    model_name: "cardiffnlp/twitter-roberta-base-sentiment-latest"
    device: "cuda"
    use_mixed_precision: true

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
```

### 4. Data Loader Factory (`core/data/`)

#### DataLoaderFactory (`data_loader_factory.py`)
- Create optimized data loaders
- Split-specific configurations
- Performance optimizations

**Usage**:
```python
from addiction_recovery_ai.core.data.data_loader_factory import DataLoaderFactory

# Create train loader
train_loader = DataLoaderFactory.create(
    train_dataset,
    config={"batch_size": 32, "shuffle": True},
    split="train"
)

# Create optimized loader
optimized_loader = DataLoaderFactory.create_optimized(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True
)
```

### 5. Plugin System (`core/plugins/`)

#### PluginManager (`plugin_manager.py`)
- Extensible plugin architecture
- Hook system
- Dynamic plugin loading

**Usage**:
```python
from addiction_recovery_ai.core.plugins.plugin_manager import Plugin, get_plugin_manager

# Create plugin
class MyPlugin(Plugin):
    def __init__(self):
        super().__init__("MyPlugin", "1.0.0")
    
    def initialize(self, config=None):
        # Initialize plugin
        pass

# Register plugin
manager = get_plugin_manager()
manager.register_plugin(MyPlugin())

# Load plugins from directory
manager.load_plugins_from_directory("plugins")

# Use hooks
def my_hook(data):
    # Process data
    return processed_data

manager.register_hook("pre_process", my_hook)
results = manager.call_hook("pre_process", input_data)
```

## 📁 Directory Structure

```
addiction_recovery_ai/
├── core/
│   ├── base/
│   │   ├── base_model.py          # Base model classes
│   │   └── base_trainer.py        # Base trainer classes
│   ├── factories/
│   │   ├── model_factory.py       # Model factory
│   │   └── trainer_factory.py    # Trainer factory
│   ├── config/
│   │   └── config_loader.py      # Configuration loader
│   ├── data/
│   │   └── data_loader_factory.py # Data loader factory
│   └── plugins/
│       └── plugin_manager.py      # Plugin system
├── config/
│   └── model_config.yaml          # Configuration file
└── ...
```

## 🎯 Benefits of Modular Architecture

### 1. Separation of Concerns
- Models, training, data loading are separate
- Each module has single responsibility
- Easy to test and maintain

### 2. Extensibility
- Add new models by registering them
- Add new trainers without modifying existing code
- Plugin system for custom functionality

### 3. Configuration Management
- Centralized YAML configuration
- Environment variable overrides
- Easy to switch between configurations

### 4. Type Safety
- Abstract base classes enforce interfaces
- Factory pattern ensures correct instantiation
- Clear contracts between modules

### 5. Testability
- Each module can be tested independently
- Mock implementations for testing
- Clear interfaces make testing easier

## 📝 Usage Examples

### Complete Workflow
```python
from addiction_recovery_ai.core.config.config_loader import get_config
from addiction_recovery_ai.core.factories.model_factory import ModelFactory, ModelBuilder
from addiction_recovery_ai.core.factories.trainer_factory import TrainerFactory
from addiction_recovery_ai.core.data.data_loader_factory import DataLoaderFactory

# Load configuration
config = get_config()

# Create model using factory
model_config = config.get_model_config("sentiment_analyzer")
model = ModelFactory.create("RecoverySentimentAnalyzer", model_config)

# Create data loaders
train_loader = DataLoaderFactory.create(
    train_dataset,
    config.get_data_config(),
    split="train"
)

val_loader = DataLoaderFactory.create(
    val_dataset,
    config.get_data_config(),
    split="val"
)

# Create trainer
trainer = TrainerFactory.create(
    "RecoveryModelTrainer",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config.get_training_config()
)

# Train
trainer.train(...)
```

### Adding New Model
```python
# 1. Create model class
from addiction_recovery_ai.core.base.base_model import BasePredictor

class MyNewModel(BasePredictor):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.model = nn.Sequential(...)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, inputs, **kwargs):
        return self.forward(inputs)
    
    def _process_batch(self, batch, **kwargs):
        return [self.predict(item) for item in batch]

# 2. Register with factory
ModelFactory.register("MyNewModel", MyNewModel)

# 3. Use in config
# config/model_config.yaml:
# models:
#   my_model:
#     type: "MyNewModel"
#     input_size: 10
#     hidden_size: 128

# 4. Create instance
model = ModelFactory.create("MyNewModel", config.get_model_config("my_model"))
```

### Adding Plugin
```python
from addiction_recovery_ai.core.plugins.plugin_manager import Plugin, get_plugin_manager

class CustomPreprocessingPlugin(Plugin):
    def __init__(self):
        super().__init__("CustomPreprocessing", "1.0.0")
    
    def initialize(self, config=None):
        # Initialize
        pass
    
    def preprocess(self, data):
        # Custom preprocessing
        return processed_data

# Register
manager = get_plugin_manager()
manager.register_plugin(CustomPreprocessingPlugin())

# Use
plugin = manager.get_plugin("CustomPreprocessing")
processed = plugin.preprocess(data)
```

## 🔧 Best Practices

### 1. Use Base Classes
- Inherit from BaseModel, BasePredictor, etc.
- Implement required abstract methods
- Use common functionality from base classes

### 2. Register with Factories
- Register all models and trainers
- Use factory for creation
- Keep configuration in YAML

### 3. Configuration Management
- Use YAML for configuration
- Support environment variable overrides
- Keep model-specific configs separate

### 4. Plugin Development
- Inherit from Plugin base class
- Implement initialize and cleanup
- Use hooks for extensibility

### 5. Testing
- Test each module independently
- Use mocks for dependencies
- Test factory registration and creation

## 📈 Summary

The modular architecture provides:

- ✅ **Base Classes**: Common interfaces and functionality
- ✅ **Factories**: Type-safe model and trainer creation
- ✅ **Configuration**: YAML-based centralized config
- ✅ **Data Loaders**: Optimized data loading
- ✅ **Plugins**: Extensible plugin system
- ✅ **Separation of Concerns**: Clear module boundaries
- ✅ **Extensibility**: Easy to add new components
- ✅ **Testability**: Independent module testing

---

**Version**: 3.4.0  
**Date**: 2025  
**Author**: Blatam Academy













