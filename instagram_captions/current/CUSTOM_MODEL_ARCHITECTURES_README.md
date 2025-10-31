# Custom Model Architectures

## Overview

The **Custom Model Architectures** system provides a comprehensive collection of custom `nn.Module` classes for various deep learning tasks. This system is designed to be easily extensible, configurable, and seamlessly integrated with the PyTorch Primary Framework System.

## 🚀 Key Features

### **Flexible Architecture Design**
- **Base Model Class**: Common functionality for all custom models
- **Modular Components**: Reusable building blocks for complex architectures
- **Configuration-Driven**: Easy model creation from YAML configuration files
- **Extensible Design**: Simple to add new model types and architectures

### **Multiple Model Types**
- **Transformer Models**: GPT-style language models with attention mechanisms
- **CNN Models**: Convolutional neural networks with ResNet-style architectures
- **RNN Models**: Recurrent neural networks (LSTM, GRU, RNN)
- **Hybrid Models**: Combinations of different architecture types

### **Advanced Features**
- **Layer Management**: Freeze/unfreeze specific layers
- **Model Analysis**: Comprehensive model information and statistics
- **Performance Optimization**: Built-in optimizations for training and inference
- **Export Support**: Multiple export formats (TorchScript, ONNX)

## 📁 Project Structure

```
custom_model_architectures/
├── custom_model_architectures.py      # Main model implementations
├── custom_model_config.yaml           # Configuration file
├── custom_model_examples.py           # Usage examples
├── CUSTOM_MODEL_ARCHITECTURES_README.md  # This documentation
└── examples/                          # Additional examples
```

## 🏗️ Architecture Components

### **Base Model Class**

The `BaseModel` class provides common functionality for all custom models:

```python
class BaseModel(nn.Module):
    """Base class for all custom models with common functionality"""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specific layers by name"""
    
    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers by name"""
```

### **Transformer Models**

Custom transformer implementation with configurable architecture:

```python
class CustomTransformerModel(BaseModel):
    """Custom transformer model with configurable architecture"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, d_ff: int, max_seq_len: int, 
                 dropout: float = 0.1, activation: str = 'gelu',
                 norm_first: bool = True, use_relative_position: bool = False,
                 tie_weights: bool = True):
```

**Features:**
- Multi-head attention with optional relative positioning
- Configurable activation functions (GELU, ReLU, Swish)
- Pre-norm and post-norm architectures
- Weight tying for embedding and output layers
- Layer normalization and dropout

### **CNN Models**

Flexible CNN architectures with multiple design patterns:

```python
class CustomCNNModel(BaseModel):
    """Custom CNN model with configurable architecture"""
    
    def __init__(self, input_channels: int, num_classes: int, 
                 base_channels: int = 64, num_layers: int = 5,
                 use_batch_norm: bool = True, use_dropout: bool = True,
                 activation: str = 'relu', architecture: str = 'standard'):
```

**Architectures:**
- **Standard**: Traditional CNN with increasing channel depth
- **ResNet**: Residual connections for better gradient flow
- **Custom**: User-defined layer configurations

**Features:**
- Batch normalization and dropout
- Multiple activation functions
- Global average pooling
- Configurable classifier layers

### **RNN Models**

Recurrent neural networks with multiple cell types:

```python
class CustomRNNModel(BaseModel):
    """Custom RNN model with configurable architecture"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.1, 
                 bidirectional: bool = False, rnn_type: str = 'lstm',
                 use_layer_norm: bool = False):
```

**RNN Types:**
- **LSTM**: Long Short-Term Memory cells
- **GRU**: Gated Recurrent Unit cells
- **RNN**: Basic recurrent cells

**Features:**
- Bidirectional processing
- Layer normalization
- Variable length sequence support
- Multiple output strategies

### **Hybrid Models**

Combinations of different architecture types:

```python
class CNNTransformerHybrid(BaseModel):
    """Hybrid model combining CNN and Transformer architectures"""
    
    def __init__(self, input_channels: int, num_classes: int, 
                 cnn_channels: int = 64, cnn_layers: int = 4,
                 d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048,
                 max_seq_len: int = 256, dropout: float = 0.1):
```

**Use Cases:**
- Image captioning
- Vision-language tasks
- Multi-modal learning

## 🛠️ Installation and Setup

### **Prerequisites**
- Python 3.8+
- PyTorch 1.12.0+
- PyYAML for configuration files

### **Quick Start**

```python
from custom_model_architectures import (
    CustomTransformerModel,
    CustomCNNModel,
    create_model_from_config
)

# Create model directly
transformer = CustomTransformerModel(
    vocab_size=10000,
    d_model=256,
    n_heads=4,
    n_layers=4,
    d_ff=1024,
    max_seq_len=128
)

# Create model from configuration
with open('custom_model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = create_model_from_config(config['transformer_models']['small'])
```

## 📊 Configuration Management

### **Configuration File Structure**

The system uses YAML configuration files for easy model management:

```yaml
transformer_models:
  small:
    type: "transformer"
    vocab_size: 10000
    d_model: 256
    n_heads: 4
    n_layers: 4
    d_ff: 1024
    max_seq_len: 256
    dropout: 0.1
    activation: "gelu"
    norm_first: true
    use_relative_position: false
    tie_weights: true
```

### **Task-Specific Configurations**

Pre-configured models for specific tasks:

```yaml
task_specific:
  instagram_caption:
    model_type: "transformer"
    config: "medium"
    customizations:
      vocab_size: 25000
      max_seq_len: 128
      dropout: 0.15
      activation: "gelu"
```

## 🚀 Usage Examples

### **Basic Model Creation**

```python
# Create transformer model
transformer = CustomTransformerModel(
    vocab_size=30000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_seq_len=512
)

# Create CNN model
cnn = CustomCNNModel(
    input_channels=3,
    num_classes=1000,
    base_channels=64,
    num_layers=5,
    architecture='resnet'
)

# Create RNN model
rnn = CustomRNNModel(
    input_size=512,
    hidden_size=256,
    num_layers=3,
    num_classes=10,
    bidirectional=True,
    rnn_type='lstm'
)
```

### **Configuration-Driven Model Creation**

```python
import yaml
from custom_model_architectures import create_model_from_config

# Load configuration
with open('custom_model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create models from configuration
small_transformer = create_model_from_config(config['transformer_models']['small'])
standard_cnn = create_model_from_config(config['cnn_models']['standard'])
hybrid_model = create_model_from_config(config['hybrid_models']['image_captioning'])
```

### **Model Training Integration**

```python
from pytorch_primary_framework_system import PyTorchPrimaryFrameworkSystem

# Create PyTorch framework system
pytorch_system = PyTorchPrimaryFrameworkSystem(config)

# Create and optimize model
model = CustomTransformerModel(...)
model = pytorch_system.optimize_model_for_pytorch(model)

# Create optimizer and scheduler
optimizer = pytorch_system.create_pytorch_optimizer(model, "adamw", lr=1e-4)
scheduler = pytorch_system.create_pytorch_scheduler(optimizer, "cosine", T_max=100)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        result = pytorch_system.train_step_pytorch(
            model=model,
            data=batch['input'],
            targets=batch['target'],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )
```

### **Advanced Model Features**

```python
# Get model information
model_info = model.get_model_info()
print(f"Total parameters: {model_info['total_parameters']:,}")
print(f"Model size: {model_info['model_size_mb']:.2f} MB")

# Freeze specific layers
model.freeze_layers(['token_embedding', 'position_embedding'])

# Check parameter status
frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
print(f"Frozen parameters: {frozen_params:,}")

# Unfreeze layers
model.unfreeze_layers(['token_embedding', 'position_embedding'])
```

## 📈 Performance Optimization

### **Memory Optimization**

```python
# Use gradient checkpointing for large models
if model.count_parameters() > 100_000_000:  # 100M parameters
    model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### **Speed Optimization**

```python
# Enable model compilation (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')

# Use channels last memory format for CNNs
if isinstance(model, CustomCNNModel):
    model = model.to(memory_format=torch.channels_last)
```

## 🔍 Model Analysis and Debugging

### **Model Summary**

```python
from custom_model_architectures import get_model_summary

# Get comprehensive model summary
summary = get_model_summary(model)
print(summary)
```

### **Parameter Analysis**

```python
# Count parameters by layer
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} parameters")

# Check parameter gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
```

### **Memory Profiling**

```python
# Track memory usage
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
    memory_reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"Memory allocated: {memory_allocated:.2f} GB")
    print(f"Memory reserved: {memory_reserved:.2f} GB")
```

## 🔄 Model Export and Deployment

### **TorchScript Export**

```python
# Export to TorchScript
model.eval()
dummy_input = torch.randint(0, 10000, (1, 128))
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model.pt")
```

### **ONNX Export**

```python
# Export to ONNX
import torch.onnx

dummy_input = torch.randint(0, 10000, (1, 128))
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## 🧪 Testing and Validation

### **Unit Tests**

```python
import pytest

def test_transformer_model():
    model = CustomTransformerModel(
        vocab_size=1000,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=64
    )
    
    input_tensor = torch.randint(0, 1000, (2, 32))
    output = model(input_tensor)
    
    assert output.shape == (2, 32, 1000)
    assert model.count_parameters() > 0

def test_cnn_model():
    model = CustomCNNModel(
        input_channels=3,
        num_classes=10,
        base_channels=16,
        num_layers=3
    )
    
    input_tensor = torch.randn(2, 3, 64, 64)
    output = model(input_tensor)
    
    assert output.shape == (2, 10)
    assert model.count_parameters() > 0
```

### **Integration Tests**

```python
def test_model_integration():
    # Test model creation from config
    config = {
        'type': 'transformer',
        'vocab_size': 1000,
        'd_model': 64,
        'n_heads': 2,
        'n_layers': 2,
        'd_ff': 256,
        'max_seq_len': 64
    }
    
    model = create_model_from_config(config)
    assert isinstance(model, CustomTransformerModel)
    
    # Test forward pass
    input_tensor = torch.randint(0, 1000, (2, 32))
    output = model(input_tensor)
    assert output.shape == (2, 32, 1000)
```

## 📚 Best Practices

### **Model Design**

1. **Inherit from BaseModel**: All custom models should inherit from `BaseModel`
2. **Implement Required Methods**: Always implement `forward()` and `_get_architecture_info()`
3. **Use Type Hints**: Provide clear type annotations for all methods
4. **Documentation**: Add comprehensive docstrings for all classes and methods

### **Configuration Management**

1. **Use YAML Files**: Store model configurations in YAML format
2. **Validation**: Validate configuration parameters before model creation
3. **Defaults**: Provide sensible default values for all parameters
4. **Environment-Specific**: Use different configurations for different environments

### **Performance Optimization**

1. **Gradient Checkpointing**: Enable for models with >100M parameters
2. **Mixed Precision**: Use automatic mixed precision for training
3. **Memory Format**: Use channels last for CNN models
4. **Model Compilation**: Enable PyTorch 2.0+ compilation when available

### **Testing and Validation**

1. **Unit Tests**: Write tests for all model components
2. **Integration Tests**: Test model integration with framework system
3. **Performance Tests**: Benchmark models for speed and memory usage
4. **Edge Cases**: Test with various input sizes and configurations

## 🚨 Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Model Compilation Errors**
   - Ensure PyTorch 2.0+
   - Check model compatibility
   - Use appropriate compilation mode

3. **Configuration Errors**
   - Validate YAML syntax
   - Check required fields
   - Verify parameter ranges

### **Debug Mode**

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model state
print(f"Model training mode: {model.training}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
```

## 🤝 Contributing

### **Adding New Model Types**

1. **Create New Class**: Inherit from `BaseModel`
2. **Implement Methods**: Override required methods
3. **Add Configuration**: Update configuration file
4. **Write Tests**: Add unit and integration tests
5. **Update Documentation**: Document new features

### **Code Style**

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for all classes and methods
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- Open source community for various libraries and tools
- Contributors to the custom model architectures system

## 📞 Support

### **Getting Help**

1. Check the documentation and examples
2. Review configuration files
3. Run unit tests to verify setup
4. Check PyTorch compatibility

### **Reporting Issues**

When reporting issues, please include:
- PyTorch version
- Python version
- Operating system
- Error traceback
- Model configuration
- Minimal reproduction code

---

**Note**: This system is designed for production use and includes comprehensive error handling, logging, and monitoring. For development and testing, consider using the debug configuration options.


