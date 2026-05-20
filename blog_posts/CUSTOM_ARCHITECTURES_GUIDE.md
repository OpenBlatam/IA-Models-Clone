# Custom PyTorch Model Architectures Guide

## Overview

This guide covers advanced custom PyTorch `nn.Module` classes designed for blog post analysis and NLP tasks. All models feature modern optimizations, proper gradient flow, and production-ready implementations.

## Table of Contents

1. [PositionalEncoding](#positionalencoding)
2. [MultiHeadAttentionWithRelativePosition](#multiheadattentionwithrelativeposition)
3. [TransformerBlock](#transformerblock)
4. [CustomTransformer](#customtransformer)
5. [Conv1DBlock](#conv1dblock)
6. [CNNFeatureExtractor](#cnnfeatureextractor)
7. [LSTMWithAttention](#lstmwithattention)
8. [Attention Mechanisms](#attention-mechanisms)
9. [CNNLSTMHybrid](#cnnlstmhybrid)
10. [TransformerCNN](#transformercnn)
11. [MultiTaskModel](#multitaskmodel)
12. [HierarchicalAttentionNetwork](#hierarchicalattentionnetwork)
13. [ResidualBlock](#residualblock)
14. [DeepResidualCNN](#deepresidualcnn)
15. [ModelFactory](#modelfactory)
16. [Best Practices](#best-practices)
17. [Performance Optimization](#performance-optimization)
18. [Testing](#testing)

## PositionalEncoding

Advanced positional encoding with learnable parameters for transformer models.

### Features
- Sinusoidal positional encoding
- Learnable positional parameters
- Dropout for regularization
- Configurable maximum sequence length

### Usage

```python
from custom_model_architectures import PositionalEncoding

# Create positional encoding
pe = PositionalEncoding(d_model=512, max_len=5000, dropout=0.1)

# Apply to input
x = torch.randn(50, 512)  # (seq_len, d_model)
output = pe(x)  # (seq_len, d_model)
```

### Parameters
- `d_model`: Embedding dimension
- `max_len`: Maximum sequence length
- `dropout`: Dropout probability

## MultiHeadAttentionWithRelativePosition

Enhanced multi-head attention with relative positional encoding.

### Features
- Relative positional encoding
- Multiple attention types
- Configurable relative position window
- Proper gradient flow

### Usage

```python
from custom_model_architectures import MultiHeadAttentionWithRelativePosition

# Create attention layer
attention = MultiHeadAttentionWithRelativePosition(
    d_model=512, 
    n_heads=8, 
    dropout=0.1,
    max_relative_position=32
)

# Apply attention
x = torch.randn(4, 20, 512)  # (batch_size, seq_len, d_model)
mask = torch.ones(4, 20, 20)  # Optional attention mask
output = attention(x, mask)
```

### Parameters
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `dropout`: Dropout probability
- `max_relative_position`: Maximum relative position distance

## TransformerBlock

Enhanced transformer block with advanced features.

### Features
- Configurable activation functions (GELU, ReLU, Swish)
- Optional relative positional attention
- Residual connections
- Layer normalization

### Usage

```python
from custom_model_architectures import TransformerBlock

# Create transformer block
block = TransformerBlock(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    activation="gelu",
    use_relative_pos=True
)

# Apply block
x = torch.randn(4, 20, 512)
mask = torch.ones(4, 20, 20)
output = block(x, mask)
```

### Parameters
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `d_ff`: Feed-forward dimension
- `dropout`: Dropout probability
- `activation`: Activation function ("gelu", "relu", "swish")
- `use_relative_pos`: Whether to use relative positional attention

## CustomTransformer

Custom transformer model with advanced features.

### Features
- Learnable positional encoding
- Multiple transformer layers
- Configurable architecture
- Production-ready implementation

### Usage

```python
from custom_model_architectures import CustomTransformer

# Create transformer model
transformer = CustomTransformer(
    vocab_size=30000,
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=2048,
    max_len=5000,
    dropout=0.1,
    activation="gelu",
    use_relative_pos=True
)

# Forward pass
x = torch.randint(0, 30000, (4, 50))  # (batch_size, seq_len)
mask = torch.ones(4, 50, 50)  # Optional mask
output = transformer(x, mask)  # (batch_size, seq_len, d_model)
```

### Parameters
- `vocab_size`: Vocabulary size
- `d_model`: Model dimension
- `n_layers`: Number of transformer layers
- `n_heads`: Number of attention heads
- `d_ff`: Feed-forward dimension
- `max_len`: Maximum sequence length
- `dropout`: Dropout probability
- `activation`: Activation function
- `use_relative_pos`: Whether to use relative positional attention

## Conv1DBlock

1D convolutional block with advanced features.

### Features
- Multiple activation functions
- Optional batch normalization
- Configurable convolution parameters
- Dropout for regularization

### Usage

```python
from custom_model_architectures import Conv1DBlock

# Create conv1d block
conv_block = Conv1DBlock(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    activation="relu",
    dropout=0.1,
    batch_norm=True
)

# Apply convolution
x = torch.randn(4, 64, 20)  # (batch_size, channels, seq_len)
output = conv_block(x)  # (batch_size, 128, 20)
```

### Parameters
- `in_channels`: Input channels
- `out_channels`: Output channels
- `kernel_size`: Convolution kernel size
- `stride`: Convolution stride
- `padding`: Convolution padding
- `dilation`: Convolution dilation
- `groups`: Convolution groups
- `bias`: Whether to use bias
- `activation`: Activation function
- `dropout`: Dropout probability
- `batch_norm`: Whether to use batch normalization

## CNNFeatureExtractor

Advanced CNN feature extractor for text processing.

### Features
- Multiple kernel sizes per layer
- Global pooling (max/avg)
- Configurable architecture
- Efficient feature extraction

### Usage

```python
from custom_model_architectures import CNNFeatureExtractor

# Create CNN extractor
extractor = CNNFeatureExtractor(
    input_dim=300,
    hidden_dims=[128, 256, 512],
    kernel_sizes=[3, 4, 5],
    dropout=0.1,
    activation="relu",
    pool_type="max"
)

# Extract features
x = torch.randn(4, 20, 300)  # (batch_size, seq_len, input_dim)
features = extractor(x)  # (batch_size, total_features)
```

### Parameters
- `input_dim`: Input dimension
- `hidden_dims`: List of hidden dimensions
- `kernel_sizes`: List of kernel sizes
- `dropout`: Dropout probability
- `activation`: Activation function
- `pool_type`: Pooling type ("max", "avg")

## LSTMWithAttention

LSTM with attention mechanism.

### Features
- Multiple attention types
- Bidirectional support
- Variable length sequences
- Packed sequence support

### Usage

```python
from custom_model_architectures import LSTMWithAttention

# Create LSTM with attention
lstm = LSTMWithAttention(
    input_size=256,
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.1,
    attention_type="dot"
)

# Forward pass
x = torch.randn(4, 20, 256)  # (batch_size, seq_len, input_size)
lengths = torch.tensor([20, 15, 18, 12])  # Variable lengths
output = lstm(x, lengths)  # (batch_size, seq_len, hidden_size*2)
```

### Parameters
- `input_size`: Input size
- `hidden_size`: Hidden size
- `num_layers`: Number of LSTM layers
- `bidirectional`: Whether to use bidirectional LSTM
- `dropout`: Dropout probability
- `attention_type`: Attention type ("dot", "general", "concat")

## Attention Mechanisms

### DotProductAttention

Simple dot product attention.

```python
from custom_model_architectures import DotProductAttention

attention = DotProductAttention(hidden_size=128)
output = attention(hidden_states)  # (batch_size, seq_len, hidden_size)
```

### GeneralAttention

General attention with learnable parameters.

```python
from custom_model_architectures import GeneralAttention

attention = GeneralAttention(hidden_size=128)
output = attention(hidden_states)  # (batch_size, seq_len, hidden_size)
```

### ConcatAttention

Concat attention mechanism.

```python
from custom_model_architectures import ConcatAttention

attention = ConcatAttention(hidden_size=128)
output = attention(hidden_states)  # (batch_size, seq_len, hidden_size)
```

## CNNLSTMHybrid

Hybrid CNN-LSTM model for text classification.

### Features
- CNN feature extraction
- LSTM with attention
- Classification head
- Variable length support

### Usage

```python
from custom_model_architectures import CNNLSTMHybrid

# Create hybrid model
model = CNNLSTMHybrid(
    vocab_size=15000,
    embed_dim=300,
    hidden_dims=[128, 256, 512],
    kernel_sizes=[3, 4, 5],
    lstm_hidden_size=256,
    num_classes=3,
    dropout=0.2,
    bidirectional=True
)

# Forward pass
x = torch.randint(0, 15000, (4, 50))  # (batch_size, seq_len)
lengths = torch.tensor([50, 40, 45, 35])  # Variable lengths
output = model(x, lengths)  # (batch_size, num_classes)
```

## TransformerCNN

Transformer-CNN hybrid model.

### Features
- Transformer encoding
- CNN feature extraction
- Classification head
- Attention mask support

### Usage

```python
from custom_model_architectures import TransformerCNN

# Create hybrid model
model = TransformerCNN(
    vocab_size=20000,
    d_model=256,
    n_layers=4,
    n_heads=8,
    d_ff=1024,
    cnn_hidden_dims=[128, 256, 512],
    cnn_kernel_sizes=[3, 4, 5],
    num_classes=10,
    dropout=0.15
)

# Forward pass
x = torch.randint(0, 20000, (4, 60))  # (batch_size, seq_len)
mask = torch.ones(4, 60, 60)  # Attention mask
output = model(x, mask)  # (batch_size, num_classes)
```

## MultiTaskModel

Multi-task learning model with shared encoder.

### Features
- Shared transformer encoder
- Task-specific heads
- Multiple task types (classification, regression, sequence)
- Efficient multi-task training

### Usage

```python
from custom_model_architectures import MultiTaskModel

# Define task configurations
task_configs = {
    "sentiment": {"num_classes": 3, "type": "classification"},
    "topic": {"num_classes": 5, "type": "classification"},
    "readability": {"num_classes": 1, "type": "regression"},
    "quality": {"num_classes": 1, "type": "regression"}
}

# Create multi-task model
model = MultiTaskModel(
    vocab_size=12000,
    d_model=384,
    n_layers=3,
    n_heads=6,
    d_ff=1536,
    task_configs=task_configs,
    dropout=0.1
)

# Forward pass for different tasks
x = torch.randint(0, 12000, (4, 50))  # (batch_size, seq_len)

for task_name in task_configs.keys():
    output = model(x, task_name)  # Task-specific output
    print(f"{task_name} output shape: {output.shape}")
```

## HierarchicalAttentionNetwork

Hierarchical attention network for document classification.

### Features
- Word-level attention
- Sentence-level attention
- Document-level classification
- Hierarchical structure

### Usage

```python
from custom_model_architectures import HierarchicalAttentionNetwork

# Create hierarchical model
model = HierarchicalAttentionNetwork(
    vocab_size=8000,
    embed_dim=200,
    hidden_size=128,
    num_classes=4,
    num_sentences=25,
    dropout=0.2
)

# Forward pass
x = torch.randint(0, 8000, (3, 25, 20))  # (batch, sentences, words)
output = model(x)  # (batch_size, num_classes)
```

## ResidualBlock

Residual block for deep networks.

### Features
- Residual connections
- Batch normalization
- Configurable kernel size
- Dropout regularization

### Usage

```python
from custom_model_architectures import ResidualBlock

# Create residual block
block = ResidualBlock(channels=64, kernel_size=3, dropout=0.1)

# Apply block
x = torch.randn(4, 64, 20)  # (batch_size, channels, seq_len)
output = block(x)  # (batch_size, 64, 20)
```

## DeepResidualCNN

Deep residual CNN for text processing.

### Features
- Multiple residual blocks
- Global pooling
- Classification head
- Deep architecture

### Usage

```python
from custom_model_architectures import DeepResidualCNN

# Create deep residual CNN
model = DeepResidualCNN(
    input_dim=300,
    hidden_dims=[64, 128, 256, 512],
    num_classes=6,
    num_residual_blocks=3,
    dropout=0.15
)

# Forward pass
x = torch.randn(5, 40, 300)  # (batch_size, seq_len, input_dim)
output = model(x)  # (batch_size, num_classes)
```

## ModelFactory

Factory class for easy model creation.

### Features
- Centralized model creation
- Configuration management
- Error handling
- Easy model instantiation

### Usage

```python
from custom_model_architectures import ModelFactory, MODEL_CONFIGS

# Create model using factory
model = ModelFactory.create_model("transformer", MODEL_CONFIGS["transformer"])

# Custom configuration
custom_config = {
    "vocab_size": 10000,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "d_ff": 1024,
    "dropout": 0.1
}

model = ModelFactory.create_model("transformer", custom_config)
```

## Best Practices

### 1. Model Initialization

```python
# Use proper initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)
```

### 2. Gradient Clipping

```python
# Apply gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(x)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Model Checkpointing

```python
# Save model checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}
torch.save(checkpoint, 'model_checkpoint.pth')

# Load model checkpoint
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Performance Optimization

### 1. JIT Compilation

```python
# Compile model for better performance
model = torch.jit.script(model)
```

### 2. Memory Optimization

```python
# Use gradient checkpointing for memory efficiency
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, x):
    return checkpoint(model, x)
```

### 3. Data Parallel Training

```python
# Use DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### 4. Efficient Data Loading

```python
# Use num_workers for efficient data loading
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    num_workers=4, 
    pin_memory=True
)
```

## Testing

### 1. Unit Tests

```python
import pytest
import torch

def test_model_output_shape():
    model = CustomTransformer(vocab_size=1000, d_model=256)
    x = torch.randint(0, 1000, (2, 10))
    output = model(x)
    assert output.shape == (2, 10, 256)

def test_model_gradients():
    model = CustomTransformer(vocab_size=1000, d_model=256)
    x = torch.randint(0, 1000, (2, 10), requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
```

### 2. Integration Tests

```python
def test_training_pipeline():
    model = CustomTransformer(vocab_size=1000, d_model=256)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    x = torch.randint(0, 1000, (4, 20))
    y = torch.randint(0, 5, (4,))
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output.mean(dim=1), y)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0
```

### 3. Performance Tests

```python
import time

def test_inference_speed():
    model = CustomTransformer(vocab_size=1000, d_model=256)
    model.eval()
    
    x = torch.randint(0, 1000, (4, 20))
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    assert avg_time < 0.01  # Should be fast
```

## Conclusion

These custom PyTorch model architectures provide a comprehensive toolkit for advanced NLP tasks. Each model is designed with modern best practices, proper gradient flow, and production-ready implementations. Use the ModelFactory for easy instantiation and follow the best practices for optimal performance.

For more advanced usage and customization, refer to the individual model documentation and test files. 