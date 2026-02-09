# Advanced Attention Mechanisms and Positional Encodings

This module provides comprehensive implementations of various attention mechanisms and positional encoding methods for transformer models, specifically designed for SEO tasks.

## Table of Contents

1. [Overview](#overview)
2. [Attention Mechanisms](#attention-mechanisms)
3. [Positional Encodings](#positional-encodings)
4. [Usage Examples](#usage-examples)
5. [Configuration](#configuration)
6. [Performance Considerations](#performance-considerations)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

The attention mechanisms and positional encodings module provides:

- **Multiple Attention Types**: Standard, Local, Sparse, and Relative Position attention
- **Various Positional Encodings**: Sinusoidal, Learned, Relative, and Rotary positional encodings
- **Factory Pattern**: Easy creation and configuration of attention mechanisms and positional encodings
- **SEO-Optimized**: Designed specifically for SEO content analysis and generation
- **Performance Optimized**: Efficient implementations with GPU support

## Attention Mechanisms

### 1. Standard Multi-Head Attention

The classic scaled dot-product attention mechanism as described in "Attention Is All You Need".

```python
from attention_mechanisms import MultiHeadAttention

attention = MultiHeadAttention(
    d_model=512,
    num_heads=8,
    dropout=0.1
)

# Forward pass
output, attention_weights = attention(
    query=x, key=x, value=x,
    mask=attention_mask,
    need_weights=True
)
```

**Features:**
- Scaled dot-product attention
- Multi-head mechanism
- Proper weight initialization
- Support for attention masks
- Optional attention weight return

### 2. Local Attention

Efficient attention mechanism that only attends to a local window around each position.

```python
from attention_mechanisms import LocalAttention

attention = LocalAttention(
    d_model=512,
    num_heads=8,
    window_size=128,  # Local window size
    dropout=0.1
)

output = attention(x, mask=attention_mask)
```

**Features:**
- Configurable window size
- Reduced computational complexity O(n × w) where w is window size
- Suitable for long sequences
- Maintains local context information

### 3. Sparse Attention

Attention mechanism using landmark tokens for efficient processing of long sequences.

```python
from attention_mechanisms import SparseAttention

attention = SparseAttention(
    d_model=512,
    num_heads=8,
    num_landmarks=64,  # Number of landmark tokens
    dropout=0.1
)

output = attention(x, mask=attention_mask)
```

**Features:**
- Uses landmark tokens for global attention
- Reduced memory usage
- Suitable for very long sequences
- Maintains global context through landmarks

### 4. Relative Position Attention

Attention mechanism that incorporates relative position information.

```python
from attention_mechanisms import AttentionWithRelativePositions

attention = AttentionWithRelativePositions(
    d_model=512,
    num_heads=8,
    max_relative_position=32,  # Maximum relative distance
    dropout=0.1
)

output = attention(x, mask=attention_mask)
```

**Features:**
- Incorporates relative position embeddings
- Better handling of sequence relationships
- Configurable maximum relative distance
- Improved performance on tasks requiring position awareness

## Positional Encodings

### 1. Sinusoidal Positional Encoding

The original positional encoding from "Attention Is All You Need".

```python
from attention_mechanisms import PositionalEncoding

pos_encoding = PositionalEncoding(
    d_model=512,
    max_len=5000,
    dropout=0.1
)

# Apply to input [seq_len, batch_size, d_model]
output = pos_encoding(x)
```

**Features:**
- Fixed sinusoidal patterns
- No learnable parameters
- Generalizes to unseen positions
- Standard in transformer implementations

### 2. Learned Positional Encoding

Trainable positional embeddings.

```python
from attention_mechanisms import LearnedPositionalEncoding

pos_encoding = LearnedPositionalEncoding(
    d_model=512,
    max_len=5000,
    dropout=0.1
)

output = pos_encoding(x)
```

**Features:**
- Learnable parameters
- Can adapt to specific tasks
- Limited to training sequence length
- Better performance on training data

### 3. Relative Positional Encoding

Positional encoding based on relative distances between positions.

```python
from attention_mechanisms import RelativePositionalEncoding

pos_encoding = RelativePositionalEncoding(
    d_model=512,
    max_relative_position=32,
    dropout=0.1
)

# Apply to input [batch_size, seq_len, d_model]
output = pos_encoding(x, seq_len)
```

**Features:**
- Based on relative distances
- Better for tasks requiring position relationships
- Configurable maximum relative distance
- Suitable for variable length sequences

### 4. Rotary Positional Encoding (RoPE)

Rotary positional encoding for better position modeling.

```python
from attention_mechanisms import RotaryPositionalEncoding

pos_encoding = RotaryPositionalEncoding(
    d_model=512,
    max_len=5000,
    dropout=0.1
)

output = pos_encoding(x, seq_len)
```

**Features:**
- Applies rotation to embeddings
- Better position modeling
- Requires even d_model
- State-of-the-art performance

## Usage Examples

### Basic Usage

```python
from attention_mechanisms import AttentionFactory, PositionalEncodingFactory
from transformer_models import TransformerConfig, SEOSpecificTransformer

# Create configuration
config = TransformerConfig(
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    attention_type="standard",
    positional_encoding_type="sinusoidal",
    vocab_size=10000
)

# Create transformer
transformer = SEOSpecificTransformer(config)

# Forward pass
outputs = transformer(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True
)
```

### Advanced Configuration

```python
# Local attention with learned positional encoding
config = TransformerConfig(
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    attention_type="local",
    attention_window_size=128,
    positional_encoding_type="learned",
    vocab_size=10000
)

# Sparse attention with relative positional encoding
config = TransformerConfig(
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    attention_type="sparse",
    attention_num_landmarks=64,
    positional_encoding_type="relative",
    positional_encoding_max_relative_position=32,
    vocab_size=10000
)
```

### Using Factories

```python
from attention_mechanisms import AttentionFactory, PositionalEncodingFactory

# Create attention mechanism
attention = AttentionFactory.create_attention(
    attention_type="local",
    d_model=512,
    num_heads=8,
    window_size=128,
    dropout=0.1
)

# Create positional encoding
pos_encoding = PositionalEncodingFactory.create_positional_encoding(
    encoding_type="rotary",
    d_model=512,
    max_len=5000,
    dropout=0.1
)
```

## Configuration

### TransformerConfig Parameters

```python
@dataclass
class TransformerConfig:
    # Basic model parameters
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 30522
    
    # Attention mechanism configuration
    attention_type: str = "standard"  # standard, local, sparse, relative
    attention_window_size: int = 128  # for local attention
    attention_num_landmarks: int = 64  # for sparse attention
    attention_max_relative_position: int = 32  # for relative attention
    
    # Positional encoding configuration
    positional_encoding_type: str = "sinusoidal"  # sinusoidal, learned, relative, rotary
    positional_encoding_max_len: int = 5000
    positional_encoding_max_relative_position: int = 32
    
    # Additional features
    use_causal_mask: bool = False
    use_padding_mask: bool = True
    return_attention_weights: bool = False
```

### Attention Types Comparison

| Attention Type | Complexity | Memory | Use Case |
|----------------|------------|--------|----------|
| Standard | O(n²) | O(n²) | General purpose |
| Local | O(n × w) | O(n × w) | Long sequences |
| Sparse | O(n × l) | O(n × l) | Very long sequences |
| Relative | O(n²) | O(n²) | Position-aware tasks |

### Positional Encoding Comparison

| Encoding Type | Learnable | Generalization | Performance |
|---------------|-----------|----------------|-------------|
| Sinusoidal | No | Yes | Good |
| Learned | Yes | No | Better |
| Relative | Yes | Yes | Good |
| Rotary | No | Yes | Best |

## Performance Considerations

### Memory Usage

- **Standard Attention**: O(n²) memory for attention weights
- **Local Attention**: O(n × w) memory, where w is window size
- **Sparse Attention**: O(n × l) memory, where l is number of landmarks
- **Relative Attention**: O(n²) memory but with relative position embeddings

### Computational Complexity

- **Standard Attention**: O(n² × d) where d is model dimension
- **Local Attention**: O(n × w × d) where w is window size
- **Sparse Attention**: O(n × l × d) where l is number of landmarks
- **Relative Attention**: O(n² × d) with additional relative position computation

### GPU Optimization

```python
# Enable mixed precision for better performance
config = TransformerConfig(
    use_mixed_precision=True,
    # ... other parameters
)

# Use gradient checkpointing for memory efficiency
config = TransformerConfig(
    gradient_checkpointing=True,
    # ... other parameters
)
```

## Best Practices

### 1. Choosing Attention Type

- **Standard**: Use for general tasks with moderate sequence lengths
- **Local**: Use for long sequences where local context is sufficient
- **Sparse**: Use for very long sequences requiring global context
- **Relative**: Use for tasks requiring position awareness

### 2. Choosing Positional Encoding

- **Sinusoidal**: Default choice, good generalization
- **Learned**: Use when training data has consistent sequence lengths
- **Relative**: Use for tasks requiring relative position information
- **Rotary**: Use for state-of-the-art performance

### 3. Configuration Guidelines

```python
# For SEO content analysis (moderate length)
config = TransformerConfig(
    attention_type="standard",
    positional_encoding_type="sinusoidal",
    hidden_size=512,
    num_layers=6
)

# For long document processing
config = TransformerConfig(
    attention_type="local",
    attention_window_size=256,
    positional_encoding_type="learned",
    hidden_size=512,
    num_layers=8
)

# For very long sequences
config = TransformerConfig(
    attention_type="sparse",
    attention_num_landmarks=128,
    positional_encoding_type="relative",
    hidden_size=512,
    num_layers=12
)
```

### 4. Memory Management

```python
# Use gradient checkpointing for large models
config = TransformerConfig(
    gradient_checkpointing=True,
    # ... other parameters
)

# Use mixed precision for faster training
config = TransformerConfig(
    use_mixed_precision=True,
    # ... other parameters
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use local or sparse attention
   - Enable gradient checkpointing
   - Use mixed precision

2. **Poor Performance**
   - Check attention type suitability for task
   - Verify positional encoding choice
   - Ensure proper initialization
   - Monitor attention weights

3. **Training Instability**
   - Reduce learning rate
   - Use proper weight initialization
   - Check gradient clipping
   - Monitor attention weight distributions

### Debugging Attention

```python
# Visualize attention weights
outputs = transformer(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True
)

attention_weights = outputs['attentions']
# attention_weights is a tuple of [batch_size, num_heads, seq_len, seq_len] tensors

# Analyze attention patterns
for layer_idx, layer_attention in enumerate(attention_weights):
    print(f"Layer {layer_idx}:")
    print(f"  Mean attention: {layer_attention.mean():.4f}")
    print(f"  Std attention: {layer_attention.std():.4f}")
    print(f"  Max attention: {layer_attention.max():.4f}")
```

### Performance Monitoring

```python
import time

# Measure inference time
start_time = time.time()
outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
end_time = time.time()

print(f"Inference time: {end_time - start_time:.4f}s")

# Measure memory usage (if using CUDA)
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

## File Structure

```
attention_mechanisms.py          # Main attention mechanisms implementation
transformer_models.py            # Updated transformer models with attention integration
example_attention_mechanisms.py  # Comprehensive usage examples
README_ATTENTION_MECHANISMS.md   # This documentation
```

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0 (for visualization)
- Seaborn >= 0.11.0 (for visualization)

## Contributing

When adding new attention mechanisms or positional encodings:

1. Implement the new mechanism in `attention_mechanisms.py`
2. Add factory support in `AttentionFactory` or `PositionalEncodingFactory`
3. Update configuration in `TransformerConfig`
4. Add tests and examples
5. Update this documentation

## License

This module is part of the SEO service and follows the same licensing terms. 