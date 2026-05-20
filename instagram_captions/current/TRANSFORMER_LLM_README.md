# Transformer and Large Language Model (LLM) System

## Overview

The Transformer and LLM System provides comprehensive implementations of Transformer architectures and Large Language Model capabilities. This system includes various attention mechanisms, positional encoding techniques, text generation utilities, and prompt engineering tools for building state-of-the-art language models.

## Key Features

### 🏗️ **Transformer Architectures**
- **Encoder-Decoder**: Complete Transformer for sequence-to-sequence tasks
- **Encoder-Only**: For classification, regression, and feature extraction
- **Decoder-Only**: For language modeling and text generation
- **Modular Design**: Reusable components for custom architectures

### 🧠 **Attention Mechanisms**
- **Multi-Head Attention**: Standard scaled dot-product attention
- **Scaled Dot-Product**: Efficient attention computation
- **Relative Positional**: Position-aware attention (configurable)
- **Sparse Attention**: Memory-efficient attention (configurable)

### 📍 **Positional Encoding**
- **Sinusoidal**: Standard positional encoding from "Attention Is All You Need"
- **Learned**: Trainable positional embeddings
- **Rotary (RoPE)**: Relative positional encoding for better generalization

### 🤖 **LLM Capabilities**
- **Text Generation**: Advanced generation with temperature, top-k, top-p sampling
- **Beam Search**: Optimal sequence generation
- **Prompt Engineering**: Zero-shot, few-shot, chain-of-thought templates
- **Model Analysis**: Attention visualization, perplexity computation

## Architecture

### Core Components

#### `PositionalEncoding`
```python
class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        # Sinusoidal positional encoding implementation
```

#### `MultiHeadAttention`
```python
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        # Multi-head attention with scaled dot-product
```

#### `TransformerBlock`
```python
class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        # Self-attention + feed-forward with residual connections
```

#### `TransformerModel`
```python
class TransformerModel(nn.Module):
    """Complete Transformer model (Encoder-Decoder)."""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 n_layers: int = 6, n_heads: int = 8, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1):
        # Complete encoder-decoder architecture
```

### LLM Utilities

#### `LLMGenerator`
```python
class LLMGenerator:
    """Large Language Model text generation utilities."""
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 1.0,
                top_k: int = 50, top_p: float = 0.9, do_sample: bool = True) -> str:
        # Advanced text generation with various sampling strategies
```

#### `PromptEngineer`
```python
class PromptEngineer:
    """Prompt engineering utilities for LLMs."""
    
    def create_prompt(self, instruction: str, template: str = 'zero_shot',
                     examples: Optional[List[str]] = None, role: Optional[str] = None,
                     format_type: Optional[str] = None) -> str:
        # Create formatted prompts for different scenarios
```

#### `LLMAnalyzer`
```python
class LLMAnalyzer:
    """Analysis utilities for LLM performance and behavior."""
    
    def analyze_attention_weights(self, input_text: str, layer_idx: int = 0) -> torch.Tensor:
        # Analyze attention patterns and weights
```

## Installation

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install matplotlib numpy pyyaml
```

### Framework Integration
The Transformer and LLM system integrates seamlessly with the existing PyTorch framework:

```python
# Import the system
from transformer_llm_system import (
    TransformerModel, LLMGenerator, PromptEngineer, create_transformer_model
)

# Use with existing framework components
from pytorch_primary_framework_system import PyTorchPrimaryFrameworkSystem
from custom_model_architectures import BaseModel
from loss_optimization_system import LossFunctions, Optimizers
```

## Quick Start

### 1. Basic Transformer Model
```python
import torch
from transformer_llm_system import create_transformer_model

# Create model from configuration
config = {
    'model_type': 'encoder_decoder',
    'src_vocab_size': 1000,
    'tgt_vocab_size': 1000,
    'd_model': 512,
    'n_layers': 6,
    'n_heads': 8,
    'd_ff': 2048
}

model = create_transformer_model(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 2. Text Generation
```python
from transformer_llm_system import LLMGenerator

# Create generator (requires tokenizer)
generator = LLMGenerator(model, tokenizer, device='cuda')

# Generate text
prompt = "The future of artificial intelligence"
generated = generator.generate(
    prompt,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
print(f"Generated: {generated}")
```

### 3. Prompt Engineering
```python
from transformer_llm_system import PromptEngineer

# Create prompt engineer
engineer = PromptEngineer()

# Zero-shot prompt
zero_shot = engineer.create_prompt(
    "Explain quantum computing",
    template='zero_shot'
)

# Few-shot prompt
examples = [
    ("What is AI?", "Artificial Intelligence is..."),
    ("What is ML?", "Machine Learning is...")
]
few_shot = engineer.create_few_shot_prompt(examples, "What is deep learning?")

# Chain-of-thought prompt
cot_prompt = engineer.create_prompt(
    "Solve: If a train travels 120 km in 2 hours, what is its speed?",
    template='chain_of_thought'
)
```

## Advanced Features

### 1. Attention Analysis
```python
from transformer_llm_system import LLMAnalyzer

# Create analyzer
analyzer = LLMAnalyzer(model, tokenizer)

# Analyze attention weights
attention_weights = analyzer.analyze_attention_weights(
    "The cat sat on the mat",
    layer_idx=0
)

# Compute perplexity
perplexity = analyzer.compute_perplexity("Sample text for evaluation")
```

### 2. Model Comparison
```python
# Compare different model sizes
configs = {
    'small': {'d_model': 256, 'n_layers': 4, 'n_heads': 8},
    'medium': {'d_model': 512, 'n_layers': 6, 'n_heads': 8},
    'large': {'d_model': 768, 'n_layers': 12, 'n_heads': 12}
}

models = {}
for name, config in configs.items():
    models[name] = create_transformer_model(config)
    params = sum(p.numel() for p in models[name].parameters())
    print(f"{name}: {params:,} parameters")
```

### 3. Performance Optimization
```python
# Mixed precision training
from torch.cuda.amp import autocast

with autocast():
    output = model(src, tgt)

# Torch compile optimization
compiled_model = torch.compile(model)
output = compiled_model(src, tgt)
```

## Configuration

### YAML Configuration
```yaml
transformer_llm_system:
  global:
    default_model_type: "encoder_decoder"
    enable_attention_analysis: true
    enable_generation: true
    device: "auto"

  architectures:
    encoder_decoder:
      enabled: true
      d_model: 512
      n_layers: 6
      n_heads: 8
      d_ff: 2048
      max_len: 5000
      dropout: 0.1

  generation:
    default_max_length: 100
    default_temperature: 1.0
    default_top_k: 50
    default_top_p: 0.9
    beam_search:
      enabled: true
      beam_size: 5
```

### Model Configurations
```python
# Pre-defined model sizes
model_configs = {
    'small': {
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 8,
        'd_ff': 1024
    },
    'medium': {
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 2048
    },
    'large': {
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'd_ff': 3072
    }
}
```

## Integration Examples

### 1. With Custom Model Architectures
```python
from custom_model_architectures import BaseModel
from transformer_llm_system import TransformerModel

class CustomTransformerModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TransformerModel(
            src_vocab_size=config['src_vocab_size'],
            tgt_vocab_size=config['tgt_vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff']
        )
    
    def forward(self, src, tgt):
        return self.transformer(src, tgt)
```

### 2. With Loss Functions and Optimizers
```python
from loss_optimization_system import LossFunctions, Optimizers

# Use cross-entropy loss for language modeling
loss_fn = LossFunctions.cross_entropy_loss

# Use Adam optimizer with learning rate scheduling
optimizer = Optimizers.create_optimizer(
    model.parameters(),
    optimizer_type='adam',
    lr=1e-4,
    weight_decay=1e-5
)
```

### 3. With Framework System
```python
from pytorch_primary_framework_system import PyTorchPrimaryFrameworkSystem

# Initialize framework with Transformer model
framework = PyTorchPrimaryFrameworkSystem()
framework.set_model(model)
framework.set_optimizer(optimizer)
framework.set_loss_function(loss_fn)

# Train the model
framework.train(train_dataloader, num_epochs=10)
```

## Examples and Demonstrations

### Running Examples
```bash
# Basic demonstration
python transformer_llm_system.py

# Advanced examples
python transformer_llm_advanced_examples.py
```

### Example Output
```
=== Transformer and LLM System Demonstration ===

Created Transformer model with 1,234,567 parameters
Output shape: torch.Size([2, 10, 1000])
Positional encoding output shape: torch.Size([50, 256])
Attention output shape: torch.Size([2, 10, 256])
Attention weights shape: torch.Size([2, 8, 10, 10])

=== Demonstration Complete ===
```

## Performance Considerations

### Memory Optimization
- **Gradient Checkpointing**: Enable for large models
- **Mixed Precision**: Use FP16 for training
- **Attention Optimization**: Flash Attention or xFormers
- **Model Parallelism**: For very large models

### Speed Optimization
- **Torch Compile**: Use `torch.compile()` for inference
- **Batch Processing**: Optimize batch sizes
- **Attention Caching**: Cache attention for generation
- **Quantization**: INT8 quantization for deployment

### Scalability
- **Distributed Training**: Multi-GPU training
- **Model Sharding**: Split large models across devices
- **Pipeline Parallelism**: For very deep models
- **Data Parallelism**: For large datasets

## Best Practices

### 1. Model Architecture
- Choose appropriate model size for your task
- Use pre-trained models when possible
- Consider computational constraints
- Validate architecture choices

### 2. Training
- Use appropriate learning rates
- Implement proper regularization
- Monitor attention patterns
- Use validation perplexity

### 3. Generation
- Tune temperature and sampling parameters
- Use appropriate prompts
- Implement safety measures
- Monitor generation quality

### 4. Analysis
- Analyze attention weights
- Monitor model behavior
- Track performance metrics
- Validate outputs

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

#### 2. Generation Issues
```python
# Adjust generation parameters
generated = generator.generate(
    prompt,
    temperature=0.7,  # Lower for more focused output
    top_k=20,        # Reduce for more diverse output
    top_p=0.8        # Adjust nucleus sampling
)
```

#### 3. Attention Issues
```python
# Check attention mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
```

## Future Enhancements

### Planned Features
- **Flash Attention 2.0**: Latest attention optimization
- **Grouped Query Attention**: Memory-efficient attention
- **Sliding Window Attention**: For long sequences
- **Multi-Query Attention**: For faster inference

### Research Integration
- **Retrieval-Augmented Generation**: RAG capabilities
- **Chain-of-Thought**: Advanced reasoning
- **Few-Shot Learning**: Improved prompting
- **Instruction Tuning**: Better task adaptation

## Contributing

### Development Setup
```bash
git clone <repository>
cd transformer-llm-system
pip install -e .
```

### Testing
```bash
python -m pytest tests/
python transformer_llm_system.py
python transformer_llm_advanced_examples.py
```

### Code Style
- Follow PEP 8 guidelines
- Add type hints
- Include docstrings
- Write unit tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Original Transformer Paper**: "Attention Is All You Need"
- **Hugging Face**: For inspiration and best practices
- **PyTorch Team**: For the excellent framework
- **Research Community**: For continuous innovations

## Support

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Attention Mechanisms](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

### Community
- GitHub Issues: For bug reports and feature requests
- Discussions: For questions and ideas
- Contributing: For development guidelines

### Contact
- Email: support@example.com
- Discord: Join our community server
- Twitter: Follow for updates

---

**Note**: This system is designed to be modular and extensible. Feel free to customize and extend it for your specific use cases!


