# Transformer Models and LLM Integration for SEO Service

This module provides comprehensive transformer architectures and Large Language Model (LLM) integration specifically designed for SEO tasks. It includes custom transformer implementations, multi-task learning capabilities, and advanced LLM integration for text generation and analysis.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Transformer Models](#transformer-models)
7. [LLM Integration](#llm-integration)
8. [Multi-Task Learning](#multi-task-learning)
9. [Advanced Features](#advanced-features)
10. [Usage Examples](#usage-examples)
11. [Best Practices](#best-practices)
12. [Performance Optimization](#performance-optimization)
13. [Troubleshooting](#troubleshooting)
14. [API Reference](#api-reference)

## Overview

The transformer models module provides:

- **Custom Transformer Architectures**: SEO-specific transformer models with advanced attention mechanisms
- **Multi-Task Learning**: Support for multiple SEO objectives simultaneously
- **LLM Integration**: Integration with popular language models for text generation and analysis
- **Advanced Attention**: Multiple attention types (standard, cosine, scaled dot-product)
- **Production Ready**: Optimized for deployment with PyTorch best practices

## Features

### Core Features
- ✅ Custom SEO-specific transformer architectures
- ✅ Multi-head attention with different attention types
- ✅ Multi-task learning for multiple SEO objectives
- ✅ LLM integration (GPT-2, T5, BERT, etc.)
- ✅ Advanced text generation and analysis
- ✅ Embedding extraction and similarity computation
- ✅ Production-ready model management
- ✅ GPU optimization and mixed precision training

### Advanced Features
- ✅ Relative position embeddings
- ✅ Gradient checkpointing for memory efficiency
- ✅ Mixed precision training
- ✅ Model saving and loading
- ✅ Pretrained model integration
- ✅ Custom attention mechanisms
- ✅ SEO-specific content analysis

## Architecture

### Transformer Components

```
Transformer Models
├── SEOSpecificTransformer
│   ├── MultiHeadAttention
│   ├── TransformerBlock
│   ├── Position Embeddings
│   └── Pooler
├── MultiTaskTransformer
│   ├── Shared Encoder
│   └── Task-Specific Heads
└── LLMIntegration
    ├── Text Generation
    ├── Content Analysis
    └── Embedding Extraction
```

### Key Classes

- **`SEOSpecificTransformer`**: Custom transformer for SEO tasks
- **`MultiTaskTransformer`**: Multi-task transformer for multiple objectives
- **`LLMIntegration`**: Integration with large language models
- **`TransformerManager`**: Model management and utilities
- **`MultiHeadAttention`**: Advanced attention mechanisms

## Installation

### Prerequisites

```bash
# Install PyTorch
pip install torch torchvision torchaudio

# Install Transformers
pip install transformers

# Install additional dependencies
pip install numpy scipy scikit-learn
```

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd blatam-academy/agents/backend/onyx/server/features/seo

# Install requirements
pip install -r requirements.deep_learning.txt
```

## Quick Start

### Basic Transformer Usage

```python
from transformer_models import TransformerConfig, SEOSpecificTransformer

# Create configuration
config = TransformerConfig(
    hidden_size=768,
    num_layers=6,
    num_heads=12,
    intermediate_size=3072
)

# Create transformer
transformer = SEOSpecificTransformer(config)

# Forward pass
input_ids = torch.randint(0, 1000, (4, 128))
attention_mask = torch.ones(4, 128)
outputs = transformer(input_ids, attention_mask)

print(f"Hidden state: {outputs['last_hidden_state'].shape}")
print(f"Pooled output: {outputs['pooler_output'].shape}")
```

### LLM Integration

```python
from transformer_models import LLMConfig, LLMIntegration

# Create LLM configuration
config = LLMConfig(
    model_type="gpt2",
    model_name="gpt2-medium",
    max_length=1024
)

# Create LLM integration
llm = LLMIntegration(config)

# Generate text
prompt = "SEO optimization tips:"
generated_text = llm.generate_text(prompt, max_length=100)
print(f"Generated: {generated_text}")

# Analyze SEO content
content = "Your SEO content here..."
analysis = llm.analyze_seo_content(content)
print(f"Analysis: {analysis}")
```

## Transformer Models

### SEOSpecificTransformer

A custom transformer architecture designed specifically for SEO tasks.

```python
from transformer_models import TransformerConfig, SEOSpecificTransformer

config = TransformerConfig(
    model_type="custom",
    hidden_size=768,
    num_layers=6,
    num_heads=12,
    intermediate_size=3072,
    dropout_rate=0.1,
    attention_dropout=0.1,
    max_position_embeddings=512
)

transformer = SEOSpecificTransformer(config)
```

#### Configuration Options

- **`hidden_size`**: Dimension of hidden layers (default: 768)
- **`num_layers`**: Number of transformer layers (default: 12)
- **`num_heads`**: Number of attention heads (default: 12)
- **`intermediate_size`**: Size of feed-forward network (default: 3072)
- **`dropout_rate`**: Dropout probability (default: 0.1)
- **`attention_dropout`**: Attention dropout probability (default: 0.1)
- **`max_position_embeddings`**: Maximum sequence length (default: 512)

### MultiTaskTransformer

A transformer that can handle multiple SEO tasks simultaneously.

```python
from transformer_models import MultiTaskTransformer

task_configs = {
    'seo_score': {
        'type': 'regression',
        'output_size': 1,
        'loss_weight': 1.0
    },
    'content_quality': {
        'type': 'classification',
        'num_classes': 5,
        'loss_weight': 1.0
    },
    'keyword_density': {
        'type': 'regression',
        'output_size': 1,
        'loss_weight': 0.5
    }
}

multi_task = MultiTaskTransformer(config, task_configs)
```

#### Task Types

- **`regression`**: Continuous value prediction
- **`classification`**: Categorical prediction
- **`ranking`**: Ranking prediction

## LLM Integration

### Supported Models

- **GPT-2**: Text generation and completion
- **BERT**: Text understanding and classification
- **T5**: Text-to-text generation
- **RoBERTa**: Robust BERT implementation
- **DistilBERT**: Distilled BERT for efficiency

### Text Generation

```python
from transformer_models import LLMConfig, LLMIntegration

config = LLMConfig(
    model_type="gpt2",
    model_name="gpt2-medium",
    temperature=0.7,
    top_p=0.9,
    top_k=50
)

llm = LLMIntegration(config)

# Generate text
prompt = "SEO optimization tips for better rankings:"
generated = llm.generate_text(prompt, max_length=200)
print(generated)
```

### Content Analysis

```python
# Analyze SEO content
content = """
This is a sample SEO content about digital marketing strategies.
It includes relevant keywords and provides valuable information.
"""

analysis = llm.analyze_seo_content(content)
print(analysis)
```

### Embedding Extraction

```python
# Get embeddings
text = "Your text here"
embeddings = llm.get_embeddings(text)
print(f"Embedding shape: {embeddings.shape}")
```

## Multi-Task Learning

### Creating Multi-Task Models

```python
from transformer_models import MultiTaskTransformer

# Define tasks
task_configs = {
    'sentiment': {
        'type': 'classification',
        'num_classes': 3,
        'loss_weight': 1.0
    },
    'topic': {
        'type': 'classification',
        'num_classes': 10,
        'loss_weight': 1.0
    },
    'readability': {
        'type': 'regression',
        'output_size': 1,
        'loss_weight': 0.5
    }
}

# Create multi-task transformer
multi_task = MultiTaskTransformer(config, task_configs)

# Forward pass
outputs = multi_task(input_ids, attention_mask)
for task_name, output in outputs.items():
    print(f"{task_name}: {output.shape}")
```

### Training Multi-Task Models

```python
# Define loss functions for each task
loss_functions = {
    'sentiment': nn.CrossEntropyLoss(),
    'topic': nn.CrossEntropyLoss(),
    'readability': nn.MSELoss()
}

# Training loop
for batch in dataloader:
    outputs = multi_task(batch['input_ids'], batch['attention_mask'])
    
    total_loss = 0
    for task_name, output in outputs.items():
        loss = loss_functions[task_name](output, batch[f'{task_name}_labels'])
        total_loss += task_configs[task_name]['loss_weight'] * loss
    
    total_loss.backward()
    optimizer.step()
```

## Advanced Features

### Custom Attention Mechanisms

```python
from transformer_models import MultiHeadAttention

# Standard attention
attention = MultiHeadAttention(
    d_model=768,
    num_heads=12,
    attention_type="standard"
)

# Cosine attention
cosine_attention = MultiHeadAttention(
    d_model=768,
    num_heads=12,
    attention_type="cosine"
)

# Scaled dot-product attention
scaled_attention = MultiHeadAttention(
    d_model=768,
    num_heads=12,
    attention_type="scaled_dot_product"
)
```

### Relative Position Embeddings

```python
# Enable relative positions
attention = MultiHeadAttention(
    d_model=768,
    num_heads=12,
    use_relative_positions=True
)
```

### Model Management

```python
from transformer_models import TransformerManager

manager = TransformerManager()

# Create models
transformer = manager.create_transformer(config, "my-transformer")
multi_task = manager.create_multi_task_transformer(config, task_configs, "my-multi-task")
llm = manager.create_llm_integration(llm_config, "my-llm")

# Save models
manager.save_model("my-transformer", "models/transformer")

# Load models
loaded_transformer = manager.load_model("my-transformer", "models/transformer")
```

## Usage Examples

### Complete Training Pipeline

```python
from deep_learning_framework import DeepLearningFramework, TrainingConfig
from transformer_models import TransformerConfig

# Create framework
config = TrainingConfig(
    model_type="transformer",
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=10
)

framework = DeepLearningFramework(config)

# Create transformer
transformer_config = TransformerConfig(
    hidden_size=768,
    num_layers=6,
    num_heads=12
)

transformer = framework.create_transformer(transformer_config, "seo-transformer")

# Train model
training_results = await framework.train_model(train_data, val_data)
print(f"Training completed: {training_results}")
```

### LLM Content Generation

```python
# Create LLM integration
llm_config = LLMConfig(
    model_type="gpt2",
    model_name="gpt2-medium",
    temperature=0.8
)

llm = framework.create_llm_integration(llm_config, "content-generator")

# Generate SEO content
prompts = [
    "Write an SEO-optimized blog post about digital marketing:",
    "Create meta descriptions for an e-commerce website:",
    "Generate title tags for a blog about technology:"
]

for prompt in prompts:
    content = framework.generate_text_with_llm("content-generator", prompt)
    print(f"Generated: {content}")
```

### SEO Analysis Pipeline

```python
# Analyze multiple pieces of content
contents = [
    "Your first SEO content here...",
    "Your second SEO content here...",
    "Your third SEO content here..."
]

analyses = []
for content in contents:
    analysis = framework.analyze_seo_content_with_llm("content-generator", content)
    analyses.append(analysis)

# Process analyses
for i, analysis in enumerate(analyses):
    print(f"Content {i+1} analysis: {analysis['analysis']}")
```

## Best Practices

### Model Configuration

1. **Choose appropriate model size**:
   - Small: 256-512 hidden size for quick experiments
   - Medium: 768 hidden size for production
   - Large: 1024+ hidden size for maximum performance

2. **Optimize attention heads**:
   - Ensure `hidden_size % num_heads == 0`
   - Use 8-16 heads for most applications

3. **Set dropout appropriately**:
   - 0.1 for most cases
   - 0.2-0.3 for overfitting prevention
   - 0.05 for large datasets

### Training Best Practices

1. **Use mixed precision training**:
   ```python
   config = TransformerConfig(use_mixed_precision=True)
   ```

2. **Enable gradient checkpointing for large models**:
   ```python
   config = TransformerConfig(gradient_checkpointing=True)
   ```

3. **Use appropriate learning rates**:
   - 1e-5 to 5e-5 for fine-tuning
   - 1e-4 to 1e-3 for training from scratch

### LLM Best Practices

1. **Choose appropriate temperature**:
   - 0.3-0.5 for focused, consistent output
   - 0.7-0.9 for creative, diverse output

2. **Use top-p and top-k sampling**:
   ```python
   config = LLMConfig(
       top_p=0.9,
       top_k=50,
       temperature=0.7
   )
   ```

3. **Set appropriate max_length**:
   - 512-1024 for most tasks
   - 2048+ for long-form content

## Performance Optimization

### GPU Optimization

```python
# Enable mixed precision
config = TransformerConfig(use_mixed_precision=True)

# Use gradient checkpointing
config = TransformerConfig(gradient_checkpointing=True)

# Optimize memory usage
config = TransformerConfig(
    use_cache=False,  # Disable KV cache for training
    max_position_embeddings=512  # Limit sequence length
)
```

### Memory Management

```python
# Clear cache periodically
import torch
torch.cuda.empty_cache()

# Use gradient accumulation
config = TrainingConfig(gradient_accumulation_steps=4)

# Monitor memory usage
from pytorch_configuration import PyTorchConfiguration
pytorch_config = PyTorchConfiguration()
memory_info = pytorch_config.get_memory_info()
print(f"GPU memory: {memory_info}")
```

### Batch Processing

```python
# Optimize batch size
config = TrainingConfig(
    batch_size=16,  # Adjust based on GPU memory
    use_mixed_precision=True
)

# Use data loading optimization
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   ```python
   # Reduce batch size
   config = TrainingConfig(batch_size=4)
   
   # Enable gradient checkpointing
   config = TransformerConfig(gradient_checkpointing=True)
   
   # Use mixed precision
   config = TransformerConfig(use_mixed_precision=True)
   ```

2. **Slow Training**:
   ```python
   # Increase batch size if memory allows
   config = TrainingConfig(batch_size=32)
   
   # Use more workers
   dataloader = DataLoader(dataset, num_workers=8)
   
   # Enable mixed precision
   config = TransformerConfig(use_mixed_precision=True)
   ```

3. **Model Not Converging**:
   ```python
   # Adjust learning rate
   config = TrainingConfig(learning_rate=1e-5)
   
   # Increase dropout
   config = TransformerConfig(dropout_rate=0.2)
   
   # Use learning rate scheduling
   config = TrainingConfig(scheduler_type="cosine_with_warmup")
   ```

### Debugging

```python
# Enable autograd monitoring
from autograd_utils import AutogradMonitor
monitor = AutogradMonitor()
monitor.enable()

# Check gradients
gradient_info = monitor.get_gradient_info()
print(f"Gradient norm: {gradient_info['gradient_norm']}")

# Monitor memory
from pytorch_configuration import PyTorchConfiguration
pytorch_config = PyTorchConfiguration()
memory_info = pytorch_config.get_memory_info()
print(f"Memory usage: {memory_info}")
```

## API Reference

### TransformerConfig

```python
@dataclass
class TransformerConfig:
    model_type: str = "bert"
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
```

### LLMConfig

```python
@dataclass
class LLMConfig:
    model_type: str = "gpt2"
    model_name: str = "gpt2-medium"
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1
    use_cache: bool = True
    use_mixed_precision: bool = True
```

### SEOSpecificTransformer

```python
class SEOSpecificTransformer(nn.Module):
    def __init__(self, config: TransformerConfig)
    def forward(self, input_ids, attention_mask=None, ...) -> Dict[str, torch.Tensor]
```

### MultiTaskTransformer

```python
class MultiTaskTransformer(nn.Module):
    def __init__(self, config: TransformerConfig, task_configs: Dict[str, Dict[str, Any]])
    def forward(self, input_ids, attention_mask=None, task_name=None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]
```

### LLMIntegration

```python
class LLMIntegration:
    def __init__(self, config: LLMConfig)
    def generate_text(self, prompt: str, max_length: Optional[int] = None) -> str
    def get_embeddings(self, text: str) -> torch.Tensor
    def analyze_seo_content(self, content: str) -> Dict[str, Any]
```

### TransformerManager

```python
class TransformerManager:
    def create_transformer(self, config: TransformerConfig, model_name: str) -> SEOSpecificTransformer
    def create_multi_task_transformer(self, config: TransformerConfig, task_configs: Dict[str, Dict[str, Any]], model_name: str) -> MultiTaskTransformer
    def create_llm_integration(self, config: LLMConfig, model_name: str) -> LLMIntegration
    def save_model(self, model_name: str, save_path: str)
    def load_model(self, model_name: str, load_path: str) -> nn.Module
```

## File Structure

```
transformer_models/
├── transformer_models.py          # Main transformer and LLM implementation
├── example_transformer_models.py  # Usage examples
├── README_TRANSFORMER_MODELS.md   # This documentation
└── tests/
    └── test_transformer_models.py # Unit tests
```

## Contributing

1. Follow the existing code style and conventions
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure all tests pass before submitting

## License

This module is part of the Blatam Academy SEO Service and follows the same licensing terms.

---

For more information, see the main [README_DEEP_LEARNING.md](README_DEEP_LEARNING.md) file. 