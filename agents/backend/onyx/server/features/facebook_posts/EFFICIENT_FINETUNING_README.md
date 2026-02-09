# 🚀 Efficient Fine-tuning & Advanced Attention System

A comprehensive implementation of efficient fine-tuning techniques (LoRA, P-tuning, Adapters) with advanced attention mechanisms and positional encodings for PyTorch and Transformers.

## 🎯 Features

### Efficient Fine-tuning Methods
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning with low-rank matrices
- **P-tuning v2**: Prefix-based tuning with virtual tokens
- **Adapter Layers**: Bottleneck layers for efficient adaptation
- **BitFit**: Bias-only fine-tuning
- **Custom Configurations**: Flexible configuration for all methods

### Advanced Attention Mechanisms
- **Multi-Head Attention**: Correctly implemented with all modern features
- **Rotary Positional Embeddings (RoPE)**: Advanced positional encoding
- **Flash Attention**: Memory-efficient attention computation
- **Relative Position Bias**: T5-style relative positioning
- **Cross-Attention**: For encoder-decoder architectures
- **Causal and Bidirectional Masks**: Flexible attention masking

### Positional Encodings
- **Rotary Embeddings (RoPE)**: State-of-the-art positional encoding
- **Absolute Positional Embeddings**: Sinusoidal and learned embeddings
- **Relative Positional Bias**: T5-style relative positioning
- **Configurable Parameters**: Flexible configuration options

## 📁 File Structure

```
efficient_finetuning_system.py    # Main efficient fine-tuning implementation
attention_mechanisms.py           # Advanced attention and positional encodings
efficient_attention_demo.py       # Comprehensive demonstration script
requirements_efficient_finetuning.txt  # Dependencies
EFFICIENT_FINETUNING_README.md   # This documentation
```

## 🛠️ Installation

### Basic Installation

```bash
# Install basic requirements
pip install torch transformers accelerate peft

# Install all requirements
pip install -r requirements_efficient_finetuning.txt
```

### Advanced Installation (Optional)

```bash
# Flash Attention (requires specific CUDA setup)
pip install flash-attn --no-build-isolation

# xFormers for memory efficiency
pip install xformers

# BitsAndBytes for quantization
pip install bitsandbytes
```

## 🚀 Quick Start

### 1. LoRA Fine-tuning

```python
from efficient_finetuning_system import create_efficient_finetuner
from transformers import TrainingArguments

# Create model with LoRA
model, fine_tuner = create_efficient_finetuner(
    model_name="gpt2",
    method="lora",
    rank=16,
    alpha=32.0,
    target_modules=["c_attn", "c_proj"]
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_output",
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    fp16=True
)

# Train
trainer = EfficientTrainer(model, fine_tuner, training_args)
# trainer.train(train_dataset)
```

### 2. P-tuning v2

```python
from efficient_finetuning_system import PtuningConfig, EfficientFineTuner
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# P-tuning configuration
config = PtuningConfig(
    num_virtual_tokens=20,
    token_dim=768,
    prefix_projection=True
)

# Apply P-tuning
fine_tuner = EfficientFineTuner(model, method="ptuning", config=config)
```

### 3. Advanced Attention

```python
from attention_mechanisms import AttentionConfig, MultiHeadAttention

# Attention configuration
config = AttentionConfig(
    hidden_size=768,
    num_heads=12,
    use_rotary=True,
    use_flash_attention=True,
    use_relative_position=True
)

# Create attention layer
attention = MultiHeadAttention(config)

# Forward pass
outputs = attention(
    hidden_states,
    attention_mask=attention_mask,
    output_attentions=True
)
```

## 📊 Efficiency Comparison

| Method | Trainable Parameters | Memory Usage | Training Speed |
|--------|---------------------|--------------|----------------|
| Full Fine-tuning | 100% | High | Slow |
| LoRA (r=16) | ~1-2% | Low | Fast |
| P-tuning v2 | <1% | Very Low | Very Fast |
| Adapter | ~3-5% | Medium | Medium |
| BitFit | <1% | Very Low | Very Fast |

## 🔧 Configuration Options

### LoRA Configuration

```python
@dataclass
class LoRAConfig:
    rank: int = 16                    # Low-rank dimension
    alpha: float = 32.0              # Scaling factor
    dropout: float = 0.1             # Dropout rate
    target_modules: List[str] = None # Target modules
    bias: str = "none"               # Bias handling
```

### P-tuning Configuration

```python
@dataclass
class PtuningConfig:
    num_virtual_tokens: int = 20     # Number of virtual tokens
    token_dim: int = 768             # Token dimension
    prefix_projection: bool = False   # Use projection layer
    num_layers: int = 12             # Number of transformer layers
```

### Attention Configuration

```python
@dataclass
class AttentionConfig:
    hidden_size: int = 768           # Hidden dimension
    num_heads: int = 12              # Number of attention heads
    use_rotary: bool = True          # Use RoPE
    use_flash_attention: bool = True  # Use Flash Attention
    use_relative_position: bool = False # Use relative position bias
```

## 🎯 Advanced Usage

### Custom LoRA Implementation

```python
from efficient_finetuning_system import LoRALinear

# Apply LoRA to specific layer
original_layer = nn.Linear(768, 768)
lora_layer = LoRALinear(
    original_layer,
    rank=16,
    alpha=32.0,
    dropout=0.1
)

# Forward pass
output = lora_layer(input_tensor)

# Merge weights for inference
lora_layer.merge_weights()
```

### Rotary Positional Embeddings

```python
from attention_mechanisms import RotaryPositionalEmbedding

# Create RoPE
rope = RotaryPositionalEmbedding(dim=64, max_position_embeddings=2048)

# Apply to query and key
q_rot, k_rot = rope(query, key, seq_len)
```

### Cross-Attention

```python
from attention_mechanisms import CrossAttention

# Create cross-attention
cross_attn = CrossAttention(config)

# Forward pass
outputs = cross_attn(
    hidden_states=decoder_states,
    encoder_hidden_states=encoder_states,
    encoder_attention_mask=encoder_mask
)
```

## 🔬 Demonstration

Run the comprehensive demonstration:

```bash
python efficient_attention_demo.py
```

This demonstrates:
- LoRA fine-tuning with attention mechanisms
- P-tuning v2 implementation
- Custom attention mechanisms
- Efficiency comparisons
- Complete training workflow

## 📈 Performance Optimization

### Memory Optimization
- Use `gradient_checkpointing` for large models
- Enable `fp16` or `bf16` mixed precision training
- Use Flash Attention for memory efficiency
- Apply gradient accumulation for large batch sizes

### Speed Optimization
- Use `torch.compile` for PyTorch 2.0+
- Enable `use_cache` for generation tasks
- Use efficient data loading with `num_workers`
- Apply model parallelism for very large models

### Example Optimization

```python
# Optimized training setup
training_args = TrainingArguments(
    fp16=True,                          # Mixed precision
    gradient_checkpointing=True,        # Memory efficiency
    dataloader_num_workers=4,          # Parallel data loading
    per_device_train_batch_size=8,     # Batch size
    gradient_accumulation_steps=4,      # Effective batch size = 32
    warmup_steps=500,                   # Learning rate warmup
    weight_decay=0.01,                  # Regularization
    logging_steps=100,                  # Logging frequency
    save_steps=1000,                    # Checkpoint frequency
)
```

## 🧪 Testing and Validation

### Unit Tests

```python
# Test LoRA implementation
def test_lora_layer():
    original = nn.Linear(768, 768)
    lora = LoRALinear(original, rank=16)
    
    x = torch.randn(2, 128, 768)
    output = lora(x)
    
    assert output.shape == x.shape
    assert len(lora.get_lora_parameters()) == 2

# Test attention mechanisms
def test_attention():
    config = AttentionConfig(hidden_size=768, num_heads=12)
    attention = MultiHeadAttention(config)
    
    x = torch.randn(2, 128, 768)
    outputs = attention(x)
    
    assert outputs[0].shape == x.shape
```

### Benchmarking

```python
import time
import torch

def benchmark_attention(config, seq_len=1024, batch_size=8):
    attention = MultiHeadAttention(config)
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Warmup
    for _ in range(10):
        _ = attention(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        _ = attention(x)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Average attention time: {avg_time:.4f}s")
```

## 🤝 Integration with Transformers

### Hugging Face Integration

```python
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Using PEFT library
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = AutoModel.from_pretrained("gpt2")
model = get_peft_model(model, peft_config)
```

### Custom Model Integration

```python
class CustomTransformerWithLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base transformer
        self.transformer = TransformerModel(config)
        
        # Apply LoRA
        self.fine_tuner = EfficientFineTuner(
            self.transformer, 
            method="lora", 
            config=LoRAConfig(rank=16)
        )
    
    def forward(self, input_ids, attention_mask=None):
        return self.transformer(input_ids, attention_mask)
```

## 📚 References and Citations

- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **P-tuning v2**: [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning](https://arxiv.org/abs/2110.07602)
- **RoPE**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **Flash Attention**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- **Adapters**: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Solutions:
   # - Reduce batch size
   # - Enable gradient checkpointing
   # - Use fp16 mixed precision
   # - Apply gradient accumulation
   ```

2. **Flash Attention Installation**
   ```bash
   # Install with specific CUDA version
   pip install flash-attn --no-build-isolation
   
   # If fails, disable flash attention
   config.use_flash_attention = False
   ```

3. **Model Loading Issues**
   ```python
   # For large models, use device mapping
   model = AutoModelForCausalLM.from_pretrained(
       "model_name",
       device_map="auto",
       torch_dtype=torch.float16
   )
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check parameter counts
def print_parameter_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,}, Trainable: {trainable:,}, Ratio: {trainable/total:.2%}")
```

## 📄 License

This implementation is provided under the MIT License. See the main project license for details.

## 🙏 Acknowledgments

- Hugging Face Transformers team for the excellent library
- Microsoft Research for LoRA
- Tsinghua University for P-tuning v2
- Stanford HAI for Flash Attention
- Google Research for various attention mechanisms

---

**Ready to implement efficient fine-tuning with advanced attention mechanisms! 🚀**






