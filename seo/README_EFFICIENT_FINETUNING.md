# Efficient Fine-tuning Techniques

This module provides comprehensive implementations of parameter-efficient fine-tuning (PEFT) techniques for transformer models, specifically designed for SEO tasks.

## Table of Contents

1. [Overview](#overview)
2. [Supported PEFT Methods](#supported-peft-methods)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Performance Comparison](#performance-comparison)
6. [Best Practices](#best-practices)
7. [Training Guidelines](#training-guidelines)
8. [Troubleshooting](#troubleshooting)

## Overview

Parameter-Efficient Fine-Tuning (PEFT) techniques allow you to fine-tune large language models with significantly fewer parameters while maintaining performance. This module implements:

- **LoRA (Low-Rank Adaptation)**: Adds low-rank matrices to existing weights
- **P-tuning**: Uses virtual tokens with learnable embeddings
- **AdaLoRA (Adaptive LoRA)**: Dynamically adjusts rank allocation
- **Prefix Tuning**: Prepends learnable prefix tokens
- **Comprehensive Training Support**: Built-in trainer with optimization

## Supported PEFT Methods

### 1. LoRA (Low-Rank Adaptation)

LoRA adds low-rank matrices to existing linear layers, significantly reducing trainable parameters.

```python
from efficient_finetuning import create_peft_config, apply_peft_to_model

# Create LoRA configuration
peft_config = create_peft_config(
    peft_type="LORA",
    r=16,                    # Rank of low-rank adaptation
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.1,        # Dropout rate
    target_modules=["w_q", "w_k", "w_v", "w_o"]  # Target modules
)

# Apply to model
peft_manager = apply_peft_to_model(model, "LORA", **peft_config.__dict__)
```

**Features:**
- Configurable rank (r) and scaling factor (alpha)
- Target specific modules
- Optional bias training
- Dropout for regularization

### 2. P-tuning

P-tuning uses virtual tokens with learnable embeddings that are processed through a small encoder.

```python
peft_config = create_peft_config(
    peft_type="P_TUNING",
    num_virtual_tokens=20,           # Number of virtual tokens
    encoder_hidden_size=128,         # Encoder hidden size
    encoder_num_layers=2,            # Number of encoder layers
    encoder_dropout=0.1              # Encoder dropout
)
```

**Features:**
- Learnable virtual tokens
- Transformer encoder for token processing
- Configurable encoder architecture
- Automatic token insertion

### 3. AdaLoRA (Adaptive LoRA)

AdaLoRA dynamically adjusts rank allocation based on importance scores.

```python
peft_config = create_peft_config(
    peft_type="ADALORA",
    target_modules=["w_q", "w_k", "w_v", "w_o"],
    init_r=12,               # Initial rank
    target_r=8,              # Target rank
    beta1=0.85,              # Importance update rate
    beta2=0.85,              # History update rate
    tinit=200,               # Initial training steps
    tfinal=1000,             # Final training steps
    deltaT=10,               # Update frequency
    orth_reg_weight=0.5      # Orthogonal regularization weight
)
```

**Features:**
- Dynamic rank allocation
- Importance-based pruning
- Orthogonal regularization
- Adaptive training schedule

### 4. Prefix Tuning

Prefix tuning prepends learnable prefix tokens to each layer.

```python
peft_config = create_peft_config(
    peft_type="PREFIX_TUNING",
    num_prefix_tokens=20,    # Number of prefix tokens
    prefix_projection=False  # Whether to use projection
)
```

**Features:**
- Layer-specific prefix tokens
- Optional projection layer
- Minimal parameter overhead
- Good for generation tasks

## Configuration

### PEFTConfig

```python
@dataclass
class PEFTConfig:
    peft_type: str = "LORA"          # PEFT method type
    task_type: str = "CAUSAL_LM"     # Task type
    inference_mode: bool = False      # Inference mode flag
    
    # LoRA specific
    lora_config: Optional[LoRAConfig] = None
    
    # P-tuning specific
    num_virtual_tokens: int = 20
    encoder_hidden_size: int = 128
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.1
    
    # Prefix tuning specific
    num_prefix_tokens: int = 20
    prefix_projection: bool = False
    
    # AdaLoRA specific
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    init_r: int = 12
    target_r: int = 8
    beta1: float = 0.85
    beta2: float = 0.85
    tinit: int = 200
    tfinal: int = 1000
    deltaT: int = 10
    orth_reg_weight: float = 0.5
```

### LoRAConfig

```python
@dataclass
class LoRAConfig:
    r: int = 16                      # Rank of adaptation
    lora_alpha: int = 32             # Scaling factor
    lora_dropout: float = 0.1        # Dropout rate
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"               # Bias training strategy
    task_type: str = "CAUSAL_LM"     # Task type
    inference_mode: bool = False     # Inference mode
    fan_in_fan_out: bool = False     # Fan in/out configuration
    modules_to_save: Optional[List[str]] = None
    init_lora_weights: bool = True   # Initialize LoRA weights
    layers_to_transform: Optional[List[int]] = None
    layers_pattern: Optional[str] = None
```

## Usage Examples

### Basic Usage

```python
from efficient_finetuning import create_peft_config, apply_peft_to_model
from transformer_models import TransformerConfig, SEOSpecificTransformer

# Create base model
config = TransformerConfig(
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    vocab_size=10000
)
model = SEOSpecificTransformer(config)

# Apply LoRA
peft_manager = apply_peft_to_model(
    model, 
    "LORA",
    r=16,
    lora_alpha=32,
    target_modules=["w_q", "w_k", "w_v", "w_o"]
)

# Get parameter statistics
param_stats = peft_manager.get_parameter_count()
print(f"Trainable parameters: {param_stats['trainable_parameters']:,}")
print(f"Trainable percentage: {param_stats['trainable_percentage']:.2f}%")
```

### Training with PEFT

```python
from efficient_finetuning import PEFTTrainer

# Create trainer
trainer = PEFTTrainer(
    model=model,
    peft_config=peft_config,
    optimizer_config={
        "lr": 1e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999)
    }
)

# Training loop
for batch in dataloader:
    metrics = trainer.train_step(batch)
    print(f"Loss: {metrics['loss']:.4f}, LR: {metrics['lr']:.6f}")
```

### Advanced Configuration

```python
# LoRA with specific target modules
peft_config = create_peft_config(
    peft_type="LORA",
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["w_q", "w_k", "v_proj", "out_proj"],
    bias="lora_only"
)

# P-tuning with custom encoder
peft_config = create_peft_config(
    peft_type="P_TUNING",
    num_virtual_tokens=30,
    encoder_hidden_size=256,
    encoder_num_layers=3,
    encoder_dropout=0.2
)

# AdaLoRA with aggressive pruning
peft_config = create_peft_config(
    peft_type="ADALORA",
    target_modules=["w_q", "w_k", "w_v", "w_o"],
    init_r=16,
    target_r=4,
    beta1=0.9,
    beta2=0.9,
    tinit=100,
    tfinal=500,
    deltaT=5,
    orth_reg_weight=1.0
)
```

### Save and Load PEFT Models

```python
# Save PEFT model
save_dir = "peft_model"
peft_manager.save_pretrained(save_dir)

# Load PEFT model
new_model = SEOSpecificTransformer(config)
new_peft_manager = EfficientFineTuningManager(new_model, peft_config)
new_peft_manager.load_pretrained(save_dir)
```

## Performance Comparison

### Parameter Efficiency

| Method | Trainable % | Memory | Speed | Best For |
|--------|-------------|--------|-------|----------|
| LoRA | 0.1-1% | Low | Fast | General purpose |
| P-tuning | 0.01-0.1% | Very Low | Fast | Generation tasks |
| AdaLoRA | 0.1-1% | Low | Medium | Adaptive scenarios |
| Prefix Tuning | 0.01-0.1% | Very Low | Fast | Generation tasks |

### Memory Usage Comparison

```python
# Example memory usage for different methods
config = TransformerConfig(hidden_size=512, num_layers=6, num_heads=8)

# Original model
original_params = 23,000,000  # 23M parameters

# PEFT methods
lora_params = 230,000         # 1% of original
p_tuning_params = 23,000      # 0.1% of original
adalora_params = 230,000      # 1% of original
prefix_params = 23,000        # 0.1% of original
```

## Best Practices

### 1. Choosing PEFT Method

- **LoRA**: Use for general fine-tuning tasks
- **P-tuning**: Use for generation tasks with limited data
- **AdaLoRA**: Use when you need adaptive rank allocation
- **Prefix Tuning**: Use for generation tasks with minimal overhead

### 2. Configuration Guidelines

```python
# For SEO content generation
peft_config = create_peft_config(
    peft_type="P_TUNING",
    num_virtual_tokens=20,
    encoder_hidden_size=128
)

# For SEO content classification
peft_config = create_peft_config(
    peft_type="LORA",
    r=16,
    lora_alpha=32,
    target_modules=["w_q", "w_k", "v_proj"]
)

# For adaptive SEO tasks
peft_config = create_peft_config(
    peft_type="ADALORA",
    init_r=12,
    target_r=8,
    orth_reg_weight=0.5
)
```

### 3. Training Recommendations

```python
# Learning rate scheduling
optimizer_config = {
    "lr": 1e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999)
}

# Use cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_training_steps
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Memory Management

```python
# Enable gradient checkpointing for large models
config = TransformerConfig(
    gradient_checkpointing=True,
    # ... other parameters
)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(**batch)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Training Guidelines

### 1. Learning Rate Selection

- **LoRA**: 1e-4 to 5e-4
- **P-tuning**: 1e-3 to 5e-3
- **AdaLoRA**: 1e-4 to 3e-4
- **Prefix Tuning**: 1e-3 to 5e-3

### 2. Training Duration

- **LoRA**: 1000-5000 steps
- **P-tuning**: 500-2000 steps
- **AdaLoRA**: 1000-3000 steps
- **Prefix Tuning**: 500-1500 steps

### 3. Evaluation Strategy

```python
# Regular evaluation during training
if step % eval_steps == 0:
    model.eval()
    with torch.no_grad():
        eval_loss = evaluate_model(model, eval_dataloader)
    
    # Save best model
    if eval_loss < best_loss:
        best_loss = eval_loss
        peft_manager.save_pretrained("best_model")
```

### 4. Hyperparameter Tuning

```python
# LoRA rank tuning
for r in [8, 16, 32, 64]:
    peft_config = create_peft_config(
        peft_type="LORA",
        r=r,
        lora_alpha=r*2
    )
    # Train and evaluate

# P-tuning virtual tokens tuning
for num_tokens in [10, 20, 30, 50]:
    peft_config = create_peft_config(
        peft_type="P_TUNING",
        num_virtual_tokens=num_tokens
    )
    # Train and evaluate
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training
   - Use P-tuning or Prefix Tuning for minimal memory usage

2. **Poor Performance**
   - Increase LoRA rank
   - Add more virtual tokens for P-tuning
   - Adjust learning rate
   - Check target module selection

3. **Training Instability**
   - Reduce learning rate
   - Increase gradient clipping threshold
   - Add weight decay
   - Use AdaLoRA with orthogonal regularization

4. **Slow Training**
   - Use LoRA instead of full fine-tuning
   - Reduce number of target modules
   - Use mixed precision training
   - Optimize data loading

### Debugging PEFT

```python
# Check parameter statistics
param_stats = peft_manager.get_parameter_count()
print(f"Trainable parameters: {param_stats['trainable_parameters']:,}")

# Check target modules
for name, module in model.named_modules():
    if any(target in name for target in peft_config.target_modules):
        print(f"Target module: {name}")

# Monitor training progress
for step, batch in enumerate(dataloader):
    metrics = trainer.train_step(batch)
    if step % 100 == 0:
        print(f"Step {step}: Loss = {metrics['loss']:.4f}")
```

### Performance Monitoring

```python
import time

# Measure training speed
start_time = time.time()
for batch in dataloader:
    metrics = trainer.train_step(batch)
end_time = time.time()

steps_per_second = len(dataloader) / (end_time - start_time)
print(f"Training speed: {steps_per_second:.2f} steps/second")

# Measure memory usage
if torch.cuda.is_available():
    memory_used = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory: {memory_used:.2f} MB")
```

## File Structure

```
efficient_finetuning.py          # Main PEFT implementation
example_efficient_finetuning.py  # Comprehensive usage examples
README_EFFICIENT_FINETUNING.md   # This documentation
```

## Dependencies

- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- NumPy >= 1.21.0

## Contributing

When adding new PEFT methods:

1. Implement the new method in `efficient_finetuning.py`
2. Add configuration support in `PEFTConfig`
3. Update factory functions
4. Add tests and examples
5. Update this documentation

## License

This module is part of the SEO service and follows the same licensing terms. 