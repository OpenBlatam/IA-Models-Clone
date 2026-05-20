# Training Improvements - Addition Removal AI

## 🚀 Training Capabilities Added

### 1. Model Trainer (`training/trainer.py`)

Complete training pipeline with:
- **Mixed Precision Training**: FP16 for faster training and reduced memory
- **Gradient Accumulation**: Handle large batch sizes efficiently
- **Gradient Clipping**: Prevent exploding gradients
- **Learning Rate Scheduling**: Automatic LR adjustment
- **Checkpointing**: Save best and latest models
- **Experiment Tracking**: TensorBoard and W&B integration

**Usage**:
```python
from addition_removal_ai import ModelTrainer
from torch.utils.data import DataLoader

trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_mixed_precision=True,
    gradient_accumulation_steps=4
)

trainer.train(
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=10,
    save_dir="./checkpoints"
)
```

### 2. LoRA Fine-tuning (`training/lora_trainer.py`)

Efficient fine-tuning with LoRA:
- **Parameter Efficient**: Only train small adapter layers
- **Memory Efficient**: Much less memory than full fine-tuning
- **Fast Training**: Faster convergence
- **Easy Integration**: Works with any transformer model

**Usage**:
```python
from addition_removal_ai import create_lora_model
from transformers import AutoModel

base_model = AutoModel.from_pretrained("bert-base-uncased")
lora_model = create_lora_model(
    base_model,
    r=8,  # LoRA rank
    lora_alpha=16,
    target_modules=["query", "value"]
)
```

### 3. Training Configuration (`config/training_config.py`)

YAML-based configuration:
- **Structured Config**: Type-safe configuration classes
- **Easy Experimentation**: Change hyperparameters easily
- **Default Templates**: Ready-to-use configurations

**Usage**:
```python
from addition_removal_ai import Config, create_default_config_file

# Create default config
create_default_config_file("config.yaml")

# Load and modify
config = Config.from_yaml("config.yaml")
config.training.batch_size = 64
config.to_yaml("config_modified.yaml")
```

## 📊 Training Features

### Mixed Precision Training
- Automatic FP16 conversion
- 2x faster training
- 50% less memory usage
- Automatic loss scaling

### Gradient Accumulation
- Simulate larger batch sizes
- Process more data per update
- Better gradient estimates

### Learning Rate Scheduling
- Linear warmup
- Cosine annealing
- Reduce on plateau
- Custom schedules

### Experiment Tracking
- TensorBoard integration
- Weights & Biases support
- Metric logging
- Model checkpointing

## 🎯 Training Workflow

### 1. Prepare Data
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = MyDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 2. Setup Model
```python
from transformers import AutoModel
from addition_removal_ai import create_lora_model

# Base model
model = AutoModel.from_pretrained("bert-base-uncased")

# Apply LoRA (optional)
if use_lora:
    model = create_lora_model(model, r=8, lora_alpha=16)
```

### 3. Configure Training
```python
from addition_removal_ai import Config

config = Config.from_yaml("training_config.yaml")
```

### 4. Train
```python
from addition_removal_ai import ModelTrainer
import torch.optim as optim

trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_mixed_precision=True
)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

trainer.train(
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=10,
    save_dir="./checkpoints"
)
```

## 📈 Best Practices

1. **Use LoRA for Fine-tuning**: Much more efficient than full fine-tuning
2. **Enable Mixed Precision**: Faster training with less memory
3. **Gradient Accumulation**: For large effective batch sizes
4. **Learning Rate Warmup**: Better convergence
5. **Early Stopping**: Prevent overfitting
6. **Experiment Tracking**: Monitor training progress
7. **Regular Checkpointing**: Save models frequently

## 🔧 Configuration Options

### Model Configuration
- `model_type`: Type of model
- `model_name`: HuggingFace model name
- `use_lora`: Enable LoRA fine-tuning
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha

### Training Configuration
- `batch_size`: Batch size
- `num_epochs`: Number of epochs
- `learning_rate`: Learning rate
- `use_mixed_precision`: Enable FP16
- `gradient_accumulation_steps`: Accumulation steps

### Optimizer Configuration
- `optimizer_type`: Optimizer (adamw, adam, sgd)
- `weight_decay`: Weight decay
- `betas`: Adam betas

### Scheduler Configuration
- `scheduler_type`: Scheduler type
- `warmup_steps`: Warmup steps
- `num_training_steps`: Total training steps

## 📝 Example Training Script

See `examples/training_example.py` for complete example.

## 🚀 Quick Start

```python
from addition_removal_ai import (
    ModelTrainer,
    create_lora_model,
    Config,
    create_default_config_file
)

# Create config
create_default_config_file("config.yaml")
config = Config.from_yaml("config.yaml")

# Setup model with LoRA
model = AutoModel.from_pretrained(config.model.model_name)
if config.model.use_lora:
    model = create_lora_model(model, r=config.model.lora_r)

# Create trainer
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_mixed_precision=config.training.use_mixed_precision
)

# Train
trainer.train(optimizer, criterion, num_epochs=config.training.num_epochs)
```

## ✨ Summary

Training capabilities added:
- ✅ Complete training pipeline
- ✅ LoRA fine-tuning support
- ✅ Mixed precision training
- ✅ Gradient accumulation
- ✅ Learning rate scheduling
- ✅ Experiment tracking (TensorBoard/W&B)
- ✅ YAML configuration
- ✅ Model checkpointing
- ✅ Best practices implementation

All training features follow deep learning best practices and are production-ready.

