# Training Guide - Addiction Recovery AI

## ✅ Training Components

### Training Structure

```
training/
├── distributed_trainer.py  # ✅ Distributed training
├── evaluator.py            # ✅ Model evaluation
├── lora_trainer.py         # ✅ LoRA fine-tuning
└── recovery_trainer.py      # ✅ Recovery model training
```

## 📦 Training Components

### `training/recovery_trainer.py` - Recovery Trainer
- **Status**: ✅ Active
- **Purpose**: Training recovery models
- **Features**: 
  - Model training
  - Checkpointing
  - Evaluation

**Usage:**
```python
from training.recovery_trainer import RecoveryTrainer

trainer = RecoveryTrainer(model, optimizer, criterion)
trainer.train(train_loader, val_loader, num_epochs=50)
```

### `training/lora_trainer.py` - LoRA Trainer
- **Status**: ✅ Active
- **Purpose**: LoRA fine-tuning
- **Features**: 
  - Parameter-efficient fine-tuning
  - LoRA adapters
  - Low-rank adaptation

**Usage:**
```python
from training.lora_trainer import LoRATrainer

trainer = LoRATrainer(base_model, rank=8, alpha=16)
trainer.finetune(train_data, num_epochs=10)
```

### `training/distributed_trainer.py` - Distributed Trainer
- **Status**: ✅ Active
- **Purpose**: Distributed training across multiple GPUs/nodes
- **Features**: 
  - Multi-GPU training
  - Distributed data parallel
  - Gradient synchronization

**Usage:**
```python
from training.distributed_trainer import DistributedTrainer

trainer = DistributedTrainer(model, world_size=4)
trainer.train(train_loader, num_epochs=50)
```

### `training/evaluator.py` - Evaluator
- **Status**: ✅ Active
- **Purpose**: Model evaluation
- **Features**: 
  - Metrics calculation
  - Model comparison
  - Performance evaluation

**Usage:**
```python
from training.evaluator import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, test_loader)
```

## 📝 Usage Examples

### Basic Training
```python
from training.recovery_trainer import RecoveryTrainer

trainer = RecoveryTrainer(model, optimizer, criterion)
history = trainer.train(
    train_loader,
    val_loader,
    num_epochs=50,
    checkpoint_dir="checkpoints"
)
```

### LoRA Fine-Tuning
```python
from training.lora_trainer import LoRATrainer

trainer = LoRATrainer(base_model, rank=8, alpha=16)
trainer.finetune(
    train_data,
    num_epochs=10,
    learning_rate=1e-4
)
```

### Distributed Training
```python
from training.distributed_trainer import DistributedTrainer

trainer = DistributedTrainer(model, world_size=4)
trainer.train(train_loader, num_epochs=50)
```

## 📚 Additional Resources

- See `CORE_GUIDE.md` for base trainer classes
- See `SCRIPTS_GUIDE.md` for training scripts
- See `core/base/base_trainer.py` for base trainer






