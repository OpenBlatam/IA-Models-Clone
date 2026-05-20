# Complete Guide - Addiction Recovery AI v3.4.0

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from addiction_recovery_ai import (
    create_sentiment_analyzer,
    create_progress_predictor,
    create_ultra_fast_inference
)

# Create models
sentiment = create_sentiment_analyzer()
predictor = create_progress_predictor()

# Ultra-fast inference
engine = create_ultra_fast_inference(predictor)

# Predict
features = torch.tensor([[0.3, 0.4, 0.5, 0.7]])
progress = engine.predict(features)
```

## 📚 Complete Workflow

### 1. Configuration-Based Setup
```python
from addiction_recovery_ai import get_config, ModelFactory

# Load config
config = get_config("config/model_config.yaml")

# Create model from config
model = ModelFactory.create(
    "RecoveryProgressPredictor",
    config.get_model_config("progress_predictor")
)
```

### 2. Data Processing
```python
from addiction_recovery_ai import (
    create_recovery_dataset,
    split_data,
    normalize_features
)

# Split data
train, val, test = split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Create dataset
dataset = create_recovery_dataset(
    train,
    feature_keys=["days_sober", "cravings", "stress", "mood"],
    target_key="progress"
)
```

### 3. Training
```python
from addiction_recovery_ai import (
    TrainerFactory,
    create_tracker,
    create_checkpoint_manager
)

# Create trainer
trainer = TrainerFactory.create(
    "RecoveryModelTrainer",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

# Setup tracking
tracker = create_tracker("experiment_v1", use_tensorboard=True)

# Setup checkpointing
checkpoint_manager = create_checkpoint_manager("checkpoints")

# Train
trainer.train(optimizer, criterion, num_epochs=50)
```

### 4. Inference
```python
from addiction_recovery_ai import create_ultra_fast_inference

# Create ultra-fast engine
engine = create_ultra_fast_inference(model)

# Predict
output = engine.predict(input_tensor)
```

## 🎯 Use Cases

### Sentiment Analysis
```python
from addiction_recovery_ai import create_sentiment_analyzer

analyzer = create_sentiment_analyzer()
result = analyzer.analyze("I'm feeling great today!")
```

### Progress Prediction
```python
from addiction_recovery_ai import create_progress_predictor

predictor = create_progress_predictor()
features = torch.tensor([[30/365, 0.3, 0.4, 0.7]])
progress = predictor.predict_progress(features)
```

### Relapse Risk
```python
from addiction_recovery_ai import create_relapse_predictor

predictor = create_relapse_predictor(bidirectional=True, use_attention=True)
sequence = torch.tensor([[...]])  # (seq_len, features)
risk = predictor.predict_risk(sequence)
```

### AI Coaching
```python
from addiction_recovery_ai import create_llm_coach

coach = create_llm_coach()
message = coach.generate_coaching_message(
    user_situation="Feeling stressed",
    days_sober=30
)
```

### Progress Visualization
```python
from addiction_recovery_ai import create_progress_visualizer

visualizer = create_progress_visualizer()
image = visualizer.generate_progress_image(
    prompt="Recovery journey",
    days_sober=60,
    progress_score=0.8
)
```

## 🔧 Advanced Features

### Experiment Tracking
```python
from addiction_recovery_ai import create_tracker

with create_tracker("experiment_v1", use_tensorboard=True, use_wandb=True) as tracker:
    tracker.log_scalar("loss", loss_value, epoch)
    tracker.log_hyperparameters({"lr": 0.001, "batch_size": 32})
```

### Model Comparison
```python
from addiction_recovery_ai import compare_models

comparison = compare_models(model1, model2, input_shape=(1, 10))
print(f"Parameter difference: {comparison['differences']['parameter_diff']}")
```

### Model Export
```python
from addiction_recovery_ai import export_model

export_model(
    model=model,
    input_shape=(1, 10),
    output_path="model.onnx",
    format="onnx"
)
```

### Async Inference
```python
from addiction_recovery_ai import create_async_engine

engine = create_async_engine(model)
output = await engine.predict_async(input_tensor)
```

## 📊 Performance Tips

### 1. Use Ultra-Fast Inference
```python
engine = create_ultra_fast_inference(model)  # 5x faster
```

### 2. Enable Caching
```python
cache = create_embedding_cache(max_size=10000)  # 10-100x faster
```

### 3. Use Batch Processing
```python
outputs = engine.predict_batch_optimized(inputs, batch_size=128)
```

### 4. Optimize Memory
```python
from addiction_recovery_ai import optimize_model_memory, get_memory_stats

optimize_model_memory(model)
stats = get_memory_stats()
```

## 🛠️ Scripts

### Training Script
```bash
python scripts/train_model.py \
    --model progress_predictor \
    --data data/train.json \
    --epochs 50 \
    --tensorboard
```

### Inference Server
```bash
python scripts/inference_server.py \
    --host 0.0.0.0 \
    --port 8000
```

## 📝 Best Practices

1. **Use Configuration Files**: Centralize settings in YAML
2. **Enable Tracking**: Use TensorBoard/WandB for experiments
3. **Save Checkpoints**: Use checkpoint manager for model versioning
4. **Optimize for Production**: Use ultra-fast inference and caching
5. **Monitor Performance**: Track metrics and memory usage
6. **Test Thoroughly**: Use evaluators and cross-validation

## 🎓 Examples

See `examples/complete_workflow.py` for a full example.

## 📚 Documentation

- `DEEP_LEARNING_ENHANCEMENTS.md`: Deep learning improvements
- `SPEED_OPTIMIZATIONS_V2.md`: Speed optimizations
- `ADVANCED_FEATURES.md`: Advanced features
- `MODULAR_ARCHITECTURE_V2.md`: Modular architecture
- `PRODUCTION_FEATURES.md`: Production features
- `ULTRA_SPEED_V2.md`: Ultra-speed optimizations

---

**Version**: 3.4.0  
**Author**: Blatam Academy













