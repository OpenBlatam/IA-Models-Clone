# Complete Refactoring Guide - Ultra-Modular Architecture

## Overview
This guide demonstrates how to use the completely refactored, ultra-modular Music Analyzer AI system. Every component is now separated, composable, and follows deep learning best practices.

## Quick Start

### 1. Create Model from Configuration

```python
from music_analyzer_ai import ModelConfig, get_factory

# Load configuration
config = ModelConfig.from_yaml("configs/my_model.yaml")

# Create complete training setup
factory = get_factory()
setup = factory.create_from_config(config)

# Access components
model = setup["model"]
optimizer = setup["optimizer"]
scheduler = setup["scheduler"]
loss = setup["loss"]
training_loop = setup["training_loop"]
callbacks = setup["callbacks"]
```

### 2. Register and Use Custom Components

```python
from music_analyzer_ai import register_model, get_registry
from music_analyzer_ai.models.modular_transformer import ModularMusicClassifier

# Register custom model
register_model("my_custom_classifier", ModularMusicClassifier)

# Use registered model
registry = get_registry()
ModelClass = registry.get_model("my_custom_classifier")
model = ModelClass(input_dim=169, embed_dim=256)
```

### 3. Compose Models from Components

```python
from music_analyzer_ai import ModelComposer
from music_analyzer_ai.models.architectures import (
    MusicFeatureEmbedding,
    MultiHeadAttention,
    ResidualFeedForward
)
import torch.nn as nn

# Build model using composer
composer = ModelComposer()
composer.add_component("embed", MusicFeatureEmbedding(169, 256), is_input=True)
composer.add_component("attn1", MultiHeadAttention(256, 8))
composer.add_component("attn2", MultiHeadAttention(256, 8))
composer.add_component("ff", ResidualFeedForward(256, 1024))
composer.add_component("output", nn.Linear(256, 10), is_output=True)

# Connect components
composer.connect("embed", "attn1")
composer.connect("attn1", "attn2")
composer.connect("attn2", "ff")
composer.connect("ff", "output")

# Build model
model = composer.build()
```

### 4. Use Model Manager

```python
from music_analyzer_ai import ModelManager

# Create manager
manager = ModelManager(device="cuda", model_dir="./models")

# Create model
model = manager.create_model(
    model_name="my_model",
    model_type="music_classifier",
    config={"input_dim": 169, "embed_dim": 256},
    compile_model=True
)

# Create inference pipeline
pipeline = manager.create_inference_pipeline(
    pipeline_name="my_pipeline",
    model_name="my_model",
    preprocess_fn=preprocess_features,
    postprocess_fn=postprocess_predictions
)

# Run inference
result = manager.predict("my_pipeline", features)
```

### 5. Training with Modular Components

```python
from music_analyzer_ai import (
    StandardTrainingLoop,
    create_optimizer,
    create_scheduler,
    MultiTaskLoss,
    EarlyStoppingCallback,
    CheckpointCallback
)

# Create training loop
loop = StandardTrainingLoop(
    model=model,
    optimizer=create_optimizer("adamw", model.parameters(), lr=1e-4),
    loss_fn=MultiTaskLoss(task_losses={
        "genre": ClassificationLoss(num_classes=10),
        "mood": ClassificationLoss(num_classes=6)
    }),
    gradient_accumulation_steps=4,
    max_grad_norm=1.0
)

# Create callbacks
callbacks = [
    EarlyStoppingCallback(patience=10),
    CheckpointCallback(save_best=True)
]

# Train
for epoch in range(num_epochs):
    train_metrics = loop.train_epoch(train_loader, epoch)
    val_metrics = loop.validate_epoch(val_loader)
    
    # Execute callbacks
    for callback in callbacks:
        callback.on_epoch_end(epoch, {**train_metrics, **val_metrics})
```

### 6. Data Processing Pipeline

```python
from music_analyzer_ai.data.transforms import (
    AudioResampler,
    AudioTrimmer,
    AudioNormalizer,
    Compose
)
from music_analyzer_ai.data.pipelines import create_standard_feature_pipeline

# Create audio transformation pipeline
audio_transforms = Compose([
    AudioResampler(target_sr=22050),
    AudioTrimmer(top_db=20.0),
    AudioNormalizer(method="peak")
])

# Create feature extraction pipeline
feature_pipeline = create_standard_feature_pipeline()

# Process audio
processed_audio, sr = audio_transforms(audio, sr)
features = feature_pipeline(processed_audio, sr)
```

### 7. Using HuggingFace Models

```python
from music_analyzer_ai.integrations.transformers_integration import (
    TransformerMusicEncoder,
    LoRATransformerWrapper
)

# Use pre-trained transformer
encoder = TransformerMusicEncoder(
    model_name="bert-base-uncased",
    num_classes=10,
    freeze_base=True
)

# Or use LoRA for efficient fine-tuning
lora_model = LoRATransformerWrapper(
    model_name="bert-base-uncased",
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"]
)
```

### 8. Diffusion Models

```python
from music_analyzer_ai.integrations.diffusion_integration import (
    DiffusionPipelineWrapper
)

# Create diffusion pipeline
pipeline = DiffusionPipelineWrapper(
    pipeline_type="stable_diffusion",
    scheduler_type="ddim",
    device="cuda"
)

# Generate
result = pipeline.generate(
    prompt="A beautiful music composition",
    num_inference_steps=50,
    guidance_scale=7.5
)
```

### 9. Evaluation Metrics

```python
from music_analyzer_ai.evaluation.modular_metrics import (
    ClassificationMetrics,
    MultiTaskMetrics
)

# Classification metrics
metrics = ClassificationMetrics(num_classes=10)
results = metrics.compute(predictions, targets)
# Returns: accuracy, precision, recall, f1_score, confusion_matrix

# Multi-task metrics
multi_metrics = MultiTaskMetrics(
    classification_tasks={"genre": 10, "mood": 6},
    regression_tasks=["energy", "tempo"]
)
results = multi_metrics.compute(predictions, targets)
```

### 10. Device Management

```python
from music_analyzer_ai.utils.device_manager import get_device_manager

# Get device manager (auto-detects best device)
device_manager = get_device_manager()

# Move to device
tensor = device_manager.move_to_device(tensor)

# Compile model
compiled_model = device_manager.compile_model(model)

# Check mixed precision
use_fp16 = device_manager.enable_mixed_precision()
```

### 11. Weight Initialization

```python
from music_analyzer_ai.utils.initialization import initialize_weights

# Initialize model weights
initialize_weights(model, strategy="xavier")
initialize_weights(model, strategy="kaiming")  # For ReLU
initialize_weights(model, strategy="transformer")  # Transformer-specific
initialize_weights(model, strategy="lstm")  # LSTM-specific
```

### 12. Input Validation

```python
from music_analyzer_ai.utils.validation import InputValidator, TensorValidator

# Validate input
is_valid = InputValidator.validate_features(
    features,
    expected_dim=169,
    name="audio_features"
)

# Sanitize input (fix NaN/Inf)
clean_features = InputValidator.sanitize_input(features)

# Check tensors
has_issues = TensorValidator.check_nan_inf(tensor, "model_output")
if has_issues:
    tensor = TensorValidator.fix_nan_inf(tensor)
```

## Complete Training Example

```python
from music_analyzer_ai import (
    ModelConfig,
    get_factory,
    StandardTrainingLoop,
    ClassificationMetrics
)
from torch.utils.data import DataLoader

# 1. Load configuration
config = ModelConfig.from_yaml("configs/music_classifier.yaml")

# 2. Create training setup
factory = get_factory()
setup = factory.create_from_config(config)

# 3. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Create metrics
metrics = ClassificationMetrics(num_classes=10)

# 5. Training loop
for epoch in range(config.training.epochs):
    # Train
    train_metrics = setup["training_loop"].train_epoch(train_loader, epoch)
    
    # Validate
    val_metrics = setup["training_loop"].validate_epoch(val_loader)
    
    # Compute detailed metrics
    all_predictions = []
    all_targets = []
    for batch in val_loader:
        outputs = setup["model"](batch["features"])
        all_predictions.append(outputs)
        all_targets.append(batch["labels"])
    
    detailed_metrics = metrics.compute(
        torch.cat(all_predictions),
        torch.cat(all_targets)
    )
    
    # Execute callbacks
    for callback in setup["callbacks"]:
        callback.on_epoch_end(epoch, {**train_metrics, **val_metrics, **detailed_metrics})
    
    # Update scheduler
    if setup["scheduler"]:
        setup["scheduler"].step()
```

## Complete Inference Example

```python
from music_analyzer_ai import ModelManager
from music_analyzer_ai.data.pipelines import create_standard_feature_pipeline
import numpy as np

# 1. Create model manager
manager = ModelManager(device="cuda")

# 2. Load or create model
model = manager.load_model(
    model_name="classifier",
    checkpoint_path="models/best_model.pt",
    model_type="music_classifier"
)

# 3. Create feature pipeline
feature_pipeline = create_standard_feature_pipeline()

# 4. Create inference pipeline
inference_pipeline = manager.create_inference_pipeline(
    pipeline_name="music_analysis",
    model_name="classifier",
    preprocess_fn=lambda x: feature_pipeline(x, sr=22050),
    postprocess_fn=lambda x: {
        "genre": int(np.argmax(x["genre_logits"])),
        "confidence": float(np.max(x["genre_logits"]))
    }
)

# 5. Run inference
audio = load_audio("song.wav")
result = manager.predict("music_analysis", audio)
print(f"Genre: {result['output']['genre']}, Confidence: {result['output']['confidence']}")
```

## Architecture Benefits

### 1. Maximum Modularity
- Every component is independent
- Easy to test in isolation
- Easy to replace or extend

### 2. Composition
- Build complex models from simple parts
- Mix and match any components
- Flexible architecture

### 3. Registry System
- Dynamic component discovery
- Easy registration of custom components
- Plugin-like extensibility

### 4. Configuration-Driven
- YAML/JSON configuration files
- Easy experimentation
- Reproducibility

### 5. Best Practices
- Proper weight initialization
- NaN/Inf detection
- Mixed precision training
- Gradient handling
- Error recovery

## File Structure

```
music_analyzer_ai/
├── core/
│   ├── registry.py          # Component registry
│   ├── composition.py       # Model composition
│   └── model_manager.py     # Model lifecycle management
│
├── factories/
│   └── unified_factory.py   # Unified factory system
│
├── config/
│   └── model_config.py     # Configuration management
│
├── models/
│   ├── architectures/       # Modular architecture components
│   └── modular_transformer.py
│
├── training/
│   ├── components/          # Training components
│   └── loops/              # Training loops
│
├── inference/
│   └── pipelines/          # Inference pipelines
│
├── data/
│   ├── transforms/         # Data transformations
│   └── pipelines/         # Feature pipelines
│
├── integrations/
│   ├── transformers_integration.py
│   └── diffusion_integration.py
│
├── evaluation/
│   └── modular_metrics.py
│
├── gradio/
│   └── components/          # Gradio components
│
└── utils/
    ├── device_manager.py
    ├── initialization.py
    └── validation.py
```

## Migration from Old Code

### Old Way
```python
from core.deep_models import DeepGenreClassifier
model = DeepGenreClassifier(input_size=169, num_genres=10)
```

### New Modular Way
```python
from music_analyzer_ai import ModelManager
manager = ModelManager()
model = manager.create_model(
    model_name="classifier",
    model_type="music_classifier",
    config={"input_dim": 169, "num_genres": 10}
)
```

## Conclusion

The refactored system provides:
- **Ultra-modular architecture**: Every component is independent
- **Composition system**: Build complex models from simple parts
- **Registry system**: Dynamic component discovery
- **Unified factory**: Consistent creation interface
- **Configuration-driven**: Easy experimentation
- **Best practices**: Following all deep learning best practices

This architecture enables rapid development, easy maintenance, and maximum code reuse.



