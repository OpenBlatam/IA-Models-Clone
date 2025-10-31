# ⚙️ YAML Configuration System

## 📋 Overview

This configuration system implements the key convention: **"Use configuration files (e.g., YAML) for hyperparameters and model settings."**

The system provides comprehensive, professional-grade configuration management for machine learning projects using YAML files, with support for environment variables, hierarchical merging, and validation.

## 🎯 Key Benefits

### ✅ **Centralized Management**
- All hyperparameters and settings in one place
- Easy to track and version control configurations
- Clear separation of code and configuration

### ✅ **Environment Support**
- Development, staging, and production configurations
- Environment variable substitution
- Easy switching between environments

### ✅ **Professional Workflow**
- Industry-standard YAML format
- Configuration validation and error handling
- Backup and versioning capabilities

### ✅ **Team Collaboration**
- Human-readable configuration files
- Easy sharing and collaboration
- Clear documentation of experiments

## 🏗️ Configuration Structure

```
configs/
├── __init__.py                    # Module exports
├── config_manager.py             # Core configuration manager
├── model_config.py               # Model configuration classes
├── training_config.py            # Training configuration classes
├── data_config.py                # Data configuration classes
├── evaluation_config.py          # Evaluation configuration classes
├── experiment_config.py          # Complete experiment configuration
├── README.md                     # This documentation
└── examples/                     # Example configuration files
    ├── model_classification.yaml  # Classification model config
    ├── model_transformer.yaml     # Transformer model config
    ├── training_standard.yaml     # Standard training config
    ├── training_standard_dev.yaml # Development overrides
    ├── experiment_complete.yaml   # Complete experiment config
    └── custom_configs/            # User-defined configurations
```

## 🔧 Core Components

### 1. **ConfigManager**
Central configuration management class that handles:
- YAML file loading and saving
- Environment variable substitution
- Configuration merging and validation
- Environment-specific overrides

### 2. **Configuration Classes**
Structured dataclasses for different aspects:
- **ModelConfig**: Model architecture and parameters
- **TrainingConfig**: Training hyperparameters and settings
- **DataConfig**: Dataset and preprocessing configuration
- **EvaluationConfig**: Evaluation metrics and settings

### 3. **YAML Files**
Human-readable configuration files with:
- Hierarchical structure
- Comments and documentation
- Environment variable support
- Validation and type checking

## 🚀 Quick Start

### 1. **Basic Usage**

```python
from configs.config_manager import ConfigManager
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig

# Initialize configuration manager
config_manager = ConfigManager("configs")

# Load model configuration
model_config = config_manager.load_config_class(
    ModelConfig, 
    "examples/model_classification.yaml"
)

# Load training configuration  
training_config = config_manager.load_config_class(
    TrainingConfig,
    "examples/training_standard.yaml"
)

print(f"Model: {model_config.architecture}")
print(f"Training: {training_config.optimizer}")
```

### 2. **Environment-Specific Configurations**

```python
# Load with environment overrides
config = config_manager.load_environment_config(
    "examples/training_standard.yaml",
    environment="dev"  # Uses training_standard_dev.yaml overrides
)

# Or use environment variable
import os
os.environ['ML_ENVIRONMENT'] = 'production'
config = config_manager.load_environment_config(
    "examples/training_standard.yaml"
)
```

### 3. **Environment Variables in YAML**

```yaml
# In your YAML file
training:
  batch_size: ${BATCH_SIZE:32}          # Use env var or default to 32
  learning_rate: ${LEARNING_RATE:0.001} # Use env var or default to 0.001
  
model:
  save_path: "${MODEL_PATH:./models}"    # Use env var or default path
```

## 📊 Configuration Examples

### **Model Configuration (model_classification.yaml)**

```yaml
# Model Architecture
model_type: "classification"
architecture: "resnet18"
input_size: [3, 224, 224]
output_size: 10
hidden_size: 512
dropout_rate: 0.2

# Pre-trained Settings
pretrained: true
pretrained_model_name: "resnet18"
freeze_backbone: false

# Device and Optimization
device: "auto"
mixed_precision: true
memory_efficient: true

# Classification-Specific
classification_config:
  num_classes: 10
  label_smoothing: 0.1
```

### **Training Configuration (training_standard.yaml)**

```yaml
# Basic Parameters
num_epochs: 100
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001

# Optimizer
optimizer: "adamw"
optimizer_params:
  betas: [0.9, 0.999]
  eps: 1e-8

# Learning Rate Scheduling
scheduler: "cosine"
scheduler_params:
  T_max: 100
  eta_min: 1e-6
warmup_epochs: 5

# Regularization
dropout_rate: 0.1
label_smoothing: 0.1
gradient_clip_norm: 1.0

# Early Stopping
early_stopping: true
patience: 15
monitor_metric: "val_loss"

# Mixed Precision
mixed_precision: true
amp_backend: "native"

# Reproducibility
seed: 42
deterministic: false
```

### **Environment Override (training_standard_dev.yaml)**

```yaml
# Development overrides for faster iteration
num_epochs: 10
batch_size: 16
log_interval: 10
early_stopping: false
num_workers: 2

# Experiment tracking
experiment_name: "dev_classification_training"
tags: 
  - "development"
  - "quick_test"
```

## 🔄 Configuration Workflows

### **1. Development Workflow**

```bash
# Set environment
export ML_ENVIRONMENT=dev

# Run with development configuration
python train.py --config examples/training_standard.yaml
```

The system automatically loads `training_standard_dev.yaml` overrides.

### **2. Production Workflow**

```bash
# Set environment variables
export ML_ENVIRONMENT=prod
export BATCH_SIZE=64
export LEARNING_RATE=0.0001
export MODEL_PATH=/prod/models

# Run with production configuration
python train.py --config examples/training_standard.yaml
```

### **3. Experiment Workflow**

```python
# Load base configuration
base_config = config_manager.load_yaml("experiments/base_experiment.yaml")

# Create experiment variations
experiments = []
for lr in [0.01, 0.001, 0.0001]:
    exp_config = config_manager.merge_configs(
        base_config, 
        {"training": {"learning_rate": lr}}
    )
    experiments.append(exp_config)

# Run experiments
for exp in experiments:
    run_experiment(exp)
```

## 🛠️ Advanced Features

### **1. Configuration Validation**

```python
# Validate required keys
config_manager.validate_config(
    config, 
    required_keys=["model", "training", "data"]
)

# Automatic validation in configuration classes
model_config = ModelConfig(
    model_type="invalid_type"  # Raises ValueError
)
```

### **2. Configuration Merging**

```python
# Deep merge configurations
merged = config_manager.merge_configs(base_config, override_config)

# Environment-specific merging
config = config_manager.load_environment_config(
    "base_config.yaml",
    environment="staging"
)
```

### **3. Configuration Backup**

```python
# Backup before modifications
backup_path = config_manager.backup_config("important_config.yaml")

# List all configurations
configs = config_manager.list_configs()
```

## 📚 Configuration Classes

### **ModelConfig**

```python
@dataclass
class ModelConfig:
    # Architecture
    model_type: str = "classification"
    architecture: str = "resnet"
    input_size: Union[int, Tuple] = 224
    output_size: int = 10
    hidden_size: int = 512
    
    # Pre-trained settings
    pretrained: bool = False
    pretrained_model_name: Optional[str] = None
    
    # Device settings
    device: str = "auto"
    mixed_precision: bool = False
    
    # Model-specific configurations
    classification_config: Dict[str, Any] = field(default_factory=dict)
    transformer_config: Dict[str, Any] = field(default_factory=dict)
```

### **TrainingConfig**

```python
@dataclass
class TrainingConfig:
    # Basic parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Optimizer
    optimizer: str = "adam"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    scheduler: Optional[str] = "cosine"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    dropout_rate: float = 0.1
    gradient_clip_norm: Optional[float] = 1.0
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    monitor_metric: str = "val_loss"
```

## 🎯 Best Practices

### **1. Configuration Organization**

```
configs/
├── models/
│   ├── classification/
│   ├── regression/
│   └── generation/
├── training/
│   ├── standard/
│   ├── fine_tuning/
│   └── distributed/
├── experiments/
│   ├── cifar10/
│   ├── imagenet/
│   └── custom/
└── environments/
    ├── dev/
    ├── staging/
    └── prod/
```

### **2. Naming Conventions**

- **Base configs**: `model_resnet.yaml`, `training_standard.yaml`
- **Environment overrides**: `training_standard_dev.yaml`
- **Experiment configs**: `experiment_cifar10_resnet.yaml`
- **Custom configs**: `custom_transformer_large.yaml`

### **3. Documentation**

```yaml
# Model Configuration for CIFAR-10 Classification
# 
# This configuration defines a ResNet-18 model for CIFAR-10 image classification.
# Key features:
# - Pre-trained ImageNet weights
# - Mixed precision training
# - Label smoothing for regularization
#
# Usage:
#   python train.py --config models/cifar10_resnet18.yaml

model_type: "classification"
# ... rest of configuration
```

### **4. Environment Variables**

```yaml
# Use environment variables for sensitive or deployment-specific values
data:
  dataset_path: "${DATASET_PATH:/default/path}"
  
logging:
  wandb_api_key: "${WANDB_API_KEY:}"
  
model:
  cache_dir: "${MODEL_CACHE:/tmp/models}"
```

## 🔧 Integration Examples

### **With Training Script**

```python
def main():
    # Load configuration
    config_manager = ConfigManager()
    
    model_config = config_manager.load_config_class(
        ModelConfig, args.model_config
    )
    training_config = config_manager.load_config_class(
        TrainingConfig, args.training_config
    )
    
    # Create model and trainer
    model = create_model(model_config)
    trainer = create_trainer(model, training_config)
    
    # Train
    trainer.train()
```

### **With Experiment Tracking**

```python
# Log configuration to experiment tracker
import wandb

wandb.init(
    project=training_config.project_name,
    config={
        "model": model_config.to_dict(),
        "training": training_config.to_dict()
    }
)
```

### **With Hyperparameter Tuning**

```python
# Use configuration system with Optuna
def objective(trial):
    # Load base configuration
    base_config = config_manager.load_config_class(
        TrainingConfig, "base_training.yaml"
    )
    
    # Override with trial suggestions
    base_config.learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    base_config.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Train and return metric
    return train_model(base_config)
```

## 🚀 Getting Started

### **1. Installation**

```bash
pip install pyyaml
```

### **2. Create Your First Configuration**

```python
from configs.config_manager import ConfigManager
from configs.model_config import ModelConfig

# Create configuration manager
config_manager = ConfigManager("my_configs")

# Create a model configuration
model_config = ModelConfig(
    model_type="classification",
    architecture="resnet18",
    input_size=(3, 224, 224),
    output_size=10,
    mixed_precision=True
)

# Save to YAML
config_manager.save_config_class(model_config, "my_model.yaml")

# Load it back
loaded_config = config_manager.load_config_class(ModelConfig, "my_model.yaml")
print(loaded_config.get_model_summary())
```

### **3. Run the Examples**

```bash
cd modular_structure
python examples/yaml_config_example.py
```

## 📊 Configuration Templates

The system includes predefined templates for common scenarios:

### **Model Templates**
- `CLASSIFICATION_CONFIGS`: ResNet, CNN, MLP configurations
- `TRANSFORMER_CONFIGS`: BERT, GPT-2, custom transformer configurations
- `DIFFUSION_CONFIGS`: Stable Diffusion, simple diffusion configurations

### **Training Templates**
- `quick_test`: Fast training for testing
- `standard_training`: Balanced configuration for most use cases
- `fine_tuning`: Optimized for transfer learning
- `heavy_regularization`: High regularization for overfitting prevention
- `distributed_training`: Multi-GPU training configuration

## 🎯 Migration Guide

### **From Hardcoded Parameters**

**Before:**
```python
# Hardcoded in script
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 100
MODEL_TYPE = "resnet18"
```

**After:**
```yaml
# training_config.yaml
learning_rate: 0.001
batch_size: 32
num_epochs: 100

# model_config.yaml  
architecture: "resnet18"
```

### **From ArgParse**

**Before:**
```python
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
```

**After:**
```python
config = config_manager.load_config_class(TrainingConfig, args.config)
# All parameters loaded from YAML
```

---

**⚙️ Key Convention Implemented**: Use configuration files (e.g., YAML) for hyperparameters and model settings

This YAML configuration system ensures your machine learning projects have professional, maintainable, and scalable configuration management that follows industry best practices!






