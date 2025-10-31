# Unified Training System for Ads Feature

## Overview

The Unified Training System consolidates all scattered training implementations from the ads feature into a clean, modular architecture following Clean Architecture principles. This system provides a unified interface for different types of training while maintaining extensibility and performance.

## 🏗️ Architecture

The training system follows a layered architecture with clear separation of concerns:

```
training/
├── __init__.py                 # Package initialization
├── base_trainer.py            # Abstract base classes and interfaces
├── pytorch_trainer.py         # PyTorch-specific trainer implementation
├── diffusion_trainer.py       # Diffusion model trainer implementation
├── multi_gpu_trainer.py       # Multi-GPU and distributed training
├── training_factory.py         # Factory pattern for trainer creation
├── experiment_tracker.py       # Experiment tracking and monitoring
├── training_optimizer.py       # Performance optimization and monitoring
├── training_demo.py           # Comprehensive system demonstration
└── README.md                  # This documentation
```

## 🎯 Key Components

### 1. Base Trainer (`base_trainer.py`)

Abstract base class that defines the common interface for all trainers:

- **TrainingConfig**: Configuration for training sessions
- **TrainingMetrics**: Training metrics and statistics
- **TrainingResult**: Results of training sessions
- **BaseTrainer**: Abstract base class with common functionality

**Features:**
- Common training lifecycle management
- Metrics collection and logging
- Callback system for extensibility
- Error handling and status tracking

### 2. PyTorch Trainer (`pytorch_trainer.py`)

Standard neural network training implementation:

- **PyTorchModelConfig**: Model configuration
- **PyTorchDataConfig**: Data handling configuration
- **SimpleModel**: Example neural network model
- **PyTorchTrainer**: PyTorch-specific trainer

**Features:**
- Mixed precision training (AMP)
- Gradient clipping and optimization
- Learning rate scheduling
- Device management (CPU/GPU/MPS)
- Checkpointing and model saving

### 3. Diffusion Trainer (`diffusion_trainer.py`)

Advanced diffusion model training implementation:

- **DiffusionModelConfig**: Diffusion model configuration
- **DiffusionTrainingConfig**: Training-specific configuration
- **AdvancedNoiseScheduler**: Advanced noise scheduling
- **DiffusionTrainer**: Diffusion model trainer

**Features:**
- Multiple noise schedule types (linear, cosine, sigma, karras)
- Various sampling methods (DDIM, Euler, DPM-Solver, etc.)
- Pipeline management and optimization
- Memory-efficient training
- Advanced loss functions

### 4. Multi-GPU Trainer (`multi_gpu_trainer.py`)

Distributed and parallel training implementation:

- **GPUConfig**: GPU configuration settings
- **MultiGPUTrainingConfig**: Multi-GPU specific settings
- **GPUMonitor**: GPU usage monitoring
- **MultiGPUTrainer**: Multi-GPU trainer

**Features:**
- DataParallel for single-node multi-GPU
- DistributedDataParallel for multi-node
- Automatic GPU detection and configuration
- Performance monitoring and optimization
- Memory management and load balancing

### 5. Training Factory (`training_factory.py`)

Factory pattern for trainer creation and management:

- **TrainerType**: Enumeration of trainer types
- **TrainerConfig**: Configuration for trainer creation
- **TrainingFactory**: Main factory class

**Features:**
- Unified trainer creation interface
- Automatic trainer selection based on requirements
- Instance management and cleanup
- Extensible trainer registration system

### 6. Experiment Tracker (`experiment_tracker.py`)

Comprehensive experiment tracking and monitoring:

- **ExperimentConfig**: Experiment configuration
- **ExperimentRun**: Individual experiment run
- **ExperimentTracker**: Main tracking class

**Features:**
- SQLite-based storage for experiments and runs
- Metrics logging and retrieval
- Artifact management
- Run comparison and analysis
- Export capabilities

### 7. Training Optimizer (`training_optimizer.py`)

Performance optimization and monitoring:

- **OptimizationConfig**: Optimization configuration
- **OptimizationResult**: Optimization results
- **TrainingOptimizer**: Main optimization class

**Features:**
- Multiple optimization levels (light, standard, aggressive, extreme)
- Mixed precision optimization
- Memory and GPU optimization
- Performance monitoring and analysis
- Optimization recommendations

## 🚀 Usage Examples

### Basic PyTorch Training

```python
from training import TrainingConfig, PyTorchTrainer, PyTorchModelConfig

# Create configuration
config = TrainingConfig(
    model_name="my_model",
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001
)

# Create trainer
trainer = PyTorchTrainer(config)

# Setup and train
await trainer.setup_training()
result = await trainer.train()
```

### Diffusion Model Training

```python
from training import DiffusionTrainer, DiffusionModelConfig

# Create diffusion trainer
diffusion_config = DiffusionModelConfig(
    model_name="runwayml/stable-diffusion-v1-5",
    scheduler_type="DDIM"
)

trainer = DiffusionTrainer(config, diffusion_config)
await trainer.setup_training()
result = await trainer.train()
```

### Multi-GPU Training

```python
from training import MultiGPUTrainer, GPUConfig

# Configure multi-GPU
gpu_config = GPUConfig(
    use_multi_gpu=True,
    batch_size_per_gpu=8
)

trainer = MultiGPUTrainer(config, gpu_config)
await trainer.setup_training()
result = await trainer.train()
```

### Using the Training Factory

```python
from training import TrainingFactory, TrainerType

# Create factory
factory = TrainingFactory()

# Create trainer via factory
trainer = factory.create_trainer(
    TrainerConfig(
        trainer_type=TrainerType.PYTORCH,
        base_config=config
    )
)

# Or create optimal trainer
optimal_trainer = factory.create_optimal_trainer(
    config, 
    {"multi_gpu": True, "diffusion_model": False}
)
```

### Experiment Tracking

```python
from training import ExperimentTracker, ExperimentConfig

# Create tracker
tracker = ExperimentTracker("./experiments")

# Create experiment
exp_config = ExperimentConfig(
    name="my_experiment",
    description="Training experiment",
    tags=["demo", "training"]
)

experiment_name = tracker.create_experiment(exp_config)
run_id = tracker.start_run(experiment_name)

# Log metrics during training
tracker.log_metrics(metrics, run_id)

# Complete run
tracker.complete_run(run_id, result)
```

### Training Optimization

```python
from training import TrainingOptimizer, OptimizationConfig, OptimizationLevel

# Create optimizer
optimizer = TrainingOptimizer(
    OptimizationConfig(level=OptimizationLevel.AGGRESSIVE)
)

# Apply optimizations to trainer
result = optimizer.optimize_trainer(trainer)

# Get recommendations
recommendations = optimizer.get_recommendations(trainer)
```

## 🔧 Configuration

### Training Configuration

```python
@dataclass
class TrainingConfig:
    model_name: str = "default_model"
    dataset_name: str = "default_dataset"
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    optimizer_name: str = "adam"
    scheduler_name: str = "cosine"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 10
```

### Optimization Configuration

```python
@dataclass
class OptimizationConfig:
    level: OptimizationLevel = OptimizationLevel.STANDARD
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False
    use_gradient_accumulation: bool = True
    use_memory_optimization: bool = True
    use_gpu_optimization: bool = True
```

## 📊 Monitoring and Metrics

The system provides comprehensive monitoring capabilities:

- **Training Metrics**: Loss, accuracy, learning rate, validation metrics
- **Performance Metrics**: Memory usage, GPU utilization, training time
- **Experiment Tracking**: Run history, artifact management, comparison
- **Optimization Metrics**: Performance improvements, memory savings

## 🔄 Extensibility

The system is designed for easy extension:

1. **New Trainer Types**: Inherit from `BaseTrainer` and implement abstract methods
2. **Custom Metrics**: Extend `TrainingMetrics` for domain-specific metrics
3. **Optimization Strategies**: Add new optimization levels and strategies
4. **Storage Backends**: Implement custom storage for experiment tracking

## 🧪 Testing

Run the comprehensive demo to test the system:

```bash
cd training
python -m training_demo
```

## 📁 Migration from Scattered Implementations

This system consolidates the following scattered files:

| Original File | New Component | Status |
|---------------|---------------|---------|
| `pytorch_example.py` | `PyTorchTrainer` | ✅ Consolidated |
| `diffusion_service.py` | `DiffusionTrainer` | ✅ Consolidated |
| `multi_gpu_training.py` | `MultiGPUTrainer` | ✅ Consolidated |
| `experiment_tracker.py` | `ExperimentTracker` | ✅ Consolidated |
| Various optimization files | `TrainingOptimizer` | ✅ Consolidated |

## 🎯 Benefits of Consolidation

1. **Unified Interface**: Consistent API across all trainer types
2. **Code Reuse**: Common functionality shared through base classes
3. **Maintainability**: Centralized codebase easier to maintain
4. **Extensibility**: Easy to add new trainer types and features
5. **Performance**: Optimized implementations with performance monitoring
6. **Testing**: Comprehensive testing and validation framework
7. **Documentation**: Clear documentation and examples

## 🚀 Future Enhancements

- **Hybrid Trainers**: Combination of multiple training approaches
- **AutoML Integration**: Automatic hyperparameter optimization
- **Cloud Integration**: Cloud-based training and deployment
- **Advanced Monitoring**: Real-time performance dashboards
- **Model Versioning**: Integrated model version control
- **Distributed Training**: Advanced multi-node training support

## 📚 Dependencies

- **PyTorch**: Core deep learning framework
- **Diffusers**: Diffusion model support
- **Transformers**: Model and tokenizer support
- **SQLite**: Experiment tracking storage
- **psutil**: System monitoring
- **GPUtil**: GPU monitoring

## 🤝 Contributing

To extend the training system:

1. Follow the existing architecture patterns
2. Implement required abstract methods
3. Add comprehensive tests
4. Update documentation
5. Follow the established coding standards

## 📄 License

This training system is part of the ads feature and follows the same licensing terms.
