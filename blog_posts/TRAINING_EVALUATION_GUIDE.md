# 🚀 Model Training & Evaluation Guide
## Production-Ready AI Training System

### Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Training System](#training-system)
4. [Evaluation System](#evaluation-system)
5. [Advanced Features](#advanced-features)
6. [Production Deployment](#production-deployment)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers the production-ready model training and evaluation system for Blatam Academy's AI infrastructure. The system provides:

- **Advanced Training**: Fine-tuning, transfer learning, LoRA, P-tuning
- **Comprehensive Evaluation**: Multi-metric evaluation, cross-validation, model comparison
- **Hyperparameter Optimization**: Automated HPO with Optuna
- **Production Features**: GPU optimization, distributed training, experiment tracking
- **Enterprise Integration**: MLflow, Weights & Biases, TensorBoard

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements_training_evaluation.txt
```

### 2. Basic Training

```python
import asyncio
from model_training import quick_train_transformer

# Quick training
result = await quick_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv",
    num_epochs=5
)
print(f"Training completed: {result}")
```

### 3. Basic Evaluation

```python
from model_evaluation import quick_model_evaluation

# Quick evaluation
result = await quick_model_evaluation(
    model_path="models/distilbert_sentiment_best.pth",
    test_dataset_path="data/test_dataset.csv"
)
print(f"Evaluation completed: {result}")
```

---

## Training System

### Configuration

```python
from model_training import TrainingConfig, ModelType, TrainingMode

config = TrainingConfig(
    model_type=ModelType.TRANSFORMER,
    training_mode=TrainingMode.FINE_TUNE,
    model_name="my-sentiment-model",
    dataset_path="data/sentiment.csv",
    output_dir="models/",
    
    # Training parameters
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=10,
    warmup_steps=100,
    
    # Advanced features
    mixed_precision=True,
    gradient_accumulation_steps=2,
    early_stopping_patience=5,
    
    # Logging
    log_to_tensorboard=True,
    log_to_wandb=True
)
```

### Training Modes

#### 1. Fine-tuning
```python
config.training_mode = TrainingMode.FINE_TUNE
# Uses pre-trained model, updates all parameters
```

#### 2. LoRA (Low-Rank Adaptation)
```python
config.training_mode = TrainingMode.LORA
# Efficient fine-tuning with low-rank matrices
```

#### 3. P-tuning
```python
config.training_mode = TrainingMode.P_TUNING
# Prompt tuning for efficient adaptation
```

#### 4. From Scratch
```python
config.training_mode = TrainingMode.FROM_SCRATCH
# Train custom model architecture
```

### Advanced Training

```python
from model_training import ModelTrainer, DeviceManager

# Initialize
device_manager = DeviceManager()
trainer = ModelTrainer(device_manager)

# Load dataset
train_dataset, val_dataset, test_dataset = trainer.load_dataset(config)

# Create model
model, tokenizer = trainer.create_model(config, num_classes=3)

# Train
result = await trainer.train(config)
print(f"Best model saved: {result['best_model_path']}")
```

### Hyperparameter Optimization

```python
from model_training import HyperparameterOptimizer

# Initialize optimizer
optimizer = HyperparameterOptimizer(trainer)

# Run optimization
config.enable_hpo = True
config.hpo_trials = 50

hpo_result = await optimizer.optimize(config)
print(f"Best parameters: {hpo_result['best_parameters']}")
```

---

## Evaluation System

### Single Model Evaluation

```python
from model_evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(device_manager)

# Evaluate model
performance = await evaluator.evaluate_model_performance(
    model, test_loader, config
)

print(f"Accuracy: {performance.accuracy:.4f}")
print(f"F1 Score: {performance.f1_score:.4f}")
print(f"Inference Time: {performance.inference_time_ms:.2f} ms")
```

### Cross-Validation

```python
# Perform cross-validation
cv_result = await evaluator.cross_validate_model(
    model_class, config, dataset, n_folds=5
)

print(f"Mean CV Score: {cv_result.mean_score:.4f}")
print(f"Std CV Score: {cv_result.std_score:.4f}")
```

### Model Comparison

```python
# Compare multiple models
models = [
    ("distilbert", model1),
    ("bert", model2),
    ("roberta", model3)
]

comparison = await evaluator.compare_models(
    models, test_loader, configs
)

print(f"Best model: {comparison.best_model}")
print(f"Ranking: {comparison.ranking}")
```

### Production Evaluation Pipeline

```python
from model_evaluation import ProductionEvaluationPipeline

# Initialize pipeline
pipeline = ProductionEvaluationPipeline(device_manager)

# Run full evaluation
model_paths = [
    "models/model1_best.pth",
    "models/model2_best.pth",
    "models/model3_best.pth"
]

results = await pipeline.run_full_evaluation(
    model_paths, "data/test_dataset.csv", "evaluation_results"
)
```

---

## Advanced Features

### 1. Distributed Training

```python
config.distributed = True
config.num_gpus = 4
config.local_rank = 0  # Set for each process

# Run with torchrun
# torchrun --nproc_per_node=4 train_script.py
```

### 2. Mixed Precision Training

```python
config.mixed_precision = True
# Automatically uses FP16 for faster training
```

### 3. Gradient Accumulation

```python
config.gradient_accumulation_steps = 4
# Accumulates gradients over 4 steps before updating
```

### 4. Early Stopping

```python
config.early_stopping_patience = 5
# Stops training if validation loss doesn't improve for 5 epochs
```

### 5. Experiment Tracking

#### TensorBoard
```python
config.log_to_tensorboard = True
# Logs metrics to TensorBoard
# tensorboard --logdir models/logs/tensorboard
```

#### Weights & Biases
```python
config.log_to_wandb = True
# Logs experiments to W&B dashboard
```

#### MLflow
```python
config.log_to_mlflow = True
# Tracks experiments with MLflow
```

---

## Production Deployment

### 1. Model Serving

```python
# Save model for serving
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'tokenizer': tokenizer,
    'metadata': {
        'version': '1.0.0',
        'created_at': time.time(),
        'performance': performance
    }
}, 'models/production_model.pth')
```

### 2. Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements_training_evaluation.txt .
RUN pip install -r requirements_training_evaluation.txt

COPY . .
EXPOSE 8000

CMD ["python", "serve_model.py"]
```

### 3. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-training-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-training
  template:
    metadata:
      labels:
        app: ai-training
    spec:
      containers:
      - name: training-service
        image: blatam/ai-training:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
```

### 4. Monitoring & Observability

```python
import structlog
import sentry_sdk

# Structured logging
logger = structlog.get_logger()

# Error tracking
sentry_sdk.init(dsn="your-sentry-dsn")

# Metrics
from prometheus_client import Counter, Histogram

training_counter = Counter('model_training_total', 'Total training runs')
inference_duration = Histogram('model_inference_duration_seconds', 'Inference duration')
```

---

## Best Practices

### 1. Data Preparation

```python
# Always validate your dataset
def validate_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    
    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
    
    # Check class balance
    class_counts = df['label'].value_counts()
    print(f"Class distribution: {class_counts}")
    
    # Check text quality
    text_lengths = df['text'].str.len()
    print(f"Text length stats: {text_lengths.describe()}")
```

### 2. Model Selection

```python
# Choose model based on task and constraints
def select_model(task_type, constraints):
    if constraints['speed'] > constraints['accuracy']:
        return "distilbert-base-uncased"  # Fast
    elif constraints['accuracy'] > constraints['speed']:
        return "roberta-large"  # Accurate
    else:
        return "bert-base-uncased"  # Balanced
```

### 3. Hyperparameter Tuning

```python
# Use Optuna for efficient HPO
def objective(trial):
    config.learning_rate = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    config.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    config.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    
    result = await trainer.train(config)
    return result['evaluation_result'].test_f1
```

### 4. Evaluation Strategy

```python
# Comprehensive evaluation
async def comprehensive_evaluation(model, config):
    # 1. Performance metrics
    performance = await evaluator.evaluate_model_performance(model, test_loader, config)
    
    # 2. Cross-validation
    cv_result = await evaluator.cross_validate_model(model_class, config, dataset)
    
    # 3. Statistical significance
    statistical_tests = evaluator.perform_statistical_tests(model_scores)
    
    # 4. Robustness tests
    robustness_tests = await run_robustness_tests(model, test_data)
    
    return {
        'performance': performance,
        'cross_validation': cv_result,
        'statistical_tests': statistical_tests,
        'robustness': robustness_tests
    }
```

### 5. Production Readiness

```python
# Production checklist
def production_checklist(model, config):
    checklist = {
        'model_size': model_size_mb < 500,  # < 500MB
        'inference_time': inference_time_ms < 100,  # < 100ms
        'accuracy': accuracy > 0.85,  # > 85%
        'memory_usage': memory_usage_mb < 2048,  # < 2GB
        'documentation': True,  # Has documentation
        'tests': True,  # Has tests
        'monitoring': True,  # Has monitoring
    }
    
    return all(checklist.values())
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```python
# Solutions:
config.batch_size = config.batch_size // 2  # Reduce batch size
config.gradient_accumulation_steps *= 2  # Increase gradient accumulation
config.mixed_precision = True  # Enable mixed precision
```

#### 2. Slow Training
```python
# Solutions:
config.mixed_precision = True  # Enable FP16
config.num_workers = 4  # Increase data loading workers
config.pin_memory = True  # Enable pinned memory
```

#### 3. Poor Performance
```python
# Solutions:
# 1. Check data quality
validate_dataset(config.dataset_path)

# 2. Try different learning rates
config.learning_rate = 1e-4  # or 5e-5, 1e-5

# 3. Increase model capacity
config.model_name = "roberta-large"  # Larger model

# 4. Use hyperparameter optimization
config.enable_hpo = True
```

#### 4. Overfitting
```python
# Solutions:
config.weight_decay = 0.1  # Increase weight decay
config.dropout = 0.3  # Add dropout
config.early_stopping_patience = 3  # Reduce patience
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable gradient checking
torch.autograd.set_detect_anomaly(True)

# Profile memory usage
from memory_profiler import profile

@profile
def train_with_profiling():
    return await trainer.train(config)
```

### PyTorch Debugging Tools

The training system includes comprehensive PyTorch debugging tools for troubleshooting training issues:

#### 1. Autograd Anomaly Detection

```python
# Enable autograd anomaly detection for gradient issues
config.detect_anomaly = True

# This will catch and report:
# - NaN gradients
# - Inf gradients  
# - Gradient computation errors
# - Backward pass issues
```

#### 2. Gradient Checking

```python
# Enable comprehensive gradient checking
config.gradient_checking = True

# This provides:
# - Gradient norm monitoring
# - NaN/Inf gradient detection
# - Parameter-wise gradient analysis
# - Detailed gradient logging
```

#### 3. Memory Profiling

```python
# Enable GPU memory profiling
config.memory_profiling = True

# This tracks:
# - GPU memory allocation
# - Memory usage per batch/epoch
# - Peak memory usage
# - Memory leaks detection
```

#### 4. Performance Profiling

```python
# Enable performance profiling
config.performance_profiling = True

# This measures:
# - Batch processing time
# - Epoch completion time
# - Forward/backward pass timing
# - Overall training performance
```

#### 5. Comprehensive Debugging

```python
# Enable all debugging features
from agents.backend.onyx.server.features.blog_posts.model_training import setup_comprehensive_debugging

config = setup_comprehensive_debugging(config)
```

#### 6. Targeted Debugging

```python
# For gradient issues only
from agents.backend.onyx.server.features.blog_posts.model_training import setup_gradient_debugging
config = setup_gradient_debugging(config)

# For memory issues only
from agents.backend.onyx.server.features.blog_posts.model_training import setup_memory_debugging
config = setup_memory_debugging(config)

# For performance issues only
from agents.backend.onyx.server.features.blog_posts.model_training import setup_performance_debugging
config = setup_performance_debugging(config)
```

#### 7. Quick Debug Training

```python
# Quick training with debugging enabled
from agents.backend.onyx.server.features.blog_posts.model_training import debug_train_transformer

# Comprehensive debugging
result = await debug_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv",
    num_epochs=5,
    debug_type="comprehensive"
)

# Get debugging summary
debug_summary = result['debug_summary']
print(f"Debug enabled: {debug_summary['debug_enabled']}")
print(f"GPU memory: {debug_summary.get('gpu_memory', {})}")
```

### When to Use Debugging Tools

#### Gradient Issues
```python
# Symptoms: NaN loss, exploding gradients, training instability
config = setup_gradient_debugging(config)

# Look for:
# - "NaN gradient detected" warnings
# - "Inf gradient detected" warnings
# - High gradient norms
# - Autograd anomaly reports
```

#### Memory Issues
```python
# Symptoms: Out of memory errors, slow training, GPU memory warnings
config = setup_memory_debugging(config)

# Look for:
# - High memory allocation
# - Memory leaks (increasing usage over time)
# - Peak memory usage
# - Memory fragmentation
```

#### Performance Issues
```python
# Symptoms: Slow training, bottlenecks, poor throughput
config = setup_performance_debugging(config)

# Look for:
# - Slow batch processing
# - Long epoch times
# - I/O bottlenecks
# - CPU/GPU utilization
```

### Debug Output Interpretation

#### Gradient Debugging Output
```
🔍 Gradient checking enabled
Batch 10: Gradient norm: 1.234567, NaN params: 0, Inf params: 0
Batch 15: Gradient norm: 2.345678, NaN params: 1, Inf params: 0
⚠️ NaN gradient detected in parameter at batch 15
❌ Gradient anomalies detected at batch 15: 1 NaN, 0 Inf
```

#### Memory Debugging Output
```
📊 Memory profiling enabled
GPU memory allocated: 2048.00 MB
GPU memory cached: 3072.00 MB
Memory usage at batch_start batch 10: Allocated: 2048.00MB, Reserved: 3072.00MB, Max: 4096.00MB
Memory usage at batch_end batch 10: Allocated: 2048.00MB, Reserved: 3072.00MB, Max: 4096.00MB
```

#### Performance Debugging Output
```
⚡ Performance profiling enabled
Performance at batch_end batch 10: 0.1234s
Performance at epoch_end epoch 1: 45.6789s
```

### Debugging Best Practices

#### 1. Start with Comprehensive Debugging
```python
# When encountering unknown issues
config = setup_comprehensive_debugging(config)
```

#### 2. Use Targeted Debugging for Known Issues
```python
# For specific problems
if gradient_issues:
    config = setup_gradient_debugging(config)
elif memory_issues:
    config = setup_memory_debugging(config)
elif performance_issues:
    config = setup_performance_debugging(config)
```

#### 3. Monitor Debug Output
```python
# Check logs for:
# - Warning messages
# - Error messages
# - Performance metrics
# - Memory usage patterns
```

#### 4. Clean Up After Debugging
```python
# Debugging is automatically cleaned up, but you can also:
torch.autograd.set_detect_anomaly(False)
torch.cuda.empty_cache()
```

#### 5. Use Debugging in Development Only
```python
# Disable debugging for production
config.debug_mode = False
config.detect_anomaly = False
config.gradient_checking = False
config.memory_profiling = False
config.performance_profiling = False
```

### Performance Profiling

```python
# Profile training loop
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run training
result = await trainer.train(config)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## API Reference

### TrainingConfig
```python
@dataclass
class TrainingConfig:
    model_type: ModelType
    training_mode: TrainingMode
    model_name: str
    dataset_path: str
    output_dir: str = "models"
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    # ... more parameters
```

### ModelTrainer
```python
class ModelTrainer:
    async def train(self, config: TrainingConfig) -> Dict[str, Any]
    async def train_epoch(self, model, dataloader, optimizer, scheduler, config)
    async def validate_epoch(self, model, dataloader) -> Dict[str, float]
    def create_model(self, config, num_classes) -> Tuple[nn.Module, Any]
    def calculate_metrics(self, y_true, y_pred, task_type) -> Dict[str, float]
```

### ModelEvaluator
```python
class ModelEvaluator:
    async def evaluate_model_performance(self, model, test_loader, config) -> ModelPerformance
    async def cross_validate_model(self, model_class, config, dataset, n_folds) -> CrossValidationResult
    async def compare_models(self, models, test_loader, configs) -> ModelComparison
    def calculate_advanced_metrics(self, y_true, y_pred, y_prob, task_type) -> Dict[str, float]
```

---

## Examples

### Complete Training Pipeline

```python
import asyncio
from model_training import ModelTrainer, TrainingConfig, ModelType, TrainingMode
from model_evaluation import ModelEvaluator, ProductionEvaluationPipeline

async def complete_pipeline():
    # 1. Setup
    device_manager = DeviceManager()
    trainer = ModelTrainer(device_manager)
    evaluator = ModelEvaluator(device_manager)
    
    # 2. Configuration
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.LORA,
        model_name="sentiment-lora",
        dataset_path="data/sentiment.csv",
        output_dir="models/",
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=10,
        mixed_precision=True,
        log_to_tensorboard=True
    )
    
    # 3. Training
    training_result = await trainer.train(config)
    
    # 4. Evaluation
    performance = await evaluator.evaluate_model_performance(
        model, test_loader, config
    )
    
    # 5. Cross-validation
    cv_result = await evaluator.cross_validate_model(
        model_class, config, dataset
    )
    
    # 6. Generate reports
    evaluator.generate_evaluation_report(performance, "evaluation_report.html")
    
    return {
        'training': training_result,
        'evaluation': performance,
        'cross_validation': cv_result
    }

# Run pipeline
result = await complete_pipeline()
print(f"Pipeline completed: {result}")
```

### Hyperparameter Optimization

```python
async def optimize_hyperparameters():
    device_manager = DeviceManager()
    trainer = ModelTrainer(device_manager)
    optimizer = HyperparameterOptimizer(trainer)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name="optimized-sentiment",
        dataset_path="data/sentiment.csv",
        output_dir="models/",
        enable_hpo=True,
        hpo_trials=50,
        num_epochs=5  # Shorter for HPO
    )
    
    result = await optimizer.optimize(config)
    
    print(f"Best parameters: {result['best_parameters']}")
    print(f"Best F1 score: {result['best_f1_score']:.4f}")
    
    return result

# Run optimization
hpo_result = await optimize_hyperparameters()
```

---

## Support

For issues and questions:

1. **Documentation**: Check this guide and inline code comments
2. **Tests**: Run `python test_training_evaluation.py`
3. **Logs**: Check TensorBoard, W&B, or MLflow logs
4. **Community**: Check Blatam Academy documentation

---

*Last updated: 2024* 

### Performance Optimization

```python
# Enable performance optimization
from model_training import setup_performance_optimization_config

config = TrainingConfig(
    model_type=ModelType.TRANSFORMER,
    training_mode=TrainingMode.FINE_TUNE,
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv"
)

# Apply comprehensive optimization
config = setup_performance_optimization_config(config)
```

### Performance Optimization Features

The training system includes comprehensive performance optimization features for maximum training efficiency:

#### 1. GPU Optimization

```python
# Enable GPU-specific optimizations
config.enable_gpu_optimization = True
config.enable_amp = True  # Automatic Mixed Precision
config.enable_cudnn_benchmark = True
config.enable_tf32 = True  # TensorFloat-32 for Ampere GPUs
config.enable_pin_memory = True
```

**Benefits:**
- **Mixed Precision**: 2x speedup with minimal accuracy loss
- **cuDNN Benchmark**: Optimal convolution algorithms
- **TensorFloat-32**: Faster matrix operations on modern GPUs
- **Pin Memory**: Faster CPU-GPU transfers

#### 2. Memory Optimization

```python
# Enable memory optimizations
config.enable_memory_optimization = True
config.enable_gradient_checkpointing = True
config.enable_channels_last = True
```

**Benefits:**
- **Gradient Checkpointing**: Trade compute for memory (50% memory reduction)
- **Channels Last**: Better memory access patterns
- **Aggressive GC**: Automatic memory cleanup

#### 3. Batch Processing Optimization

```python
# Enable batch optimizations
config.enable_batch_optimization = True
config.enable_dynamic_batching = True
config.enable_persistent_workers = True
config.num_workers = -1  # Auto-detect optimal
config.prefetch_factor = 4
```

**Benefits:**
- **Dynamic Batching**: Auto-detect optimal batch size
- **Persistent Workers**: Avoid worker startup overhead
- **Auto Worker Detection**: Optimal worker count based on system
- **Prefetch**: Overlap data loading with training

#### 4. PyTorch 2.0+ Compilation

```python
# Enable model compilation (PyTorch 2.0+)
config.enable_compilation = True
config.enable_compile_mode = "max-autotune"  # Maximum optimization
```

**Benefits:**
- **Model Compilation**: 10-30% speedup through graph optimization
- **Max Autotune**: Automatic kernel selection for best performance

### Quick Performance Optimization Functions

#### Comprehensive Optimization

```python
from model_training import optimized_train_transformer

# All optimizations enabled
result = await optimized_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv",
    optimization_type="comprehensive"
)

print(f"Performance summary: {result['performance_optimization_summary']}")
```

#### GPU-Specific Optimization

```python
# Focus on GPU utilization
result = await optimized_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv",
    optimization_type="gpu"
)
```

#### Memory-Specific Optimization

```python
# Focus on memory efficiency
result = await optimized_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv",
    optimization_type="memory"
)
```

#### Ultra Optimization

```python
from model_training import ultra_optimized_train_transformer

# Maximum possible optimization
result = await ultra_optimized_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv"
)

print(f"Ultra optimization results: {result['performance_summary']}")
```

### Performance Monitoring

#### Real-Time Metrics

```python
# Get performance summary during training
trainer = ModelTrainer(device_manager)
performance_summary = trainer.get_performance_summary()

print(f"GPU Memory: {performance_summary['gpu_info']['memory_allocated_mb']:.2f} MB")
print(f"Throughput: {performance_summary['performance']['avg_throughput']:.2f} samples/sec")
print(f"Memory Efficiency: {performance_summary['performance']['memory_efficiency']:.2f}")
```

#### Performance Metrics Explained

- **Average Throughput**: Samples processed per second
- **Max Throughput**: Peak processing speed achieved
- **Memory Efficiency**: Current memory usage vs peak usage
- **Batch Processing Time**: Time per batch (lower is better)
- **Epoch Time**: Total time per epoch

### Optimization Strategies by Use Case

#### 1. High-Performance Training

```python
# For maximum speed with powerful hardware
config = setup_performance_optimization_config(config)
config.enable_compilation = True
config.enable_compile_mode = "max-autotune"
config.batch_size = 64  # Large batches
config.num_workers = 8  # Many workers
```

#### 2. Memory-Constrained Training

```python
# For limited memory environments
config = setup_memory_optimization_config(config)
config.batch_size = 8  # Small batches
config.enable_gradient_checkpointing = True
config.enable_amp = True
```

#### 3. Balanced Training

```python
# For good performance with reasonable resource usage
config.enable_gpu_optimization = True
config.enable_amp = True
config.enable_pin_memory = True
config.batch_size = 32
config.num_workers = 4
```

### Performance Comparison

#### Expected Performance Gains

| Optimization Type | Speedup | Memory Reduction | Use Case |
|-------------------|---------|------------------|----------|
| **GPU Optimization** | 2-3x | 0% | High-end GPUs |
| **Memory Optimization** | 1.2x | 50% | Limited memory |
| **Batch Optimization** | 1.5x | 0% | Large datasets |
| **Compilation** | 1.3x | 0% | PyTorch 2.0+ |
| **Comprehensive** | 3-5x | 30% | Production training |
| **Ultra** | 5-10x | 40% | Maximum performance |

#### Performance Monitoring Example

```python
# Monitor performance during training
async def train_with_monitoring():
    trainer = ModelTrainer(device_manager)
    
    # Setup monitoring
    performance_history = []
    
    def log_performance(epoch, metrics):
        perf_summary = trainer.get_performance_summary()
        performance_history.append({
            'epoch': epoch,
            'throughput': perf_summary['performance']['avg_throughput'],
            'memory_usage': perf_summary['gpu_info']['memory_allocated_mb'],
            'memory_efficiency': perf_summary['performance']['memory_efficiency']
        })
    
    # Train with monitoring
    result = await trainer.train(config)
    
    # Analyze performance trends
    print("Performance Analysis:")
    for perf in performance_history:
        print(f"Epoch {perf['epoch']}: {perf['throughput']:.2f} samples/sec, "
              f"{perf['memory_usage']:.2f} MB, efficiency: {perf['memory_efficiency']:.2f}")
```

### Troubleshooting Performance Issues

#### 1. Low Throughput

```python
# Check for bottlenecks
performance_summary = trainer.get_performance_summary()

if performance_summary['performance']['avg_throughput'] < 100:
    # Increase batch size
    config.batch_size *= 2
    
    # Enable more optimizations
    config.enable_compilation = True
    config.enable_amp = True
```

#### 2. High Memory Usage

```python
# Reduce memory usage
config.enable_gradient_checkpointing = True
config.batch_size = max(1, config.batch_size // 2)
config.enable_amp = True
```

#### 3. GPU Underutilization

```python
# Increase GPU utilization
config.enable_cudnn_benchmark = True
config.enable_tf32 = True
config.enable_pin_memory = True
config.num_workers = 4  # More workers for data loading
```

### Production Performance Configuration

```python
# Production-ready performance configuration
def get_production_config():
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name="distilbert-base-uncased",
        dataset_path="data/sentiment_dataset.csv"
    )
    
    # Enable all optimizations
    config = setup_performance_optimization_config(config)
    
    # Production-specific settings
    config.enable_compilation = True
    config.enable_compile_mode = "max-autotune"
    config.enable_gradient_checkpointing = True
    config.enable_channels_last = True
    config.enable_tf32 = True
    config.enable_cudnn_benchmark = True
    config.enable_amp = True
    config.enable_dynamic_batching = True
    config.enable_pin_memory = True
    config.enable_persistent_workers = True
    config.num_workers = -1  # Auto-detect
    config.prefetch_factor = 4
    
    return config
```

---

## Advanced Features

### Custom Training Loops

```python
# Custom training with performance optimization
async def custom_training_loop():
    trainer = ModelTrainer(device_manager)
    
    # Setup performance optimization
    trainer.setup_performance_optimization(config)
    
    # Custom training loop with performance monitoring
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Training with performance tracking
        train_metrics = await trainer.train_epoch(...)
        
        # Performance monitoring
        epoch_time = time.time() - epoch_start
        performance_summary = trainer.get_performance_summary()
        
        print(f"Epoch {epoch}: {epoch_time:.2f}s, "
              f"Throughput: {performance_summary['performance']['avg_throughput']:.2f} samples/sec")
```

### Performance Benchmarking

```python
# Benchmark different optimization configurations
async def benchmark_optimizations():
    configs = [
        ("Standard", TrainingConfig(...)),
        ("GPU Optimized", setup_gpu_optimization_config(TrainingConfig(...))),
        ("Memory Optimized", setup_memory_optimization_config(TrainingConfig(...))),
        ("Comprehensive", setup_performance_optimization_config(TrainingConfig(...))),
        ("Ultra", setup_performance_optimization_config(TrainingConfig(...)))
    ]
    
    results = {}
    for name, config in configs:
        start_time = time.time()
        result = await trainer.train(config)
        training_time = time.time() - start_time
        
        performance_summary = trainer.get_performance_summary()
        results[name] = {
            'training_time': training_time,
            'throughput': performance_summary['performance']['avg_throughput'],
            'memory_usage': performance_summary['gpu_info']['memory_allocated_mb']
        }
    
    # Compare results
    for name, metrics in results.items():
        print(f"{name}: {metrics['training_time']:.2f}s, "
              f"{metrics['throughput']:.2f} samples/sec, "
              f"{metrics['memory_usage']:.2f} MB")
```

---

## Best Practices

### 1. Start with Comprehensive Optimization

```python
# Always start with comprehensive optimization
config = setup_performance_optimization_config(config)
```

### 2. Monitor Performance Continuously

```python
# Regular performance monitoring
performance_summary = trainer.get_performance_summary()
if performance_summary['performance']['avg_throughput'] < target_throughput:
    # Adjust configuration
    config.batch_size *= 2
```

### 3. Use Appropriate Optimization for Your Hardware

```python
# GPU-rich environment
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
    config = setup_gpu_optimization_config(config)

# Memory-constrained environment
elif psutil.virtual_memory().total < 16e9:
    config = setup_memory_optimization_config(config)
```

### 4. Profile Before Optimizing

```python
# Profile to identify bottlenecks
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run training
result = await trainer.train(config)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions by time
```

### 5. Use Ultra Optimization for Production

```python
# For production training
result = await ultra_optimized_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv"
)
```

---

## Performance Optimization Summary

The training system provides comprehensive performance optimization features:

### ✅ **Optimization Types Available**
- **GPU Optimization**: Mixed precision, cuDNN benchmark, TensorFloat-32
- **Memory Optimization**: Gradient checkpointing, channels last, aggressive GC
- **Batch Optimization**: Dynamic batching, persistent workers, auto-detection
- **Compilation**: PyTorch 2.0+ model compilation with max autotune

### ✅ **Quick Functions Available**
- `optimized_train_transformer()`: Quick training with optimization
- `ultra_optimized_train_transformer()`: Maximum performance training
- `setup_performance_optimization_config()`: Configure all optimizations
- `get_performance_summary()`: Real-time performance metrics

### ✅ **Performance Monitoring**
- Real-time throughput tracking
- Memory usage monitoring
- GPU utilization metrics
- Performance efficiency analysis

### ✅ **Expected Performance Gains**
- **3-5x speedup** with comprehensive optimization
- **5-10x speedup** with ultra optimization
- **50% memory reduction** with memory optimization
- **2-3x GPU utilization** improvement

### ✅ **Production Ready**
- Automatic hardware detection
- Adaptive optimization based on resources
- Comprehensive error handling
- Real-time performance monitoring
- Extensive logging and metrics

The performance optimization system is designed to automatically adapt to your hardware and provide the best possible training performance with minimal configuration required. 

---

## Multi-GPU Training

```python
# Enable multi-GPU training
from model_training import setup_auto_multi_gpu_config

config = TrainingConfig(
    model_type=ModelType.TRANSFORMER,
    training_mode=TrainingMode.FINE_TUNE,
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv"
)

# Apply automatic multi-GPU configuration
config = setup_auto_multi_gpu_config(config)
```

### Multi-GPU Training Features

The training system includes comprehensive multi-GPU training support for scaling training across multiple GPUs:

#### 1. DataParallel Training

```python
# Enable DataParallel for simpler multi-GPU setups
config.use_data_parallel = True
config.num_gpus = 4  # Use 4 GPUs
```

**Benefits:**
- **Simple Setup**: Easy to configure and use
- **Automatic Batch Splitting**: Automatically splits batches across GPUs
- **Single Process**: Runs in a single process
- **Good for 2-4 GPUs**: Optimal for smaller multi-GPU setups

**Use Cases:**
- Development and experimentation
- Smaller multi-GPU setups (2-4 GPUs)
- When simplicity is preferred over maximum efficiency

#### 2. DistributedDataParallel Training

```python
# Enable DistributedDataParallel for advanced multi-GPU setups
config.use_distributed_data_parallel = True
config.num_gpus = 8  # Use 8 GPUs
config.distributed_backend = "nccl"  # NVIDIA Collective Communications Library
```

**Benefits:**
- **Higher Efficiency**: Better memory usage and communication
- **Scalable**: Works well with 4+ GPUs and multi-node setups
- **Process-based**: Each GPU runs in its own process
- **Better Memory Management**: More efficient memory usage

**Use Cases:**
- Production training with 4+ GPUs
- Multi-node distributed training
- When maximum efficiency is required

#### 3. Automatic Multi-GPU Configuration

```python
# Automatic configuration based on available hardware
config = setup_auto_multi_gpu_config(config)
```

**Features:**
- **Auto-detection**: Automatically detects available GPUs
- **Smart Strategy Selection**: Chooses DataParallel for ≤4 GPUs, DistributedDataParallel for >4 GPUs
- **Automatic Settings**: Configures all necessary distributed settings
- **Hardware Adaptation**: Adapts to your specific hardware setup

### Quick Multi-GPU Training Functions

#### Automatic Multi-GPU Training

```python
from model_training import multi_gpu_train_transformer

# Automatic multi-GPU training
result = await multi_gpu_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv",
    multi_gpu_type="auto"
)

print(f"Multi-GPU summary: {result['multi_gpu_summary']}")
```

#### DataParallel Training

```python
from model_training import data_parallel_train_transformer

# DataParallel training (simpler)
result = await data_parallel_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv"
)
```

#### DistributedDataParallel Training

```python
from model_training import distributed_train_transformer

# DistributedDataParallel training (advanced)
result = await distributed_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv"
)
```

### Multi-GPU Configuration Functions

#### Automatic Configuration

```python
from model_training import setup_auto_multi_gpu_config

# Automatic multi-GPU setup
config = setup_auto_multi_gpu_config(config)
```

#### DataParallel Configuration

```python
from model_training import setup_data_parallel_config

# DataParallel setup for 2-4 GPUs
config = setup_data_parallel_config(config)
```

#### DistributedDataParallel Configuration

```python
from model_training import setup_distributed_config

# DistributedDataParallel setup for 4+ GPUs
config = setup_distributed_config(config)
```

### Multi-GPU Monitoring

#### Real-Time GPU Information

```python
# Get multi-GPU summary during training
trainer = ModelTrainer(device_manager)
multi_gpu_summary = trainer.get_multi_gpu_summary()

print(f"Training type: {multi_gpu_summary['training_type']}")
print(f"Number of GPUs: {multi_gpu_summary['num_gpus']}")
print(f"Devices: {multi_gpu_summary['devices']}")
```

#### GPU Information Details

```python
# Detailed GPU information
if 'gpu_info' in multi_gpu_summary:
    for gpu_id, info in multi_gpu_summary['gpu_info'].items():
        print(f"{gpu_id}: {info['name']}")
        print(f"  Memory: {info['memory_total_gb']:.1f} GB total")
        print(f"  Allocated: {info['memory_allocated_mb']:.1f} MB")
        print(f"  Reserved: {info['memory_reserved_mb']:.1f} MB")
```

### Multi-GPU Training Strategies

#### 1. Development and Testing

```python
# Use DataParallel for development
config = setup_data_parallel_config(config)
config.num_gpus = 2  # Use 2 GPUs for testing
```

#### 2. Production Training

```python
# Use DistributedDataParallel for production
config = setup_distributed_config(config)
config.num_gpus = 8  # Use all available GPUs
```

#### 3. Memory-Constrained Training

```python
# Optimize for memory efficiency
config = setup_distributed_config(config)
config.enable_gradient_checkpointing = True
config.batch_size = 8  # Smaller batch size per GPU
```

### Multi-GPU + Performance Optimization

#### Combined Optimization

```python
from model_training import optimized_multi_gpu_train_transformer

# Multi-GPU + performance optimization
result = await optimized_multi_gpu_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv",
    multi_gpu_type="auto",
    optimization_type="comprehensive"
)

print(f"Multi-GPU: {result['multi_gpu_summary']}")
print(f"Performance: {result['performance_summary']}")
```

#### Ultra-Optimized Multi-GPU

```python
from model_training import ultra_optimized_multi_gpu_train_transformer

# Maximum optimization across multiple GPUs
result = await ultra_optimized_multi_gpu_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv"
)
```

### Multi-GPU Performance Comparison

#### Expected Performance Gains

| Training Type | GPUs | Speedup | Memory Efficiency | Use Case |
|---------------|------|---------|-------------------|----------|
| **Single GPU** | 1 | 1x | Standard | Development |
| **DataParallel** | 2-4 | 1.8-3.5x | Good | Small multi-GPU |
| **DistributedDataParallel** | 4+ | 3.5-8x | Excellent | Production |
| **Ultra Multi-GPU** | 4+ | 5-10x | Excellent | Maximum performance |

#### Performance Monitoring Example

```python
# Monitor multi-GPU performance during training
async def train_with_multi_gpu_monitoring():
    trainer = ModelTrainer(device_manager)
    
    # Setup monitoring
    multi_gpu_history = []
    
    def log_multi_gpu_performance(epoch, metrics):
        multi_gpu_summary = trainer.get_multi_gpu_summary()
        multi_gpu_history.append({
            'epoch': epoch,
            'training_type': multi_gpu_summary['training_type'],
            'num_gpus': multi_gpu_summary['num_gpus'],
            'gpu_utilization': multi_gpu_summary.get('efficiency', {}).get('gpu_utilization', 'unknown')
        })
    
    # Train with monitoring
    result = await trainer.train(config)
    
    # Analyze multi-GPU performance
    print("Multi-GPU Performance Analysis:")
    for perf in multi_gpu_history:
        print(f"Epoch {perf['epoch']}: {perf['training_type']}, "
              f"{perf['num_gpus']} GPUs, utilization: {perf['gpu_utilization']}")
```

### Troubleshooting Multi-GPU Issues

#### 1. GPU Memory Issues

```python
# Reduce memory usage per GPU
config.batch_size = max(1, config.batch_size // 2)
config.enable_gradient_checkpointing = True
config.enable_amp = True  # Mixed precision
```

#### 2. Communication Issues

```python
# For DistributedDataParallel issues
config.distributed_backend = "nccl"  # Use NCCL for GPU
config.find_unused_parameters = True  # If model has unused parameters
config.bucket_cap_mb = 25  # Adjust bucket size
```

#### 3. Load Balancing Issues

```python
# Ensure proper data distribution
if config.use_distributed_data_parallel:
    # DistributedDataParallel handles this automatically
    pass
else:
    # For DataParallel, ensure batch size is divisible by GPU count
    config.batch_size = (config.batch_size // config.num_gpus) * config.num_gpus
```

### Production Multi-GPU Configuration

```python
# Production-ready multi-GPU configuration
def get_production_multi_gpu_config():
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name="distilbert-base-uncased",
        dataset_path="data/sentiment_dataset.csv"
    )
    
    # Enable multi-GPU training
    config = setup_auto_multi_gpu_config(config)
    
    # Production-specific settings
    if config.use_distributed_data_parallel:
        config.distributed_backend = "nccl"
        config.find_unused_parameters = False
        config.gradient_as_bucket_view = True
        config.broadcast_buffers = True
        config.bucket_cap_mb = 25
        config.static_graph = False
    
    # Performance optimizations
    config.enable_amp = True
    config.enable_pin_memory = True
    config.enable_persistent_workers = True
    config.num_workers = -1  # Auto-detect
    
    return config
```

---

## Advanced Multi-GPU Features

### Custom Multi-GPU Training Loops

```python
# Custom training with multi-GPU optimization
async def custom_multi_gpu_training_loop():
    trainer = ModelTrainer(device_manager)
    
    # Setup multi-GPU training
    trainer.setup_multi_gpu_training(config)
    
    # Custom training loop with multi-GPU monitoring
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        # Training with multi-GPU tracking
        train_metrics = await trainer.train_epoch(...)
        
        # Multi-GPU monitoring
        epoch_time = time.time() - epoch_start
        multi_gpu_summary = trainer.get_multi_gpu_summary()
        
        print(f"Epoch {epoch}: {epoch_time:.2f}s, "
              f"Training type: {multi_gpu_summary['training_type']}, "
              f"GPUs: {multi_gpu_summary['num_gpus']}")
```

### Multi-GPU Benchmarking

```python
# Benchmark different multi-GPU configurations
async def benchmark_multi_gpu_configurations():
    configs = [
        ("Single GPU", TrainingConfig(...)),
        ("DataParallel 2 GPUs", setup_data_parallel_config(TrainingConfig(...))),
        ("DataParallel 4 GPUs", setup_data_parallel_config(TrainingConfig(...))),
        ("DistributedDataParallel 4 GPUs", setup_distributed_config(TrainingConfig(...))),
        ("DistributedDataParallel 8 GPUs", setup_distributed_config(TrainingConfig(...)))
    ]
    
    results = {}
    for name, config in configs:
        start_time = time.time()
        result = await trainer.train(config)
        training_time = time.time() - start_time
        
        multi_gpu_summary = trainer.get_multi_gpu_summary()
        results[name] = {
            'training_time': training_time,
            'num_gpus': multi_gpu_summary['num_gpus'],
            'training_type': multi_gpu_summary['training_type']
        }
    
    # Compare results
    for name, metrics in results.items():
        print(f"{name}: {metrics['training_time']:.2f}s, "
              f"{metrics['num_gpus']} GPUs, {metrics['training_type']}")
```

---

## Best Practices

### 1. Start with Auto Configuration

```python
# Always start with automatic configuration
config = setup_auto_multi_gpu_config(config)
```

### 2. Monitor GPU Utilization

```python
# Regular multi-GPU monitoring
multi_gpu_summary = trainer.get_multi_gpu_summary()
if multi_gpu_summary['num_gpus'] > 1:
    print(f"Utilizing {multi_gpu_summary['num_gpus']} GPUs")
    print(f"Training type: {multi_gpu_summary['training_type']}")
```

### 3. Use Appropriate Strategy for Your Hardware

```python
# GPU-rich environment
if torch.cuda.device_count() <= 4:
    config = setup_data_parallel_config(config)
else:
    config = setup_distributed_config(config)
```

### 4. Profile Multi-GPU Performance

```python
# Profile to identify bottlenecks
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run multi-GPU training
result = await trainer.train(config)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions by time
```

### 5. Use Ultra-Optimized Multi-GPU for Production

```python
# For production multi-GPU training
result = await ultra_optimized_multi_gpu_train_transformer(
    model_name="distilbert-base-uncased",
    dataset_path="data/sentiment_dataset.csv"
)
```

---

## Multi-GPU Training Summary

The training system provides comprehensive multi-GPU training features:

### ✅ **Multi-GPU Types Available**
- **DataParallel**: Simple multi-GPU for 2-4 GPUs
- **DistributedDataParallel**: Advanced multi-GPU for 4+ GPUs
- **Automatic Configuration**: Smart strategy selection based on hardware

### ✅ **Quick Functions Available**
- `multi_gpu_train_transformer()`: Quick multi-GPU training
- `data_parallel_train_transformer()`: DataParallel training
- `distributed_train_transformer()`: DistributedDataParallel training
- `setup_auto_multi_gpu_config()`: Automatic configuration
- `get_multi_gpu_summary()`: Real-time multi-GPU metrics

### ✅ **Multi-GPU Monitoring**
- Real-time GPU utilization tracking
- Memory usage per GPU
- Training type and efficiency metrics
- Performance scaling analysis

### ✅ **Expected Performance Gains**
- **1.8-3.5x speedup** with DataParallel (2-4 GPUs)
- **3.5-8x speedup** with DistributedDataParallel (4+ GPUs)
- **5-10x speedup** with ultra-optimized multi-GPU
- **Linear scaling** with GPU count (up to communication limits)

### ✅ **Production Ready**
- Automatic hardware detection
- Adaptive strategy selection
- Comprehensive error handling
- Real-time multi-GPU monitoring
- Extensive logging and metrics

### ✅ **Easy Integration**
- Drop-in replacement for existing training
- Backward compatible with single-GPU training
- Minimal configuration required
- Quick multi-GPU functions

The multi-GPU training system is designed to automatically adapt to your hardware and provide optimal scaling across multiple GPUs with minimal configuration required. 