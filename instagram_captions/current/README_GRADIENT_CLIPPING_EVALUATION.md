# Gradient Clipping, NaN/Inf Handling & Evaluation Metrics System

## Overview

This comprehensive system implements robust gradient clipping, proper handling of NaN/Inf values, and advanced evaluation metrics for production-ready deep learning workflows. The system is designed to handle various NLP tasks including classification, regression, and text generation with appropriate metrics for each task type.

## Key Features

### 🔧 Gradient Clipping & NaN/Inf Handling
- **Advanced Gradient Clipping**: Support for both norm-based and value-based gradient clipping
- **Comprehensive NaN/Inf Detection**: Monitors gradients, weights, and loss values
- **Recovery Strategies**: Automatic gradient zeroing, batch skipping, and training restart options
- **Real-time Monitoring**: Continuous tracking of gradient statistics and anomaly detection
- **Performance Profiling**: Built-in PyTorch profiler integration for optimization

### 📊 Advanced Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, Cohen's Kappa, Matthews Correlation
- **Regression Metrics**: MSE, MAE, RMSE, R², MAPE, SMAPE, Huber Loss, Log-Cosh Loss
- **Text Generation Metrics**: Perplexity, BLEU, ROUGE, METEOR, BERT Score, Semantic Similarity
- **Custom Metrics**: Extensible framework for custom metric implementations
- **Multi-task Evaluation**: Support for evaluating multiple tasks simultaneously

### 🚀 Production-Ready Features
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support
- **Multi-GPU Training**: DataParallel and DistributedDataParallel support
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Checkpointing**: Automatic model saving with best model preservation
- **Early Stopping**: Configurable early stopping with patience
- **Error Handling**: Robust error handling and recovery mechanisms

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Comprehensive Training System                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Gradient System │  │ Evaluation      │  │ Training     │ │
│  │                 │  │ System          │  │ System       │ │
│  │ • Monitoring    │  │ • Classification│  │ • DataLoader │ │
│  │ • Clipping      │  │ • Regression    │  │ • Optimizer  │ │
│  │ • NaN/Inf       │  │ • Text Gen      │  │ • Scheduler  │ │
│  │ • Recovery      │  │ • Custom        │  │ • Checkpoint │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
current/
├── gradient_clipping_nan_handling_system.py      # Core gradient handling
├── advanced_evaluation_metrics_system.py         # Comprehensive metrics
├── comprehensive_training_system.py              # Integrated training system
├── requirements_nlp_optimized.txt                # Dependencies
├── config/
│   └── nlp_config.yaml                          # Configuration files
├── README_GRADIENT_CLIPPING_EVALUATION.md       # This documentation
└── examples/
    ├── classification_example.py                # Classification training
    ├── regression_example.py                    # Regression training
    └── text_generation_example.py               # Text generation training
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements_nlp_optimized.txt
```

### 2. Basic Usage

```python
from comprehensive_training_system import ComprehensiveTrainingSystem, ComprehensiveTrainingConfig
from gradient_clipping_nan_handling_system import GradientConfig
from advanced_evaluation_metrics_system import ClassificationMetricsConfig

# Configuration
config = ComprehensiveTrainingConfig(
    model_name="gpt2",
    model_type="causal_lm",
    batch_size=16,
    learning_rate=2e-5,
    gradient_clip_val=1.0,
    gradient_clip_norm=1.0,
    check_gradients=True,
    check_weights=True,
    check_loss=True,
    zero_nan_gradients=True
)

# Create training system
training_system = ComprehensiveTrainingSystem(config)

# Load model and setup training
training_system.load_model()
training_system.setup_training(train_dataset, val_dataset)

# Start training
results = training_system.train(task_type="classification")
```

### 3. Advanced Configuration

```python
# Gradient configuration
gradient_config = GradientConfig(
    gradient_clip_val=1.0,
    gradient_clip_norm=1.0,
    gradient_clip_algorithm="norm",  # "norm" or "value"
    check_gradients=True,
    check_weights=True,
    check_loss=True,
    zero_nan_gradients=True,
    skip_nan_batches=False,
    restart_training_on_nan=False,
    max_nan_restarts=3
)

# Evaluation configuration
evaluation_config = ClassificationMetricsConfig(
    compute_accuracy=True,
    compute_precision=True,
    compute_recall=True,
    compute_f1=True,
    compute_roc_auc=True,
    compute_pr_auc=True,
    compute_confusion_matrix=True,
    compute_classification_report=True,
    compute_cohen_kappa=True,
    compute_matthews_corrcoef=True,
    average_method="weighted"
)
```

## Detailed Components

### 1. Gradient Clipping & NaN/Inf Handling System

#### GradientMonitor
Monitors and handles gradient issues in real-time:

```python
from gradient_clipping_nan_handling_system import GradientMonitor, GradientConfig

# Setup monitoring
gradient_config = GradientConfig(
    check_gradients=True,
    check_weights=True,
    check_loss=True,
    zero_nan_gradients=True
)

monitor = GradientMonitor(gradient_config)

# Check gradients during training
grad_stats = monitor.check_gradients(model)
weight_stats = monitor.check_weights(model)
loss_valid = monitor.check_loss(loss)
```

#### GradientClipper
Implements advanced gradient clipping strategies:

```python
from gradient_clipping_nan_handling_system import GradientClipper

clipper = GradientClipper(gradient_config)

# Clip gradients by norm
grad_norm = clipper.clip_gradients(model, optimizer)

# Or clip by value
grad_config.gradient_clip_algorithm = "value"
clipper.clip_gradients(model, optimizer)
```

#### RobustTrainingSystem
Complete training system with robust gradient handling:

```python
from gradient_clipping_nan_handling_system import RobustTrainingSystem

training_system = RobustTrainingSystem(model, gradient_config, evaluation_config)
training_system.setup_training(learning_rate=2e-5, weight_decay=0.01)

# Training step with automatic gradient handling
step_metrics = training_system.train_step(batch, task_type="classification")
```

### 2. Advanced Evaluation Metrics System

#### Classification Metrics
Comprehensive classification evaluation:

```python
from advanced_evaluation_metrics_system import AdvancedClassificationMetrics, ClassificationMetricsConfig

config = ClassificationMetricsConfig(
    compute_accuracy=True,
    compute_precision=True,
    compute_recall=True,
    compute_f1=True,
    compute_roc_auc=True,
    compute_pr_auc=True,
    compute_confusion_matrix=True,
    compute_classification_report=True,
    compute_cohen_kappa=True,
    compute_matthews_corrcoef=True
)

evaluator = AdvancedClassificationMetrics(config)

# Evaluate classification
metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
per_class_metrics = evaluator.compute_per_class_metrics(y_true, y_pred, y_proba)
```

#### Regression Metrics
Advanced regression evaluation:

```python
from advanced_evaluation_metrics_system import AdvancedRegressionMetrics, RegressionMetricsConfig

config = RegressionMetricsConfig(
    compute_mse=True,
    compute_mae=True,
    compute_rmse=True,
    compute_r2=True,
    compute_mape=True,
    compute_smape=True,
    compute_huber_loss=True,
    compute_correlation=True
)

evaluator = AdvancedRegressionMetrics(config)
metrics = evaluator.compute_metrics(y_true, y_pred)
```

#### Text Generation Metrics
Comprehensive text generation evaluation:

```python
from advanced_evaluation_metrics_system import AdvancedTextGenerationMetrics, TextGenerationMetricsConfig

config = TextGenerationMetricsConfig(
    compute_perplexity=True,
    compute_bleu=True,
    compute_rouge=True,
    compute_meteor=True,
    compute_bert_score=True,
    compute_semantic_similarity=True
)

evaluator = AdvancedTextGenerationMetrics(config)
metrics = evaluator.compute_metrics(references, predictions)
```

#### Custom Metrics
Extensible custom metrics framework:

```python
from advanced_evaluation_metrics_system import CustomMetricsEvaluator, CustomMetricsConfig

def custom_f1_score(y_true, y_pred, **kwargs):
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)

def custom_balanced_accuracy(y_true, y_pred, **kwargs):
    return balanced_accuracy_score(y_true, y_pred)

config = CustomMetricsConfig(
    custom_metrics={
        "custom_f1": custom_f1_score,
        "balanced_accuracy": custom_balanced_accuracy
    },
    metric_weights={
        "accuracy": 0.3,
        "f1": 0.3,
        "roc_auc": 0.2,
        "custom_f1": 0.2
    }
)

evaluator = CustomMetricsEvaluator(config)
custom_metrics = evaluator.compute_custom_metrics(y_true, y_pred)
weighted_score = evaluator.compute_weighted_score(metrics)
```

### 3. Comprehensive Training System

#### Complete Training Workflow
Integrated training system with all features:

```python
from comprehensive_training_system import ComprehensiveTrainingSystem, ComprehensiveTrainingConfig

# Configuration
config = ComprehensiveTrainingConfig(
    model_name="gpt2",
    model_type="causal_lm",
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=10,
    gradient_clip_val=1.0,
    gradient_clip_norm=1.0,
    check_gradients=True,
    check_weights=True,
    check_loss=True,
    zero_nan_gradients=True,
    evaluation_interval=100,
    save_interval=1000,
    log_interval=10,
    use_tensorboard=True,
    save_best_model=True,
    save_last_model=True
)

# Create and setup training system
training_system = ComprehensiveTrainingSystem(config)
training_system.load_model()
training_system.setup_training(train_dataset, val_dataset)

# Start training
results = training_system.train(task_type="classification")
```

## Configuration Options

### Gradient Configuration
- `gradient_clip_val`: Maximum gradient value for value-based clipping
- `gradient_clip_norm`: Maximum gradient norm for norm-based clipping
- `gradient_clip_algorithm`: "norm" or "value"
- `check_gradients`: Enable gradient monitoring
- `check_weights`: Enable weight monitoring
- `check_loss`: Enable loss monitoring
- `zero_nan_gradients`: Automatically zero NaN gradients
- `skip_nan_batches`: Skip batches with NaN values
- `restart_training_on_nan`: Restart training when NaN detected
- `max_nan_restarts`: Maximum number of restart attempts

### Evaluation Configuration
- `compute_accuracy`: Enable accuracy computation
- `compute_precision`: Enable precision computation
- `compute_recall`: Enable recall computation
- `compute_f1`: Enable F1 score computation
- `compute_roc_auc`: Enable ROC AUC computation
- `compute_pr_auc`: Enable PR AUC computation
- `compute_confusion_matrix`: Enable confusion matrix
- `compute_classification_report`: Enable detailed classification report
- `average_method`: Averaging method for multi-class metrics

### Training Configuration
- `model_name`: Pre-trained model name
- `model_type`: Model type ("causal_lm", "sequence_classification", "custom")
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `use_amp`: Enable Automatic Mixed Precision
- `use_data_parallel`: Enable DataParallel for multi-GPU
- `use_distributed`: Enable DistributedDataParallel
- `evaluation_interval`: Steps between evaluations
- `save_interval`: Steps between checkpoints
- `log_interval`: Steps between logging

## Monitoring & Logging

### TensorBoard Integration
```python
# Training metrics are automatically logged to TensorBoard
# View with: tensorboard --logdir runs
```

### Weights & Biases Integration
```python
config = ComprehensiveTrainingConfig(
    use_wandb=True,
    wandb_project="nlp_training"
)
```

### Custom Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training system automatically logs:
# - Training loss and metrics
# - Gradient statistics
# - NaN/Inf detection
# - Model checkpoints
# - Performance profiling
```

## Performance Optimization

### Mixed Precision Training
```python
config = ComprehensiveTrainingConfig(
    use_amp=True,  # Enable Automatic Mixed Precision
    gradient_accumulation_steps=4  # Gradient accumulation for larger effective batch size
)
```

### Multi-GPU Training
```python
config = ComprehensiveTrainingConfig(
    use_data_parallel=True,  # For single machine, multiple GPUs
    use_distributed=True,    # For distributed training
    num_workers=4            # DataLoader workers
)
```

### Performance Profiling
```python
config = ComprehensiveTrainingConfig(
    enable_profiling=True,
    profile_interval=100
)

# Profiling results saved to checkpoints/training_profile.json
# View with Chrome DevTools or TensorBoard
```

## Error Handling & Recovery

### Automatic Recovery
```python
config = ComprehensiveTrainingConfig(
    zero_nan_gradients=True,      # Zero NaN gradients automatically
    skip_nan_batches=False,       # Skip problematic batches
    restart_training_on_nan=False, # Restart training on NaN detection
    max_nan_restarts=3            # Maximum restart attempts
)
```

### Manual Recovery
```python
# Check training statistics
stats = training_system.get_training_summary()
print(f"NaN rate: {stats['nan_rate']}")
print(f"Inf rate: {stats['inf_rate']}")

# Load from checkpoint
training_system.load_model("checkpoints/best_model.pt")
```

## Best Practices

### 1. Gradient Clipping
- Use norm-based clipping for most cases: `gradient_clip_algorithm="norm"`
- Set `gradient_clip_norm=1.0` as a good starting point
- Monitor gradient norms during training

### 2. NaN/Inf Handling
- Always enable gradient and weight checking
- Use `zero_nan_gradients=True` for automatic recovery
- Monitor NaN/Inf rates in training logs
- Consider reducing learning rate if NaN rates are high

### 3. Evaluation Metrics
- Choose metrics appropriate for your task
- Use weighted averaging for imbalanced datasets
- Implement custom metrics for domain-specific requirements
- Monitor multiple metrics to get comprehensive evaluation

### 4. Training Configuration
- Start with smaller batch sizes and increase gradually
- Use learning rate scheduling (CosineAnnealingLR)
- Enable early stopping to prevent overfitting
- Use checkpointing to save best models

### 5. Performance Optimization
- Enable AMP for faster training and lower memory usage
- Use appropriate number of DataLoader workers
- Profile training to identify bottlenecks
- Use gradient accumulation for larger effective batch sizes

## Troubleshooting

### Common Issues

#### High NaN/Inf Rates
```python
# Reduce learning rate
config.learning_rate = 1e-5

# Increase gradient clipping
config.gradient_clip_norm = 0.5

# Check data preprocessing
# Ensure inputs are properly normalized
```

#### Poor Training Performance
```python
# Enable profiling to identify bottlenecks
config.enable_profiling = True

# Increase batch size if memory allows
config.batch_size = 32

# Use gradient accumulation
config.gradient_accumulation_steps = 8
```

#### Memory Issues
```python
# Reduce batch size
config.batch_size = 8

# Enable gradient checkpointing
# (implement in model if needed)

# Use mixed precision
config.use_amp = True
```

### Debug Mode
```python
import torch
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

# Check gradients manually
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
```

## Examples

### Classification Training
```python
# See examples/classification_example.py for complete example
```

### Regression Training
```python
# See examples/regression_example.py for complete example
```

### Text Generation Training
```python
# See examples/text_generation_example.py for complete example
```

## Contributing

To extend the system:

1. **Add Custom Metrics**: Implement new metrics in `CustomMetricsEvaluator`
2. **Add Gradient Handling**: Extend `GradientMonitor` for new detection methods
3. **Add Training Features**: Extend `ComprehensiveTrainingSystem` for new capabilities
4. **Add Model Types**: Support new model architectures in the training system

## License

This system is part of the Blatam Academy NLP project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration options
3. Examine the example implementations
4. Check the logging output for detailed error messages

---

**Note**: This system is designed for production use and includes comprehensive error handling, monitoring, and optimization features. Always test with your specific use case and data before deploying to production.




