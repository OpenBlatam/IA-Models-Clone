# PyTorch Debugging Guide

## Overview

This guide covers the comprehensive PyTorch debugging tools implemented in the ads generation system. These tools help identify and resolve training issues, gradient problems, and model anomalies.

## Features

### 1. Autograd Anomaly Detection
- **`detect_anomaly()`**: Automatically detects backward pass anomalies
- **Context Management**: Safe enable/disable of debugging features
- **Detailed Error Reporting**: Full stack traces and tensor information

### 2. Tensor Analysis
- **NaN/Inf Detection**: Automatic detection of numerical anomalies
- **Statistical Analysis**: Mean, std, min, max values
- **Memory Usage Tracking**: Tensor memory consumption monitoring
- **Shape and Type Validation**: Comprehensive tensor metadata analysis

### 3. Gradient Monitoring
- **Gradient Explosion Detection**: Identifies exploding gradients
- **Gradient Vanishing Detection**: Identifies vanishing gradients
- **Gradient Norm Analysis**: Tracks gradient magnitude changes
- **Parameter-wise Analysis**: Individual parameter gradient monitoring

### 4. Model Validation
- **Parameter Consistency**: Validates model parameter integrity
- **Output Validation**: Checks model outputs for anomalies
- **Memory Profiling**: Tracks model memory usage
- **Component Analysis**: Analyzes individual model components

## Usage

### Basic Autograd Debugging

```python
from onyx.server.features.ads.pytorch_debug_utils import PyTorchDebugger, debug_context

# Using context manager
with debug_context(DebugLevel.BASIC) as debugger:
    # Your training code here
    loss = model(inputs)
    loss.backward()
    optimizer.step()

# Using debugger directly
debugger = PyTorchDebugger(DebugLevel.ADVANCED)
with debugger.autograd_anomaly_detection():
    # Training code
    pass
```

### Training Debugger

```python
from onyx.server.features.ads.pytorch_debug_utils import TrainingDebugger

# Initialize training debugger
debugger = TrainingDebugger(DebugLevel.ADVANCED)

# Monitor training steps
for step in range(num_steps):
    loss = model(inputs)
    loss.backward()
    
    # Monitor step for anomalies
    step_info = debugger.monitor_training_step(model, loss, step)
    
    optimizer.step()

# Get training summary
summary = debugger.get_training_summary()
print(f"Anomaly rate: {summary['anomaly_rate']:.2%}")
```

### Diffusion Model Debugging

```python
from onyx.server.features.ads.pytorch_debug_utils import DiffusionModelDebugger

# Initialize diffusion debugger
debugger = DiffusionModelDebugger(DebugLevel.BASIC)

# Analyze diffusion pipeline
pipeline_analysis = debugger.analyze_diffusion_pipeline(pipeline)
print(f"Total parameters: {pipeline_analysis['total_parameters']}")

# Monitor diffusion steps
for step in range(num_inference_steps):
    # Your diffusion step
    latents = pipeline.unet(latents, timestep, encoder_hidden_states)
    
    # Monitor step
    step_info = debugger.monitor_diffusion_step(pipeline, step, latents)
```

### Tensor Analysis

```python
from onyx.server.features.ads.pytorch_debug_utils import PyTorchDebugger

debugger = PyTorchDebugger()

# Analyze individual tensor
tensor_info = debugger.analyze_tensor(my_tensor, "my_tensor")
print(f"Has NaN: {tensor_info.has_nan}")
print(f"Mean value: {tensor_info.mean_value}")

# Analyze model parameters
param_analysis = debugger.analyze_model_parameters(model)
for name, info in param_analysis.items():
    if info.has_nan:
        print(f"Parameter {name} has NaN values")
```

### Gradient Analysis

```python
# Analyze gradients
gradient_info = debugger.analyze_gradients(model)

# Check for gradient explosion
exploded_params = debugger.check_gradient_explosion(model, threshold=10.0)
if exploded_params:
    print(f"Gradient explosion in: {exploded_params}")

# Check for gradient vanishing
vanished_params = debugger.check_gradient_vanishing(model, threshold=1e-6)
if vanished_params:
    print(f"Gradient vanishing in: {vanished_params}")
```

## Integration with Training Logger

### Automatic Debugging in Training

```python
from onyx.server.features.ads.training_logger import AsyncTrainingLogger

# Initialize logger with autograd debugging
logger = AsyncTrainingLogger(
    user_id=123,
    model_name="microsoft/DialoGPT-medium",
    enable_autograd_debug=True  # Enables PyTorch debugging
)

# Start training (autograd debugging automatically enabled)
logger.start_training(total_epochs=10, total_steps=1000)

# Check tensor anomalies during training
tensor_info = await logger.check_tensor_anomalies_async(loss_tensor, "loss")
if tensor_info["has_nan"]:
    logger.log_warning("Loss tensor contains NaN values")

# Check gradient anomalies
gradient_info = await logger.check_gradient_anomalies_async(model)
if gradient_info["params_with_nan_grad"] > 0:
    logger.log_warning("Model has parameters with NaN gradients")
```

### Custom Training Loop with Debugging

```python
async def debug_training_loop(model, dataloader, optimizer, logger):
    """Training loop with comprehensive debugging."""
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            try:
                # Forward pass
                outputs = model(batch)
                loss = outputs.loss
                
                # Check loss tensor
                loss_info = await logger.check_tensor_anomalies_async(loss, "loss")
                if loss_info["has_nan"] or loss_info["has_inf"]:
                    logger.log_error(
                        ValueError("Loss contains NaN/Inf values"),
                        TrainingPhase.TRAINING,
                        {"epoch": epoch, "step": step, "loss_info": loss_info}
                    )
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                gradient_info = await logger.check_gradient_anomalies_async(model)
                if gradient_info["params_with_nan_grad"] > 0:
                    logger.log_warning(
                        f"Gradient anomalies detected at step {step}",
                        TrainingPhase.TRAINING
                    )
                
                # Update progress
                await logger.update_progress_async(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    learning_rate=optimizer.param_groups[0]['lr']
                )
                
                optimizer.step()
                optimizer.zero_grad()
                
            except Exception as e:
                logger.log_error(e, TrainingPhase.TRAINING, {
                    "epoch": epoch,
                    "step": step
                })
                raise
```

## API Endpoints

### Debug Summary
```bash
GET /training-logs/debug/summary
```

Response:
```json
{
  "training_debugger": {
    "total_anomalies": 5,
    "debug_history_length": 100,
    "total_steps": 1000,
    "anomaly_rate": 0.005,
    "gradient_explosion_rate": 0.002
  },
  "diffusion_debugger": {
    "total_anomalies": 2,
    "debug_history_length": 50
  },
  "general_debugger": {
    "total_anomalies": 3,
    "debug_history_length": 75
  }
}
```

### Training Debug Summary
```bash
GET /training-logs/debug/training/summary
```

Response:
```json
{
  "total_steps": 1000,
  "anomaly_steps": 5,
  "gradient_explosions": 2,
  "gradient_vanishing": 1,
  "loss_anomalies": 3,
  "anomaly_rate": 0.005,
  "loss_anomaly_rate": 0.003,
  "gradient_explosion_rate": 0.002,
  "gradient_vanishing_rate": 0.001
}
```

### Set Debug Level
```bash
POST /training-logs/debug/level/advanced
```

### Clear Debug History
```bash
POST /training-logs/debug/clear
```

## Debug Levels

### Basic Level
- Tensor NaN/Inf detection
- Basic gradient analysis
- Simple anomaly reporting

### Advanced Level
- Detailed tensor statistics
- Gradient explosion/vanishing detection
- Memory usage tracking
- Model parameter analysis

### Extreme Level
- Full tensor history
- Real-time monitoring
- Detailed error context
- Performance impact

## Common Issues and Solutions

### 1. NaN Loss Values
```python
# Problem: Loss becomes NaN
if torch.isnan(loss):
    # Check input data
    input_info = debugger.analyze_tensor(inputs, "model_inputs")
    
    # Check model parameters
    param_info = debugger.analyze_model_parameters(model)
    
    # Check learning rate
    if learning_rate > 1.0:
        logger.log_warning("Learning rate too high")
```

### 2. Gradient Explosion
```python
# Problem: Gradients become very large
exploded_params = debugger.check_gradient_explosion(model, threshold=10.0)
if exploded_params:
    # Solutions:
    # 1. Reduce learning rate
    # 2. Apply gradient clipping
    # 3. Check input scaling
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. Gradient Vanishing
```python
# Problem: Gradients become very small
vanished_params = debugger.check_gradient_vanishing(model, threshold=1e-6)
if vanished_params:
    # Solutions:
    # 1. Increase learning rate
    # 2. Use different activation functions
    # 3. Check weight initialization
    # 4. Use skip connections
```

### 4. Memory Issues
```python
# Problem: Out of memory errors
memory_profile = debugger.profile_memory_usage(model)
total_memory = memory_profile["total"]

if total_memory > available_memory:
    # Solutions:
    # 1. Reduce batch size
    # 2. Use gradient checkpointing
    # 3. Move to CPU if possible
    # 4. Use mixed precision training
```

## Best Practices

### 1. Enable Debugging During Development
```python
# Always enable debugging during development
logger = AsyncTrainingLogger(
    user_id=123,
    model_name="model_name",
    enable_autograd_debug=True
)
```

### 2. Monitor Critical Tensors
```python
# Monitor loss, gradients, and key tensors
await logger.check_tensor_anomalies_async(loss, "loss")
await logger.check_gradient_anomalies_async(model)
```

### 3. Set Appropriate Thresholds
```python
# Use reasonable thresholds for your model
exploded_params = debugger.check_gradient_explosion(model, threshold=10.0)
vanished_params = debugger.check_gradient_vanishing(model, threshold=1e-6)
```

### 4. Regular Debug Summary
```python
# Check debug summary regularly
summary = debugger.get_debug_summary()
if summary["total_anomalies"] > threshold:
    logger.log_warning("High anomaly rate detected")
```

### 5. Clean Up Debug History
```python
# Clear debug history periodically
debugger.clear_debug_history()
```

## Performance Considerations

### Debugging Overhead
- **Basic Level**: Minimal overhead (~1-2%)
- **Advanced Level**: Moderate overhead (~5-10%)
- **Extreme Level**: Significant overhead (~15-25%)

### Memory Usage
- Debug history can consume significant memory
- Clear history regularly in long training runs
- Use appropriate debug levels for production

### When to Use Each Level
- **Development**: Use Advanced or Extreme level
- **Testing**: Use Basic or Advanced level
- **Production**: Use Basic level or disable debugging

## Troubleshooting

### Debugging Not Working
1. Check if PyTorch version supports `detect_anomaly()`
2. Verify autograd debugging is enabled
3. Check for proper context manager usage

### High Anomaly Rates
1. Review model architecture
2. Check data preprocessing
3. Adjust learning rate
4. Verify weight initialization

### Memory Issues
1. Reduce batch size
2. Use gradient checkpointing
3. Enable mixed precision training
4. Monitor debug history size

This comprehensive PyTorch debugging system provides the tools needed to identify and resolve training issues in the ads generation system. 