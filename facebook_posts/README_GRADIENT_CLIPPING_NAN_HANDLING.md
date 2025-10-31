# Advanced Gradient Clipping and NaN/Inf Handling System

A comprehensive implementation of gradient clipping techniques and numerical stability management for deep learning training.

## Features

### 🎯 Gradient Clipping Methods

#### 1. **Norm-based Clipping**
- L2 norm clipping with configurable threshold
- Global norm clipping across all parameters
- Efficient implementation with minimal memory overhead

#### 2. **Value-based Clipping**
- Direct value clipping with upper/lower bounds
- Configurable thresholds for different parameter types
- Maintains gradient direction while limiting magnitude

#### 3. **Layer-wise Clipping**
- Individual thresholds for different layers
- Customizable per-layer clipping strategies
- Useful for models with varying gradient scales

#### 4. **Percentile-based Clipping**
- Dynamic threshold based on gradient distribution
- Configurable percentile (e.g., 90th, 95th percentile)
- Adaptive to training dynamics

#### 5. **Exponential Moving Average Clipping**
- Adaptive thresholds using exponential moving average
- Smooth adaptation to gradient changes
- Configurable smoothing factor and minimum threshold

#### 6. **Adaptive Clipping**
- Automatic threshold adjustment based on gradient history
- Learning rate adaptation
- Prevents excessive clipping

### 🚨 NaN/Inf Detection & Handling

#### 1. **Comprehensive Detection**
- NaN detection in gradients, parameters, and loss
- Inf detection for overflow prevention
- Configurable thresholds for different numerical issues

#### 2. **Handling Strategies**
- **Detect**: Log and monitor without intervention
- **Replace**: Substitute with safe values
- **Skip**: Skip problematic updates
- **Restore**: Restore from checkpoints (requires checkpoint management)
- **Gradient Zeroing**: Zero problematic gradients
- **Adaptive**: Dynamic handling based on severity
- **Gradient Scaling**: Scale instead of zeroing

#### 3. **Severity-based Handling**
- Automatic severity assessment
- Different strategies for different issue types
- Configurable response levels

## Quick Start

### Basic Usage

```python
from gradient_clipping_nan_handling import (
    GradientClippingConfig, 
    NaNHandlingConfig, 
    NumericalStabilityManager
)

# Configure gradient clipping
clipping_config = GradientClippingConfig(
    clipping_type=ClippingType.NORM,
    max_norm=1.0,
    monitor_clipping=True
)

# Configure NaN handling
nan_config = NaNHandlingConfig(
    handling_type=NaNHandlingType.ADAPTIVE,
    detect_nan=True,
    detect_inf=True,
    detect_overflow=True
)

# Create stability manager
stability_manager = NumericalStabilityManager(clipping_config, nan_config)

# Use in training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    # Apply stability measures
    stability_result = stability_manager.step(model, loss, optimizer)
    
    optimizer.step()
```

### Training Wrapper

```python
from gradient_clipping_nan_handling import create_training_wrapper

# Create wrapper
wrapper = create_training_wrapper(clipping_config, nan_config)

# Use in training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    # Automatic stability measures
    stability_result = wrapper(model, loss, optimizer)
    
    optimizer.step()

# Save and visualize results
wrapper.save_histories()
wrapper.plot_histories("training_stability.png")
```

## Configuration Options

### Gradient Clipping Configuration

```python
@dataclass
class GradientClippingConfig:
    # Basic settings
    clipping_type: ClippingType = ClippingType.NORM
    max_norm: float = 1.0
    max_value: float = 1.0
    
    # Layer-wise settings
    layer_wise_enabled: bool = False
    layer_norm_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Percentile settings
    percentile_enabled: bool = False
    percentile_threshold: float = 95.0
    
    # Exponential settings
    exponential_enabled: bool = False
    exponential_alpha: float = 0.9
    exponential_min_threshold: float = 0.1
    
    # Monitoring
    monitor_clipping: bool = True
    log_clipping_stats: bool = True
    save_clipping_history: bool = True
```

### NaN Handling Configuration

```python
@dataclass
class NaNHandlingConfig:
    # Handling strategy
    handling_type: NaNHandlingType = NaNHandlingType.DETECT
    
    # Detection settings
    detect_nan: bool = True
    detect_inf: bool = True
    detect_overflow: bool = True
    
    # Thresholds
    nan_threshold: float = 1e-6
    inf_threshold: float = 1e6
    overflow_threshold: float = 1e6
    
    # Replacement values
    nan_replacement: float = 0.0
    inf_replacement: float = 1e6
    overflow_replacement: float = 1e6
    
    # Monitoring
    monitor_nan: bool = True
    log_nan_stats: bool = True
    save_nan_history: bool = True
```

## Advanced Usage

### Custom Layer-wise Thresholds

```python
clipping_config = GradientClippingConfig(
    clipping_type=ClippingType.LAYER_WISE,
    layer_wise_enabled=True,
    layer_norm_thresholds={
        'encoder.weight': 0.8,
        'decoder.weight': 1.2,
        'attention.weight': 0.6
    }
)
```

### Adaptive NaN Handling

```python
nan_config = NaNHandlingConfig(
    handling_type=NaNHandlingType.ADAPTIVE,
    detect_nan=True,
    detect_inf=True,
    detect_overflow=True
)
```

### Exponential Clipping

```python
clipping_config = GradientClippingConfig(
    clipping_type=ClippingType.EXPONENTIAL,
    exponential_enabled=True,
    exponential_alpha=0.95,  # Smoothing factor
    exponential_min_threshold=0.5  # Minimum threshold
)
```

## Monitoring and Visualization

### Automatic Logging

The system automatically logs:
- Gradient clipping statistics
- NaN/Inf detection events
- Handling actions taken
- Stability scores over time

### History Tracking

```python
# Save histories
stability_manager.save_histories()

# Plot results
stability_manager.plot_stability_history("stability_analysis.png")
stability_manager.gradient_clipper.plot_clipping_history("clipping_analysis.png")
stability_manager.nan_handler.plot_nan_history("nan_analysis.png")
```

### Metrics Available

- **Clipping Statistics**: Clipping ratios, gradient norms, clipped norms
- **Numerical Issues**: NaN/Inf/Overflow counts, handling actions
- **Stability Scores**: Overall numerical stability metrics
- **Training Dynamics**: Evolution of stability over training steps

## Best Practices

### 1. **Choose Appropriate Clipping Type**
- **Norm**: General purpose, good for most cases
- **Layer-wise**: When layers have different gradient scales
- **Percentile**: For dynamic threshold adaptation
- **Exponential**: For smooth threshold adaptation

### 2. **Configure NaN Handling**
- **Detect**: For monitoring and debugging
- **Adaptive**: For production training
- **Gradient Scaling**: When you want to preserve gradient information

### 3. **Monitor Training**
- Enable logging and history saving
- Plot results to understand training dynamics
- Adjust thresholds based on observed behavior

### 4. **Integration with Training Loops**
- Apply stability measures before optimizer step
- Monitor stability scores for training health
- Use training wrapper for automatic integration

## Performance Considerations

### Memory Usage
- Minimal overhead for basic clipping
- Configurable history saving
- Efficient tensor operations

### Computational Cost
- Fast detection algorithms
- Optimized clipping implementations
- Configurable monitoring levels

### Scalability
- Works with models of any size
- Efficient for large parameter counts
- Parallel processing support (configurable)

## Troubleshooting

### Common Issues

1. **Excessive Clipping**
   - Reduce `max_norm` threshold
   - Use adaptive or exponential clipping
   - Check learning rate

2. **Frequent NaN/Inf**
   - Enable adaptive handling
   - Check data preprocessing
   - Verify model architecture

3. **Performance Issues**
   - Disable detailed monitoring
   - Reduce history saving frequency
   - Use efficient clipping types

### Debug Mode

```python
# Enable detailed logging
clipping_config.log_clipping_stats = True
nan_config.log_nan_stats = True

# Check stability scores
stability_result = stability_manager.step(model, loss, optimizer)
print(f"Stability: {stability_result['stability_score']:.4f}")
```

## Examples

See the demonstration function in the main file for comprehensive examples of all features.

## Dependencies

- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0
- tqdm >= 4.60.0

## License

This implementation is part of the Blatam Academy project and follows the project's licensing terms.






