# Advanced Noise Scheduler and Sampling Methods System

## Overview

This system implements a comprehensive framework for noise schedulers and sampling methods in diffusion models, providing multiple beta schedules, advanced noise schedulers, and various sampling algorithms. It's designed to work seamlessly with PyTorch and integrates with the Hugging Face Diffusers library.

## Features

### 🎯 **Multiple Beta Schedules**
- **Linear**: Standard linear interpolation between beta_start and beta_end
- **Cosine**: Cosine-based schedule for smoother transitions
- **Quadratic**: Quadratic interpolation for custom curves
- **Sigmoid**: Sigmoid-based schedule for gradual changes
- **Exponential**: Exponential growth/decay schedule
- **Custom**: Extensible framework for custom schedules

### 🔄 **Advanced Noise Schedulers**
- **DDPM**: Denoising Diffusion Probabilistic Models
- **DDIM**: Denoising Diffusion Implicit Models
- **PNDM**: Pseudo Numerical Methods
- **Euler**: Euler method for ODE solving
- **Heun**: 2nd order Runge-Kutta method

### 🎲 **Sampling Methods**
- **DDPM**: Standard probabilistic sampling
- **DDIM**: Deterministic sampling with configurable noise
- **Ancestral**: Ancestral sampling with noise injection
- **Euler**: Euler method sampling
- **Heun**: Heun method sampling
- **Classifier-free Guidance**: Advanced guidance techniques

### ⚙️ **Configuration Management**
- YAML/JSON configuration files
- Pydantic-based validation
- Runtime configuration updates
- Pipeline comparison tools

## Architecture

```
AdvancedDiffusionSystem
├── NoiseSchedulerFactory
│   ├── LinearNoiseScheduler
│   ├── CosineNoiseScheduler
│   ├── QuadraticNoiseScheduler
│   ├── SigmoidNoiseScheduler
│   └── ExponentialNoiseScheduler
├── SamplerFactory
│   ├── DDPMSampler
│   ├── DDIMSampler
│   ├── AncestralSampler
│   ├── EulerSampler
│   └── HeunSampler
└── DiffusionPipeline
    ├── Noise Scheduler
    ├── Sampler
    └── Generation Logic
```

## Quick Start

### 1. Basic Usage

```python
from noise_scheduler_sampling_system import (
    AdvancedDiffusionSystem, 
    NoiseSchedulerConfig, 
    BetaSchedule, 
    SamplingMethod
)

# Create system
system = AdvancedDiffusionSystem()

# Create pipeline with cosine scheduler and DDIM sampler
config = NoiseSchedulerConfig(
    beta_schedule=BetaSchedule.COSINE,
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
)

pipeline = system.create_pipeline(
    "cosine_ddim", 
    config, 
    SamplingMethod.DDIM
)

# Generate samples
shape = (1, 3, 64, 64)  # Batch, channels, height, width
samples = pipeline.sample(
    model=your_model,
    shape=shape,
    num_inference_steps=50,
    guidance_scale=7.5
)
```

### 2. Multiple Schedulers Comparison

```python
# Create multiple pipelines
configs = {
    'linear': NoiseSchedulerConfig(beta_schedule=BetaSchedule.LINEAR),
    'cosine': NoiseSchedulerConfig(beta_schedule=BetaSchedule.COSINE),
    'quadratic': NoiseSchedulerConfig(beta_schedule=BetaSchedule.QUADRATIC),
    'sigmoid': NoiseSchedulerConfig(beta_schedule=BetaSchedule.SIGMOID)
}

for name, config in configs.items():
    system.create_pipeline(name, config, SamplingMethod.DDPM)

# Compare all schedulers
results = system.compare_schedulers(
    shape=(1, 3, 64, 64),
    num_inference_steps=20
)
```

### 3. Custom Configuration

```python
# Advanced configuration
config = NoiseSchedulerConfig(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule=BetaSchedule.COSINE,
    clip_sample=True,
    prediction_type="epsilon",
    thresholding=False,
    dynamic_thresholding_ratio=0.995,
    sample_max_value=1.0,
    timestep_spacing="leading",
    steps_offset=1,
    use_clipped_model_output=True,
    variance_type="fixed_small",
    clip_sample_range=1.0,
    sample_padding_threshold=0.0,
    sample_padding_norm=1.0
)
```

## API Reference

### Core Classes

#### `NoiseSchedulerConfig`
Configuration dataclass for noise schedulers with comprehensive parameters.

#### `BaseNoiseScheduler`
Abstract base class for all noise schedulers with common functionality.

#### `BaseSampler`
Abstract base class for all sampling methods.

#### `DiffusionPipeline`
Complete pipeline integrating scheduler and sampler with generation logic.

#### `AdvancedDiffusionSystem`
High-level system managing multiple pipelines and configurations.

### Key Methods

#### `DiffusionPipeline.sample()`
```python
def sample(
    self,
    model: nn.Module,
    shape: Tuple[int, ...],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    classifier_free_guidance: bool = True,
    **kwargs
) -> torch.Tensor
```

#### `AdvancedDiffusionSystem.compare_schedulers()`
```python
def compare_schedulers(
    self,
    shape: Tuple[int, ...],
    num_inference_steps: int = 50,
    **kwargs
) -> Dict[str, torch.Tensor]
```

## Beta Schedule Types

### Linear Schedule
```
β_t = β_start + (β_end - β_start) * t / T
```
- **Pros**: Simple, predictable
- **Cons**: May not be optimal for all tasks
- **Use case**: General purpose, baseline comparisons

### Cosine Schedule
```
α̅_t = cos²((t/T + 0.008) / 1.008 * π/2)
β_t = 1 - α̅_t / α̅_{t-1}
```
- **Pros**: Smooth transitions, often better quality
- **Cons**: More complex computation
- **Use case**: High-quality generation, research

### Quadratic Schedule
```
β_t = β_start + (β_end - β_start) * (t/T)²
```
- **Pros**: Gradual start, faster end
- **Cons**: May be too aggressive
- **Use case**: Tasks requiring early stability

### Sigmoid Schedule
```
β_t = β_start + (β_end - β_start) * σ(12 * t/T - 6)
```
- **Pros**: Very gradual start and end
- **Cons**: Complex, may be too slow
- **Use case**: Fine-grained control tasks

### Exponential Schedule
```
β_t = β_start * (β_end / β_start)^(t/T)
```
- **Pros**: Exponential growth/decay
- **Cons**: May be unstable
- **Use case**: Specific mathematical requirements

## Sampling Methods

### DDPM (Denoising Diffusion Probabilistic Models)
- **Type**: Probabilistic
- **Noise**: Adds noise at each step
- **Quality**: High
- **Speed**: Standard
- **Use case**: General purpose, high quality

### DDIM (Denoising Diffusion Implicit Models)
- **Type**: Deterministic (when η=0)
- **Noise**: Configurable noise injection
- **Quality**: High
- **Speed**: Fast (fewer steps)
- **Use case**: Fast generation, deterministic results

### Ancestral Sampling
- **Type**: Probabilistic
- **Noise**: Always adds noise
- **Quality**: Good
- **Speed**: Standard
- **Use case**: Exploration, diversity

### Euler Method
- **Type**: Deterministic
- **Noise**: No noise
- **Quality**: Good
- **Speed**: Fast
- **Use case**: Fast inference, ODE solving

### Heun Method
- **Type**: Deterministic
- **Noise**: No noise
- **Quality**: Very good
- **Speed**: Medium
- **Use case**: High quality, ODE solving

## Configuration Files

### YAML Configuration Example
```yaml
linear_ddpm:
  num_train_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: "linear"
  clip_sample: true
  prediction_type: "epsilon"
  thresholding: false
  dynamic_thresholding_ratio: 0.995
  sample_max_value: 1.0
  timestep_spacing: "leading"
  steps_offset: 1
  use_clipped_model_output: true
  variance_type: "fixed_small"
  clip_sample_range: 1.0
  sample_padding_threshold: 0.0
  sample_padding_norm: 1.0

cosine_ddim:
  num_train_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: "cosine"
  clip_sample: true
  prediction_type: "epsilon"
  thresholding: false
  dynamic_thresholding_ratio: 0.995
  sample_max_value: 1.0
  timestep_spacing: "leading"
  steps_offset: 1
  use_clipped_model_output: true
  variance_type: "fixed_small"
  clip_sample_range: 1.0
  sample_padding_threshold: 0.0
  sample_padding_norm: 1.0
```

### Loading Configuration
```python
# Load from YAML
system = AdvancedDiffusionSystem("config.yaml")

# Load from JSON
system = AdvancedDiffusionSystem("config.json")

# Save current configuration
system.save_config("current_config.yaml")
```

## Performance Optimization

### GPU Utilization
- Automatic device detection (CUDA/CPU)
- Batch processing support
- Memory-efficient operations

### Sampling Optimization
- Configurable inference steps
- Early stopping capabilities
- Gradient-free sampling

### Memory Management
- Efficient tensor operations
- Minimal memory allocation
- Automatic cleanup

## Integration Examples

### With Hugging Face Diffusers
```python
from diffusers import StableDiffusionPipeline
from noise_scheduler_sampling_system import AdvancedDiffusionSystem

# Create custom system
system = AdvancedDiffusionSystem()

# Integrate with existing pipeline
diffusers_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
custom_pipeline = system.create_pipeline("custom", config, SamplingMethod.DDIM)

# Use custom scheduler for generation
samples = custom_pipeline.sample(
    model=diffusers_pipeline.unet,
    shape=(1, 4, 64, 64),
    num_inference_steps=20
)
```

### With Custom Models
```python
class CustomUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Your UNet implementation
    
    def forward(self, x, t, **kwargs):
        # Your forward pass
        return predicted_noise

# Create pipeline with custom model
pipeline = system.create_pipeline("custom", config, SamplingMethod.DDPM)
samples = pipeline.sample(
    model=CustomUNet(),
    shape=(1, 3, 256, 256),
    num_inference_steps=100
)
```

## Error Handling

### Common Issues
- **Configuration Errors**: Invalid parameter values
- **Memory Errors**: Large batch sizes or high resolution
- **Device Errors**: CUDA/CPU compatibility issues

### Debugging
- Comprehensive logging
- Error categorization
- Recovery suggestions
- Performance profiling

## Testing

### Unit Tests
```bash
python -m pytest test_noise_scheduler_sampling_system.py -v
```

### Integration Tests
```bash
python -m pytest test_integration.py -v
```

### Performance Tests
```bash
python -m pytest test_performance.py -v
```

## Dependencies

### Core Dependencies
```
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

### Optional Dependencies
```
yaml>=6.0
pydantic>=2.0.0
```

## Contributing

### Adding New Schedulers
1. Inherit from `BaseNoiseScheduler`
2. Implement `_get_beta_schedule()` method
3. Add to `NoiseSchedulerFactory`
4. Update tests

### Adding New Samplers
1. Inherit from `BaseSampler`
2. Implement `step()` method
3. Add to `SamplerFactory`
4. Update tests

### Code Style
- Follow PEP 8
- Use type hints
- Add comprehensive docstrings
- Include unit tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example code
- Consult the API reference

## Changelog

### v1.0.0
- Initial release
- Basic noise schedulers (Linear, Cosine)
- Basic samplers (DDPM, DDIM)
- Configuration management

### v1.1.0
- Added Quadratic, Sigmoid, Exponential schedulers
- Added Ancestral, Euler, Heun samplers
- Enhanced configuration options
- Performance improvements

### v1.2.0
- Advanced diffusion system
- Pipeline comparison tools
- YAML/JSON configuration support
- Comprehensive testing suite


