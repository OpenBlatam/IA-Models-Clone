# Advanced Noise Schedulers and Sampling Methods Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Noise Scheduler Types](#noise-scheduler-types)
4. [Sampling Methods](#sampling-methods)
5. [Advanced Techniques](#advanced-techniques)
6. [Implementation Details](#implementation-details)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Introduction

Noise schedulers and sampling methods are fundamental components of diffusion models that control the noise addition and removal processes. This guide provides a comprehensive overview of advanced noise schedulers and sampling techniques used in modern diffusion models.

### Key Concepts

- **Noise Scheduler**: Controls the noise schedule during training and inference
- **Sampling Method**: Determines how to generate samples from the model
- **Beta Schedule**: Defines the noise level at each timestep
- **Denoising Process**: The reverse process of removing noise step by step

## Mathematical Foundations

### Forward Diffusion Process

The forward diffusion process gradually adds noise to data according to a predefined schedule:

```
q(x_t | x_0) = N(x_t; √(ᾱ_t) * x_0, (1 - ᾱ_t) * I)
```

Where:
- `x_t` is the noisy sample at timestep t
- `x_0` is the original sample
- `ᾱ_t = ∏(1 - β_i)` is the cumulative product of alphas
- `β_t` is the noise schedule at timestep t

### Reverse Diffusion Process

The reverse process learns to denoise step by step:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

Where:
- `μ_θ` is the predicted mean
- `Σ_θ` is the predicted variance
- `θ` represents the model parameters

### Beta Schedule Types

#### Linear Schedule
```python
β_t = β_start + (β_end - β_start) * t / T
```

#### Cosine Schedule
```python
ᾱ_t = cos((t/T + 0.008) / 1.008 * π/2)²
β_t = 1 - ᾱ_t / ᾱ_{t-1}
```

#### Quadratic Schedule
```python
β_t = (β_start + (β_end - β_start) * t / T)²
```

## Noise Scheduler Types

### 1. DDPM Scheduler

The original Denoising Diffusion Probabilistic Models scheduler:

```python
config = NoiseSchedulerConfig(
    scheduler_type="ddpm",
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)
```

**Characteristics:**
- Simple and stable
- Good for training
- Slower sampling

### 2. DDIM Scheduler

Deterministic sampling with faster inference:

```python
config = NoiseSchedulerConfig(
    scheduler_type="ddim",
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    eta=0.0  # Deterministic when eta=0
)
```

**Characteristics:**
- Deterministic sampling
- Faster inference
- Fewer steps required

### 3. DPM-Solver Scheduler

Advanced solver with better efficiency:

```python
config = NoiseSchedulerConfig(
    scheduler_type="dpm_solver",
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    use_karras_sigmas=True
)
```

**Characteristics:**
- Very fast sampling
- High quality results
- Advanced numerical methods

### 4. Euler Scheduler

Simple ODE solver:

```python
config = NoiseSchedulerConfig(
    scheduler_type="euler",
    prediction_type="epsilon"
)
```

**Characteristics:**
- Simple implementation
- Good baseline
- Moderate speed

### 5. PNDM Scheduler

Pseudo numerical methods:

```python
config = NoiseSchedulerConfig(
    scheduler_type="pndm",
    num_train_timesteps=1000
)
```

**Characteristics:**
- Fast sampling
- Good quality
- Memory efficient

## Sampling Methods

### 1. Standard Sampling

Basic denoising process:

```python
def standard_sampling(model, latents, scheduler, num_steps=50):
    scheduler.set_timesteps(num_steps)
    
    for t in scheduler.timesteps:
        # Predict noise
        noise_pred = model(latents, t)
        
        # Denoise step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
```

### 2. Classifier-Free Guidance

Conditional generation with guidance:

```python
def classifier_free_guidance_sampling(
    model, prompt_embeds, uncond_embeds, latents, 
    scheduler, guidance_scale=7.5
):
    # Duplicate latents for guidance
    latents = torch.cat([latents] * 2)
    
    for t in scheduler.timesteps:
        # Predict noise for both conditioned and unconditioned
        noise_pred = model(latents, t, torch.cat([uncond_embeds, prompt_embeds]))
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        # Apply guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Denoise step
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
```

### 3. Temperature Sampling

Control randomness with temperature:

```python
def temperature_sampling(model, latents, scheduler, temperature=1.0):
    for t in scheduler.timesteps:
        noise_pred = model(latents, t)
        
        # Apply temperature scaling
        if temperature != 1.0:
            noise_pred = noise_pred / temperature
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
```

### 4. Noise Injection Sampling

Add controlled noise during sampling:

```python
def noise_injection_sampling(
    model, latents, scheduler, injection_strength=0.1
):
    for i, t in enumerate(scheduler.timesteps):
        noise_pred = model(latents, t)
        
        # Inject additional noise
        if injection_strength > 0:
            extra_noise = torch.randn_like(noise_pred) * injection_strength
            noise_pred = noise_pred + extra_noise
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
```

### 5. Adaptive Sampling

Dynamic step adjustment based on change magnitude:

```python
def adaptive_sampling(model, latents, scheduler, threshold=0.1):
    i = 0
    while i < len(scheduler.timesteps):
        t = scheduler.timesteps[i]
        
        noise_pred = model(latents, t)
        result = scheduler.step(noise_pred, t, latents)
        
        # Check if we need more steps
        if i < len(scheduler.timesteps) - 1:
            change = torch.norm(latents - result.prev_sample, dim=1).mean()
            
            if change > threshold:
                # Insert additional step
                pass
        
        latents = result.prev_sample
        i += 1
    
    return latents
```

## Advanced Techniques

### 1. Karras Sigma Modification

Improved noise schedule for better sampling:

```python
def apply_karras_modification(betas):
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0
    
    # Convert to sigmas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)
    
    # Apply Karras modification
    sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_max), len(sigmas)))
    sigmas = sigmas * (1.0 + torch.linspace(0, 1, len(sigmas)) ** rho)
    
    # Convert back to betas
    alphas_cumprod = 1.0 / (1.0 + sigmas ** 2)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1.0 - alphas
    
    return betas
```

### 2. Dynamic Thresholding

Adaptive clipping for better quality:

```python
def dynamic_thresholding(sample, threshold_ratio=0.995):
    # Calculate dynamic threshold
    s = torch.quantile(torch.abs(sample), threshold_ratio, dim=1, keepdim=True)
    s = torch.clamp(s, min=1.0)
    
    # Apply thresholding
    sample = torch.clamp(sample, -s, s) / s
    
    return sample
```

### 3. Multi-Step Sampling

Combine multiple sampling methods:

```python
def multi_step_sampling(model, latents, schedulers, weights=None):
    if weights is None:
        weights = [1.0] * len(schedulers)
    
    results = []
    for scheduler, weight in zip(schedulers, weights):
        result = standard_sampling(model, latents, scheduler)
        results.append(result * weight)
    
    return torch.stack(results).sum(dim=0)
```

## Implementation Details

### Configuration Management

```python
@dataclass
class NoiseSchedulerConfig:
    # Basic parameters
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    prediction_type: str = "epsilon"
    
    # Scheduler type
    scheduler_type: str = "ddpm"
    
    # Sampling parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    temperature: float = 1.0
    
    # Advanced parameters
    use_karras_sigmas: bool = False
    clip_sample: bool = True
    thresholding: bool = False
```

### Scheduler Factory

```python
def create_scheduler(config: NoiseSchedulerConfig):
    if config.scheduler_type == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type
        )
    elif config.scheduler_type == "ddim":
        return DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            eta=config.eta
        )
    # ... other schedulers
```

### Memory Management

```python
def memory_efficient_sampling(model, latents, scheduler):
    # Use gradient checkpointing
    with torch.no_grad():
        for t in scheduler.timesteps:
            # Clear cache periodically
            if t % 10 == 0:
                torch.cuda.empty_cache()
            
            noise_pred = model(latents, t)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
```

## Performance Optimization

### 1. Mixed Precision

```python
def mixed_precision_sampling(model, latents, scheduler):
    scaler = GradScaler()
    
    with autocast():
        for t in scheduler.timesteps:
            noise_pred = model(latents, t)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
```

### 2. Batch Processing

```python
def batch_sampling(model, latents_batch, scheduler):
    # Process multiple samples in parallel
    batch_size = latents_batch.shape[0]
    
    for t in scheduler.timesteps:
        noise_pred = model(latents_batch, t)
        latents_batch = scheduler.step(noise_pred, t, latents_batch).prev_sample
    
    return latents_batch
```

### 3. Caching

```python
class CachedScheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.cache = {}
    
    def step(self, model_output, timestep, sample):
        key = (timestep, sample.shape)
        if key not in self.cache:
            self.cache[key] = self.scheduler.step(model_output, timestep, sample)
        return self.cache[key]
```

## Best Practices

### 1. Scheduler Selection

- **Training**: Use DDPM or DDIM for stability
- **Fast Inference**: Use DPM-Solver or PNDM
- **High Quality**: Use DDIM with low eta or DPM-Solver++
- **Memory Constrained**: Use PNDM or Euler

### 2. Step Count Optimization

```python
def optimize_step_count(scheduler_type, quality_requirement):
    step_counts = {
        "ddpm": 1000,
        "ddim": 50,
        "dpm_solver": 20,
        "pndm": 50,
        "euler": 100
    }
    
    base_steps = step_counts[scheduler_type]
    
    if quality_requirement == "high":
        return base_steps * 2
    elif quality_requirement == "low":
        return base_steps // 2
    else:
        return base_steps
```

### 3. Guidance Scale Tuning

```python
def tune_guidance_scale(model, prompt, uncond_prompt, latents, scheduler):
    scales = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]
    results = []
    
    for scale in scales:
        result = classifier_free_guidance_sampling(
            model, prompt, uncond_prompt, latents, scheduler, scale
        )
        results.append(result)
    
    return results
```

### 4. Error Handling

```python
def robust_sampling(model, latents, scheduler, max_retries=3):
    for attempt in range(max_retries):
        try:
            return standard_sampling(model, latents, scheduler)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    raise RuntimeError("Sampling failed after maximum retries")
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

**Problem**: CUDA out of memory during sampling

**Solutions**:
```python
# Reduce batch size
latents = latents[:1]

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()

# Use CPU offloading
latents = latents.cpu()
```

#### 2. Quality Issues

**Problem**: Poor sample quality

**Solutions**:
```python
# Increase step count
scheduler.set_timesteps(100)

# Adjust guidance scale
guidance_scale = 10.0

# Use better scheduler
config.scheduler_type = "dpm_solver"

# Enable thresholding
config.thresholding = True
```

#### 3. Speed Issues

**Problem**: Sampling too slow

**Solutions**:
```python
# Use faster scheduler
config.scheduler_type = "dpm_solver"

# Reduce step count
scheduler.set_timesteps(20)

# Use mixed precision
with autocast():
    # sampling code

# Enable optimizations
config.use_karras_sigmas = True
```

#### 4. Numerical Instability

**Problem**: NaN or inf values

**Solutions**:
```python
# Check for NaN
if torch.isnan(noise_pred).any():
    noise_pred = torch.nan_to_num(noise_pred)

# Clip values
noise_pred = torch.clamp(noise_pred, -10, 10)

# Use stable scheduler
config.scheduler_type = "ddpm"
```

### Debugging Tools

```python
def debug_sampling(model, latents, scheduler):
    debug_info = {
        'latents_shape': latents.shape,
        'latents_range': (latents.min().item(), latents.max().item()),
        'latents_mean': latents.mean().item(),
        'latents_std': latents.std().item()
    }
    
    for t in scheduler.timesteps:
        noise_pred = model(latents, t)
        
        debug_info[f'timestep_{t}'] = {
            'noise_pred_range': (noise_pred.min().item(), noise_pred.max().item()),
            'noise_pred_mean': noise_pred.mean().item(),
            'noise_pred_std': noise_pred.std().item()
        }
        
        if torch.isnan(noise_pred).any():
            debug_info[f'timestep_{t}']['has_nan'] = True
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return debug_info
```

## Examples

### Complete Sampling Pipeline

```python
def complete_sampling_pipeline():
    # Configuration
    config = NoiseSchedulerConfig(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="cosine",
        scheduler_type="dpm_solver",
        num_inference_steps=20,
        guidance_scale=7.5,
        use_karras_sigmas=True
    )
    
    # Create components
    scheduler = AdvancedNoiseScheduler(config)
    sampling_methods = AdvancedSamplingMethods(config)
    
    # Initialize latents
    latents = torch.randn(1, 4, 64, 64)
    
    # Sample with different methods
    samples_standard = sampling_methods.standard_sampling(
        model, latents, scheduler
    )
    
    samples_temp = sampling_methods.temperature_sampling(
        model, latents, scheduler, temperature=0.8
    )
    
    samples_noise = sampling_methods.noise_injection_sampling(
        model, latents, scheduler, injection_strength=0.1
    )
    
    return samples_standard, samples_temp, samples_noise
```

### Scheduler Comparison

```python
def compare_schedulers():
    schedulers = ["ddpm", "ddim", "dpm_solver", "pndm", "euler"]
    results = {}
    
    for scheduler_type in schedulers:
        config = NoiseSchedulerConfig(scheduler_type=scheduler_type)
        scheduler = AdvancedNoiseScheduler(config)
        
        # Measure performance
        start_time = time.time()
        samples = standard_sampling(model, latents, scheduler)
        end_time = time.time()
        
        results[scheduler_type] = {
            'samples': samples,
            'time': end_time - start_time,
            'memory': torch.cuda.memory_allocated() / 1e9
        }
    
    return results
```

### Advanced Usage

```python
def advanced_sampling_example():
    # Multi-scheduler ensemble
    schedulers = [
        AdvancedNoiseScheduler(NoiseSchedulerConfig(scheduler_type="ddpm")),
        AdvancedNoiseScheduler(NoiseSchedulerConfig(scheduler_type="ddim")),
        AdvancedNoiseScheduler(NoiseSchedulerConfig(scheduler_type="dpm_solver"))
    ]
    
    # Sample with each scheduler
    samples = []
    for scheduler in schedulers:
        sample = standard_sampling(model, latents, scheduler)
        samples.append(sample)
    
    # Ensemble results
    ensemble_sample = torch.stack(samples).mean(dim=0)
    
    return ensemble_sample
```

This guide provides a comprehensive overview of noise schedulers and sampling methods. The implementation includes advanced techniques, performance optimizations, and practical examples for production use. 