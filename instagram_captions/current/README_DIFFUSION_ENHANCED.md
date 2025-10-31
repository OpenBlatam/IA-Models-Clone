# 🚀 Enhanced Diffusion Models System

A production-ready, highly modular diffusion model framework with advanced noise schedulers, sampling methods, and seamless integration with the Hugging Face Diffusers library.

## ✨ Features

### 🔧 **Advanced Noise Schedulers**
- **Linear**: Standard linear beta schedule
- **Cosine**: Improved cosine schedule for better quality
- **Quadratic**: Quadratic interpolation schedule
- **Sigmoid**: Smooth sigmoid transition schedule
- **Exponential**: Exponential growth schedule

### 🎯 **Advanced Sampling Methods**
- **DDPM**: Standard diffusion sampling
- **DDIM**: Deterministic sampling with configurable stochasticity
- **Ancestral**: Ancestral sampling for better exploration
- **Euler**: First-order numerical integration
- **Heun**: Second-order numerical integration (higher accuracy)

### 🏗️ **Enhanced UNet Architecture**
- **Modular design** with separate blocks
- **Time embedding** integration
- **Attention mechanisms** with configurable heads
- **Residual connections** throughout
- **Proper weight initialization**

### ⚡ **Advanced Training Features**
- **Mixed precision training** with automatic scaling
- **EMA (Exponential Moving Average)** for stable inference
- **Gradient checkpointing** for memory efficiency
- **XFormers optimization** when available
- **Learning rate scheduling** with cosine annealing

### 🔌 **Diffusers Library Integration**
- **Seamless integration** with Hugging Face Diffusers
- **Fallback to custom implementations** when not available
- **Production-ready pipelines** (StableDiffusion, StableDiffusionXL)

### 📊 **Configuration Management**
- **YAML configuration** support
- **Comprehensive parameters** for all aspects
- **Easy serialization/deserialization**

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements_diffusion_enhanced.txt
```

### Basic Usage

```python
from diffusion_models_system import DiffusionConfig, DiffusionPipeline

# Create configuration
config = DiffusionConfig(
    in_channels=3,
    out_channels=3,
    model_channels=128,
    schedule_type="cosine",
    sampling_method="ddim",
    use_mixed_precision=True,
    use_ema=True
)

# Create pipeline
pipeline = DiffusionPipeline(config)

# Train model
history = pipeline.train(dataloader, num_epochs=100)

# Generate samples
samples = pipeline.generate(batch_size=4, num_steps=50)
```

## 📁 Project Structure

```
diffusion_models_system/
├── diffusion_models_system.py      # Main system implementation
├── requirements_diffusion_enhanced.txt  # Dependencies
├── README_DIFFUSION_ENHANCED.md   # This documentation
└── config/
    └── diffusion_config.yaml      # Example configuration
```

## 🔧 Configuration

### Model Architecture
```yaml
# Model architecture
in_channels: 3
out_channels: 3
model_channels: 128
num_res_blocks: 2
attention_resolutions: [16, 8]
dropout: 0.1
channel_mult: [1, 2, 4, 8]
num_heads: 8
```

### Diffusion Process
```yaml
# Diffusion process
beta_start: 0.0001
beta_end: 0.02
num_diffusion_timesteps: 1000
schedule_type: "cosine"  # linear, cosine, quadratic, sigmoid, exponential
```

### Training
```yaml
# Training
learning_rate: 1e-4
batch_size: 16
num_epochs: 100
gradient_clip_val: 1.0
use_mixed_precision: true
use_ema: true
```

### Sampling
```yaml
# Sampling
sampling_method: "ddim"  # ddpm, ddim, ancestral, euler, heun
num_inference_steps: 50
guidance_scale: 7.5
classifier_free_guidance: true
```

## 🎯 Advanced Usage

### Custom Noise Scheduler

```python
from diffusion_models_system import NoiseScheduler

class CustomNoiseScheduler(NoiseScheduler):
    def __init__(self, custom_param: float):
        self.custom_param = custom_param
        # Implement custom logic
    
    def get_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        # Custom beta calculation
        pass
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        # Custom alpha calculation
        pass
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        # Custom sigma calculation
        pass
```

### Custom Sampling Method

```python
from diffusion_models_system import SamplingMethod

class CustomSampling(SamplingMethod):
    def sample(self, model, x_t, t, scheduler, **kwargs):
        # Implement custom sampling logic
        return x_prev
```

### Factory Integration

```python
# Register custom components
NoiseSchedulerFactory.schedulers["custom"] = CustomNoiseScheduler
SamplingMethodFactory.samplers["custom"] = CustomSampling

# Use custom components
config = DiffusionConfig(
    schedule_type="custom",
    sampling_method="custom"
)
```

## 🔌 Diffusers Integration

### Automatic Pipeline Setup

The system automatically detects and integrates with the Diffusers library:

```python
# Automatic integration when available
if DIFFUSERS_AVAILABLE:
    # Creates native Diffusers pipeline
    self.diffusers_pipeline = DiffusionPipeline(
        unet=self.model,
        scheduler=self.diffusers_scheduler
    )
```

### Using Diffusers Schedulers

```python
# Direct access to Diffusers schedulers
from diffusers import DDIMScheduler, DDPMScheduler

# Use with custom model
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02
)
```

## 📊 Performance Optimization

### Mixed Precision Training

```python
config = DiffusionConfig(
    use_mixed_precision=True,
    gradient_accumulation_steps=4
)

# Automatic mixed precision with gradient scaling
with torch.cuda.amp.autocast():
    predicted_noise = model(x_t, t)
    loss = F.mse_loss(predicted_noise, noise)
```

### Memory Optimization

```python
config = DiffusionConfig(
    use_gradient_checkpointing=True,
    use_xformers=True
)

# Enables gradient checkpointing and XFormers
model.gradient_checkpointing_enable()
torch.backends.xformers.enable()
```

### EMA for Stable Inference

```python
config = DiffusionConfig(
    use_ema=True,
    ema_decay=0.9999
)

# Automatic EMA updates during training
# Use EMA model for inference
model = trainer.ema_model if hasattr(trainer, 'ema_model') else trainer.model
```

## 🧪 Testing and Validation

### Unit Tests

```bash
pytest test_diffusion_models_system.py -v
```

### Integration Tests

```python
# Test complete pipeline
def test_pipeline_integration():
    config = DiffusionConfig(num_epochs=1)
    pipeline = DiffusionPipeline(config)
    
    # Test training
    dummy_data = torch.randn(10, 3, 64, 64)
    dataloader = DataLoader(dummy_data, batch_size=2)
    history = pipeline.train(dataloader, num_epochs=1)
    
    # Test generation
    samples = pipeline.generate(batch_size=2, num_steps=10)
    assert samples.shape == (2, 3, 64, 64)
```

## 📈 Monitoring and Logging

### Training Progress

```python
# Automatic logging during training
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in dataloader:
        loss_dict = trainer.train_step(batch)
        epoch_losses.append(loss_dict["loss"])
    
    avg_loss = np.mean(epoch_losses)
    logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")
```

### Checkpoint Management

```python
# Automatic checkpointing
if (epoch + 1) % 10 == 0:
    checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
    trainer.save_checkpoint(checkpoint_path)

# Load checkpoints
trainer.load_checkpoint("checkpoint_epoch_50.pt")
```

## 🚀 Production Deployment

### Model Export

```python
# Save complete model
pipeline.save_model("production_model.pt")

# Load in production
production_pipeline = DiffusionPipeline.load_model("production_model.pt")
```

### Configuration Management

```python
# Save configuration
config.save_yaml("production_config.yaml")

# Load configuration
config = DiffusionConfig.from_yaml("production_config.yaml")
```

### Performance Monitoring

```python
# Monitor training metrics
training_history = pipeline.train(dataloader)

# Analyze performance
epochs = [h["epoch"] for h in training_history]
losses = [h["loss"] for h in training_history]

plt.plot(epochs, losses)
plt.title("Training Loss")
plt.show()
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Slow Training**
   - Enable XFormers optimization
   - Use appropriate noise schedule
   - Optimize data loading

3. **Poor Sample Quality**
   - Increase number of inference steps
   - Try different sampling methods
   - Adjust guidance scale

### Debug Mode

```python
# Enable PyTorch anomaly detection
torch.autograd.set_detect_anomaly(True)

# Enable gradient anomaly detection
torch.autograd.detect_anomaly()
```

## 📚 References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the examples

---

**Built with ❤️ for the AI/ML community**
