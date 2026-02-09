# Diffusion Models System

A comprehensive system for training, inference, and analysis of diffusion models, with a focus on Stable Diffusion and related architectures.

## 🎯 Overview

This system provides a complete framework for working with diffusion models, including:

- **Model Management**: Load and configure diffusion models (Stable Diffusion, DDPM, DDIM)
- **Training Pipeline**: Complete training workflow with optimization and checkpointing
- **Inference Engine**: High-quality image generation with configurable parameters
- **Quality Analysis**: Comprehensive analysis of generated images and model performance
- **Memory Optimization**: Advanced techniques for efficient GPU memory usage
- **Reporting System**: Detailed reports and analytics for experiments

## 🏗️ Architecture

### Core Components

1. **DiffusionModelManager**: Central manager for model loading, configuration, and inference
2. **DiffusionTrainer**: Handles training workflow, optimization, and checkpointing
3. **DiffusionAnalyzer**: Analyzes image quality, performance metrics, and generates reports
4. **DiffusionDataProcessor**: Manages data loading and preprocessing for training

### Key Features

- **Multi-Model Support**: Stable Diffusion, DDPM, DDIM, and custom architectures
- **Memory Optimization**: Attention slicing, VAE slicing, XFormers integration
- **Advanced Training**: Multiple optimizers, schedulers, and loss functions
- **Quality Metrics**: Brightness, contrast, sharpness, and color diversity analysis
- **Performance Monitoring**: Generation time, memory usage, and throughput tracking

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_diffusion_models.txt

# For GPU acceleration (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage

```python
from diffusion_models_system import DiffusionConfig, TrainingConfig, create_diffusion_system

# Create configurations
diffusion_config = DiffusionConfig(
    model_name="runwayml/stable-diffusion-v1-5",
    num_inference_steps=50,
    guidance_scale=7.5
)

training_config = TrainingConfig(
    learning_rate=1e-5,
    num_epochs=100,
    batch_size=1
)

# Create system
model_manager, trainer, analyzer = create_diffusion_system(
    diffusion_config, training_config
)

# Generate images
images = model_manager.generate_image(
    prompt="A beautiful sunset over mountains",
    negative_prompt="blurry, low quality",
    num_images=1
)

# Analyze quality
quality_metrics = analyzer.analyze_generation_quality(images, ["prompt"])
print(f"Image sharpness: {quality_metrics['sharpness']:.4f}")
```

## 📋 Configuration

### DiffusionConfig

```python
@dataclass
class DiffusionConfig:
    # Model settings
    model_name: str = "runwayml/stable-diffusion-v1-5"
    model_type: str = "stable_diffusion"
    use_pipeline: bool = True
    
    # Training settings
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    
    # Inference settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    
    # Memory optimization
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    
    # Quality settings
    height: int = 512
    width: int = 512
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Basic training
    learning_rate: float = 1e-5
    num_epochs: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer: str = "adamw"  # adamw, lion, adafactor
    weight_decay: float = 0.01
    warmup_steps: int = 500
    lr_scheduler: str = "cosine"
    
    # Loss and regularization
    loss_type: str = "l2"  # l2, l1, huber
    gradient_clip_norm: float = 1.0
```

## 🎨 Image Generation

### Basic Generation

```python
# Single image generation
images = model_manager.generate_image(
    prompt="A futuristic city with flying cars",
    negative_prompt="blurry, low quality, distorted",
    num_images=1,
    num_inference_steps=50,
    guidance_scale=7.5
)

# Batch generation
images = model_manager.generate_image(
    prompt=["Prompt 1", "Prompt 2", "Prompt 3"],
    negative_prompt=["Negative 1", "Negative 2", "Negative 3"],
    num_images=1
)
```

### Advanced Generation

```python
# Custom parameters
images = model_manager.generate_image(
    prompt="A magical forest",
    height=768,
    width=768,
    num_inference_steps=100,
    guidance_scale=15.0,
    eta=0.1
)
```

## 🏋️ Training

### Training Workflow

```python
# Setup training
trainer = DiffusionTrainer(model_manager, training_config)

# Training loop
for epoch in range(training_config.num_epochs):
    for batch in train_dataloader:
        # Training step
        step_result = trainer.train_step(batch, epoch)
        
        # Log metrics
        if step_result['global_step'] % 100 == 0:
            print(f"Step {step_result['global_step']}: Loss {step_result['loss']:.6f}")
    
    # Validation
    val_metrics = trainer.validate(val_dataloader)
    print(f"Epoch {epoch}: Val Loss {val_metrics['val_loss']:.6f}")
    
    # Save checkpoint
    if val_metrics['val_loss'] < trainer.best_loss:
        trainer.save_checkpoint(f"best_model_epoch_{epoch}.pt")
```

### Custom Training

```python
# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        return mse + 0.1 * perceptual

# Custom optimizer
optimizer = torch.optim.AdamW(
    model_manager.unet.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

## 🔍 Analysis & Quality

### Image Quality Analysis

```python
# Analyze single image
quality_metrics = analyzer.analyze_generation_quality(
    images=[image],
    prompts=["prompt"]
)

print(f"Brightness: {quality_metrics['brightness']:.4f}")
print(f"Contrast: {quality_metrics['contrast']:.4f}")
print(f"Sharpness: {quality_metrics['sharpness']:.4f}")
print(f"Color Diversity: {quality_metrics['color_diversity']:.4f}")
```

### Performance Analysis

```python
# Analyze generation performance
performance_metrics = analyzer.analyze_model_performance(
    generation_times=[1.2, 1.1, 1.3],
    memory_usage=[2048, 2048, 2048]
)

print(f"Average generation time: {performance_metrics['generation_time']['mean']:.4f}s")
print(f"Images per second: {performance_metrics['throughput']['images_per_second']:.2f}")
```

### Comprehensive Reports

```python
# Generate comprehensive report
report = analyzer.create_generation_report(
    images=images,
    prompts=prompts,
    generation_times=generation_times,
    quality_metrics=quality_metrics
)

# Save report
with open("generation_report.json", "w") as f:
    json.dump(report, f, indent=2)
```

## 🚀 Memory Optimization

### Attention Slicing

```python
# Enable attention slicing for memory efficiency
diffusion_config = DiffusionConfig(
    enable_attention_slicing=True,
    enable_vae_slicing=True
)
```

### XFormers Integration

```python
# Enable XFormers for memory-efficient attention
diffusion_config = DiffusionConfig(
    enable_xformers_memory_efficient_attention=True
)
```

### Model Offloading

```python
# Enable CPU offloading for large models
diffusion_config = DiffusionConfig(
    enable_model_cpu_offload=True
)
```

## 📊 Monitoring & Logging

### Training Metrics

```python
# Monitor training progress
for step_result in training_results:
    print(f"Step {step_result['global_step']}")
    print(f"  Loss: {step_result['loss']:.6f}")
    print(f"  Learning Rate: {step_result['learning_rate']:.2e}")
    print(f"  Epoch: {step_result['epoch']}")
```

### Quality Tracking

```python
# Track quality over time
quality_history = []
for epoch in range(num_epochs):
    # Generate test images
    test_images = model_manager.generate_image(test_prompts)
    
    # Analyze quality
    quality = analyzer.analyze_generation_quality(test_images, test_prompts)
    quality_history.append(quality)
    
    print(f"Epoch {epoch}: Sharpness {quality['sharpness']:.4f}")
```

## 🔧 Advanced Features

### Custom Schedulers

```python
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

# Use DDIM scheduler
scheduler = DDIMScheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)

# Use DPM-Solver++ scheduler
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_name,
    subfolder="scheduler"
)
```

### EMA Integration

```python
# Enable Exponential Moving Average
diffusion_config = DiffusionConfig(
    use_ema=True
)

# EMA model will be automatically created and updated
```

### Gradient Checkpointing

```python
# Enable gradient checkpointing for memory efficiency
diffusion_config = DiffusionConfig(
    use_gradient_checkpointing=True
)
```

## 📁 File Structure

```
diffusion_models_system/
├── diffusion_models_system.py      # Main system implementation
├── diffusion_models_demo.py        # Comprehensive demonstration
├── requirements_diffusion_models.txt # Dependencies
├── DIFFUSION_MODELS_README.md      # This file
├── generated_images/               # Generated images output
├── checkpoints/                    # Training checkpoints
└── reports/                        # Analysis reports
```

## 🧪 Testing

### Run Demo

```bash
python diffusion_models_demo.py
```

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest -v --cov=diffusion_models_system
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Enable attention slicing and VAE slicing
2. **Slow Generation**: Reduce inference steps or use faster schedulers
3. **Poor Quality**: Increase guidance scale or inference steps
4. **Model Loading Errors**: Check internet connection and model names

### Performance Tips

1. **Use Mixed Precision**: Enable AMP for faster training
2. **Optimize Batch Size**: Balance memory usage and training efficiency
3. **Use XFormers**: Enable for memory-efficient attention
4. **Gradient Checkpointing**: Enable for large models

## 🔗 Integration

### With Experiment Tracking

```python
from experiment_tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker("diffusion_experiment")

# Log training metrics
tracker.log_metric("train_loss", step_result['loss'])
tracker.log_metric("learning_rate", step_result['learning_rate'])

# Log generated images
tracker.log_image("generated_image", images[0])
```

### With Gradio Interface

```python
import gradio as gr

def generate_image_interface(prompt, negative_prompt):
    images = model_manager.generate_image(prompt, negative_prompt)
    return images[0]

# Create Gradio interface
interface = gr.Interface(
    fn=generate_image_interface,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt")
    ],
    outputs=gr.Image(label="Generated Image")
)
```

## 📚 References

- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [DDIM Paper](https://arxiv.org/abs/2010.02502)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the Diffusers library
- Stability AI for Stable Diffusion
- The open-source AI community

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo scripts

---

**Note**: This system requires significant computational resources, especially for training. GPU acceleration is highly recommended for optimal performance.






