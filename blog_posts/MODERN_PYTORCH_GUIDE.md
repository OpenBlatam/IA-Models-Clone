# Modern PyTorch, Transformers, Diffusers, and Gradio Best Practices Guide

## Overview

This guide covers modern best practices and up-to-date APIs for the deep learning ecosystem, focusing on PyTorch 2.0+, Transformers, Diffusers, and Gradio. It provides production-ready implementations with the latest optimizations and features.

## Key Features

### 1. PyTorch 2.0+ Optimizations
- **torch.compile**: Automatic model compilation for performance
- **torch.func**: Functional programming for PyTorch
- **torch.export**: Model serialization and deployment
- **Mixed Precision**: Automatic mixed precision training
- **Memory Optimization**: Gradient checkpointing and efficient attention

### 2. Modern Transformer Training
- **Latest Transformers Library**: Up-to-date model architectures
- **Quantization**: 4-bit and 8-bit quantization for memory efficiency
- **Accelerate Integration**: Distributed training and optimization
- **PEFT Support**: Parameter-efficient fine-tuning
- **Modern Training Loop**: Best practices for transformer training

### 3. Advanced Diffusion Models
- **Latest Diffusers**: State-of-the-art diffusion pipelines
- **Multiple Schedulers**: DDPM, DDIM, DPM-Solver, and more
- **Memory Optimization**: Attention slicing and model offloading
- **ControlNet Support**: Advanced control mechanisms
- **Image-to-Image**: Inpainting and upscaling capabilities

### 4. Modern Gradio Interfaces
- **Gradio 4.0+**: Latest UI components and themes
- **Real-time Updates**: Live model inference and training
- **Advanced Components**: Custom components and layouts
- **Performance Optimization**: Efficient interface design
- **Integration**: Seamless integration with all components

## PyTorch 2.0+ Best Practices

### 1. Model Compilation with torch.compile

```python
from modern_pytorch_practices import ModernPyTorchPractices

practices = ModernPyTorchPractices()

# Compile model for better performance
compiled_model = practices.demonstrate_torch_compile(model)

# Different compilation modes
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # For inference
    fullgraph=True,
    dynamic=True
)

# Maximum optimization mode
compiled_model = torch.compile(
    model,
    mode="max-autotune",  # For maximum performance
    fullgraph=True
)
```

### 2. Functional Programming with torch.func

```python
from torch.func import functional_call, vmap, grad

# Functional model call
def model_fn(params, x):
    return functional_call(model, params, (x,))

# Vectorized operations
batched_input = torch.randn(10, 3, 224, 224)
batched_output = vmap(model_fn, in_dims=(None, 0))(params, batched_input)

# Gradient computation
grad_fn = grad(model_fn)
gradients = grad_fn(params, batched_input[0])
```

### 3. Model Export with torch.export

```python
# Export model for deployment
exported_model = torch.export(
    model,
    (example_input,),
    dynamic_shapes={"x": {0: "batch_size"}},
    strict=False
)

# Save exported model
torch.save(exported_model, "exported_model.pt")
```

### 4. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def training_step(data, target):
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss
```

### 5. Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use channels_last memory format
model = model.to(memory_format=torch.channels_last)

# Memory efficient attention
model.config.use_memory_efficient_attention = True
```

## Modern Transformer Training

### 1. Configuration and Setup

```python
from modern_pytorch_practices import TransformerConfig, ModernTransformerTrainer

config = TransformerConfig(
    model_name="bert-base-uncased",
    task="classification",
    num_labels=2,
    max_length=512,
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    fp16=True,
    dataloader_num_workers=4
)

trainer = ModernTransformerTrainer(config)
```

### 2. Quantization for Memory Efficiency

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 3. Training with Modern Practices

```python
# Prepare dataset
train_dataset = trainer.prepare_dataset(train_texts, train_labels)
val_dataset = trainer.prepare_dataset(val_texts, val_labels)

# Train model
trainer_result = trainer.train(train_texts, train_labels, val_texts, val_labels)

# Make predictions
predictions = trainer.predict(["This is a test sentence"])
```

### 4. Different Tasks

```python
# Classification
config = TransformerConfig(
    model_name="bert-base-uncased",
    task="classification",
    num_labels=3
)

# Generation
config = TransformerConfig(
    model_name="gpt2",
    task="generation"
)

# Token Classification
config = TransformerConfig(
    model_name="bert-base-uncased",
    task="token_classification",
    num_labels=10
)
```

## Advanced Diffusion Models

### 1. Pipeline Configuration

```python
from modern_pytorch_practices import DiffusionConfig, ModernDiffusionPipeline

config = DiffusionConfig(
    model_name="runwayml/stable-diffusion-v1-5",
    scheduler_name="DPMSolverMultistepScheduler",
    num_inference_steps=20,
    guidance_scale=7.5,
    width=512,
    height=512,
    batch_size=1,
    use_attention_processor=True,
    enable_memory_efficient_attention=True
)

pipeline = ModernDiffusionPipeline(config)
```

### 2. Text-to-Image Generation

```python
# Generate single image
image = pipeline.generate_image(
    prompt="A beautiful sunset over mountains",
    negative_prompt="blurry, low quality"
)

# Generate multiple images
prompts = [
    "A cute cat playing with yarn",
    "A majestic dragon flying over a castle"
]
images = pipeline.generate_image_batch(prompts)
```

### 3. Image-to-Image Generation

```python
# Load input image
input_image = PIL.Image.open("input.jpg")

# Generate image-to-image
result = pipeline.img2img_generation(
    image=input_image,
    prompt="A blue version of this image",
    strength=0.8
)
```

### 4. Inpainting

```python
# Load image and mask
image = PIL.Image.open("image.jpg")
mask = PIL.Image.open("mask.png")

# Perform inpainting
result = pipeline.inpainting(
    image=image,
    mask=mask,
    prompt="A beautiful flower in the masked area"
)
```

### 5. Different Schedulers

```python
# DDPM Scheduler
config = DiffusionConfig(
    scheduler_name="DDPMScheduler",
    num_inference_steps=1000
)

# DDIM Scheduler
config = DiffusionConfig(
    scheduler_name="DDIMScheduler",
    num_inference_steps=50
)

# DPM-Solver Scheduler
config = DiffusionConfig(
    scheduler_name="DPMSolverMultistepScheduler",
    num_inference_steps=20
)

# Euler Scheduler
config = DiffusionConfig(
    scheduler_name="EulerDiscreteScheduler",
    num_inference_steps=30
)
```

## Modern Gradio Interfaces

### 1. Basic Interface Setup

```python
from modern_pytorch_practices import ModernGradioInterface

interface = ModernGradioInterface()

# Create transformer interface
transformer_interface = interface.create_transformer_interface()

# Create diffusion interface
diffusion_interface = interface.create_diffusion_interface()

# Create combined interface
combined_interface = interface.create_combined_interface()
```

### 2. Custom Interface Components

```python
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft(), title="Modern Deep Learning") as interface:
    gr.Markdown("# 🤖 Modern Deep Learning Interface")
    
    with gr.Tabs():
        with gr.TabItem("Transformers"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_name = gr.Dropdown(
                        choices=["bert-base-uncased", "roberta-base", "gpt2"],
                        value="bert-base-uncased",
                        label="Model"
                    )
                    
                    task = gr.Dropdown(
                        choices=["classification", "generation"],
                        value="classification",
                        label="Task"
                    )
                
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        lines=3,
                        label="Input Text",
                        placeholder="Enter your text here..."
                    )
                    
                    generate_button = gr.Button("🚀 Generate", variant="primary")
                    output = gr.JSON(label="Output")
```

### 3. Real-time Training Interface

```python
def create_training_interface():
    with gr.Blocks(theme=gr.themes.Glass()) as interface:
        gr.Markdown("# 🎯 Real-time Training Interface")
        
        with gr.Row():
            with gr.Column():
                # Training configuration
                config_inputs = [
                    gr.Dropdown(choices=["bert", "gpt2"], label="Model"),
                    gr.Slider(minimum=1, maximum=10, value=3, label="Epochs"),
                    gr.Slider(minimum=1e-6, maximum=1e-3, value=2e-5, label="Learning Rate")
                ]
                
                # Training data
                train_texts = gr.Textbox(lines=5, label="Training Texts")
                train_labels = gr.Textbox(lines=5, label="Training Labels")
                
                train_button = gr.Button("🚀 Start Training", variant="primary")
            
            with gr.Column():
                # Training progress
                progress_bar = gr.Progress()
                training_logs = gr.Textbox(lines=10, label="Training Logs")
                
                # Real-time metrics
                loss_plot = gr.Plot(label="Training Loss")
                accuracy_plot = gr.Plot(label="Accuracy")
        
        # Event handlers
        train_button.click(
            train_model,
            inputs=config_inputs + [train_texts, train_labels],
            outputs=[training_logs, loss_plot, accuracy_plot],
            show_progress=True
        )
    
    return interface
```

### 4. Advanced Diffusion Interface

```python
def create_diffusion_interface():
    with gr.Blocks(theme=gr.themes.Monochrome()) as interface:
        gr.Markdown("# 🎨 Advanced Diffusion Interface")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model configuration
                model_name = gr.Dropdown(
                    choices=["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1"],
                    value="runwayml/stable-diffusion-v1-5",
                    label="Model"
                )
                
                scheduler = gr.Dropdown(
                    choices=["DPMSolverMultistepScheduler", "EulerDiscreteScheduler"],
                    value="DPMSolverMultistepScheduler",
                    label="Scheduler"
                )
                
                num_steps = gr.Slider(minimum=1, maximum=50, value=20, label="Steps")
                guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, label="Guidance")
            
            with gr.Column(scale=2):
                # Generation
                prompt = gr.Textbox(lines=3, label="Prompt")
                negative_prompt = gr.Textbox(lines=2, label="Negative Prompt")
                
                generate_button = gr.Button("🎨 Generate", variant="primary")
                output_image = gr.Image(label="Generated Image")
        
        # Event handler
        generate_button.click(
            generate_image,
            inputs=[model_name, scheduler, num_steps, guidance_scale, prompt, negative_prompt],
            outputs=output_image
        )
    
    return interface
```

## Integration with Existing System

### 1. Modern Deep Learning System

```python
from modern_pytorch_practices import ModernDeepLearningSystem

# Initialize system
system = ModernDeepLearningSystem()

# Run transformer experiment
transformer_config = TransformerConfig(
    model_name="bert-base-uncased",
    task="classification",
    num_labels=2,
    num_epochs=3
)

trainer_result, optimized_model = system.run_transformer_experiment(
    transformer_config,
    train_texts=["positive", "negative"],
    train_labels=[1, 0]
)

# Run diffusion experiment
diffusion_config = DiffusionConfig(
    model_name="runwayml/stable-diffusion-v1-5",
    num_inference_steps=20
)

images = system.run_diffusion_experiment(
    diffusion_config,
    prompts=["A beautiful landscape", "A cute animal"]
)

# Launch interface
system.launch_interface(port=7860)
```

### 2. Integration with Experiment Tracking

```python
from experiment_tracking import experiment_tracking

# Integrate with experiment tracking
with experiment_tracking("modern_pytorch_experiment", config) as tracker:
    # Train transformer
    trainer_result, optimized_model = system.run_transformer_experiment(
        transformer_config, train_texts, train_labels
    )
    
    # Log metrics
    tracker.log_metrics({
        "transformer_accuracy": 0.95,
        "transformer_loss": 0.12
    })
    
    # Generate diffusion images
    images = system.run_diffusion_experiment(diffusion_config, prompts)
    
    # Log images
    for i, image in enumerate(images):
        if image:
            image_path = f"generated_image_{i}.png"
            image.save(image_path)
            tracker.log_image(image_path, f"generated_image_{i}")
```

### 3. Integration with Version Control

```python
from version_control import version_control

# Integrate with version control
with version_control("modern_pytorch_project", auto_commit=True) as vc:
    # Version transformer model
    model_version = vc.version_model(
        model_name="bert_classifier",
        model_path="final_model/pytorch_model.bin",
        metadata={
            "architecture": "bert-base-uncased",
            "task": "classification",
            "accuracy": 0.95
        },
        description="Trained BERT classifier for sentiment analysis"
    )
    
    # Version diffusion pipeline
    pipeline_version = vc.version_model(
        model_name="stable_diffusion",
        model_path="diffusion_pipeline",
        metadata={
            "model": "runwayml/stable-diffusion-v1-5",
            "scheduler": "DPMSolverMultistepScheduler"
        },
        description="Optimized Stable Diffusion pipeline"
    )
```

## Performance Optimization

### 1. PyTorch 2.0+ Optimizations

```python
# Enable PyTorch 2.0 optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use channels_last memory format
model = model.to(memory_format=torch.channels_last)

# Compile model for maximum performance
model = torch.compile(model, mode="max-autotune", fullgraph=True)
```

### 2. Memory Optimization

```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Memory efficient attention
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()
pipeline.enable_model_cpu_offload()

# Use attention processor 2.0
pipeline.unet.set_attn_processor(AttnProcessor2_0())
```

### 3. Batch Processing

```python
# Efficient batch processing
def process_batch(texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = model(batch)
        results.extend(batch_results)
    return results
```

## Best Practices

### 1. Model Selection

```python
# Choose appropriate model size
models = {
    "small": "distilbert-base-uncased",  # 66M parameters
    "medium": "bert-base-uncased",       # 110M parameters
    "large": "bert-large-uncased",       # 340M parameters
    "xl": "microsoft/DialoGPT-large"     # 774M parameters
}

# Consider task requirements
task_models = {
    "classification": "bert-base-uncased",
    "generation": "gpt2",
    "translation": "t5-base",
    "summarization": "facebook/bart-base"
}
```

### 2. Training Configuration

```python
# Optimal learning rates
learning_rates = {
    "bert": 2e-5,
    "gpt2": 5e-5,
    "t5": 1e-4,
    "roberta": 1e-5
}

# Batch sizes for different GPU memory
batch_sizes = {
    "8GB": 8,
    "16GB": 16,
    "24GB": 32,
    "40GB": 64
}
```

### 3. Diffusion Configuration

```python
# Optimal inference steps
step_configs = {
    "fast": {"steps": 10, "scheduler": "DPMSolverMultistepScheduler"},
    "balanced": {"steps": 20, "scheduler": "EulerDiscreteScheduler"},
    "quality": {"steps": 50, "scheduler": "DDIMScheduler"}
}

# Guidance scales
guidance_scales = {
    "creative": 3.0,
    "balanced": 7.5,
    "precise": 15.0
}
```

## Error Handling

### 1. Model Loading Errors

```python
try:
    model = AutoModel.from_pretrained("model-name")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Fallback to smaller model
    model = AutoModel.from_pretrained("distilbert-base-uncased")
```

### 2. CUDA Memory Errors

```python
try:
    model = model.cuda()
except RuntimeError as e:
    if "out of memory" in str(e):
        # Enable memory optimizations
        model.gradient_checkpointing_enable()
        torch.cuda.empty_cache()
        model = model.cuda()
```

### 3. Generation Errors

```python
def safe_generate(pipeline, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return pipeline.generate_image(prompt)
        except Exception as e:
            logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return None
```

## Monitoring and Logging

### 1. Performance Monitoring

```python
import time
import psutil
import GPUtil

def monitor_performance():
    # CPU usage
    cpu_percent = psutil.cpu_percent()
    
    # Memory usage
    memory = psutil.virtual_memory()
    
    # GPU usage
    gpu_usage = 0
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = gpus[0].load * 100
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "gpu_usage": gpu_usage
    }
```

### 2. Model Performance

```python
def benchmark_model(model, input_data, num_runs=100):
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time * 1000  # Convert to milliseconds
```

## Conclusion

This guide provides comprehensive coverage of modern best practices for PyTorch, Transformers, Diffusers, and Gradio. Key takeaways include:

- **PyTorch 2.0+**: Leverage torch.compile, torch.func, and torch.export for optimal performance
- **Transformers**: Use quantization, modern training loops, and task-specific configurations
- **Diffusers**: Implement memory optimizations, multiple schedulers, and advanced pipelines
- **Gradio**: Create modern, responsive interfaces with real-time updates
- **Integration**: Seamlessly integrate with experiment tracking and version control
- **Performance**: Apply memory optimizations and batch processing for efficiency
- **Monitoring**: Implement comprehensive logging and performance monitoring

These practices ensure production-ready, efficient, and maintainable deep learning systems. 