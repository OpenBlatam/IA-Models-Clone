# Official Documentation Reference Guide

This guide provides comprehensive references to official documentation best practices and up-to-date APIs for PyTorch, Transformers, Diffusers, and Gradio.

## Table of Contents

1. [PyTorch Best Practices](#pytorch-best-practices)
2. [Transformers Best Practices](#transformers-best-practices)
3. [Diffusers Best Practices](#diffusers-best-practices)
4. [Gradio Best Practices](#gradio-best-practices)
5. [Version Compatibility](#version-compatibility)
6. [Performance Optimization](#performance-optimization)
7. [Migration Guides](#migration-guides)
8. [Code Examples](#code-examples)

## PyTorch Best Practices

### Official Documentation
- **Main Documentation**: https://pytorch.org/docs/stable/
- **Tutorials**: https://pytorch.org/tutorials/
- **GitHub**: https://github.com/pytorch/pytorch
- **Current Version**: 2.1.0
- **Minimum Supported**: 1.13.0

### Critical Best Practices

#### 1. Automatic Mixed Precision (AMP)
```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Best Practices:**
- Use `GradScaler` for gradient scaling
- Wrap forward pass in `autocast()` context
- Scale loss before backward pass
- Update scaler after optimizer step
- Can provide 2-3x speedup on modern GPUs
- Reduces memory usage by ~50%

#### 2. Efficient Data Loading
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Multiprocessing
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True,  # Reduce worker startup overhead
    drop_last=True  # Consistent batch sizes
)
```

**Best Practices:**
- Use `num_workers=4-8` for CPU-bound data loading
- Enable `pin_memory=True` for GPU training
- Use `persistent_workers=True` for efficiency
- Set appropriate `batch_size` based on GPU memory

#### 3. Model Checkpointing
```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'config': config
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

#### 4. Distributed Training
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```

### Performance Optimization

#### 1. Memory Management
```python
# Clear cache
torch.cuda.empty_cache()

# Use gradient checkpointing
model = torch.utils.checkpoint.checkpoint_sequential(
    model, input, segments=2
)

# Profile memory usage
torch.cuda.memory_summary()
```

#### 2. Model Optimization
```python
# JIT compilation
scripted_model = torch.jit.script(model)

# TorchScript optimization
traced_model = torch.jit.trace(model, example_input)

# Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## Transformers Best Practices

### Official Documentation
- **Main Documentation**: https://huggingface.co/docs/transformers/
- **Model Hub**: https://huggingface.co/models
- **GitHub**: https://github.com/huggingface/transformers
- **Current Version**: 4.35.0
- **Minimum Supported**: 4.20.0

### Critical Best Practices

#### 1. Model Loading
```python
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Load model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# For specific tasks
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# With device mapping for large models
model = AutoModel.from_pretrained(
    "microsoft/DialoGPT-large",
    device_map="auto",
    torch_dtype=torch.float16
)
```

**Best Practices:**
- Use `AutoModel` classes for flexibility
- Load tokenizer from same model
- Use `device_map="auto"` for large models
- Specify `torch_dtype` for memory efficiency

#### 2. Tokenization
```python
# Basic tokenization
inputs = tokenizer("Hello world!", return_tensors="pt")

# Batch tokenization
texts = ["Hello world!", "How are you?"]
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# For training
inputs = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt",
    add_special_tokens=True
)
```

**Best Practices:**
- Use `padding` and `truncation` for batch processing
- Set appropriate `max_length`
- Use `return_tensors='pt'` for PyTorch
- Handle special tokens properly

#### 3. Training with Trainer
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    eval_steps=1000,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,  # Mixed precision
    dataloader_num_workers=4,
    gradient_accumulation_steps=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

#### 4. Custom Training Loop
```python
from transformers import get_scheduler

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

### Performance Optimization

#### 1. Memory Efficiency
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Use 8-bit optimizers
from transformers import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=5e-5)

# Model parallelism
model = AutoModel.from_pretrained(
    "microsoft/DialoGPT-large",
    device_map="balanced",
    max_memory={0: "10GB", 1: "10GB"}
)
```

#### 2. Caching and Optimization
```python
# Cache tokenized datasets
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# Use accelerate for distributed training
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

## Diffusers Best Practices

### Official Documentation
- **Main Documentation**: https://huggingface.co/docs/diffusers/
- **Model Hub**: https://huggingface.co/models?pipeline_tag=text-to-image
- **GitHub**: https://github.com/huggingface/diffusers
- **Current Version**: 0.24.0
- **Minimum Supported**: 0.18.0

### Critical Best Practices

#### 1. Pipeline Usage
```python
from diffusers import DiffusionPipeline
import torch

# Load pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Move to GPU
pipeline = pipeline.to("cuda")

# Generate images
image = pipeline("A beautiful sunset").images[0]
image.save("sunset.png")

# Batch generation
prompts = ["A cat", "A dog", "A bird"]
images = pipeline(prompts, num_inference_steps=50)
```

**Best Practices:**
- Use `torch.float16` for memory efficiency
- Set appropriate `num_inference_steps`
- Use batch processing for multiple images
- Enable attention slicing for large models

#### 2. Memory Optimization
```python
# Enable attention slicing
pipeline.enable_attention_slicing()

# Enable model offloading
pipeline.enable_model_cpu_offload()

# Use sequential CPU offloading
pipeline.enable_sequential_cpu_offload()

# Enable xformers memory efficient attention
pipeline.enable_xformers_memory_efficient_attention()

# Use VAE slicing
pipeline.enable_vae_slicing()
```

#### 3. Custom Training
```python
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

# Initialize model and scheduler
model = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5")
scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5")

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=1000)

for batch in dataloader:
    noise = torch.randn_like(batch)
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch.shape[0],))
    noisy_batch = scheduler.add_noise(batch, noise, timesteps)
    
    noise_pred = model(noisy_batch, timesteps).sample
    loss = F.mse_loss(noise_pred, noise)
    
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
```

#### 4. Advanced Features
```python
# ControlNet
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

# LoRA fine-tuning
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### Performance Optimization

#### 1. Inference Optimization
```python
# Use torch.compile (PyTorch 2.0+)
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")

# Use attention slicing
pipeline.enable_attention_slicing(slice_size="max")

# Use VAE tiling
pipeline.enable_vae_tiling()

# Use model offloading
pipeline.enable_model_cpu_offload()
```

#### 2. Memory Management
```python
# Clear cache
torch.cuda.empty_cache()

# Use gradient checkpointing
model.enable_gradient_checkpointing()

# Use 8-bit optimizers
from transformers import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=1e-4)
```

## Gradio Best Practices

### Official Documentation
- **Main Documentation**: https://gradio.app/docs/
- **Examples**: https://gradio.app/docs/examples
- **GitHub**: https://github.com/gradio-app/gradio
- **Current Version**: 4.0.0
- **Minimum Supported**: 3.50.0

### Critical Best Practices

#### 1. Interface Creation
```python
import gradio as gr

def predict(text):
    # Your model prediction logic here
    return f"Prediction: {text}"

# Create interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input Text", placeholder="Enter text here..."),
    outputs=gr.Textbox(label="Output"),
    title="My Model Demo",
    description="Enter text to get predictions",
    examples=[
        ["Hello world"],
        ["How are you?"],
        ["This is a test"]
    ],
    cache_examples=True
)

# Launch
interface.launch(server_name="0.0.0.0", server_port=7860)
```

**Best Practices:**
- Use appropriate input/output components
- Provide clear labels and descriptions
- Handle errors gracefully
- Use examples for better UX
- Enable caching for expensive operations

#### 2. Advanced Components
```python
import gradio as gr

def process_image(image, text, slider_value):
    # Process image and text
    return processed_image, f"Processed: {text} (slider: {slider_value})"

# Advanced interface with Blocks
with gr.Blocks(title="Advanced Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Advanced Image Processing Demo")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Input Image", type="pil")
            text_input = gr.Textbox(label="Input Text", placeholder="Enter description...")
            slider = gr.Slider(minimum=0, maximum=100, value=50, label="Processing Intensity")
            submit_btn = gr.Button("Process", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Column(scale=1):
            image_output = gr.Image(label="Output Image")
            text_output = gr.Textbox(label="Output Text")
    
    # Event handlers
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, text_input, slider],
        outputs=[image_output, text_output]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", None, ""),
        inputs=[],
        outputs=[image_input, text_input, image_output, text_output]
    )

demo.launch()
```

#### 3. Error Handling
```python
import gradio as gr
import traceback

def predict_with_error_handling(text):
    try:
        # Your model prediction logic here
        if not text:
            raise ValueError("Input text cannot be empty")
        
        result = model.predict(text)
        return result
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg

interface = gr.Interface(
    fn=predict_with_error_handling,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Textbox(label="Output"),
    title="Error Handling Demo"
)
```

#### 4. Deployment
```python
# Local deployment
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Create public link
    debug=True
)

# Hugging Face Spaces deployment
# Create app.py with your interface
# Add requirements.txt
# Push to Hugging Face Spaces

# Docker deployment
# Create Dockerfile
# Build and run container
```

### Performance Optimization

#### 1. Caching and Queuing
```python
# Enable caching
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    cache_examples=True
)

# Use queuing for heavy operations
interface.launch(
    max_threads=4,
    queue=True,
    concurrency_count=2
)
```

#### 2. Component Optimization
```python
# Use appropriate components
gr.Image(type="pil")  # For image processing
gr.Audio(type="numpy")  # For audio processing
gr.Video()  # For video processing

# Use batch processing
def batch_predict(texts):
    return [predict(text) for text in texts]

interface = gr.Interface(
    fn=batch_predict,
    inputs=gr.Textbox(lines=5, label="Input Texts (one per line)"),
    outputs=gr.JSON(label="Batch Results")
)
```

## Version Compatibility

### PyTorch Compatibility Matrix

| PyTorch Version | CUDA Version | Python Version | Key Features |
|----------------|--------------|----------------|--------------|
| 2.1.0 | 11.8, 12.1 | 3.8-3.11 | torch.compile, improved AMP |
| 2.0.0 | 11.7, 12.1 | 3.8-3.11 | torch.compile, better performance |
| 1.13.0 | 11.6, 11.7 | 3.7-3.10 | Stable release, good compatibility |

### Transformers Compatibility

| Transformers Version | PyTorch Version | Key Features |
|---------------------|-----------------|--------------|
| 4.35.0 | 1.13.0+ | Latest models, improved performance |
| 4.30.0 | 1.13.0+ | Stable release, good compatibility |
| 4.20.0 | 1.13.0+ | Minimum supported version |

### Diffusers Compatibility

| Diffusers Version | Transformers Version | Key Features |
|------------------|---------------------|--------------|
| 0.24.0 | 4.30.0+ | Latest diffusion models |
| 0.21.0 | 4.25.0+ | Stable release |
| 0.18.0 | 4.20.0+ | Minimum supported version |

### Gradio Compatibility

| Gradio Version | Python Version | Key Features |
|---------------|----------------|--------------|
| 4.0.0 | 3.8+ | Latest UI components |
| 3.50.0 | 3.8+ | Stable release |
| 3.40.0 | 3.8+ | Minimum supported version |

## Performance Optimization

### General Optimization Tips

1. **Use Mixed Precision**: Enable AMP for 2-3x speedup
2. **Optimize Data Loading**: Use appropriate num_workers and pin_memory
3. **Use Gradient Accumulation**: For large effective batch sizes
4. **Enable Caching**: Cache expensive operations
5. **Use Model Parallelism**: For very large models
6. **Profile Your Code**: Use torch.profiler to identify bottlenecks

### Memory Optimization

1. **Gradient Checkpointing**: Trade compute for memory
2. **Model Offloading**: Move parts to CPU when not needed
3. **Attention Slicing**: Reduce memory usage in attention layers
4. **Use 8-bit Optimizers**: Reduce optimizer memory usage
5. **Clear Cache**: Regularly clear GPU cache

### Distributed Training

1. **Use DistributedDataParallel**: For multi-GPU training
2. **Use DistributedSampler**: For proper data distribution
3. **Optimize Communication**: Use NCCL backend
4. **Use Gradient Accumulation**: For large effective batch sizes

## Migration Guides

### PyTorch 1.13 to 2.0

**Breaking Changes:**
- `torch.jit.script` behavior changes
- Some deprecated functions removed

**New Features:**
- `torch.compile` for automatic optimization
- Improved AMP performance
- Better memory management

**Migration Steps:**
1. Update PyTorch to 2.0
2. Test existing code
3. Enable `torch.compile` where beneficial
4. Update deprecated function calls

### Transformers 4.20 to 4.35

**Breaking Changes:**
- Some model loading APIs changed
- Tokenizer behavior updates

**New Features:**
- Better model parallelism
- Improved memory efficiency
- New model architectures

**Migration Steps:**
1. Update Transformers
2. Test model loading
3. Update tokenizer usage if needed
4. Enable new features where beneficial

### Diffusers 0.18 to 0.24

**Breaking Changes:**
- Some pipeline APIs changed
- Model loading updates

**New Features:**
- Better memory optimization
- New diffusion models
- Improved training utilities

**Migration Steps:**
1. Update Diffusers
2. Test pipeline usage
3. Update model loading if needed
4. Enable new optimizations

## Code Examples

### Complete Training Example

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
import gradio as gr

# Model setup
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=1000)

# Training loop with AMP
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

# Gradio interface
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return f"Prediction: {outputs.logits.argmax().item()}"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input Text"),
    outputs=gr.Textbox(label="Output"),
    title="BERT Classification Demo"
)

interface.launch()
```

### Diffusion Model Example

```python
import torch
from diffusers import DiffusionPipeline
import gradio as gr

# Load pipeline with optimizations
pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()

def generate_image(prompt, num_steps=50):
    image = pipeline(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=7.5
    ).images[0]
    return image

# Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A beautiful sunset..."),
        gr.Slider(minimum=10, maximum=100, value=50, label="Inference Steps")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Stable Diffusion Demo"
)

interface.launch()
```

## Conclusion

This guide provides comprehensive references to official documentation best practices for PyTorch, Transformers, Diffusers, and Gradio. Always refer to the official documentation for the most up-to-date information and follow the recommended practices for optimal performance and compatibility.

### Key Takeaways

1. **Always use the latest stable versions** when possible
2. **Enable mixed precision training** for better performance
3. **Optimize data loading** with appropriate settings
4. **Use official APIs** and avoid deprecated functions
5. **Handle errors gracefully** in production code
6. **Profile and optimize** based on your specific use case
7. **Follow security best practices** for deployment
8. **Keep dependencies updated** regularly

### Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Hugging Face Courses](https://huggingface.co/course) 