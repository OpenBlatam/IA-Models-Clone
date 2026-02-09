# Performance Optimization Guide

## Advanced Optimization Techniques for AI Video Generation

This guide covers the latest performance optimization techniques for PyTorch, Transformers, Diffusers, and Gradio to maximize efficiency in AI video generation.

---

## 🚀 PyTorch 2.0+ Optimizations

### 1. Torch Compile (PyTorch 2.0+)

#### **Basic Compilation**
```python
import torch
import torch.nn as nn

class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Compile model for faster inference
model = VideoModel()
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # Optimize for inference
    fullgraph=True,          # Compile entire graph
    dynamic=True             # Support dynamic shapes
)

# Usage
x = torch.randn(1, 3, 16, 256, 256)
with torch.no_grad():
    output = compiled_model(x)
```

#### **Advanced Compilation Modes**
```python
# Different compilation modes for different use cases
modes = {
    "default": "default",           # Balanced optimization
    "reduce-overhead": "reduce-overhead",  # Minimize Python overhead
    "max-autotune": "max-autotune"  # Maximum optimization
}

for mode_name, mode in modes.items():
    compiled_model = torch.compile(model, mode=mode)
    print(f"Compiled with {mode_name} mode")
```

### 2. Flash Attention and Memory Efficient Attention

#### **Enable Flash Attention**
```python
# Enable all attention optimizations
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Use in attention layers
class OptimizedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )
    
    def forward(self, x):
        # Flash attention is automatically used when available
        return self.attention(x, x, x)[0]

# Test performance improvement
import time

def benchmark_attention(model, x, iterations=100):
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        _ = model(x)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / iterations

# Compare with and without flash attention
model_standard = OptimizedAttention(512, 8)
model_flash = OptimizedAttention(512, 8)

x = torch.randn(32, 100, 512).cuda()

time_standard = benchmark_attention(model_standard, x)
time_flash = benchmark_attention(model_flash, x)

print(f"Standard attention: {time_standard:.4f}s")
print(f"Flash attention: {time_flash:.4f}s")
print(f"Speedup: {time_standard/time_flash:.2f}x")
```

### 3. Mixed Precision Training

#### **Advanced Mixed Precision**
```python
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def train_step(self, batch, targets):
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(batch)
            loss = F.mse_loss(outputs, targets)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

# Usage
trainer = MixedPrecisionTrainer(model, optimizer)
for batch, targets in dataloader:
    loss = trainer.train_step(batch, targets)
```

### 4. Memory Optimization

#### **Gradient Checkpointing**
```python
from torch.utils.checkpoint import checkpoint

class LargeVideoModel(nn.Module):
    def __init__(self, num_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(512, 8) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            # Use checkpointing to save memory
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# Memory usage comparison
def measure_memory_usage(model, x):
    torch.cuda.reset_peak_memory_stats()
    _ = model(x)
    return torch.cuda.max_memory_allocated() / 1024**3  # GB

model_standard = LargeVideoModel()
model_checkpoint = LargeVideoModel()

x = torch.randn(1, 100, 512).cuda()

memory_standard = measure_memory_usage(model_standard, x)
memory_checkpoint = measure_memory_usage(model_checkpoint, x)

print(f"Standard model: {memory_standard:.2f} GB")
print(f"Checkpointed model: {memory_checkpoint:.2f} GB")
print(f"Memory reduction: {memory_standard/memory_checkpoint:.2f}x")
```

---

## 🔄 Transformers Optimizations

### 1. Model Quantization

#### **8-bit Quantization**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with 8-bit quantization
model_8bit = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Performance comparison
def benchmark_model(model, input_ids, iterations=100):
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_ids)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / iterations

# Compare quantization levels
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()

# Load different quantization levels
model_fp16 = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    torch_dtype=torch.float16,
    device_map="auto"
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    load_in_8bit=True,
    device_map="auto"
)

# Benchmark
time_fp16 = benchmark_model(model_fp16, input_ids)
time_8bit = benchmark_model(model_8bit, input_ids)

print(f"FP16 model: {time_fp16:.4f}s")
print(f"8-bit model: {time_8bit:.4f}s")
print(f"Speedup: {time_fp16/time_8bit:.2f}x")
```

#### **4-bit Quantization with BitsAndBytes**
```python
from transformers import BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with 4-bit quantization
model_4bit = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    quantization_config=bnb_config,
    device_map="auto"
)

# Memory usage comparison
def get_model_memory_usage(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_memory / 1024**3  # GB

memory_fp16 = get_model_memory_usage(model_fp16)
memory_4bit = get_model_memory_usage(model_4bit)

print(f"FP16 model: {memory_fp16:.2f} GB")
print(f"4-bit model: {memory_4bit:.2f} GB")
print(f"Memory reduction: {memory_fp16/memory_4bit:.2f}x")
```

### 2. Efficient Tokenization

#### **Batch Tokenization Optimization**
```python
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Optimize batch tokenization
def optimized_tokenize_batch(texts, max_length=512):
    # Use padding and truncation for efficiency
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False  # Skip if not needed
    )
    return inputs

# Performance comparison
texts = ["Hello world"] * 100

# Standard tokenization
start_time = time.time()
for text in texts:
    _ = tokenizer(text)
standard_time = time.time() - start_time

# Optimized batch tokenization
start_time = time.time()
_ = optimized_tokenize_batch(texts)
batch_time = time.time() - start_time

print(f"Standard tokenization: {standard_time:.4f}s")
print(f"Batch tokenization: {batch_time:.4f}s")
print(f"Speedup: {standard_time/batch_time:.2f}x")
```

### 3. Generation Optimization

#### **Optimized Text Generation**
```python
def optimized_generation(model, tokenizer, prompt, max_length=100):
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Optimized generation config
    generation_config = {
        "max_length": max_length,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,  # Enable KV cache
        "return_dict_in_generate": True,
        "output_scores": False,  # Disable if not needed
        "output_attentions": False,  # Disable if not needed
        "output_hidden_states": False  # Disable if not needed
    }
    
    # Generate with optimizations
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Benchmark generation
prompt = "Once upon a time"
iterations = 10

start_time = time.time()
for _ in range(iterations):
    _ = optimized_generation(model, tokenizer, prompt)
generation_time = (time.time() - start_time) / iterations

print(f"Average generation time: {generation_time:.4f}s")
```

---

## 🎨 Diffusers Optimizations

### 1. Pipeline Optimizations

#### **Memory Efficient Pipeline**
```python
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

def create_optimized_pipeline(model_name="runwayml/stable-diffusion-v1-5"):
    # Load pipeline with optimizations
    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Use optimized scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    
    # Enable memory optimizations
    pipeline.enable_attention_slicing(slice_size="auto")
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()
    
    # Use xformers if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    # Compile UNet for faster inference
    if hasattr(torch, 'compile'):
        pipeline.unet = torch.compile(
            pipeline.unet,
            mode="reduce-overhead",
            fullgraph=True
        )
    
    return pipeline

# Performance comparison
def benchmark_pipeline(pipeline, prompt, num_inference_steps=50):
    torch.cuda.synchronize()
    start_time = time.time()
    
    _ = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        height=512,
        width=512
    )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    return end_time - start_time

# Create optimized pipeline
pipeline = create_optimized_pipeline()
pipeline = pipeline.to("cuda")

# Benchmark
prompt = "A beautiful landscape painting"
time_optimized = benchmark_pipeline(pipeline, prompt)

print(f"Optimized pipeline time: {time_optimized:.2f}s")
```

### 2. Video Generation Optimization

#### **Optimized Video Pipeline**
```python
from diffusers import TextToVideoPipeline

def create_optimized_video_pipeline():
    # Load video generation pipeline
    pipeline = TextToVideoPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16
    )
    
    # Enable optimizations
    pipeline.enable_attention_slicing()
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()
    
    # Use optimized scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    
    return pipeline

# Video generation with optimizations
def generate_optimized_video(pipeline, prompt, num_frames=16):
    # Optimized generation parameters
    video_frames = pipeline(
        prompt,
        num_inference_steps=50,  # Reduced for speed
        height=256,              # Reduced resolution
        width=256,
        num_frames=num_frames,
        fps=8,                   # Reduced FPS
        guidance_scale=7.5       # Optimized guidance
    ).frames
    
    return video_frames

# Benchmark video generation
pipeline = create_optimized_video_pipeline()
pipeline = pipeline.to("cuda")

prompt = "A cat walking in the rain"
start_time = time.time()
video_frames = generate_optimized_video(pipeline, prompt)
video_time = time.time() - start_time

print(f"Video generation time: {video_time:.2f}s")
print(f"Generated {len(video_frames)} frames")
```

### 3. Custom Training Optimization

#### **Optimized Training Loop**
```python
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
import torch.nn.functional as F

class OptimizedTrainer:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()
    
    def train_step(self, batch, noise_scheduler):
        self.optimizer.zero_grad()
        
        # Add noise
        noise = torch.randn_like(batch)
        batch_size = batch.shape[0]
        timesteps = torch.randint(
            0, 
            noise_scheduler.num_train_timesteps, 
            (batch_size,),
            device=batch.device
        ).long()
        
        noisy_batch = noise_scheduler.add_noise(batch, noise, timesteps)
        
        # Forward pass with mixed precision
        with autocast():
            noise_pred = self.model(noisy_batch, timesteps).sample
            loss = F.mse_loss(noise_pred, noise, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape))))
            loss = loss.mean()
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return loss.item()

# Usage
model = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet"
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=1000)
noise_scheduler = DDPMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

trainer = OptimizedTrainer(model, optimizer, scheduler)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = trainer.train_step(batch, noise_scheduler)
```

---

## 🎛️ Gradio Optimizations

### 1. Interface Performance

#### **Optimized Interface Design**
```python
import gradio as gr
import asyncio
import threading

def create_optimized_interface():
    # Optimized video generation function
    def generate_video_optimized(prompt, num_frames, height, width, progress=gr.Progress()):
        # Simulate progress updates
        for i in range(num_frames):
            progress(i / num_frames, desc=f"Generating frame {i+1}/{num_frames}")
            time.sleep(0.1)  # Simulate processing
        
        # Return generated video
        return "generated_video.mp4"
    
    # Create optimized interface
    interface = gr.Interface(
        fn=generate_video_optimized,
        inputs=[
            gr.Textbox(
                label="Prompt",
                placeholder="Describe the video...",
                lines=3,
                max_lines=10,
                show_label=True,
                container=True,
                scale=1
            ),
            gr.Slider(
                minimum=8,
                maximum=64,
                value=16,
                step=8,
                label="Frames",
                show_label=True,
                container=True,
                scale=1
            ),
            gr.Slider(
                minimum=256,
                maximum=1024,
                value=512,
                step=64,
                label="Height",
                show_label=True,
                container=True,
                scale=1
            ),
            gr.Slider(
                minimum=256,
                maximum=1024,
                value=512,
                step=64,
                label="Width",
                show_label=True,
                container=True,
                scale=1
            )
        ],
        outputs=gr.Video(
            label="Generated Video",
            show_label=True,
            container=True,
            scale=1
        ),
        title="Optimized AI Video Generation",
        description="Generate videos with optimized performance",
        examples=[
            ["A cat walking in the rain", 16, 512, 512],
            ["A sunset over the ocean", 24, 512, 512]
        ],
        cache_examples=True,  # Cache example outputs
        theme=gr.themes.Soft(),
        show_progress=True,  # Show progress bars
        queue=True  # Enable queuing for long operations
    )
    
    return interface

# Launch with optimizations
interface = create_optimized_interface()
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    enable_queue=True,
    max_threads=40,  # Increase thread pool
    show_api=False,  # Disable API docs for performance
    quiet=False,
    show_error=True,
    favicon_path=None,
    app_kwargs=None,
    inbrowser=True,
    prevent_thread_lock=False,
    show_tips=True,
    height=500,
    show_btn=None,
    server_protocol="http"
)
```

### 2. Async Processing

#### **Async Video Generation**
```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class AsyncVideoGenerator:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def generate_video_async(self, prompt, num_frames, height, width, progress=gr.Progress()):
        # Run video generation in thread pool
        loop = asyncio.get_event_loop()
        
        def generate_video_sync():
            # Simulate video generation
            for i in range(num_frames):
                progress(i / num_frames, desc=f"Generating frame {i+1}/{num_frames}")
                time.sleep(0.1)
            return "generated_video.mp4"
        
        # Execute in thread pool
        result = await loop.run_in_executor(
            self.executor,
            generate_video_sync
        )
        
        return result

# Create async interface
generator = AsyncVideoGenerator()

def create_async_interface():
    with gr.Blocks(title="Async Video Generation") as demo:
        gr.Markdown("# Async AI Video Generation")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", placeholder="Describe the video...")
                num_frames = gr.Slider(8, 64, 16, step=8, label="Frames")
                height = gr.Slider(256, 1024, 512, step=64, label="Height")
                width = gr.Slider(256, 1024, 512, step=64, label="Width")
                generate_btn = gr.Button("Generate Video", variant="primary")
            
            with gr.Column(scale=2):
                output_video = gr.Video(label="Generated Video")
                status = gr.Textbox(label="Status", interactive=False)
        
        # Async event handler
        async def generate_video_handler(prompt, num_frames, height, width, progress=gr.Progress()):
            try:
                video = await generator.generate_video_async(
                    prompt, num_frames, height, width, progress
                )
                return video, "Video generated successfully!"
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        generate_btn.click(
            fn=generate_video_handler,
            inputs=[prompt, num_frames, height, width],
            outputs=[output_video, status],
            api_name="generate_video",
            show_progress=True,
            queue=True
        )
    
    return demo

# Launch async interface
demo = create_async_interface()
demo.launch(
    enable_queue=True,
    max_threads=40,
    show_progress=True
)
```

### 3. Caching and Optimization

#### **Result Caching**
```python
import hashlib
import json
import os

class VideoCache:
    def __init__(self, cache_dir="video_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, prompt, num_frames, height, width):
        # Create unique cache key
        params = {
            "prompt": prompt,
            "num_frames": num_frames,
            "height": height,
            "width": width
        }
        key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return key
    
    def get_cached_video(self, prompt, num_frames, height, width):
        key = self.get_cache_key(prompt, num_frames, height, width)
        cache_file = os.path.join(self.cache_dir, f"{key}.mp4")
        
        if os.path.exists(cache_file):
            return cache_file
        return None
    
    def cache_video(self, prompt, num_frames, height, width, video_path):
        key = self.get_cache_key(prompt, num_frames, height, width)
        cache_file = os.path.join(self.cache_dir, f"{key}.mp4")
        
        # Copy video to cache
        import shutil
        shutil.copy2(video_path, cache_file)
        return cache_file

# Create cached video generation
cache = VideoCache()

def generate_video_with_cache(prompt, num_frames, height, width, progress=gr.Progress()):
    # Check cache first
    cached_video = cache.get_cached_video(prompt, num_frames, height, width)
    if cached_video:
        progress(1.0, desc="Loading from cache")
        return cached_video
    
    # Generate new video
    progress(0, desc="Generating video...")
    video_path = generate_video(prompt, num_frames, height, width, progress)
    
    # Cache the result
    cache.cache_video(prompt, num_frames, height, width, video_path)
    
    return video_path

# Interface with caching
interface = gr.Interface(
    fn=generate_video_with_cache,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(8, 64, 16, step=8, label="Frames"),
        gr.Slider(256, 1024, 512, step=64, label="Height"),
        gr.Slider(256, 1024, 512, step=64, label="Width")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="Cached Video Generation",
    description="Generate videos with intelligent caching"
)
```

---

## 📊 Performance Monitoring

### 1. Memory and GPU Monitoring

```python
import psutil
import GPUtil
import time

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def get_system_stats(self):
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU stats
        gpu_stats = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature
                })
        except:
            pass
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / 1024**3,
            "memory_total_gb": memory.total / 1024**3,
            "gpu_stats": gpu_stats,
            "uptime": time.time() - self.start_time
        }
    
    def monitor_training(self, model, dataloader, num_epochs):
        """Monitor training performance."""
        stats_history = []
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                # Get stats before training step
                stats = self.get_system_stats()
                
                # Training step
                loss = model.training_step(batch)
                
                # Record stats
                stats["epoch"] = epoch
                stats["batch"] = batch_idx
                stats["loss"] = loss
                stats_history.append(stats)
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: Loss {loss:.4f}")
                    print(f"GPU Memory: {stats['gpu_stats'][0]['memory_used']}MB")
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        return stats_history

# Usage
monitor = PerformanceMonitor()
stats = monitor.get_system_stats()
print(f"CPU: {stats['cpu_percent']}%")
print(f"Memory: {stats['memory_percent']}%")
print(f"GPU Memory: {stats['gpu_stats'][0]['memory_used']}MB")
```

### 2. Performance Profiling

```python
import torch.profiler as profiler

def profile_model(model, input_data):
    """Profile model performance."""
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with profiler.record_function("model_inference"):
            output = model(input_data)
    
    # Print profiling results
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
    
    return output

# Profile video generation
def profile_video_generation(pipeline, prompt):
    """Profile video generation pipeline."""
    
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with profiler.record_function("video_generation"):
            video_frames = pipeline(
                prompt,
                num_inference_steps=50,
                height=256,
                width=256,
                num_frames=16
            )
    
    # Print profiling results
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))
    
    return video_frames
```

This performance optimization guide provides comprehensive techniques for maximizing efficiency in AI video generation using the latest features from all major libraries. 