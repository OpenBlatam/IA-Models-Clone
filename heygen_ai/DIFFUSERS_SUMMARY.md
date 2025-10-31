# Diffusers Library Integration Summary for HeyGen AI

## Overview
Comprehensive integration with the Hugging Face Diffusers library for state-of-the-art diffusion models, providing high-level APIs for image generation, training, and inference with optimized performance and memory management following PEP 8 style guidelines.

## Core Components

### 1. **Diffusers Manager** (`diffusers_integration.py`)

#### Diffusers Manager Implementation
- **DiffusersManager**: High-level manager for diffusion pipeline operations
- **Pipeline Management**: Automatic loading and configuration of diffusion pipelines
- **Optimization Support**: Built-in performance and memory optimizations
- **Flexible Configuration**: Configurable model selection, schedulers, and parameters

#### Diffusers Manager Features
```python
# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    scheduler_type="DDIMScheduler",  # "DDIMScheduler", "PNDMScheduler", "EulerDiscreteScheduler"
    torch_dtype="float16",  # "float16", "float32"
    device="cuda",
    use_safetensors=True,
    enable_attention_slicing=True,
    enable_vae_slicing=True,
    enable_model_cpu_offload=False,
    enable_sequential_cpu_offload=False,
    enable_xformers_memory_efficient_attention=True,
    guidance_scale=7.5,
    num_inference_steps=50,
    height=512,
    width=512,
    num_images_per_prompt=1,
    eta=0.0,
    negative_prompt="",
    seed=None
)

# Create manager
manager = DiffusersManager(config)

# Load pipeline
manager.load_pipeline()

# Generate images
images = manager.generate_images(
    prompt="A beautiful landscape with mountains and a lake, digital art",
    num_images=2,
    guidance_scale=7.5,
    num_inference_steps=50
)

# Generate with latents
latents = torch.randn(1, 4, 64, 64)
image = manager.generate_images_with_latents(
    prompt="A futuristic city",
    latents=latents,
    guidance_scale=7.5
)

# Encode and decode
text_embeddings = manager.encode_prompt("A beautiful sunset")
decoded_image = manager.decode_latents(latents)

# Access components
scheduler = manager.get_scheduler()
unet = manager.get_unet()
vae = manager.get_vae()
text_encoder = manager.get_text_encoder()
```

#### Pipeline Loading Implementation
```python
def load_pipeline(self) -> None:
    """Load the diffusion pipeline."""
    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler
        
        # Load pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.torch_dtype,
            use_safetensors=self.config.use_safetensors
        )
        
        # Configure scheduler
        if self.config.scheduler_type == "DDIMScheduler":
            self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        elif self.config.scheduler_type == "PNDMScheduler":
            self.scheduler = PNDMScheduler.from_config(self.pipeline.scheduler.config)
        elif self.config.scheduler_type == "EulerDiscreteScheduler":
            self.scheduler = EulerDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        
        self.pipeline.scheduler = self.scheduler
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable optimizations
        if self.config.enable_attention_slicing:
            self.pipeline.enable_attention_slicing()
        
        if self.config.enable_vae_slicing:
            self.pipeline.enable_vae_slicing()
        
        if self.config.enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        
        if self.config.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        
        if self.config.enable_xformers_memory_efficient_attention:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except ImportError:
                logger.warning("xformers not available, skipping memory efficient attention")
        
        logger.info(f"Loaded pipeline with scheduler: {self.config.scheduler_type}")
        
    except ImportError as e:
        logger.error(f"Diffusers library not available: {e}")
        raise
```

#### Image Generation Implementation
```python
def generate_images(
    self,
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_images: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    seed: Optional[int] = None
) -> List[torch.Tensor]:
    """Generate images using the pipeline."""
    if self.pipeline is None:
        raise ValueError("Pipeline not loaded. Call load_pipeline() first.")
    
    # Set parameters
    if negative_prompt is None:
        negative_prompt = self.config.negative_prompt
    if num_images is None:
        num_images = self.config.num_images_per_prompt
    if guidance_scale is None:
        guidance_scale = self.config.guidance_scale
    if num_inference_steps is None:
        num_inference_steps = self.config.num_inference_steps
    if height is None:
        height = self.config.height
    if width is None:
        width = self.config.width
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Generate images
    with torch.no_grad():
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            eta=self.config.eta
        ).images
    
    return images
```

### 2. **Diffusers Training Manager**

#### Training Manager Implementation
- **DiffusersTrainingManager**: Manager for training diffusion models
- **Model Loading**: Load individual components for training
- **Training Loop**: Optimized training step implementation
- **Checkpoint Management**: Save and load training checkpoints

#### Training Manager Features
```python
# Create training manager
training_manager = DiffusersTrainingManager(config)

# Load models for training
models = training_manager.load_models_for_training()

# Access components
vae = models["vae"]
unet = models["unet"]
text_encoder = models["text_encoder"]
tokenizer = models["tokenizer"]
noise_scheduler = models["noise_scheduler"]

# Create optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

# Training step
batch = {
    "pixel_values": torch.randn(2, 3, 512, 512),
    "input_ids": torch.randint(0, 1000, (2, 77))
}

metrics = training_manager.training_step(
    batch=batch,
    models=models,
    optimizer=optimizer,
    noise_scheduler=noise_scheduler
)

# Save checkpoint
training_manager.save_checkpoint(
    models=models,
    optimizer=optimizer,
    epoch=10,
    save_path="checkpoint.pt"
)
```

#### Model Loading for Training
```python
def load_models_for_training(self):
    """Load models for training."""
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=self.torch_dtype
        )
        
        # Load UNet
        unet = UNet2DConditionModel.from_pretrained(
            self.config.model_id,
            subfolder="unet",
            torch_dtype=self.torch_dtype
        )
        
        # Load text encoder
        text_encoder = CLIPTextModel.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype
        )
        
        # Load tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer"
        )
        
        # Load scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.model_id,
            subfolder="scheduler"
        )
        
        # Freeze VAE and text encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Move to device
        vae = vae.to(self.device)
        unet = unet.to(self.device)
        text_encoder = text_encoder.to(self.device)
        
        return {
            "vae": vae,
            "unet": unet,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "noise_scheduler": noise_scheduler
        }
        
    except ImportError as e:
        logger.error(f"Required libraries not available: {e}")
        raise
```

#### Training Step Implementation
```python
def training_step(
    self,
    batch: Dict[str, torch.Tensor],
    models: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    noise_scheduler
) -> Dict[str, float]:
    """Training step."""
    vae = models["vae"]
    unet = models["unet"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    
    # Get batch data
    pixel_values = batch["pixel_values"].to(self.device)
    input_ids = batch["input_ids"].to(self.device)
    
    # Encode images
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    
    # Encode text
    encoder_hidden_states = text_encoder(input_ids)[0]
    
    # Sample noise
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()
    
    # Add noise
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Predict noise
    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    
    # Compute loss
    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {"loss": loss.item()}
```

### 3. **Diffusers Inference Manager**

#### Inference Manager Implementation
- **DiffusersInferenceManager**: Manager for step-by-step inference
- **Component Access**: Direct access to individual model components
- **Custom Generation**: Customizable generation process
- **Latent Manipulation**: Direct manipulation of latent representations

#### Inference Manager Features
```python
# Create inference manager
inference_manager = DiffusersInferenceManager(config)

# Load models for inference
models = inference_manager.load_models_for_inference()

# Generate image step by step
image = inference_manager.generate_image_step_by_step(
    prompt="A serene forest with sunlight filtering through trees, nature photography",
    models=models,
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    seed=42
)

# Access components
vae = models["vae"]
unet = models["unet"]
text_encoder = models["text_encoder"]
tokenizer = models["tokenizer"]
scheduler = models["scheduler"]
```

#### Step-by-Step Generation Implementation
```python
def generate_image_step_by_step(
    self,
    prompt: str,
    models: Dict[str, Any],
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None
) -> torch.Tensor:
    """Generate image step by step."""
    vae = models["vae"]
    unet = models["unet"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    scheduler = models["scheduler"]
    
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Tokenize prompts
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    uncond_input = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=text_input.input_ids.shape[-1],
        truncation=True,
        return_tensors="pt"
    )
    
    # Encode text
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Initialize latents
    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        device=self.device,
        dtype=self.torch_dtype
    )
    latents = latents * scheduler.init_noise_sigma
    
    # Denoising loop
    scheduler.set_timesteps(num_inference_steps)
    
    for t in scheduler.timesteps:
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode latents
    latents = 1 / vae.config.scaling_factor * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    
    return image
```

## Complete Usage Examples

### 1. **Basic Pipeline Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    scheduler_type="DDIMScheduler",
    torch_dtype="float16",
    device="cuda",
    guidance_scale=7.5,
    num_inference_steps=50,
    height=512,
    width=512
)

# Create manager
manager = DiffusersManager(config)

# Load pipeline
manager.load_pipeline()

# Generate images
prompt = "A beautiful landscape with mountains and a lake, digital art"
images = manager.generate_images(
    prompt=prompt,
    num_images=2,
    guidance_scale=7.5,
    num_inference_steps=50
)

print(f"Generated {len(images)} images")
print(f"Image type: {type(images[0])}")
print(f"Image size: {images[0].size}")
```

### 2. **Different Schedulers Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager

schedulers = ["DDIMScheduler", "PNDMScheduler", "EulerDiscreteScheduler"]
results = {}

for scheduler_type in schedulers:
    print(f"Testing {scheduler_type}...")
    
    # Create configuration
    config = DiffusersConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        scheduler_type=scheduler_type,
        torch_dtype="float16",
        device="cuda",
        guidance_scale=7.5,
        num_inference_steps=50
    )
    
    # Create manager
    manager = DiffusersManager(config)
    
    # Load pipeline
    manager.load_pipeline()
    
    # Generate image
    prompt = "A futuristic city at night, neon lights, cyberpunk style"
    start_time = time.time()
    images = manager.generate_images(
        prompt=prompt,
        num_images=1,
        guidance_scale=7.5,
        num_inference_steps=50
    )
    end_time = time.time()
    
    results[scheduler_type] = {
        "images": images,
        "time": end_time - start_time
    }
    
    print(f"{scheduler_type} - Time: {results[scheduler_type]['time']:.2f}s")
```

### 3. **Guidance Scale Comparison Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    scheduler_type="DDIMScheduler",
    torch_dtype="float16",
    device="cuda",
    num_inference_steps=50
)

# Create manager
manager = DiffusersManager(config)

# Load pipeline
manager.load_pipeline()

# Test different guidance scales
guidance_scales = [1.0, 3.0, 7.5, 15.0]
prompt = "A majestic dragon flying over a medieval castle, fantasy art"
results = {}

for guidance_scale in guidance_scales:
    print(f"Testing guidance scale: {guidance_scale}")
    
    start_time = time.time()
    images = manager.generate_images(
        prompt=prompt,
        num_images=1,
        guidance_scale=guidance_scale,
        num_inference_steps=50
    )
    end_time = time.time()
    
    results[guidance_scale] = {
        "images": images,
        "time": end_time - start_time
    }
    
    print(f"Guidance scale {guidance_scale} - Time: {results[guidance_scale]['time']:.2f}s")
```

### 4. **Step-by-Step Generation Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersInferenceManager

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    torch_dtype="float16",
    device="cuda"
)

# Create inference manager
manager = DiffusersInferenceManager(config)

# Load models
models = manager.load_models_for_inference()

# Generate image step by step
prompt = "A serene forest with sunlight filtering through trees, nature photography"
image = manager.generate_image_step_by_step(
    prompt=prompt,
    models=models,
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    seed=42
)

print(f"Generated image with shape: {image.shape}")
print(f"Image statistics - Min: {image.min():.4f}, Max: {image.max():.4f}, Mean: {image.mean():.4f}")
```

### 5. **Training Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersTrainingManager

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    torch_dtype="float16",
    device="cuda"
)

# Create training manager
manager = DiffusersTrainingManager(config)

# Load models for training
models = manager.load_models_for_training()

# Create optimizer
optimizer = torch.optim.AdamW(models["unet"].parameters(), lr=1e-4)

# Sample training data (in practice, this would come from a dataset)
batch_size = 2
height, width = 512, 512
channels = 3

# Mock training step
batch = {
    "pixel_values": torch.randn(batch_size, channels, height, width),
    "input_ids": torch.randint(0, 1000, (batch_size, 77))  # Mock tokenized text
}

# Training step
metrics = manager.training_step(
    batch=batch,
    models=models,
    optimizer=optimizer,
    noise_scheduler=models["noise_scheduler"]
)

print(f"Training loss: {metrics['loss']:.6f}")
```

### 6. **Model Components Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    torch_dtype="float16",
    device="cuda"
)

# Create manager
manager = DiffusersManager(config)

# Load pipeline
manager.load_pipeline()

# Get individual components
scheduler = manager.get_scheduler()
unet = manager.get_unet()
vae = manager.get_vae()
text_encoder = manager.get_text_encoder()

print(f"Scheduler type: {type(scheduler).__name__}")
print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
print(f"Text encoder parameters: {sum(p.numel() for p in text_encoder.parameters()):,}")

# Test encoding and decoding
prompt = "A beautiful sunset over the ocean"
text_embeddings = manager.encode_prompt(prompt)

print(f"Text embeddings shape: {text_embeddings.shape}")

# Test VAE encoding/decoding
test_image = torch.randn(1, 3, 512, 512)
latents = vae.encode(test_image).latent_dist.sample()
decoded_image = vae.decode(latents).sample

print(f"Original image shape: {test_image.shape}")
print(f"Latents shape: {latents.shape}")
print(f"Decoded image shape: {decoded_image.shape}")
```

### 7. **Performance Optimization Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager

# Test different optimization settings
optimization_configs = [
    {
        "name": "Default",
        "enable_attention_slicing": False,
        "enable_vae_slicing": False,
        "enable_xformers_memory_efficient_attention": False
    },
    {
        "name": "Attention Slicing",
        "enable_attention_slicing": True,
        "enable_vae_slicing": False,
        "enable_xformers_memory_efficient_attention": False
    },
    {
        "name": "VAE Slicing",
        "enable_attention_slicing": False,
        "enable_vae_slicing": True,
        "enable_xformers_memory_efficient_attention": False
    },
    {
        "name": "XFormers",
        "enable_attention_slicing": False,
        "enable_vae_slicing": False,
        "enable_xformers_memory_efficient_attention": True
    },
    {
        "name": "All Optimizations",
        "enable_attention_slicing": True,
        "enable_vae_slicing": True,
        "enable_xformers_memory_efficient_attention": True
    }
]

results = {}

for opt_config in optimization_configs:
    print(f"Testing {opt_config['name']}...")
    
    # Create configuration
    config = DiffusersConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        scheduler_type="DDIMScheduler",
        torch_dtype="float16",
        device="cuda",
        enable_attention_slicing=opt_config["enable_attention_slicing"],
        enable_vae_slicing=opt_config["enable_vae_slicing"],
        enable_xformers_memory_efficient_attention=opt_config["enable_xformers_memory_efficient_attention"]
    )
    
    # Create manager
    manager = DiffusersManager(config)
    
    # Load pipeline
    manager.load_pipeline()
    
    # Generate image and measure time
    prompt = "A futuristic robot in a neon-lit city, sci-fi art"
    start_time = time.time()
    
    try:
        images = manager.generate_images(
            prompt=prompt,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        end_time = time.time()
        
        results[opt_config["name"]] = {
            "time": end_time - start_time,
            "success": True,
            "images": images
        }
        
        print(f"{opt_config['name']} - Time: {results[opt_config['name']]['time']:.2f}s")
        
    except Exception as e:
        results[opt_config["name"]] = {
            "time": None,
            "success": False,
            "error": str(e)
        }
        print(f"{opt_config['name']} failed: {e}")
```

### 8. **Different Models Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager

models_to_test = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "CompVis/stable-diffusion-v1-4"
]

results = {}

for model_id in models_to_test:
    print(f"Testing model: {model_id}")
    
    try:
        # Create configuration
        config = DiffusersConfig(
            model_id=model_id,
            scheduler_type="DDIMScheduler",
            torch_dtype="float16",
            device="cuda",
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # Create manager
        manager = DiffusersManager(config)
        
        # Load pipeline
        manager.load_pipeline()
        
        # Generate image
        prompt = "A majestic eagle soaring over snow-capped mountains, wildlife photography"
        start_time = time.time()
        images = manager.generate_images(
            prompt=prompt,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        end_time = time.time()
        
        results[model_id] = {
            "time": end_time - start_time,
            "success": True,
            "images": images
        }
        
        print(f"{model_id} - Time: {results[model_id]['time']:.2f}s")
        
    except Exception as e:
        results[model_id] = {
            "time": None,
            "success": False,
            "error": str(e)
        }
        print(f"{model_id} failed: {e}")
```

### 9. **Memory Usage Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager
import psutil
import gc

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    scheduler_type="DDIMScheduler",
    torch_dtype="float16",
    device="cuda"
)

# Memory analysis
process = psutil.Process()

# Initial memory
initial_memory = process.memory_info().rss / 1024 / 1024  # MB
print(f"Initial memory: {initial_memory:.2f} MB")

# Create manager
manager = DiffusersManager(config)
manager_memory = process.memory_info().rss / 1024 / 1024  # MB
print(f"After creating manager: {manager_memory:.2f} MB (+{manager_memory - initial_memory:.2f} MB)")

# Load pipeline
manager.load_pipeline()
pipeline_memory = process.memory_info().rss / 1024 / 1024  # MB
print(f"After loading pipeline: {pipeline_memory:.2f} MB (+{pipeline_memory - manager_memory:.2f} MB)")

# Generate image
prompt = "A peaceful garden with blooming flowers, impressionist painting"
start_time = time.time()
images = manager.generate_images(
    prompt=prompt,
    num_images=1,
    guidance_scale=7.5,
    num_inference_steps=50
)
end_time = time.time()
generation_memory = process.memory_info().rss / 1024 / 1024  # MB

print(f"After generation: {generation_memory:.2f} MB (+{generation_memory - pipeline_memory:.2f} MB)")
print(f"Generation time: {end_time - start_time:.2f}s")

# Clean up
del manager, images
gc.collect()
torch.cuda.empty_cache()

final_memory = process.memory_info().rss / 1024 / 1024  # MB
print(f"After cleanup: {final_memory:.2f} MB")
```

### 10. **Custom Scheduler Example**
```python
from diffusers import DDIMScheduler, EulerDiscreteScheduler
from .diffusers_integration import DiffusersConfig, DiffusersManager

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    torch_dtype="float16",
    device="cuda"
)

# Create manager
manager = DiffusersManager(config)

# Load pipeline
manager.load_pipeline()

# Test custom scheduler parameters
custom_schedulers = {
    "DDIM_eta_0": DDIMScheduler.from_config(manager.pipeline.scheduler.config, eta=0.0),
    "DDIM_eta_1": DDIMScheduler.from_config(manager.pipeline.scheduler.config, eta=1.0),
    "Euler": EulerDiscreteScheduler.from_config(manager.pipeline.scheduler.config)
}

results = {}

for name, scheduler in custom_schedulers.items():
    print(f"Testing {name}...")
    
    # Set scheduler
    manager.pipeline.scheduler = scheduler
    
    # Generate image
    prompt = "A magical forest with glowing mushrooms and fairy lights, fantasy art"
    start_time = time.time()
    images = manager.generate_images(
        prompt=prompt,
        num_images=1,
        guidance_scale=7.5,
        num_inference_steps=50
    )
    end_time = time.time()
    
    results[name] = {
        "time": end_time - start_time,
        "images": images
    }
    
    print(f"{name} - Time: {results[name]['time']:.2f}s")
```

### 11. **Batch Generation Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    scheduler_type="DDIMScheduler",
    torch_dtype="float16",
    device="cuda"
)

# Create manager
manager = DiffusersManager(config)

# Load pipeline
manager.load_pipeline()

# Batch of prompts
prompts = [
    "A serene lake at sunset, landscape photography",
    "A futuristic city with flying cars, sci-fi art",
    "A cozy cottage in the woods, digital art",
    "A majestic dragon breathing fire, fantasy art"
]

results = {}

for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
    
    start_time = time.time()
    images = manager.generate_images(
        prompt=prompt,
        num_images=2,  # Generate 2 variations
        guidance_scale=7.5,
        num_inference_steps=50
    )
    end_time = time.time()
    
    results[f"prompt_{i+1}"] = {
        "prompt": prompt,
        "time": end_time - start_time,
        "images": images
    }
    
    print(f"Prompt {i+1} - Time: {results[f'prompt_{i+1}']['time']:.2f}s")
```

### 12. **Seed Consistency Example**
```python
from .diffusers_integration import DiffusersConfig, DiffusersManager

# Create configuration
config = DiffusersConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    scheduler_type="DDIMScheduler",
    torch_dtype="float16",
    device="cuda"
)

# Create manager
manager = DiffusersManager(config)

# Load pipeline
manager.load_pipeline()

# Test seed consistency
prompt = "A beautiful butterfly on a flower, macro photography"
seeds = [42, 42, 123, 123]  # Test same seeds
results = {}

for i, seed in enumerate(seeds):
    print(f"Generating with seed {seed} (run {i+1})")
    
    start_time = time.time()
    images = manager.generate_images(
        prompt=prompt,
        num_images=1,
        guidance_scale=7.5,
        num_inference_steps=50,
        seed=seed
    )
    end_time = time.time()
    
    results[f"seed_{seed}_run_{i+1}"] = {
        "seed": seed,
        "time": end_time - start_time,
        "images": images
    }
    
    print(f"Seed {seed} run {i+1} - Time: {results[f'seed_{seed}_run_{i+1}']['time']:.2f}s")
```

## Key Benefits

### 1. **High-Level API**
- **Easy Integration**: Simple API for loading and using diffusion models
- **Automatic Optimization**: Built-in performance and memory optimizations
- **Flexible Configuration**: Configurable model selection and parameters

### 2. **Multiple Model Support**
- **Stable Diffusion**: Support for various Stable Diffusion models
- **Custom Models**: Easy integration with custom trained models
- **Model Comparison**: Built-in tools for comparing different models

### 3. **Performance Optimizations**
- **Attention Slicing**: Memory-efficient attention computation
- **VAE Slicing**: Memory-efficient VAE encoding/decoding
- **XFormers**: Memory-efficient attention with xformers
- **Model Offloading**: CPU offloading for memory management

### 4. **Training Support**
- **Component Access**: Direct access to individual model components
- **Training Loop**: Optimized training step implementation
- **Checkpoint Management**: Save and load training checkpoints

### 5. **Inference Flexibility**
- **Step-by-Step Generation**: Customizable generation process
- **Latent Manipulation**: Direct manipulation of latent representations
- **Custom Schedulers**: Support for custom noise schedulers

### 6. **Production-Ready Features**
- **Error Handling**: Robust error handling and validation
- **Memory Management**: Efficient memory usage and cleanup
- **Performance Monitoring**: Built-in performance analysis tools
- **Extensible**: Easy to extend with new features

### 7. **Comprehensive Examples**
- **Basic Usage**: Simple examples for getting started
- **Advanced Features**: Complex examples for advanced users
- **Performance Analysis**: Benchmarking and memory analysis
- **Model Comparison**: Tools for comparing different models and configurations

The Diffusers library integration provides a comprehensive framework for working with state-of-the-art diffusion models, offering high-level APIs, performance optimizations, and extensive customization options while maintaining ease of use and production readiness. 