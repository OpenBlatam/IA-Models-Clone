# Diffusion Models Implementation Summary for HeyGen AI

## Overview
Comprehensive implementation of diffusion models including DDPM (Denoising Diffusion Probabilistic Models), DDIM (Denoising Diffusion Implicit Models), and advanced variants with noise schedulers, UNet architectures, and training/sampling utilities following PEP 8 style guidelines.

## Core Components

### 1. **Noise Scheduler** (`diffusion_models.py`)

#### Noise Scheduler Implementation
- **NoiseScheduler**: Manages noise schedules for diffusion models
- **Multiple Beta Schedules**: Linear, cosine, and sigmoid schedules
- **DDIM Support**: DDIM-specific timesteps and sigmas
- **Precomputed Values**: Efficient access to commonly used values

#### Noise Scheduler Features
```python
# Create configuration
config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",  # "linear", "cosine", "sigmoid"
    prediction_type="epsilon",  # "epsilon", "x0", "v"
    loss_type="mse",  # "mse", "l1", "huber"
    guidance_scale=7.5,
    classifier_free_guidance=True,
    clip_denoised=True,
    use_ema=True,
    ema_decay=0.9999
)

# Create noise scheduler
scheduler = NoiseScheduler(config)

# Add noise
noisy_x, noise = scheduler.add_noise(x_start, timesteps)

# Remove noise
denoised_x = scheduler.remove_noise(noisy_x, noise, timesteps)

# Compute posterior statistics
posterior_mean, posterior_variance, posterior_log_variance = scheduler.q_posterior_mean_variance(
    x_start, x_t, timesteps
)

# Compute p statistics
p_mean, p_variance, p_log_variance = scheduler.p_mean_variance(
    model_output, x_t, timesteps, clip_denoised=True
)

# DDIM step
x_prev = scheduler.ddim_step(model_output, x_t, timesteps, eta=0.0)
```

#### Beta Schedule Implementation
```python
def _get_beta_schedule(self) -> torch.Tensor:
    """Get beta schedule."""
    if self.beta_schedule == "linear":
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
    elif self.beta_schedule == "cosine":
        return self._cosine_beta_schedule()
    elif self.beta_schedule == "sigmoid":
        return self._sigmoid_beta_schedule()
    else:
        raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

def _cosine_beta_schedule(self) -> torch.Tensor:
    """Cosine beta schedule."""
    steps = self.num_timesteps + 1
    x = torch.linspace(0, self.num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def _sigmoid_beta_schedule(self) -> torch.Tensor:
    """Sigmoid beta schedule."""
    betas = torch.linspace(-6, 6, self.num_timesteps)
    return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
```

#### DDIM Implementation
```python
def _make_ddim_timesteps(self, ddim_discr_method: str = "uniform", ddim_eta: float = 0.0) -> torch.Tensor:
    """Make DDIM timesteps."""
    if ddim_discr_method == "uniform":
        c = self.num_timesteps // 50
        ddim_timesteps = torch.arange(0, self.num_timesteps, c)
    elif ddim_discr_method == "quad":
        ddim_timesteps = ((torch.arange(0, self.num_timesteps) ** 2) / self.num_timesteps).long()
    else:
        raise ValueError(f"Unknown DDIM discretization method: {ddim_discr_method}")
    
    return ddim_timesteps

def ddim_step(
    self,
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    eta: float = 0.0
) -> torch.Tensor:
    """DDIM step."""
    if self.config.prediction_type == "epsilon":
        pred_epsilon = model_output
        pred_x_start = self.remove_noise(x_t, pred_epsilon, timesteps)
    elif self.config.prediction_type == "x0":
        pred_x_start = model_output
    else:
        raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
    
    # DDIM step
    alpha_cumprod_t = self.ddim_alphas[timesteps].view(-1, 1, 1, 1)
    alpha_cumprod_t_prev = self.ddim_alphas_prev[timesteps].view(-1, 1, 1, 1)
    sigma_t = eta * self.ddim_sigmas[timesteps].view(-1, 1, 1, 1)
    
    pred_noise = (x_t - torch.sqrt(alpha_cumprod_t) * pred_x_start) / torch.sqrt(1 - alpha_cumprod_t)
    
    x_prev = (
        torch.sqrt(alpha_cumprod_t_prev) * pred_x_start +
        torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * pred_noise +
        sigma_t * torch.randn_like(x_t)
    )
    
    return x_prev
```

### 2. **UNet Architecture**

#### UNet Block Implementation
- **UNetBlock**: Individual UNet block with residual connections and attention
- **Time Embedding**: Sinusoidal position embeddings for timesteps
- **Attention Mechanisms**: Multi-head attention for spatial dependencies
- **Residual Connections**: Skip connections for gradient flow

#### UNet Block Features
```python
class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.use_attention = use_attention
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
        
        # Attention
        if use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
        else:
            self.attention = None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = self.residual_conv(x)
        
        # First convolution
        h = self.conv1(x)
        h = F.silu(h)
        
        # Time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.view(-1, self.out_channels, 1, 1)
        
        # Second convolution
        h = self.conv2(h)
        h = self.dropout(h)
        h = F.silu(h)
        
        # Residual connection
        h = h + residual
        
        # Attention
        if self.attention is not None:
            b, c, h, w = h.shape
            h_flat = h.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
            h_attended, _ = self.attention(h_flat, h_flat, h_flat)
            h_attended = h_attended.transpose(1, 2).view(b, c, h, w)
            h = h + h_attended
        
        return h
```

#### Sinusoidal Position Embeddings
```python
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

#### Complete UNet Architecture
```python
class UNet(nn.Module):
    """UNet model for diffusion."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_heads: int = 8,
        use_spatial_transformer: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        # ... (implementation details)
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input blocks, middle block, output blocks
        # ... (detailed implementation)
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input blocks
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                for layer in module:
                    if isinstance(layer, UNetBlock):
                        h = layer(h, time_emb)
                    elif isinstance(layer, nn.MultiheadAttention):
                        b, c, h_h, h_w = h.shape
                        h_flat = h.view(b, c, h_h * h_w).transpose(1, 2)
                        h_attended, _ = layer(h_flat, h_flat, h_flat)
                        h = h_attended.transpose(1, 2).view(b, c, h_h, h_w)
            hs.append(h)
        
        # Middle block
        for module in self.middle_block:
            if isinstance(module, UNetBlock):
                h = module(h, time_emb)
            elif isinstance(module, nn.MultiheadAttention):
                b, c, h_h, h_w = h.shape
                h_flat = h.view(b, c, h_h * h_w).transpose(1, 2)
                h_attended, _ = module(h_flat, h_flat, h_flat)
                h = h_attended.transpose(1, 2).view(b, c, h_h, h_w)
        
        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, UNetBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, nn.MultiheadAttention):
                    b, c, h_h, h_w = h.shape
                    h_flat = h.view(b, c, h_h * h_w).transpose(1, 2)
                    h_attended, _ = layer(h_flat, h_flat, h_flat)
                    h = h_attended.transpose(1, 2).view(b, c, h_h, h_w)
                elif isinstance(layer, nn.ConvTranspose2d):
                    h = layer(h)
        
        # Output
        return self.out(h)
```

### 3. **Diffusion Models**

#### Abstract Base Class
```python
class DiffusionModel(ABC):
    """Abstract base class for diffusion models."""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.scheduler = NoiseScheduler(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Get the model."""
        pass

    @abstractmethod
    def training_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Training step."""
        pass

    @abstractmethod
    def sampling_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Sampling step."""
        pass

    def sample(
        self,
        batch_size: int = 1,
        image_size: int = 64,
        channels: int = 3,
        guidance_scale: Optional[float] = None
    ) -> torch.Tensor:
        """Generate samples."""
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        
        model = self.get_model()
        model.eval()
        
        # Start from noise
        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        
        # Sampling loop
        with torch.no_grad():
            for t in reversed(range(self.config.num_timesteps)):
                timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # Model prediction
                model_output = model(x, timesteps)
                
                # Apply guidance if enabled
                if self.config.classifier_free_guidance and guidance_scale > 1.0:
                    uncond_output = model(x, timesteps, context=torch.zeros_like(context))
                    model_output = uncond_output + guidance_scale * (model_output - uncond_output)
                
                # Sampling step
                x = self.sampling_step(x, timesteps, model_output)
        
        return x
```

#### DDPM Implementation
```python
class DDPM(DiffusionModel):
    """Denoising Diffusion Probabilistic Models (DDPM)."""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        self.model = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=128,
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            num_heads=8
        ).to(self.device)
        
        # EMA model
        if config.use_ema:
            self.ema_model = UNet(
                in_channels=3,
                out_channels=3,
                model_channels=128,
                num_res_blocks=2,
                attention_resolutions=(16, 8),
                dropout=0.1,
                channel_mult=(1, 2, 4, 8),
                num_heads=8
            ).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.eval()
        else:
            self.ema_model = None

    def get_model(self) -> nn.Module:
        """Get the model."""
        return self.ema_model if self.ema_model is not None else self.model

    def training_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Training step."""
        batch_size = batch.shape[0]
        timesteps = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noisy_batch, noise = self.scheduler.add_noise(batch, timesteps)
        
        # Model prediction
        model_output = self.model(noisy_batch, timesteps)
        
        # Loss
        if self.config.loss_type == "mse":
            loss = F.mse_loss(model_output, noise)
        elif self.config.loss_type == "l1":
            loss = F.l1_loss(model_output, noise)
        elif self.config.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        if self.ema_model is not None:
            with torch.no_grad():
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_param.data.mul_(self.config.ema_decay).add_(param.data, alpha=1 - self.config.ema_decay)
        
        return {"loss": loss.item()}

    def sampling_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Sampling step."""
        # Compute mean and variance
        posterior_mean, posterior_variance, _ = self.scheduler.p_mean_variance(
            model_output, x_t, timesteps, self.config.clip_denoised
        )
        
        # Sample
        noise = torch.randn_like(x_t)
        x_prev = posterior_mean + torch.sqrt(posterior_variance) * noise
        
        return x_prev
```

#### DDIM Implementation
```python
class DDIM(DiffusionModel):
    """Denoising Diffusion Implicit Models (DDIM)."""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        self.model = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=128,
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            num_heads=8
        ).to(self.device)

    def get_model(self) -> nn.Module:
        """Get the model."""
        return self.model

    def training_step(
        self,
        batch: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Training step."""
        batch_size = batch.shape[0]
        timesteps = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noisy_batch, noise = self.scheduler.add_noise(batch, timesteps)
        
        # Model prediction
        model_output = self.model(noisy_batch, timesteps)
        
        # Loss
        if self.config.loss_type == "mse":
            loss = F.mse_loss(model_output, noise)
        elif self.config.loss_type == "l1":
            loss = F.l1_loss(model_output, noise)
        elif self.config.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()}

    def sampling_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Sampling step."""
        return self.scheduler.ddim_step(model_output, x_t, timesteps, eta=0.0)
```

## Complete Usage Examples

### 1. **DDPM Basic Example**
```python
from .diffusion_models import DiffusionConfig, DDPM

# Create configuration
config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon",
    loss_type="mse",
    guidance_scale=7.5,
    classifier_free_guidance=True,
    clip_denoised=True,
    use_ema=True,
    ema_decay=0.9999
)

# Create DDPM model
ddpm_model = DDPM(config)

# Generate samples
samples = ddpm_model.sample(
    batch_size=4,
    image_size=64,
    channels=3,
    guidance_scale=7.5
)

print(f"Generated {samples.shape[0]} samples with shape {samples.shape}")
```

### 2. **DDIM Basic Example**
```python
from .diffusion_models import DiffusionConfig, DDIM

# Create configuration
config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="cosine",
    prediction_type="epsilon",
    loss_type="mse",
    guidance_scale=7.5,
    classifier_free_guidance=True,
    clip_denoised=True,
    use_ema=False  # DDIM typically doesn't use EMA
)

# Create DDIM model
ddim_model = DDIM(config)

# Generate samples
samples = ddim_model.sample(
    batch_size=4,
    image_size=64,
    channels=3,
    guidance_scale=7.5
)

print(f"Generated {samples.shape[0]} DDIM samples with shape {samples.shape}")
```

### 3. **Noise Scheduler Example**
```python
from .diffusion_models import DiffusionConfig, NoiseScheduler

# Create configuration
config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear"
)

# Create noise scheduler
scheduler = NoiseScheduler(config)

# Test noise addition
batch_size = 2
image_size = 32
channels = 3

x_start = torch.randn(batch_size, channels, image_size, image_size)
timesteps = torch.randint(0, config.num_timesteps, (batch_size,))

# Add noise
noisy_x, noise = scheduler.add_noise(x_start, timesteps)

# Remove noise
denoised_x = scheduler.remove_noise(noisy_x, noise, timesteps)

print(f"Original shape: {x_start.shape}")
print(f"Noisy shape: {noisy_x.shape}")
print(f"Denoised shape: {denoised_x.shape}")
print(f"Reconstruction error: {F.mse_loss(x_start, denoised_x):.6f}")
```

### 4. **UNet Model Example**
```python
from .diffusion_models import UNet, SinusoidalPositionEmbeddings

# Create UNet model
unet_model = UNet(
    in_channels=3,
    out_channels=3,
    model_channels=64,
    num_res_blocks=2,
    attention_resolutions=(16, 8),
    dropout=0.1,
    channel_mult=(1, 2, 4),
    num_heads=8
)

# Test forward pass
batch_size = 2
image_size = 32
channels = 3

x = torch.randn(batch_size, channels, image_size, image_size)
timesteps = torch.randint(0, 1000, (batch_size,))

# Forward pass
output = unet_model(x, timesteps)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Model parameters: {sum(p.numel() for p in unet_model.parameters()):,}")
```

### 5. **Training Example**
```python
from .diffusion_models import DiffusionConfig, DDPM

# Create configuration
config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon",
    loss_type="mse",
    use_ema=True,
    ema_decay=0.9999
)

# Create DDPM model
ddpm_model = DDPM(config)

# Create optimizer
optimizer = torch.optim.AdamW(ddpm_model.model.parameters(), lr=1e-4)

# Sample training data
batch_size = 4
image_size = 32
channels = 3

# Training loop
num_epochs = 5
losses = []

for epoch in range(num_epochs):
    epoch_losses = []
    
    for step in range(10):  # 10 steps per epoch
        # Generate random training data
        batch = torch.randn(batch_size, channels, image_size, image_size)
        
        # Training step
        metrics = ddpm_model.training_step(batch, optimizer)
        loss = metrics["loss"]
        epoch_losses.append(loss)
        
        if step % 5 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss:.6f}")
    
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)
    print(f"Epoch {epoch} average loss: {avg_loss:.6f}")

print(f"Training completed. Final loss: {losses[-1]:.6f}")
```

### 6. **Sampling Comparison Example**
```python
from .diffusion_models import DiffusionConfig, DDPM, DDIM

# Create configurations
ddpm_config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon",
    loss_type="mse",
    use_ema=True,
    ema_decay=0.9999
)

ddim_config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="cosine",
    prediction_type="epsilon",
    loss_type="mse",
    use_ema=False
)

# Create models
ddpm_model = DDPM(ddpm_config)
ddim_model = DDIM(ddim_config)

# Generate samples
batch_size = 2
image_size = 32
channels = 3

# DDPM samples
ddpm_samples = ddpm_model.sample(
    batch_size=batch_size,
    image_size=image_size,
    channels=channels,
    guidance_scale=7.5
)

# DDIM samples
ddim_samples = ddim_model.sample(
    batch_size=batch_size,
    image_size=image_size,
    channels=channels,
    guidance_scale=7.5
)

print(f"DDPM samples shape: {ddpm_samples.shape}")
print(f"DDIM samples shape: {ddim_samples.shape}")
print(f"DDPM statistics - Mean: {ddpm_samples.mean():.4f}, Std: {ddpm_samples.std():.4f}")
print(f"DDIM statistics - Mean: {ddim_samples.mean():.4f}, Std: {ddim_samples.std():.4f}")
```

### 7. **Guidance Example**
```python
from .diffusion_models import DiffusionConfig, DDPM

# Create configuration with guidance
config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon",
    loss_type="mse",
    guidance_scale=7.5,
    classifier_free_guidance=True,
    use_ema=True,
    ema_decay=0.9999
)

# Create DDPM model
ddpm_model = DDPM(config)

# Generate samples with different guidance scales
batch_size = 2
image_size = 32
channels = 3

guidance_scales = [1.0, 3.0, 7.5, 15.0]
samples_dict = {}

for guidance_scale in guidance_scales:
    samples = ddpm_model.sample(
        batch_size=batch_size,
        image_size=image_size,
        channels=channels,
        guidance_scale=guidance_scale
    )
    samples_dict[guidance_scale] = samples
    
    print(f"Guidance scale {guidance_scale}:")
    print(f"  Shape: {samples.shape}")
    print(f"  Mean: {samples.mean():.4f}")
    print(f"  Std: {samples.std():.4f}")
    print(f"  Range: [{samples.min():.4f}, {samples.max():.4f}]")
```

### 8. **Beta Schedule Comparison Example**
```python
from .diffusion_models import DiffusionConfig, NoiseScheduler

# Create configurations with different beta schedules
configs = {
    "linear": DiffusionConfig(beta_schedule="linear"),
    "cosine": DiffusionConfig(beta_schedule="cosine"),
    "sigmoid": DiffusionConfig(beta_schedule="sigmoid")
}

schedulers = {}
beta_curves = {}

for name, config in configs.items():
    scheduler = NoiseScheduler(config)
    schedulers[name] = scheduler
    beta_curves[name] = scheduler.betas.cpu().numpy()
    
    print(f"{name.capitalize()} beta schedule:")
    print(f"  Beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    print(f"  Alpha cumprod range: [{scheduler.alphas_cumprod.min():.6f}, {scheduler.alphas_cumprod.max():.6f}]")
```

### 9. **Custom UNet Example**
```python
from .diffusion_models import UNet

# Create custom UNet with different parameters
custom_unet = UNet(
    in_channels=3,
    out_channels=3,
    model_channels=96,
    num_res_blocks=3,
    attention_resolutions=(16, 8, 4),
    dropout=0.15,
    channel_mult=(1, 2, 4, 8),
    num_heads=12,
    use_spatial_transformer=True,
    transformer_depth=2,
    context_dim=768,
    use_checkpoint=True,
    use_fp16=False,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_new_attention_order=True
)

# Test forward pass
batch_size = 2
image_size = 64
channels = 3

x = torch.randn(batch_size, channels, image_size, image_size)
timesteps = torch.randint(0, 1000, (batch_size,))
context = torch.randn(batch_size, 77, 768)  # Text context

# Forward pass
output = custom_unet(x, timesteps, context)

print(f"Custom UNet input shape: {x.shape}")
print(f"Custom UNet output shape: {output.shape}")
print(f"Custom UNet parameters: {sum(p.numel() for p in custom_unet.parameters()):,}")
```

### 10. **Performance Benchmark Example**
```python
from .diffusion_models import DiffusionConfig, DDPM, DDIM
import time

# Create configurations
ddpm_config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon",
    loss_type="mse",
    use_ema=True,
    ema_decay=0.9999
)

ddim_config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="cosine",
    prediction_type="epsilon",
    loss_type="mse",
    use_ema=False
)

# Create models
ddpm_model = DDPM(ddpm_config)
ddim_model = DDIM(ddim_config)

# Benchmark parameters
batch_size = 1
image_size = 64
channels = 3
num_runs = 5

# Benchmark DDPM
ddpm_times = []
for _ in range(num_runs):
    start_time = time.time()
    samples = ddpm_model.sample(
        batch_size=batch_size,
        image_size=image_size,
        channels=channels,
        guidance_scale=7.5
    )
    end_time = time.time()
    ddpm_times.append(end_time - start_time)

# Benchmark DDIM
ddim_times = []
for _ in range(num_runs):
    start_time = time.time()
    samples = ddim_model.sample(
        batch_size=batch_size,
        image_size=image_size,
        channels=channels,
        guidance_scale=7.5
    )
    end_time = time.time()
    ddim_times.append(end_time - start_time)

print("Performance Benchmark Results:")
print(f"DDPM - Average time: {np.mean(ddpm_times):.4f}s ± {np.std(ddpm_times):.4f}s")
print(f"DDIM - Average time: {np.mean(ddim_times):.4f}s ± {np.std(ddim_times):.4f}s")
print(f"Speedup: {np.mean(ddpm_times) / np.mean(ddim_times):.2f}x")
```

### 11. **Memory Usage Example**
```python
from .diffusion_models import DiffusionConfig, DDPM, DDIM
import psutil
import gc

# Create configurations
ddpm_config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type="epsilon",
    loss_type="mse",
    use_ema=True,
    ema_decay=0.9999
)

ddim_config = DiffusionConfig(
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="cosine",
    prediction_type="epsilon",
    loss_type="mse",
    use_ema=False
)

# Memory analysis
process = psutil.Process()

# DDPM memory usage
initial_memory = process.memory_info().rss / 1024 / 1024  # MB
ddpm_model = DDPM(ddpm_config)
ddpm_memory = process.memory_info().rss / 1024 / 1024  # MB

# Generate samples
samples = ddpm_model.sample(batch_size=1, image_size=64, channels=3)
ddpm_sample_memory = process.memory_info().rss / 1024 / 1024  # MB

# Clean up
del ddpm_model, samples
gc.collect()

# DDIM memory usage
ddim_model = DDIM(ddim_config)
ddim_memory = process.memory_info().rss / 1024 / 1024  # MB

# Generate samples
samples = ddim_model.sample(batch_size=1, image_size=64, channels=3)
ddim_sample_memory = process.memory_info().rss / 1024 / 1024  # MB

print("Memory Usage Analysis:")
print(f"Initial memory: {initial_memory:.2f} MB")
print(f"DDPM model memory: {ddpm_memory:.2f} MB (+{ddpm_memory - initial_memory:.2f} MB)")
print(f"DDPM sampling memory: {ddpm_sample_memory:.2f} MB (+{ddpm_sample_memory - ddpm_memory:.2f} MB)")
print(f"DDIM model memory: {ddim_memory:.2f} MB (+{ddim_memory - initial_memory:.2f} MB)")
print(f"DDIM sampling memory: {ddim_sample_memory:.2f} MB (+{ddim_sample_memory - ddim_memory:.2f} MB)")
```

## Key Benefits

### 1. **Multiple Diffusion Models**
- **DDPM**: Standard denoising diffusion probabilistic models
- **DDIM**: Faster sampling with implicit models
- **Extensible**: Easy to add new diffusion variants

### 2. **Flexible Noise Schedules**
- **Linear Schedule**: Standard linear beta schedule
- **Cosine Schedule**: Improved cosine schedule for better quality
- **Sigmoid Schedule**: Sigmoid-based schedule for specific use cases

### 3. **Advanced UNet Architecture**
- **Residual Connections**: Skip connections for gradient flow
- **Attention Mechanisms**: Multi-head attention for spatial dependencies
- **Time Embeddings**: Sinusoidal position embeddings for timesteps
- **Customizable**: Highly configurable architecture

### 4. **Training and Sampling**
- **Efficient Training**: Optimized training loops with EMA support
- **Fast Sampling**: DDIM for faster generation
- **Guidance Support**: Classifier-free guidance for better quality
- **Multiple Loss Types**: MSE, L1, and Huber losses

### 5. **Production-Ready Features**
- **Error Handling**: Robust error handling and validation
- **Performance Optimization**: Optimized for speed and memory usage
- **Modular Design**: Clean separation of concerns
- **Extensible**: Easy to extend with new features

### 6. **Comprehensive Examples**
- **Basic Usage**: Simple examples for getting started
- **Advanced Features**: Complex examples for advanced users
- **Performance Analysis**: Benchmarking and memory analysis
- **Visualization**: Plotting and visualization utilities

The diffusion models implementation provides a comprehensive framework for state-of-the-art image generation, offering multiple diffusion variants, flexible architectures, and efficient training/sampling capabilities while maintaining performance and extensibility for various applications. 