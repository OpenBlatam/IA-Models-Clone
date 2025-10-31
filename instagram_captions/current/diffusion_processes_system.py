"""
Diffusion Processes System
Comprehensive implementation of forward and reverse diffusion processes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiffusionConfig:
    """Configuration for diffusion processes"""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    image_size: int = 64
    in_channels: int = 3
    out_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.1
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_heads: int = 8
    use_scale_shift_norm: bool = True
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Mixed precision training settings
    use_mixed_precision: bool = True
    use_grad_scaler: bool = True
    # Multi-GPU settings
    use_data_parallel: bool = False
    use_distributed: bool = False
    rank: int = 0
    world_size: int = 1
    # Training optimization settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    clip_grad: bool = True

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """Residual block with optional up/down sampling"""
    
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0, 
                 up: bool = False, down: bool = False, use_scale_shift_norm: bool = True):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * channels if use_scale_shift_norm else channels)
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        if up:
            self.h_upd = nn.Upsample(scale_factor=2, mode='nearest')
        elif down:
            self.h_upd = nn.AvgPool2d(2)
        else:
            self.h_upd = nn.Identity()
            
        if up or down:
            self.h_skip = nn.Conv2d(channels, channels, 1)
        else:
            self.h_skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out.unsqueeze(-1).unsqueeze(-1)
            h = self.out_layers(h)
            
        return self.h_upd(h) + self.h_skip(x)

class AttentionBlock(nn.Module):
    """Multi-head self-attention block"""
    
    def __init__(self, channels: int, num_heads: int = 1, num_head_channels: int = -1):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.num_head_channels = num_head_channels
        head_dim = channels // self.num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        scale = 1 / math.sqrt(math.sqrt(C))
        
        attn = torch.einsum("bchw,bcij->bhwij", q * scale, k * scale)
        attn = torch.softmax(attn, dim=-1)
        
        h = torch.einsum("bhwij,bcij->bchw", attn, v)
        h = self.proj(h)
        
        return x + h

class UNetModel(nn.Module):
    """UNet model for diffusion processes"""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.in_channels = config.in_channels
        self.model_channels = config.model_channels
        
        # Time embedding
        time_embed_dim = config.model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(config.model_channels),
            nn.Linear(config.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(config.in_channels, config.model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling
        input_block_chans = [config.model_channels]
        ch = config.model_channels
        ds = 1
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, config.dropout, 
                            down=ds > 1, use_scale_shift_norm=config.use_scale_shift_norm)
                ]
                ch = mult * config.model_channels
                if ds in config.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=config.num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            if level != len(config.channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([ResBlock(ch, time_embed_dim, config.dropout, down=True)]))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock(ch, time_embed_dim, config.dropout, use_scale_shift_norm=config.use_scale_shift_norm),
            AttentionBlock(ch, num_heads=config.num_heads),
            ResBlock(ch, time_embed_dim, config.dropout, use_scale_shift_norm=config.use_scale_shift_norm)
        ])
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(config.channel_mult))[::-1]:
            for i in range(config.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, config.dropout, 
                            up=ds > 1, use_scale_shift_norm=config.use_scale_shift_norm)
                ]
                ch = mult * config.model_channels
                if ds in config.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=config.num_heads))
                if level and i == config.num_res_blocks:
                    layers.append(ResBlock(ch, time_embed_dim, config.dropout, up=True))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, config.out_channels, 3, padding=1)
        )
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Time embedding
        emb = self.time_embed(timesteps)
        
        # Input blocks
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            hs.append(h)
        
        # Middle block
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, emb)
            else:
                h = module(h)
        
        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
        
        return self.out(h)

class DiffusionProcesses:
    """Implements forward and reverse diffusion processes with mixed precision training"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = config.device
        
        # Initialize beta schedule
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Initialize UNet model
        self.model = UNetModel(config).to(self.device)
        
        # Multi-GPU setup
        if self.config.use_data_parallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        elif self.config.use_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.config.rank],
                output_device=self.config.rank
            )
            logger.info(f"Using DistributedDataParallel on rank {self.config.rank}")
        
        # Mixed precision training setup
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler() if self.config.use_grad_scaler else None
            logger.info("Mixed precision training enabled")
        
        logger.info(f"Initialized DiffusionProcesses with {config.num_timesteps} timesteps")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule for diffusion process"""
        if self.config.beta_end <= self.config.beta_start:
            raise ValueError("beta_end must be greater than beta_start")
        
        # Linear schedule
        betas = torch.linspace(
            self.config.beta_start, 
            self.config.beta_end, 
            self.config.num_timesteps,
            device=self.device
        )
        
        return betas
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0)
        Adds noise to x_0 according to timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get alpha values for timestep t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # q(x_t | x_0) = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training the diffusion model with mixed precision
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy, target = self.q_sample(x_start, t, noise)
        
        # Predict noise with mixed precision
        if self.config.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                predicted_noise = self.model(x_noisy, t)
                loss = F.mse_loss(predicted_noise, target, reduction='none')
                loss = loss.mean(dim=[1, 2, 3])
        else:
            predicted_noise = self.model(x_noisy, t)
            loss = F.mse_loss(predicted_noise, target, reduction='none')
            loss = loss.mean(dim=[1, 2, 3])
        
        return {
            'loss': loss.mean(),
            'pred': predicted_noise,
            'target': target
        }
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        Single denoising step
        """
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t_index]
        
        # Predict noise
        model_output = self.model(x, t)
        
        # DDPM sampling
        pred_original = (x - sqrt_one_minus_alphas_cumprod_t * model_output) * sqrt_recip_alphas_cumprod_t
        
        # Add noise for next step
        if t_index > 0:
            noise = torch.randn_like(x)
            x_prev = pred_original + torch.sqrt(betas_t) * noise
        else:
            x_prev = pred_original
            
        return x_prev
    
    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int, ...], num_timesteps: Optional[int] = None) -> torch.Tensor:
        """
        Complete reverse diffusion process
        Generates samples from noise
        """
        if num_timesteps is None:
            num_timesteps = self.config.num_timesteps
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Reverse diffusion loop
        for i in tqdm(reversed(range(0, num_timesteps)), desc="Sampling", total=num_timesteps):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        
        return x
    
    @torch.no_grad()
    def sample(self, batch_size: int = 1, num_timesteps: Optional[int] = None) -> torch.Tensor:
        """
        Generate samples using the trained model
        """
        shape = (batch_size, self.config.in_channels, self.config.image_size, self.config.image_size)
        return self.p_sample_loop(shape, num_timesteps)
    
    def train_step(self, x_start: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Single training step with mixed precision and gradient accumulation
        """
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # Compute loss
        loss_dict = self.p_losses(x_start, t)
        loss = loss_dict['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with mixed precision
        if self.config.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.clip_grad:
            if self.config.use_mixed_precision and self.scaler is not None:
                self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step (only on accumulation steps)
        if (self.config.gradient_accumulation_steps == 1 or 
            (hasattr(self, '_step_count') and (self._step_count + 1) % self.config.gradient_accumulation_steps == 0)):
            
            if self.config.use_mixed_precision and self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Update step count
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            't_mean': t.float().mean().item(),
            'grad_norm': self._get_grad_norm()
        }
    
    def _get_grad_norm(self) -> float:
        """Get the gradient norm for monitoring"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'betas': self.betas,
            'alphas_cumprod': self.alphas_cumprod
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

class DiffusionTrainer:
    """Training wrapper for diffusion processes with mixed precision and optimization"""
    
    def __init__(self, diffusion: DiffusionProcesses, config: DiffusionConfig):
        self.diffusion = diffusion
        self.config = config
        self.optimizer = torch.optim.AdamW(
            diffusion.model.parameters(), 
            lr=1e-4, 
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000
        )
        
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch with mixed precision"""
        self.diffusion.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(self.config.device)
            
            # Training step
            step_metrics = self.diffusion.train_step(x, self.optimizer)
            total_loss += step_metrics['loss']
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def train(self, dataloader: torch.utils.data.DataLoader, num_epochs: int, 
              save_path: Optional[str] = None) -> List[Dict[str, float]]:
        """Train the diffusion model"""
        metrics_history = []
        
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(dataloader)
            metrics_history.append(epoch_metrics)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_metrics['loss']:.6f} - LR: {epoch_metrics['lr']:.2e}")
            
            # Save model periodically
            if save_path and (epoch + 1) % 10 == 0:
                self.diffusion.save_model(f"{save_path}_epoch_{epoch+1}.pt")
        
        # Save final model
        if save_path:
            self.diffusion.save_model(f"{save_path}_final.pt")
        
        return metrics_history

def create_diffusion_config(**kwargs) -> DiffusionConfig:
    """Create diffusion configuration with default values"""
    return DiffusionConfig(**kwargs)

def main():
    """Main function to demonstrate diffusion processes with mixed precision"""
    # Configuration with mixed precision and optimization
    config = create_diffusion_config(
        num_timesteps=100,
        image_size=32,
        in_channels=3,
        model_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        use_mixed_precision=True,
        use_grad_scaler=True,
        gradient_accumulation_steps=2,
        clip_grad=True,
        max_grad_norm=1.0
    )
    
    # Initialize diffusion processes
    diffusion = DiffusionProcesses(config)
    
    # Create dummy data
    batch_size = 4
    x_start = torch.randn(batch_size, config.in_channels, config.image_size, config.image_size, device=config.device)
    
    # Test forward diffusion
    t = torch.randint(0, config.num_timesteps, (batch_size,), device=config.device)
    x_t, noise = diffusion.q_sample(x_start, t)
    logger.info(f"Forward diffusion: x_start shape {x_start.shape} -> x_t shape {x_t.shape}")
    
    # Test reverse diffusion (single step)
    x_prev = diffusion.p_sample(x_t, t, t[0].item())
    logger.info(f"Reverse diffusion step: x_t shape {x_t.shape} -> x_prev shape {x_prev.shape}")
    
    # Test complete sampling
    samples = diffusion.sample(batch_size=2, num_timesteps=50)
    logger.info(f"Generated samples shape: {samples.shape}")
    
    # Test training step with mixed precision
    trainer = DiffusionTrainer(diffusion, config)
    step_metrics = trainer.train_step(x_start, trainer.optimizer)
    logger.info(f"Training step metrics: {step_metrics}")
    
    logger.info("Diffusion processes system with mixed precision test completed successfully!")

if __name__ == "__main__":
    main()



