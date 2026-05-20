from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from diffusers import (
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image
import json
import wandb
from tqdm import tqdm
import gc
                import bitsandbytes as bnb
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Diffusion Model Training - Production Training Workflows
=======================================================

Advanced training workflows for diffusion models using Diffusers library.
Features: Custom training loops, loss functions, optimization, scheduling,
gradient accumulation, mixed precision, and production monitoring.
"""

    UNet2DConditionModel, AutoencoderKL, DDIMScheduler,
    DDPMScheduler, StableDiffusionPipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for diffusion model training."""
    # Model configuration
    model_name: str = "runwayml/stable-diffusion-v1-5"
    pretrained_model_name_or_path: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    subfolder: Optional[str] = None
    
    # Training configuration
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    
    # Data configuration
    train_data_dir: str = "data/train"
    validation_data_dir: Optional[str] = None
    train_batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 4
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    
    # Diffusion configuration
    noise_offset: float = 0.0
    input_perturbation: float = 0.0
    prediction_type: str = "epsilon"
    snr_gamma: Optional[float] = None
    scale_factor: float = 0.18215
    scale_by_std: bool = False
    
    # Optimization configuration
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Logging and saving
    output_dir: str = "outputs"
    logging_dir: str = "logs"
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    save_only_model: bool = False
    validation_steps: int = 100
    validation_prompt: Optional[str] = None
    num_validation_images: int = 4
    validation_guidance_scale: float = 7.5
    validation_num_inference_steps: int = 50
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = None
    dataloader_pin_memory: bool = True
    
    # Advanced configuration
    enable_xformers_memory_efficient_attention: bool = True
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_compilation: bool = False
    enable_peft: bool = False
    peft_config: Optional[Dict[str, Any]] = None


class DiffusionDataset(Dataset):
    """Dataset for diffusion model training."""
    
    def __init__(self, data_dir: str, tokenizer: CLIPTokenizer, size: int = 512,
                 center_crop: bool = False, random_flip: bool = False):
        
    """__init__ function."""
self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        # Find all image files
        self.image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
            self.image_files.extend(self.data_dir.glob(ext))
            self.image_files.extend(self.data_dir.glob(ext.upper()))
        
        logger.info(f"Found {len(self.image_files)} images in {data_dir}")
    
    def __len__(self) -> Any:
        return len(self.image_files)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        image_path = self.image_files[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Center crop if requested
        if self.center_crop:
            crop = min(image.size)
            image = image.crop((
                (image.size[0] - crop) // 2,
                (image.size[1] - crop) // 2,
                (image.size[0] + crop) // 2,
                (image.size[1] + crop) // 2
            ))
        
        # Resize
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Random flip
        if self.random_flip and torch.rand(1).item() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Tokenize caption (using filename as caption)
        caption = image_path.stem
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(),
            "attention_mask": tokens.attention_mask.squeeze()
        }


class DiffusionTrainer:
    """Advanced trainer for diffusion models."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        
        # Set seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        # Initialize components
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Initialize wandb if available
        try:
            wandb.init(project="diffusion-training", config=config.__dict__)
            self.use_wandb = True
        except:
            self.use_wandb = False
            logger.warning("Wandb not available, logging disabled")
    
    async def setup_models(self) -> Any:
        """Setup all model components."""
        logger.info("Setting up models...")
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_name,
            subfolder="tokenizer",
            revision=self.config.revision,
            variant=self.config.variant
        )
        
        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.model_name,
            subfolder="text_encoder",
            revision=self.config.revision,
            variant=self.config.variant
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_name,
            subfolder="vae",
            revision=self.config.revision,
            variant=self.config.variant
        )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.model_name,
            subfolder="unet",
            revision=self.config.revision,
            variant=self.config.variant
        )
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.model_name,
            subfolder="scheduler"
        )
        
        # Move models to device
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)
        
        # Set to eval mode for VAE and text encoder
        self.vae.eval()
        self.text_encoder.eval()
        
        # Freeze VAE and text encoder
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        logger.info("Models setup complete")
    
    def setup_optimization(self) -> Any:
        """Setup optimizer and learning rate scheduler."""
        logger.info("Setting up optimization...")
        
        # Setup optimizer
        if self.config.use_8bit_adam:
            try:
                self.optimizer = bnb.optim.AdamW8bit(
                    self.unet.parameters(),
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                    eps=self.config.adam_epsilon
                )
            except ImportError:
                logger.warning("8-bit Adam not available, using regular AdamW")
                self.optimizer = optim.AdamW(
                    self.unet.parameters(),
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                    eps=self.config.adam_epsilon
                )
        else:
            self.optimizer = optim.AdamW(
                self.unet.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.adam_epsilon
            )
        
        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.max_train_steps,
            num_cycles=self.config.lr_num_cycles,
            power=self.config.lr_power
        )
        
        # Setup mixed precision
        if self.config.mixed_precision == "fp16":
            self.scaler = GradScaler()
        
        # Setup gradient checkpointing
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        
        logger.info("Optimization setup complete")
    
    def setup_dataset(self) -> Tuple[DiffusionDataset, Optional[DiffusionDataset]]:
        """Setup training and validation datasets."""
        logger.info("Setting up datasets...")
        
        # Training dataset
        train_dataset = DiffusionDataset(
            data_dir=self.config.train_data_dir,
            tokenizer=self.tokenizer,
            size=self.config.resolution,
            center_crop=self.config.center_crop,
            random_flip=self.config.random_flip
        )
        
        # Validation dataset
        val_dataset = None
        if self.config.validation_data_dir:
            val_dataset = DiffusionDataset(
                data_dir=self.config.validation_data_dir,
                tokenizer=self.tokenizer,
                size=self.config.resolution,
                center_crop=self.config.center_crop,
                random_flip=False  # No random flip for validation
            )
        
        logger.info(f"Training dataset: {len(train_dataset)} samples")
        if val_dataset:
            logger.info(f"Validation dataset: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def compute_loss(self, model_pred: torch.Tensor, target: torch.Tensor, 
                    timesteps: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        if self.config.prediction_type == "epsilon":
            target = target
        elif self.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        if self.config.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute SNR
            snr = self.noise_scheduler.step(timesteps, model_pred, target).snr
            mse_loss_weights = torch.where(
                snr < self.config.snr_gamma,
                torch.ones_like(snr),
                torch.zeros_like(snr)
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        
        return loss
    
    async def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.unet.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get latents
        with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * self.config.scale_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        if self.config.noise_offset:
            noise += self.config.noise_offset * torch.randn_like(latents)
        
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,))
        timesteps = timesteps.long().to(self.device)
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Add input perturbation
        if self.config.input_perturbation:
            noisy_latents = noisy_latents + self.config.input_perturbation * torch.randn_like(noisy_latents)
        
        # Get text embeddings
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
        
        # Predict noise
        if self.config.mixed_precision == "fp16":
            with autocast():
                noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        else:
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Compute loss
        loss = self.compute_loss(noise_pred, noise, timesteps)
        
        # Backward pass
        if self.config.mixed_precision == "fp16":
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.config.mixed_precision == "fp16":
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        if self.config.mixed_precision == "fp16":
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        
        return loss.item()
    
    async def validation_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single validation step."""
        self.unet.eval()
        
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get latents
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * self.config.scale_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,))
            timesteps = timesteps.long().to(self.device)
            
            # Add noise
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            
            # Predict noise
            if self.config.mixed_precision == "fp16":
                with autocast():
                    noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            else:
                noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Compute loss
            loss = self.compute_loss(noise_pred, noise, timesteps)
            
            return loss.item()
    
    async def generate_validation_images(self, pipeline: StableDiffusionPipeline) -> List[Image.Image]:
        """Generate validation images."""
        if not self.config.validation_prompt:
            return []
        
        images = []
        for i in range(self.config.num_validation_images):
            result = pipeline(
                prompt=self.config.validation_prompt,
                guidance_scale=self.config.validation_guidance_scale,
                num_inference_steps=self.config.validation_num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(i)
            )
            images.extend(result.images)
        
        return images
    
    async def save_checkpoint(self, step: int, loss: float):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save UNet
        unet_path = output_dir / f"unet_step_{step}.safetensors"
        self.unet.save_pretrained(unet_path)
        
        # Save optimizer state
        optimizer_path = output_dir / f"optimizer_step_{step}.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        scheduler_path = output_dir / f"scheduler_step_{step}.pt"
        torch.save(self.lr_scheduler.state_dict(), scheduler_path)
        
        # Save training state
        state = {
            "step": step,
            "epoch": self.epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "global_step": self.global_step
        }
        state_path = output_dir / f"training_state_step_{step}.json"
        with open(state_path, "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(state, f)
        
        logger.info(f"Saved checkpoint at step {step}")
    
    async def train(self) -> Any:
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup components
        await self.setup_models()
        self.setup_optimization()
        train_dataset, val_dataset = self.setup_dataset()
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.dataloader_pin_memory
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.dataloader_pin_memory
            )
        
        # Calculate total steps
        if self.config.max_train_steps is None:
            self.config.max_train_steps = self.config.num_train_epochs * len(train_dataloader)
        
        # Training loop
        progress_bar = tqdm(range(self.config.max_train_steps), desc="Training")
        progress_bar.set_description("Training")
        
        for epoch in range(self.config.num_train_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                # Training step
                loss = await self.training_step(batch)
                epoch_loss += loss
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "epoch": epoch,
                    "step": self.global_step,
                    "loss": f"{loss:.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss,
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": self.global_step
                    })
                
                # Validation
                if val_dataloader and self.global_step % self.config.validation_steps == 0:
                    val_loss = 0.0
                    val_steps = 0
                    
                    for val_batch in val_dataloader:
                        val_loss += await self.validation_step(val_batch)
                        val_steps += 1
                        if val_steps >= 10:  # Limit validation steps
                            break
                    
                    avg_val_loss = val_loss / val_steps
                    
                    if self.use_wandb:
                        wandb.log({
                            "val/loss": avg_val_loss,
                            "val/step": self.global_step
                        })
                    
                    logger.info(f"Validation loss: {avg_val_loss:.4f}")
                    
                    # Save best model
                    if avg_val_loss < self.best_loss:
                        self.best_loss = avg_val_loss
                        await self.save_checkpoint(self.global_step, avg_val_loss)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    await self.save_checkpoint(self.global_step, loss)
                
                self.global_step += 1
                
                # Check if training is complete
                if self.global_step >= self.config.max_train_steps:
                    break
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save final checkpoint
        await self.save_checkpoint(self.global_step, loss)
        logger.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()


async def main():
    """Main function for training."""
    config = TrainingConfig(
        train_data_dir="data/train",
        validation_data_dir="data/val",
        output_dir="outputs",
        num_train_epochs=10,
        train_batch_size=1,
        learning_rate=1e-5,
        save_steps=500,
        validation_steps=100
    )
    
    trainer = DiffusionTrainer(config)
    await trainer.train()


match __name__:
    case "__main__":
    asyncio.run(main()) 