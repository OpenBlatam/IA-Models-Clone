"""
Comprehensive Diffusion Models System for Training, Inference, and Analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline, DDIMScheduler, DDPM, UNet2DConditionModel,
    AutoencoderKL, DiffusionPipeline, DPMSolverMultistepScheduler,
    EMAModel, AttnProcessor2_0, DiffusionScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
import numpy as np
import PIL.Image
import logging
import time
from pathlib import Path
import json
import warnings


@dataclass
class DiffusionConfig:
    """Configuration for diffusion models."""
    # Model settings
    model_name: str = "runwayml/stable-diffusion-v1-5"
    model_type: str = "stable_diffusion"  # stable_diffusion, ddpm, ddim, custom
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
    use_classifier_free_guidance: bool = True
    
    # Memory optimization
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    enable_model_cpu_offload: bool = False
    
    # Quality settings
    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1
    
    # Advanced settings
    use_ema: bool = True
    use_gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    use_amp: bool = True


@dataclass
class TrainingConfig:
    """Configuration for diffusion model training."""
    # Basic training
    learning_rate: float = 1e-5
    num_epochs: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer: str = "adamw"  # adamw, lion, adafactor
    weight_decay: float = 0.01
    warmup_steps: int = 500
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    
    # Loss and regularization
    loss_type: str = "l2"  # l2, l1, huber
    label_smoothing: float = 0.0
    gradient_clip_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 10
    evaluation_steps: int = 100
    
    # Data
    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
    image_size: int = 512
    center_crop: bool = True
    random_flip: bool = True


class DiffusionModelManager:
    """Manages diffusion models for training, inference, and analysis."""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self._setup_logging()
        
        # Initialize components
        self.pipeline = None
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        
        self._load_models()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the diffusion model manager."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_models(self):
        """Load diffusion models and components."""
        try:
            if self.config.use_pipeline:
                self._load_pipeline()
            else:
                self._load_individual_components()
            
            self._apply_optimizations()
            self.logger.info("✅ Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _load_pipeline(self):
        """Load complete diffusion pipeline."""
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.use_amp else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Extract components for training
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = self.pipeline.scheduler
    
    def _load_individual_components(self):
        """Load individual model components."""
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.model_name,
            subfolder="unet",
            torch_dtype=torch.float16 if self.config.use_amp else torch.float32
        )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_name,
            subfolder="vae",
            torch_dtype=torch.float16 if self.config.use_amp else torch.float32
        )
        
        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.model_name,
            subfolder="text_encoder",
            torch_dtype=torch.float16 if self.config.use_amp else torch.float32
        )
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model_name,
            subfolder="tokenizer"
        )
        
        # Load scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config.model_name,
            subfolder="scheduler"
        )
        
        # Move to device
        self.unet = self.unet.to(self.device)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.scheduler = self.scheduler.to(self.device)
    
    def _apply_optimizations(self):
        """Apply memory and performance optimizations."""
        if self.config.enable_attention_slicing and hasattr(self.pipeline, 'enable_attention_slicing'):
            self.pipeline.enable_attention_slicing()
            self.logger.info("✅ Attention slicing enabled")
        
        if self.config.enable_vae_slicing and hasattr(self.pipeline, 'enable_vae_slicing'):
            self.pipeline.enable_vae_slicing()
            self.logger.info("✅ VAE slicing enabled")
        
        if self.config.enable_xformers_memory_efficient_attention:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                self.logger.info("✅ XFormers memory efficient attention enabled")
            except Exception as e:
                self.logger.warning(f"⚠️ XFormers not available: {e}")
        
        if self.config.enable_model_cpu_offload and hasattr(self.pipeline, 'enable_model_cpu_offload'):
            self.pipeline.enable_model_cpu_offload()
            self.logger.info("✅ Model CPU offload enabled")
        
        if self.config.use_gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.logger.info("✅ Gradient checkpointing enabled")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        **kwargs
    ) -> List[PIL.Image.Image]:
        """Generate images using the diffusion model."""
        try:
            # Override config with kwargs
            generation_kwargs = {
                'num_inference_steps': self.config.num_inference_steps,
                'guidance_scale': self.config.guidance_scale,
                'eta': self.config.eta,
                'height': self.config.height,
                'width': self.config.width,
                'num_images_per_prompt': num_images,
                **kwargs
            }
            
            # Generate images
            if self.pipeline:
                images = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **generation_kwargs
                ).images
            else:
                images = self._generate_with_components(
                    prompt, negative_prompt, **generation_kwargs
                )
            
            self.logger.info(f"✅ Generated {len(images)} images")
            return images
            
        except Exception as e:
            self.logger.error(f"❌ Error generating images: {e}")
            raise
    
    def _generate_with_components(
        self,
        prompt: str,
        negative_prompt: str = "",
        **kwargs
    ) -> List[PIL.Image.Image]:
        """Generate images using individual components."""
        # Tokenize prompts
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode text
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Generate latents
        latents = self._generate_latents(text_embeddings, **kwargs)
        
        # Decode latents to images
        images = self._decode_latents(latents)
        
        return images
    
    def _generate_latents(
        self,
        text_embeddings: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> torch.Tensor:
        """Generate latents using the UNet and scheduler."""
        # Setup timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare latents
        latents = torch.randn(
            (text_embeddings.shape[0] // 2, 4, self.config.height // 8, self.config.width // 8),
            device=self.device,
            dtype=text_embeddings.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
    def _decode_latents(self, latents: torch.Tensor) -> List[PIL.Image.Image]:
        """Decode latents to images using the VAE."""
        # Scale latents
        latents = 1 / 0.18215 * latents
        
        # Decode
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        
        # Convert to PIL images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        
        pil_images = [PIL.Image.fromarray(image) for image in images]
        return pil_images
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the loaded models."""
        info = {
            'config': self.config.__dict__,
            'device': str(self.device),
            'models_loaded': {
                'pipeline': self.pipeline is not None,
                'unet': self.unet is not None,
                'vae': self.vae is not None,
                'text_encoder': self.text_encoder is not None,
                'tokenizer': self.tokenizer is not None,
                'scheduler': self.scheduler is not None
            }
        }
        
        if self.unet:
            info['unet_params'] = sum(p.numel() for p in self.unet.parameters())
            info['unet_trainable_params'] = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        
        if self.vae:
            info['vae_params'] = sum(p.numel() for p in self.vae.parameters())
        
        if self.text_encoder:
            info['text_encoder_params'] = sum(p.numel() for p in self.text_encoder.parameters())
        
        return info


class DiffusionTrainer:
    """Handles training of diffusion models."""
    
    def __init__(self, model_manager: DiffusionModelManager, config: TrainingConfig):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
    
    def _setup_optimizer(self):
        """Setup optimizer for training."""
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model_manager.unet.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "lion":
            try:
                from lion_pytorch import Lion
                self.optimizer = Lion(
                    self.model_manager.unet.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            except ImportError:
                self.logger.warning("⚠️ Lion optimizer not available, using AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.model_manager.unet.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.num_epochs
            )
        elif self.config.lr_scheduler == "constant":
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")
    
    def _setup_loss_function(self):
        """Setup loss function for training."""
        if self.config.loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        elif self.config.loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif self.config.loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, float]:
        """Perform a single training step."""
        self.model_manager.unet.train()
        
        # Move batch to device
        pixel_values = batch["pixel_values"].to(self.model_manager.device)
        input_ids = batch["input_ids"].to(self.model_manager.device)
        
        # Encode text
        with torch.no_grad():
            text_embeddings = self.model_manager.text_encoder(input_ids)[0]
        
        # Sample noise and timesteps
        noise = torch.randn_like(pixel_values)
        timesteps = torch.randint(
            0,
            self.model_manager.scheduler.config.num_train_timesteps,
            (pixel_values.shape[0],),
            device=pixel_values.device
        ).long()
        
        # Add noise to images
        noisy_images = self.model_manager.scheduler.add_noise(
            pixel_values, noise, timesteps
        )
        
        # Predict noise
        noise_pred = self.model_manager.unet(
            noisy_images,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Calculate loss
        loss = self.loss_fn(noise_pred, noise)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model_manager.unet.parameters(),
                self.config.gradient_clip_norm
            )
        
        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'epoch': epoch,
            'global_step': self.global_step
        }
    
    def validate(
        self,
        val_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Validate the model on validation data."""
        self.model_manager.unet.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.model_manager.device)
                input_ids = batch["input_ids"].to(self.model_manager.device)
                
                # Encode text
                text_embeddings = self.model_manager.text_encoder(input_ids)[0]
                
                # Sample noise and timesteps
                noise = torch.randn_like(pixel_values)
                timesteps = torch.randint(
                    0,
                    self.model_manager.scheduler.config.num_train_timesteps,
                    (pixel_values.shape[0],),
                    device=pixel_values.device
                ).long()
                
                # Add noise to images
                noisy_images = self.model_manager.scheduler.add_noise(
                    pixel_values, noise, timesteps
                )
                
                # Predict noise
                noise_pred = self.model_manager.unet(
                    noisy_images,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Calculate loss
                loss = self.loss_fn(noise_pred, noise)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'val_loss': avg_loss,
            'epoch': self.epoch
        }
    
    def save_checkpoint(self, path: str, metadata: Dict[str, Any] = None):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model_manager.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'training_state': {
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_loss': self.best_loss
            },
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"✅ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.model_manager.device)
        
        # Load model state
        self.model_manager.unet.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['training_state']['global_step']
        self.epoch = checkpoint['training_state']['epoch']
        self.best_loss = checkpoint['training_state']['best_loss']
        
        self.logger.info(f"✅ Checkpoint loaded from {path}")


class DiffusionAnalyzer:
    """Analyzes diffusion models and their outputs."""
    
    def __init__(self, model_manager: DiffusionModelManager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
    
    def analyze_generation_quality(
        self,
        images: List[PIL.Image.Image],
        prompts: List[str]
    ) -> Dict[str, Any]:
        """Analyze the quality of generated images."""
        try:
            # Convert PIL images to tensors
            image_tensors = []
            for img in images:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                image_tensors.append(img_tensor)
            
            # Stack tensors
            batch_tensor = torch.cat(image_tensors, dim=0)
            
            # Calculate quality metrics
            quality_metrics = {
                'brightness': self._calculate_brightness(batch_tensor),
                'contrast': self._calculate_contrast(batch_tensor),
                'sharpness': self._calculate_sharpness(batch_tensor),
                'color_diversity': self._calculate_color_diversity(batch_tensor),
                'image_count': len(images),
                'prompts': prompts
            }
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing generation quality: {e}")
            return {}
    
    def _calculate_brightness(self, images: torch.Tensor) -> float:
        """Calculate average brightness of images."""
        # Convert to grayscale and calculate mean
        grayscale = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        return grayscale.mean().item()
    
    def _calculate_contrast(self, images: torch.Tensor) -> float:
        """Calculate average contrast of images."""
        # Calculate standard deviation of pixel values
        return images.std().item()
    
    def _calculate_sharpness(self, images: torch.Tensor) -> float:
        """Calculate average sharpness of images using Laplacian variance."""
        # Convert to grayscale
        grayscale = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        
        # Apply Laplacian filter
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=images.device).unsqueeze(0).unsqueeze(0)
        
        sharpness_scores = []
        for i in range(grayscale.shape[0]):
            img = grayscale[i:i+1]
            laplacian = F.conv2d(img, laplacian_kernel, padding=1)
            sharpness_scores.append(laplacian.var().item())
        
        return np.mean(sharpness_scores)
    
    def _calculate_color_diversity(self, images: torch.Tensor) -> float:
        """Calculate color diversity using histogram analysis."""
        # Calculate color histograms
        histograms = []
        for i in range(images.shape[0]):
            img = images[i]
            # Convert to 8-bit for histogram calculation
            img_8bit = (img * 255).byte()
            
            # Calculate histogram for each channel
            channel_hists = []
            for c in range(3):
                hist = torch.histc(img_8bit[c], bins=256, min=0, max=255)
                channel_hists.append(hist)
            
            histograms.append(torch.stack(channel_hists))
        
        # Calculate average histogram
        avg_histogram = torch.stack(histograms).mean(dim=0)
        
        # Calculate entropy as diversity measure
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        normalized_hist = avg_histogram / (avg_histogram.sum() + epsilon)
        entropy = -(normalized_hist * torch.log(normalized_hist + epsilon)).sum()
        
        return entropy.item()
    
    def analyze_model_performance(
        self,
        generation_times: List[float],
        memory_usage: List[float]
    ) -> Dict[str, Any]:
        """Analyze model performance metrics."""
        if not generation_times:
            return {}
        
        performance_metrics = {
            'generation_time': {
                'mean': np.mean(generation_times),
                'std': np.std(generation_times),
                'min': np.min(generation_times),
                'max': np.max(generation_times),
                'total': np.sum(generation_times)
            },
            'memory_usage': {
                'mean': np.mean(memory_usage) if memory_usage else 0,
                'std': np.std(memory_usage) if memory_usage else 0,
                'min': np.min(memory_usage) if memory_usage else 0,
                'max': np.max(memory_usage) if memory_usage else 0
            },
            'throughput': {
                'images_per_second': len(generation_times) / np.sum(generation_times),
                'average_time_per_image': np.mean(generation_times)
            }
        }
        
        return performance_metrics
    
    def create_generation_report(
        self,
        images: List[PIL.Image.Image],
        prompts: List[str],
        generation_times: List[float],
        quality_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a comprehensive generation report."""
        report = {
            'summary': {
                'total_images': len(images),
                'total_prompts': len(prompts),
                'generation_time_total': sum(generation_times),
                'generation_time_average': np.mean(generation_times)
            },
            'quality_analysis': quality_metrics,
            'performance_analysis': self.analyze_model_performance(generation_times, []),
            'prompt_analysis': {
                'prompts': prompts,
                'prompt_lengths': [len(p) for p in prompts],
                'average_prompt_length': np.mean([len(p) for p in prompts])
            },
            'timestamp': time.time()
        }
        
        return report


class DiffusionDataProcessor:
    """Handles data processing for diffusion model training."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_dataset(
        self,
        data_dir: str,
        tokenizer: Any,
        image_size: int = 512
    ) -> torch.utils.data.Dataset:
        """Create a dataset for diffusion model training."""
        # This is a placeholder for dataset creation
        # In practice, you would implement proper dataset loading
        # based on your specific data format and requirements
        
        class DiffusionDataset(torch.utils.data.Dataset):
            def __init__(self, data_dir: str, tokenizer: Any, image_size: int):
                self.data_dir = Path(data_dir)
                self.tokenizer = tokenizer
                self.image_size = image_size
                
                # Placeholder for data loading logic
                self.data = []
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                # Placeholder for data loading
                # In practice, you would load images and text here
                return {
                    'pixel_values': torch.randn(3, self.image_size, self.image_size),
                    'input_ids': torch.randint(0, 1000, (77,))
                }
        
        return DiffusionDataset(data_dir, tokenizer, image_size)
    
    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> torch.utils.data.DataLoader:
        """Create a dataloader for training."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids
        }


def create_diffusion_system(
    diffusion_config: DiffusionConfig,
    training_config: TrainingConfig
) -> Tuple[DiffusionModelManager, DiffusionTrainer, DiffusionAnalyzer]:
    """Create a complete diffusion system."""
    # Create model manager
    model_manager = DiffusionModelManager(diffusion_config)
    
    # Create trainer
    trainer = DiffusionTrainer(model_manager, training_config)
    
    # Create analyzer
    analyzer = DiffusionAnalyzer(model_manager)
    
    return model_manager, trainer, analyzer






