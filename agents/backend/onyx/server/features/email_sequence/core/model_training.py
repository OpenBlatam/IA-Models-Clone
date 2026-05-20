from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import math
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from diffusers import (
from transformers import (
from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Model Training and Evaluation for Email Sequence System

Advanced training and evaluation pipeline for diffusion models with
proper loss functions, metrics, validation, and performance monitoring.
"""



    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler
)
    CLIPTextModel,
    CLIPTokenizer,
    get_linear_schedule_with_warmup
)


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Model parameters
    model_name: str = "runwayml/stable-diffusion-v1-5"
    unet_config: Dict[str, Any] = None
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Scheduler parameters
    lr_scheduler: str = "cosine"  # cosine, linear, reduce_lr_on_plateau
    warmup_steps: int = 500
    num_training_steps: int = 10000
    
    # Loss parameters
    loss_type: str = "mse"  # mse, huber, smooth_l1
    loss_weight: float = 1.0
    
    # Validation parameters
    validation_split: float = 0.1
    validation_steps: int = 100
    save_best_model: bool = True
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and saving
    log_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./trained_models"
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


class EmailSequenceDataset(Dataset):
    """Custom dataset for email sequence training"""
    
    def __init__(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        tokenizer,
        max_length: int = 77
    ):
        
    """__init__ function."""
self.sequences = sequences
        self.subscribers = subscribers
        self.templates = templates
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Preprocess data
        self.processed_data = self._preprocess_data()
        
        logger.info(f"Dataset initialized with {len(self.processed_data)} samples")
    
    def _preprocess_data(self) -> List[Dict[str, Any]]:
        """Preprocess sequences for training"""
        processed_data = []
        
        for sequence in self.sequences:
            for step in sequence.steps:
                # Create context for each step
                context = self._create_context(sequence, step)
                
                # Tokenize text
                tokens = self.tokenizer(
                    context,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                processed_data.append({
                    "input_ids": tokens["input_ids"].squeeze(),
                    "attention_mask": tokens["attention_mask"].squeeze(),
                    "sequence_id": sequence.id,
                    "step_order": step.order,
                    "content_length": len(step.content or ""),
                    "delay_hours": step.delay_hours or 0
                })
        
        return processed_data
    
    def _create_context(
        self,
        sequence: EmailSequence,
        step: SequenceStep
    ) -> str:
        """Create context string for training"""
        
        # Get random subscriber and template for variety
        subscriber = random.choice(self.subscribers)
        template = random.choice(self.templates)
        
        context_parts = [
            f"Sequence: {sequence.name}",
            f"Step: {step.order}",
            f"Subscriber: {subscriber.first_name} {subscriber.last_name}",
            f"Company: {subscriber.company}",
            f"Interests: {', '.join(subscriber.interests)}",
            f"Template: {template.name}",
            f"Content: {step.content or 'No content'}"
        ]
        
        return " | ".join(context_parts)
    
    def __len__(self) -> Any:
        return len(self.processed_data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return self.processed_data[idx]


class DiffusionModel(nn.Module):
    """Custom diffusion model for email sequences"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Initialize UNet
        if config.unet_config is None:
            config.unet_config = {
                "sample_size": 64,
                "in_channels": 4,
                "out_channels": 4,
                "down_block_types": (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D"
                ),
                "up_block_types": (
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D"
                ),
                "block_out_channels": (320, 640, 1280, 1280),
                "layers_per_block": 2,
                "cross_attention_dim": 768,
                "attention_head_dim": 8,
                "use_linear_projection": True
            }
        
        self.unet = UNet2DConditionModel(**config.unet_config)
        
        # Initialize text encoder
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze text encoder
        self.text_encoder.requires_grad_(False)
        
        # Initialize VAE
        self.vae = AutoencoderKL.from_pretrained(config.model_name, subfolder="vae")
        self.vae.requires_grad_(False)
        
        # Initialize scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        
        logger.info("Diffusion Model initialized")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass"""
        
        # Encode text
        text_embeddings = self.text_encoder(input_ids, attention_mask=attention_mask)[0]
        
        # Predict noise
        noise_pred = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        return noise_pred
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embeddings"""
        tokens = self.text_tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids)[0]
        
        return text_embeddings


class TrainingMetrics:
    """Training metrics tracking and evaluation"""
    
    def __init__(self) -> Any:
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "gradient_norm": [],
            "epoch": [],
            "step": []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest metrics"""
        return {key: values[-1] if values else 0.0 for key, values in self.metrics.items()}
    
    def is_best_model(self, val_loss: float) -> bool:
        """Check if current model is best"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.metrics["train_loss"])
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        
        # Validation loss
        axes[0, 1].plot(self.metrics["val_loss"])
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Loss")
        
        # Learning rate
        axes[1, 0].plot(self.metrics["learning_rate"])
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("LR")
        
        # Gradient norm
        axes[1, 1].plot(self.metrics["gradient_norm"])
        axes[1, 1].set_title("Gradient Norm")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Norm")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class ModelTrainer:
    """Advanced model trainer for diffusion models"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = DiffusionModel(config)
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize metrics
        self.metrics = TrainingMetrics()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Setup mixed precision
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info("Model Trainer initialized")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        
        if self.config.lr_scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_training_steps,
                eta_min=1e-6
            )
        elif self.config.lr_scheduler == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.num_training_steps
            )
        elif self.config.lr_scheduler == "reduce_lr_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")
    
    async def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None
    ) -> Dict[str, List[float]]:
        """Complete training loop"""
        
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        early_stopping_counter = 0
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training epoch
            train_loss = await self._train_epoch(train_dataloader)
            training_history["train_loss"].append(train_loss)
            
            # Validation
            if val_dataloader:
                val_loss = await self._validate(val_dataloader)
                training_history["val_loss"].append(val_loss)
                
                # Check if best model
                if self.metrics.is_best_model(val_loss):
                    await self._save_model("best_model.pth")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                # Early stopping
                if early_stopping_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss if val_dataloader else train_loss)
            else:
                self.scheduler.step()
            
            training_history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss if val_dataloader else 'N/A':.4f}")
        
        return training_history
    
    async def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {self.epoch + 1}")
        
        for batch in progress_bar:
            # Prepare batch
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Create random latents
            latents = torch.randn(
                input_ids.shape[0], 4, 64, 64, device=self.device
            )
            
            # Add noise
            timesteps = torch.randint(
                0, self.model.noise_scheduler.num_train_timesteps,
                (input_ids.shape[0],), device=self.device
            )
            
            noisy_latents = self.model.noise_scheduler.add_noise(latents, timesteps)[0]
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    noise_pred = self.model(
                        input_ids, attention_mask, noisy_latents, timesteps
                    )
                    loss = F.mse_loss(noise_pred, latents)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                noise_pred = self.model(
                    input_ids, attention_mask, noisy_latents, timesteps
                )
                loss = F.mse_loss(noise_pred, latents)
                
                loss.backward()
                
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log metrics
            if self.global_step % self.config.log_steps == 0:
                self.metrics.update({
                    "train_loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "step": self.global_step,
                    "epoch": self.epoch
                })
        
        return total_loss / num_batches
    
    async def _validate(self, dataloader: DataLoader) -> float:
        """Validate model"""
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Create random latents
                latents = torch.randn(
                    input_ids.shape[0], 4, 64, 64, device=self.device
                )
                
                # Add noise
                timesteps = torch.randint(
                    0, self.model.noise_scheduler.num_train_timesteps,
                    (input_ids.shape[0],), device=self.device
                )
                
                noisy_latents = self.model.noise_scheduler.add_noise(latents, timesteps)[0]
                
                # Forward pass
                noise_pred = self.model(
                    input_ids, attention_mask, noisy_latents, timesteps
                )
                loss = F.mse_loss(noise_pred, latents)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Update metrics
        self.metrics.update({
            "val_loss": avg_loss,
            "step": self.global_step,
            "epoch": self.epoch
        })
        
        return avg_loss
    
    async def _save_model(self, filename: str):
        """Save model checkpoint"""
        
        save_path = Path(self.config.output_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.metrics.best_val_loss
        }
        
        torch.save(checkpoint, save_path)
        self.metrics.best_model_path = str(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    async def load_model(self, checkpoint_path: str):
        """Load model checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.metrics.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        logger.info(f"Model loaded from {checkpoint_path}")


class ModelEvaluator:
    """Model evaluation and testing"""
    
    def __init__(self, model: DiffusionModel, device: str = "cuda"):
        
    """__init__ function."""
self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        logger.info("Model Evaluator initialized")
    
    async def evaluate_model(
        self,
        test_dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        
        self.model.eval()
        
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Create random latents
                latents = torch.randn(
                    input_ids.shape[0], 4, 64, 64, device=self.device
                )
                
                # Add noise
                timesteps = torch.randint(
                    0, self.model.noise_scheduler.num_train_timesteps,
                    (input_ids.shape[0],), device=self.device
                )
                
                noisy_latents = self.model.noise_scheduler.add_noise(latents, timesteps)[0]
                
                # Forward pass
                noise_pred = self.model(
                    input_ids, attention_mask, noisy_latents, timesteps
                )
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, latents)
                total_loss += loss.item()
                
                # Store predictions and targets
                predictions.extend(noise_pred.cpu().numpy().flatten())
                targets.extend(latents.cpu().numpy().flatten())
        
        # Calculate metrics
        metrics = {
            "test_loss": total_loss / len(test_dataloader),
            "mse": mean_squared_error(targets, predictions),
            "mae": mean_absolute_error(targets, predictions),
            "r2_score": r2_score(targets, predictions)
        }
        
        return metrics
    
    async def generate_samples(
        self,
        prompts: List[str],
        num_samples: int = 4
    ) -> List[torch.Tensor]:
        """Generate samples from trained model"""
        
        self.model.eval()
        samples = []
        
        for prompt in prompts:
            # Encode text
            text_embeddings = self.model.encode_text(prompt).to(self.device)
            
            # Initialize latents
            latents = torch.randn(
                num_samples, 4, 64, 64, device=self.device
            )
            
            # Denoising loop
            for t in range(self.model.noise_scheduler.num_train_timesteps - 1, -1, -1):
                timesteps = torch.tensor([t], device=self.device)
                
                # Predict noise
                noise_pred = self.model.unet(
                    latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings.repeat(num_samples, 1, 1)
                ).sample
                
                # Denoise step
                latents = self.model.noise_scheduler.step(
                    noise_pred, timesteps, latents
                )["prev_sample"]
            
            # Decode latents
            with torch.no_grad():
                images = self.model.vae.decode(latents / 0.18215).sample
            
            samples.extend(images)
        
        return samples
    
    async def get_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        return {
            "model_info": {
                "device": str(self.device),
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            "performance_metrics": {
                "memory_usage": self._get_memory_usage(),
                "inference_time": self._measure_inference_time()
            }
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
                "gpu_memory_free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3  # GB
            }
        else:
            return {"cpu_memory": "N/A"}
    
    def _measure_inference_time(self) -> float:
        """Measure average inference time"""
        self.model.eval()
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (1, 77)).to(self.device)
        attention_mask = torch.ones(1, 77).to(self.device)
        latents = torch.randn(1, 4, 64, 64).to(self.device)
        timesteps = torch.tensor([500]).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(input_ids, attention_mask, latents, timesteps)
        
        # Measure inference time
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(input_ids, attention_mask, latents, timesteps)
            times.append(time.time() - start_time)
        
        return np.mean(times) 