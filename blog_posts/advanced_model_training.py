from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from diffusers import (
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np
import logging
import os
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
from sklearn.model_selection import train_test_split, KFold
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import math
            from torchmetrics.image.fid import FrechetInceptionDistance
from typing import Any, List, Dict, Optional
import asyncio
"""
Advanced Model Training and Evaluation System
Comprehensive training and evaluation framework for diffusion models
with advanced features, proper PyTorch integration, and production-ready capabilities
"""

    UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler,
    StableDiffusionPipeline, StableDiffusionXLPipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Model configuration
    model_type: str = "unet"  # "unet", "vae", "text_encoder", "full_pipeline"
    model_id: str = "runwayml/stable-diffusion-v1-5"
    pretrained_model_path: Optional[str] = None
    
    # Training configuration
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data configuration
    train_data_dir: str = "data/train"
    val_data_dir: str = "data/val"
    image_size: int = 512
    num_workers: int = 4
    
    # Optimization configuration
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_xformers: bool = True
    compile_model: bool = False
    
    # Scheduler configuration
    scheduler_type: str = "cosine"  # "linear", "cosine", "exponential", "step"
    scheduler_warmup_steps: int = 1000
    scheduler_t_max: int = 1000
    
    # Loss configuration
    loss_type: str = "mse"  # "mse", "huber", "smooth_l1", "focal"
    loss_weight: float = 1.0
    
    # Evaluation configuration
    eval_steps: int = 500
    save_steps: int = 1000
    log_steps: int = 100
    num_eval_samples: int = 4
    
    # Output configuration
    output_dir: str = "outputs"
    save_total_limit: int = 3
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    # Monitoring configuration
    use_wandb: bool = True
    use_tensorboard: bool = True
    log_images: bool = True
    
    # Advanced configuration
    use_ema: bool = True
    ema_decay: float = 0.9999
    use_ddp: bool = False
    ddp_backend: str = "nccl"
    
    # Custom configuration
    custom_loss_fn: Optional[Callable] = None
    custom_metrics: List[str] = field(default_factory=list)
    additional_training_kwargs: Dict[str, Any] = field(default_factory=dict)


class DiffusionDataset(Dataset):
    """Dataset for diffusion model training"""
    
    def __init__(self, data_dir: str, tokenizer, image_size: int = 512, transform=None):
        
    """__init__ function."""
self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.transform = transform
        
        # Load data files
        self.image_files = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
        self.caption_files = list(self.data_dir.glob("*.txt"))
        
        # Create caption mapping
        self.caption_map = {}
        for caption_file in self.caption_files:
            image_name = caption_file.stem
            with open(caption_file, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                caption = f.read().strip()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            self.caption_map[image_name] = caption
    
    def __len__(self) -> Any:
        return len(self.image_files)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Resize image
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get caption
        image_name = image_path.stem
        caption = self.caption_map.get(image_name, "A beautiful image")
        
        # Tokenize caption
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "caption": caption
        }


class EarlyStopping:
    """Early stopping utility to stop training when validation loss does not improve."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        
    """__init__ function."""
self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_step = 0

    def __call__(self, val_loss: float, step: int):
        
    """__call__ function."""
if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_step = step
            if self.verbose:
                print(f"EarlyStopping: New best val_loss {val_loss:.6f} at step {step}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter} steps (best {self.best_loss:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping early at step {step}")


class AdvancedTrainer:
    """Advanced trainer for diffusion models"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.writer = None
        self.ema_model = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Setup components
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_monitoring()
        self._setup_ema()
    
    def _setup_model(self) -> Any:
        """Setup model based on configuration"""
        if self.config.model_type == "unet":
            self.model = UNet2DConditionModel.from_pretrained(
                self.config.model_id,
                subfolder="unet",
                use_safetensors=True
            )
        elif self.config.model_type == "vae":
            self.model = AutoencoderKL.from_pretrained(
                self.config.model_id,
                subfolder="vae",
                use_safetensors=True
            )
        elif self.config.model_type == "text_encoder":
            self.model = CLIPTextModel.from_pretrained(
                self.config.model_id,
                subfolder="text_encoder",
                use_safetensors=True
            )
        elif self.config.model_type == "full_pipeline":
            self.model = StableDiffusionPipeline.from_pretrained(
                self.config.model_id,
                use_safetensors=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Load pretrained weights if specified
        if self.config.pretrained_model_path:
            self.model.load_state_dict(torch.load(self.config.pretrained_model_path))
        
        # Move to device
        self.model.to(self.device)
        
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        
        # Compile model if requested
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def _setup_optimizer(self) -> Any:
        """Setup optimizer"""
        # Get trainable parameters
        trainable_params = list(self.model.parameters())
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup gradient scaler for mixed precision
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
    
    def _setup_scheduler(self) -> Any:
        """Setup learning rate scheduler"""
        if self.config.scheduler_type == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.scheduler_warmup_steps
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler_t_max,
                eta_min=1e-6
            )
        elif self.config.scheduler_type == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        elif self.config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1000,
                gamma=0.5
            )
        elif self.config.scheduler_type == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=True
            )
        else:
            self.scheduler = None
    
    def _setup_monitoring(self) -> Any:
        """Setup monitoring tools"""
        # TensorBoard
        if self.config.use_tensorboard:
            self.writer = SummaryWriter(log_dir=f"{self.config.output_dir}/tensorboard")
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.init(
                project="diffusion-training",
                config=self.config.__dict__,
                name=f"{self.config.model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _setup_ema(self) -> Any:
        """Setup Exponential Moving Average"""
        if self.config.use_ema:
            self.ema_model = self._create_ema_model()
    
    def _create_ema_model(self) -> Any:
        """Create EMA model"""
        ema_model = type(self.model)()
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        return ema_model
    
    def _update_ema(self) -> Any:
        """Update EMA model"""
        if self.ema_model is not None:
            with torch.no_grad():
                for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_param.data.mul_(self.config.ema_decay).add_(
                        param.data, alpha=1 - self.config.ema_decay
                    )
    
    def _compute_loss(self, batch, model_output) -> Any:
        """Compute loss based on configuration"""
        if self.config.custom_loss_fn:
            return self.config.custom_loss_fn(batch, model_output)
        
        if self.config.loss_type == "mse":
            return F.mse_loss(model_output.sample, batch["target"], reduction="mean")
        elif self.config.loss_type == "huber":
            return F.huber_loss(model_output.sample, batch["target"], reduction="mean")
        elif self.config.loss_type == "smooth_l1":
            return F.smooth_l1_loss(model_output.sample, batch["target"], reduction="mean")
        elif self.config.loss_type == "focal":
            return self._focal_loss(model_output.sample, batch["target"])
        else:
            return F.mse_loss(model_output.sample, batch["target"], reduction="mean")
    
    def _focal_loss(self, pred, target, alpha=0.25, gamma=2.0) -> Any:
        """Focal loss implementation"""
        ce_loss = F.mse_loss(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to monitoring tools"""
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.log(metrics, step=step)
        
        # Console logging
        if step % self.config.log_steps == 0:
            logger.info(f"Step {step}: {metrics}")
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            "step": step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_loss": self.best_loss,
            "config": self.config.__dict__
        }
        
        # Save EMA model if available
        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Cleanup old checkpoints"""
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint.unlink()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with robust gradient clipping and NaN/Inf handling"""
        self.model.train()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Forward pass
        with autocast() if self.config.use_mixed_precision else torch.no_grad():
            model_output = self.model(**batch)
            loss = self._compute_loss(batch, model_output)
        # NaN/Inf loss check
        if not torch.isfinite(loss):
            logger.warning(f"NaN or Inf loss detected at step {self.global_step}. Skipping update.")
            self.optimizer.zero_grad()
            return {"loss": float('nan')}
        # Backward pass
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        # Gradient NaN/Inf check and clipping
        grad_norm = None
        skip_update = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    logger.warning(f"NaN or Inf in gradients of {name} at step {self.global_step}. Skipping update.")
                    skip_update = True
                    break
        if not skip_update:
            # Gradient clipping
            if self.config.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if not math.isfinite(grad_norm):
                logger.warning(f"Non-finite grad norm ({grad_norm}) at step {self.global_step}. Skipping update.")
                skip_update = True
        # Optimizer step
        if not skip_update:
            if self.config.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            # Scheduler step
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            # Update EMA
            self._update_ema()
        # Reset gradients
        self.optimizer.zero_grad()
        return {"loss": loss.item() if not skip_update else float('nan'), "grad_norm": grad_norm if grad_norm is not None else float('nan')}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with autocast() if self.config.use_mixed_precision else torch.no_grad():
                    model_output = self.model(**batch)
                    loss = self._compute_loss(batch, model_output)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop with early stopping and flexible LR scheduling"""
        logger.info("Starting training...")
        early_stopper = EarlyStopping(patience=10, min_delta=1e-4, verbose=True)
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            for batch in progress_bar:
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                progress_bar.set_postfix({"loss": metrics["loss"]})
                if self.global_step % self.config.log_steps == 0:
                    self._log_metrics(metrics, self.global_step)
                if val_loader and self.global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate(val_loader)
                    self._log_metrics(val_metrics, self.global_step)
                    # Early stopping check
                    early_stopper(val_metrics["val_loss"], self.global_step)
                    # Save best model
                    if val_metrics["val_loss"] < self.best_loss:
                        self.best_loss = val_metrics["val_loss"]
                        self._save_checkpoint(self.global_step, is_best=True)
                    # ReduceLROnPlateau support
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["val_loss"])
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(self.global_step)
                self.global_step += 1
                if early_stopper.early_stop:
                    logger.info(f"Early stopping triggered at step {self.global_step}")
                    break
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            # Step scheduler if not ReduceLROnPlateau
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            if early_stopper.early_stop:
                break
        self._save_checkpoint(self.global_step)
        logger.info("Training completed!")


class ModelEvaluator:
    """Advanced model evaluator for diffusion models"""
    def __init__(self, model, device: torch.device):
        
    """__init__ function."""
self.model = model
        self.device = device
        self.metrics = {}
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)

    def evaluate_quality(self, generated_images: List[Image.Image], reference_images: List[Image.Image]) -> Dict[str, float]:
        """Evaluate image quality metrics: FID, LPIPS, SSIM, PSNR"""
        fid_score = self._calculate_fid(generated_images, reference_images)
        lpips_score = self._calculate_lpips(generated_images, reference_images)
        ssim_score = self._calculate_ssim(generated_images, reference_images)
        psnr_score = self._calculate_psnr(generated_images, reference_images)
        return {
            "fid": fid_score,
            "lpips": lpips_score,
            "ssim": ssim_score,
            "psnr": psnr_score
        }

    def _calculate_fid(self, generated_images: List[Image.Image], reference_images: List[Image.Image]) -> float:
        """Calculate FID score using torchmetrics or a placeholder if unavailable."""
        try:
            fid = FrechetInceptionDistance(feature=64).to(self.device)
            for img in generated_images:
                arr = np.asarray(img).astype(np.uint8)
                arr = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
                fid.update(arr, real=False)
            for img in reference_images:
                arr = np.asarray(img).astype(np.uint8)
                arr = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
                fid.update(arr, real=True)
            return float(fid.compute().cpu().item())
        except Exception as e:
            logger.warning(f"FID calculation failed: {e}")
            return -1.0

    def _calculate_lpips(self, generated_images: List[Image.Image], reference_images: List[Image.Image]) -> float:
        """Calculate LPIPS score."""
        scores = []
        for gen, ref in zip(generated_images, reference_images):
            gen_t = self._pil_to_tensor(gen).to(self.device)
            ref_t = self._pil_to_tensor(ref).to(self.device)
            score = self.lpips_fn(gen_t, ref_t).item()
            scores.append(score)
        return float(np.mean(scores)) if scores else -1.0

    def _calculate_ssim(self, generated_images: List[Image.Image], reference_images: List[Image.Image]) -> float:
        """Calculate SSIM score."""
        scores = []
        for gen, ref in zip(generated_images, reference_images):
            gen_arr = np.asarray(gen).astype(np.float32)
            ref_arr = np.asarray(ref).astype(np.float32)
            s = ssim(gen_arr, ref_arr, channel_axis=-1, data_range=255)
            scores.append(s)
        return float(np.mean(scores)) if scores else -1.0

    def _calculate_psnr(self, generated_images: List[Image.Image], reference_images: List[Image.Image]) -> float:
        """Calculate PSNR score."""
        scores = []
        for gen, ref in zip(generated_images, reference_images):
            gen_arr = np.asarray(gen).astype(np.float32)
            ref_arr = np.asarray(ref).astype(np.float32)
            s = psnr(ref_arr, gen_arr, data_range=255)
            scores.append(s)
        return float(np.mean(scores)) if scores else -1.0

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t

    def evaluate_diversity(self, generated_images: List[Image.Image]) -> Dict[str, float]:
        """Evaluate image diversity metrics"""
        # Inception Score
        inception_score = self._calculate_inception_score(generated_images)
        
        # Diversity Score
        diversity_score = self._calculate_diversity_score(generated_images)
        
        return {
            "inception_score": 0.0,  # Placeholder
            "diversity_score": 0.0   # Placeholder
        }
    
    def _calculate_inception_score(self, images: List[Image.Image]) -> float:
        """Calculate Inception Score"""
        # Simplified Inception Score calculation
        return 0.0  # Placeholder
    
    def _calculate_diversity_score(self, images: List[Image.Image]) -> float:
        """Calculate Diversity Score"""
        # Simplified Diversity Score calculation
        return 0.0  # Placeholder


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 4,
    cross_validation: bool = False,
    n_splits: int = 5,
    random_seed: int = 42
) -> dict:
    """
    Create efficient PyTorch DataLoaders for train/val/test splits.
    Optionally supports KFold cross-validation.
    Returns a dict with keys: 'train', 'val', 'test' or 'folds' (if cross_validation=True).
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    if cross_validation:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        folds = []
        for train_idx, val_idx in kf.split(indices):
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
            folds.append({'train': train_loader, 'val': val_loader})
        return {'folds': folds}
    else:
        test_size = int(test_split * dataset_size)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size - test_size
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}


def main():
    """Example usage of advanced model training"""
    
    # Configuration
    config = TrainingConfig(
        model_type="unet",
        model_id="runwayml/stable-diffusion-v1-5",
        batch_size=2,
        num_epochs=10,
        learning_rate=1e-4,
        use_mixed_precision=True,
        use_gradient_checkpointing=True,
        eval_steps=100,
        save_steps=200,
        log_steps=10
    )
    
    # Create trainer
    trainer = AdvancedTrainer(config)
    
    # Create datasets
    # train_dataset = DiffusionDataset(...)
    # Efficient data loading and splitting
    data_loaders = create_data_loaders(
        train_dataset,
        batch_size=config.batch_size,
        val_split=0.1,
        test_split=0.1,
        shuffle=True,
        num_workers=config.num_workers,
        cross_validation=False,  # Set True for KFold
        n_splits=5
    )
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    print("Advanced model training completed!")


match __name__:
    case "__main__":
    main() 