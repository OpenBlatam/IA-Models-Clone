import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our existing components
from .custom_nn_modules import FacebookDiffusionUNet
from .forward_reverse_diffusion import (
    DiffusionConfig, ForwardDiffusionProcess, ReverseDiffusionProcess, 
    DiffusionTraining, BetaSchedule
)
from .production_final_optimizer import OptimizedFacebookProductionSystem, OptimizationConfig


@dataclass
class DiffusionTrainingConfig:
    """Configuration for diffusion model training"""
    # Diffusion parameters
    num_timesteps: int = 1000
    beta_schedule: BetaSchedule = BetaSchedule.COSINE
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 100
    gradient_accumulation_steps: int = 4
    
    # Model parameters
    image_size: int = 256
    in_channels: int = 3
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (8, 16)
    dropout: float = 0.1
    
    # Sampling parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # Data parameters
    data_dir: str = "facebook_content_images"
    save_dir: str = "diffusion_models"
    
    # Hardware
    use_gpu: bool = True
    mixed_precision: bool = True


class FacebookImageDataset(Dataset):
    """Dataset for Facebook content images"""
    
    def __init__(self, data_dir: str, image_size: int = 256, transform=None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Find all image files
        self.image_files = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
        
        if not self.image_files:
            # Create synthetic data for demonstration
            self._create_synthetic_data()
        
        self.logger = logging.getLogger("FacebookImageDataset")
        self.logger.info(f"Loaded {len(self.image_files)} images from {data_dir}")
    
    def _create_synthetic_data(self):
        """Create synthetic image data for demonstration"""
        import cv2
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic images
        for i in range(100):
            # Create random image
            img = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # Add some structure to make it more realistic
            cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), 3)
            cv2.circle(img, (150, 150), 50, (0, 255, 0), -1)
            
            # Save image
            img_path = self.data_dir / f"synthetic_image_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)
        
        self.image_files = list(self.data_dir.glob("*.jpg"))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        import cv2
        
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        
        if self.transform:
            img = self.transform(img)
        
        return img


class IntegratedDiffusionTrainer:
    """Integrated diffusion model trainer for Facebook content optimization"""
    
    def __init__(self, config: DiffusionTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        
        # Initialize logging
        self.logger = logging.getLogger("IntegratedDiffusionTrainer")
        self.logger.setLevel(logging.INFO)
        
        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize diffusion processes
        self._initialize_diffusion_processes()
        
        # Initialize model
        self._initialize_model()
        
        # Initialize training components
        self._initialize_training_components()
        
        self.logger.info(f"Initialized diffusion trainer on device: {self.device}")
    
    def _initialize_diffusion_processes(self):
        """Initialize diffusion processes"""
        # Create diffusion configuration
        diffusion_config = DiffusionConfig(
            num_timesteps=self.config.num_timesteps,
            beta_schedule=self.config.beta_schedule,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end
        )
        
        # Initialize processes
        self.forward_process = ForwardDiffusionProcess(diffusion_config)
        self.reverse_process = ReverseDiffusionProcess(diffusion_config)
        self.diffusion_training = DiffusionTraining(diffusion_config)
        
        self.logger.info("Initialized diffusion processes")
    
    def _initialize_model(self):
        """Initialize the diffusion UNet model"""
        self.model = FacebookDiffusionUNet(
            image_size=self.config.image_size,
            in_channels=self.config.in_channels,
            model_channels=self.config.model_channels,
            num_res_blocks=self.config.num_res_blocks,
            attention_resolutions=self.config.attention_resolutions,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    def _initialize_training_components(self):
        """Initialize training components"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Gradient scaler for mixed precision
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info("Initialized training components")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train the diffusion model"""
        self.logger.info("Starting diffusion model training...")
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            train_losses.append(train_loss)
            
            # Validation phase
            if val_loader:
                val_loss = self._validate_epoch(val_loader, epoch)
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best_model.pth", epoch, val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}" +
                (f", Val Loss: {val_loss:.4f}" if val_loader else "")
            )
            
            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch, train_loss)
        
        # Save final model
        self._save_checkpoint("final_model.pth", self.config.epochs - 1, train_losses[-1])
        
        self.logger.info("Training completed!")
        return train_losses, val_losses
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(self.device)
            
            # Gradient accumulation
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(images)
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss = self._compute_loss(images)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(self.device)
                loss = self._compute_loss(images)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _compute_loss(self, images: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss"""
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise to images
        noisy_images, noise = self.forward_process.q_sample(images, t)
        
        # Predict noise
        predicted_noise = self.model(noisy_images, t)
        
        # Compute loss
        loss = self.criterion(predicted_noise, noise)
        
        return loss
    
    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': vars(self.config)
        }
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['loss']
    
    def generate_images(self, num_images: int = 4, guidance_scale: float = 7.5) -> torch.Tensor:
        """Generate images using the trained diffusion model"""
        self.model.eval()
        
        with torch.no_grad():
            # Start from random noise
            x = torch.randn(num_images, self.config.in_channels, self.config.image_size, self.config.image_size, device=self.device)
            
            # Generate images
            generated_images = self.reverse_process.p_sample_loop(
                self.model, x, num_inference_steps=self.config.num_inference_steps
            )
        
        return generated_images
    
    def generate_content_variations(self, base_image: torch.Tensor, num_variations: int = 4) -> torch.Tensor:
        """Generate variations of a base image"""
        self.model.eval()
        
        # Add noise to base image
        t = torch.randint(0, self.config.num_timesteps, (1,), device=self.device)
        noisy_image, _ = self.forward_process.q_sample(base_image.unsqueeze(0), t)
        
        # Generate variations
        variations = []
        for _ in range(num_variations):
            with torch.no_grad():
                variation = self.reverse_process.p_sample_loop(
                    self.model, noisy_image, num_inference_steps=self.config.num_inference_steps
                )
                variations.append(variation)
        
        return torch.cat(variations, dim=0)


class EnhancedFacebookProductionSystem(OptimizedFacebookProductionSystem):
    """Enhanced production system with integrated diffusion training"""
    
    def __init__(self, config: OptimizationConfig, diffusion_config: Optional[DiffusionTrainingConfig] = None):
        super().__init__(config)
        
        # Initialize diffusion trainer if config provided
        self.diffusion_trainer = None
        if diffusion_config:
            self.diffusion_trainer = IntegratedDiffusionTrainer(diffusion_config)
        
        self.logger.info("Enhanced Facebook Production System initialized with diffusion training")
    
    def train_with_diffusion(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train both the main models and diffusion model"""
        if self.diffusion_trainer is None:
            self.logger.warning("No diffusion trainer configured")
            return
        
        # Train diffusion model
        self.logger.info("Training diffusion model...")
        diffusion_losses = self.diffusion_trainer.train(train_loader, val_loader)
        
        # Train main models (existing functionality)
        self.logger.info("Training main models...")
        super().train()
        
        return diffusion_losses
    
    def optimize_content_with_images(self, text: str, content_type: str, base_image: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Optimize content with image generation capabilities"""
        # Get text-based optimization
        text_results = self.optimize_content(text, content_type)
        
        # Generate images if diffusion model is available
        if self.diffusion_trainer is not None:
            if base_image is not None:
                # Generate variations of base image
                image_variations = self.diffusion_trainer.generate_content_variations(base_image)
                text_results['image_variations'] = image_variations
            else:
                # Generate new images
                generated_images = self.diffusion_trainer.generate_images()
                text_results['generated_images'] = generated_images
        
        return text_results
    
    def save_complete_system(self, save_dir: str):
        """Save the complete system including diffusion model"""
        super()._save_checkpoint(f"{save_dir}/main_models.pth")
        
        if self.diffusion_trainer:
            self.diffusion_trainer._save_checkpoint(f"{save_dir}/diffusion_model.pth", 0, 0.0)
        
        self.logger.info(f"Complete system saved to {save_dir}")
    
    def load_complete_system(self, save_dir: str):
        """Load the complete system including diffusion model"""
        super().load_checkpoint(f"{save_dir}/main_models.pth")
        
        if self.diffusion_trainer:
            self.diffusion_trainer.load_checkpoint(f"{save_dir}/diffusion_model.pth")
        
        self.logger.info(f"Complete system loaded from {save_dir}")


def create_enhanced_optimization_system():
    """Create and configure the enhanced optimization system with diffusion training"""
    
    # Main optimization config
    main_config = OptimizationConfig(
        learning_rate=1e-4,
        batch_size=32,
        epochs=50,
        use_gpu=True,
        mixed_precision=True,
        gradient_accumulation_steps=2
    )
    
    # Diffusion training config
    diffusion_config = DiffusionTrainingConfig(
        num_timesteps=1000,
        beta_schedule=BetaSchedule.COSINE,
        learning_rate=1e-4,
        batch_size=16,
        epochs=100,
        image_size=256,
        use_gpu=True,
        mixed_precision=True
    )
    
    # Create enhanced system
    system = EnhancedFacebookProductionSystem(main_config, diffusion_config)
    
    return system


def main():
    """Main training and demonstration function"""
    # Create system
    system = create_enhanced_optimization_system()
    
    # Create datasets
    train_dataset = FacebookImageDataset(
        data_dir=system.diffusion_trainer.config.data_dir,
        image_size=system.diffusion_trainer.config.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=system.diffusion_trainer.config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Train the complete system
    print("Training enhanced Facebook optimization system with diffusion models...")
    diffusion_losses = system.train_with_diffusion(train_loader)
    
    # Demonstrate image generation
    print("Generating sample images...")
    generated_images = system.diffusion_trainer.generate_images(num_images=4)
    
    # Save results
    system.save_complete_system("enhanced_facebook_system")
    
    print("Enhanced system training completed!")
    print(f"Generated {generated_images.shape[0]} images")
    
    return system


if __name__ == "__main__":
    main()


