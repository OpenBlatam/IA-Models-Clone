"""
Training Module for Dermatology AI Models
Implements efficient data loading, training loops, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Callable, Any
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class SkinDataset(Dataset):
    """
    PyTorch Dataset for skin images
    Implements proper data loading with augmentation support
    """
    
    def __init__(
        self,
        images: List[np.ndarray],
        labels: Optional[Dict[str, List]] = None,
        transform: Optional[Callable] = None,
        target_size: tuple = (224, 224)
    ):
        self.images = images
        self.labels = labels or {}
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image = self.images[idx]
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            from PIL import Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Resize
        image = image.resize(self.target_size)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image = transform(image)
        
        # Prepare labels
        sample = {'image': image}
        
        for key, values in self.labels.items():
            if idx < len(values):
                sample[key] = torch.tensor(values[idx], dtype=torch.float32)
        
        return sample


class Trainer:
    """
    Training class with best practices:
    - Efficient data loading
    - Mixed precision training
    - Gradient clipping
    - Early stopping
    - Learning rate scheduling
    - Proper evaluation metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        early_stopping_patience: int = 10,
        experiment_tracker: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device != "cpu"
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        self.experiment_tracker = experiment_tracker
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        metric_values = {name: [] for name in (metrics or {}).keys()}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self._compute_loss(criterion, outputs, batch)
            else:
                outputs = self.model(images)
                loss = self._compute_loss(criterion, outputs, batch)
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )
                optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            
            if metrics:
                for name, metric_fn in metrics.items():
                    metric_val = metric_fn(outputs, batch)
                    metric_values[name].append(metric_val)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}")
                break
        
        # Average metrics
        avg_loss = running_loss / len(self.train_loader)
        avg_metrics = {
            name: np.mean(values) if values else 0.0
            for name, values in metric_values.items()
        }
        
        return {
            'loss': avg_loss,
            **avg_metrics
        }
    
    def validate(
        self,
        criterion: nn.Module,
        metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        running_loss = 0.0
        metric_values = {name: [] for name in (metrics or {}).keys()}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self._compute_loss(criterion, outputs, batch)
                else:
                    outputs = self.model(images)
                    loss = self._compute_loss(criterion, outputs, batch)
                
                running_loss += loss.item()
                
                if metrics:
                    for name, metric_fn in metrics.items():
                        metric_val = metric_fn(outputs, batch)
                        metric_values[name].append(metric_val)
        
        avg_loss = running_loss / len(self.val_loader)
        avg_metrics = {
            name: np.mean(values) if values else 0.0
            for name, values in metric_values.items()
        }
        
        return {
            'loss': avg_loss,
            **avg_metrics
        }
    
    def _compute_loss(
        self,
        criterion: nn.Module,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss from outputs and batch"""
        # This is a placeholder - implement based on your model's output format
        if 'conditions' in outputs and 'conditions' in batch:
            return criterion(outputs['conditions'], batch['conditions'])
        elif isinstance(outputs, torch.Tensor) and 'labels' in batch:
            return criterion(outputs, batch['labels'])
        else:
            # Default: MSE loss for regression
            if isinstance(outputs, dict):
                # Multi-task loss
                total_loss = 0.0
                for key, pred in outputs.items():
                    if key in batch:
                        total_loss += nn.functional.mse_loss(pred, batch[key])
                return total_loss
            else:
                return nn.functional.mse_loss(outputs, batch.get('target', outputs))
    
    def fit(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """Train the model"""
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}, Mixed Precision: {self.use_mixed_precision}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(optimizer, criterion, metrics)
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)
            
            # Validate
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.validate(criterion, metrics)
                self.training_history['val_loss'].append(val_metrics.get('loss', 0))
                self.training_history['val_metrics'].append(val_metrics)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics.get('loss', 'N/A')} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Experiment tracking
            if self.experiment_tracker:
                from core.experiment_tracker import ExperimentMetrics
                self.experiment_tracker.log_metrics(ExperimentMetrics(
                    experiment_id=self.experiment_tracker.current_experiment,
                    epoch=epoch + 1,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics.get('loss'),
                    learning_rate=optimizer.param_groups[0]['lr']
                ))
            
            # Early stopping
            if self.val_loader:
                val_loss = val_metrics.get('loss', float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    if checkpoint_dir:
                        self.save_checkpoint(
                            checkpoint_dir / f"best_model_epoch_{epoch + 1}.pt",
                            optimizer,
                            epoch
                        )
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    optimizer,
                    epoch
                )
        
        logger.info("Training completed")
    
    def save_checkpoint(
        self,
        path: Path,
        optimizer: optim.Optimizer,
        epoch: int
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path, optimizer: Optional[optim.Optimizer] = None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        })
        
        logger.info(f"Checkpoint loaded: {path}")


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """Create data loaders with optimal settings"""
    loaders = {}
    
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    if val_dataset:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    if test_dataset:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
    
    return loaders













