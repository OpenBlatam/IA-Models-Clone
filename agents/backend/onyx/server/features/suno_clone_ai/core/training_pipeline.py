"""
Training Pipeline for Music Generation Models

Implements:
- Efficient data loading with PyTorch DataLoader
- Proper train/validation/test splits
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Mixed precision training
- Experiment tracking
"""

import logging
import os
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import experiment tracking libraries
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class MusicDataset(Dataset):
    """
    Dataset for music generation training.
    
    Handles loading and preprocessing of audio-text pairs.
    """
    
    def __init__(
        self,
        data_path: str,
        sample_rate: int = 32000,
        max_duration: int = 30,
        transform: Optional[Callable] = None
    ):
        """
        Initialize music dataset.
        
        Args:
            data_path: Path to data directory or JSON file
            sample_rate: Target sample rate
            max_duration: Maximum duration in seconds
            transform: Optional transform function
        """
        self.data_path = Path(data_path)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.transform = transform
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from path."""
        if self.data_path.is_file() and self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                return json.load(f)
        elif self.data_path.is_dir():
            # Load from directory structure
            data = []
            for file in self.data_path.glob("*.json"):
                with open(file, 'r') as f:
                    data.extend(json.load(f))
            return data
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with 'text', 'audio', and metadata
        """
        item = self.data[idx]
        
        # Load audio if path provided
        if 'audio_path' in item:
            import torchaudio
            audio, sr = torchaudio.load(item['audio_path'])
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            # Trim to max duration
            max_samples = self.max_duration * self.sample_rate
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
        else:
            # Use preprocessed audio array
            audio = torch.from_numpy(np.array(item['audio'])).float()
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
        
        # Apply transform if provided
        if self.transform:
            audio = self.transform(audio)
        
        return {
            'text': item['text'],
            'audio': audio,
            'metadata': item.get('metadata', {})
        }


class EarlyStopping:
    """
    Early stopping callback to prevent overfitting.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for loss/metric
            restore_best_weights: Restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score (loss or metric)
            model: Model to save weights from
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self._save_weights(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_weights(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self._restore_weights(model)
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def _save_weights(self, model: nn.Module) -> None:
        """Save model weights."""
        self.best_weights = model.state_dict().copy()
    
    def _restore_weights(self, model: nn.Module) -> None:
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


class TrainingPipeline:
    """
    Complete training pipeline with best practices.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_mixed_precision: bool = True,
        gradient_clip_norm: float = 1.0,
        device: Optional[str] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            test_dataset: Test dataset (optional)
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            use_mixed_precision: Enable mixed precision training
            gradient_clip_norm: Gradient clipping norm
            device: Device to use
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.gradient_clip_norm = gradient_clip_norm
        
        # Setup DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0
            )
        
        self.test_loader = None
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0
            )
        
        # Mixed precision scaler
        self.scaler = None
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = None
        
        # Experiment tracking
        self.writer = None
        self.use_wandb = False
    
    def setup_training(
        self,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        log_dir: str = "./logs"
    ) -> None:
        """
        Setup training components.
        
        Args:
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            early_stopping: Early stopping callback
            use_wandb: Use Weights & Biases for tracking
            use_tensorboard: Use TensorBoard for tracking
            log_dir: Directory for logs
        """
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        
        # Setup experiment tracking
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(project="music-generation")
            self.use_wandb = True
        
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            log_path = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(str(log_path))
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            try:
                # Move to device
                audio = batch['audio'].to(self.device)
                text = batch['text']
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(audio, text)
                        loss = self.criterion(output, audio)
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(audio, text)
                    loss = self.criterion(output, audio)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm
                        )
                    
                    self.optimizer.step()
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss detected, skipping batch")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                logger.error(f"Error in training batch: {e}", exc_info=True)
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate model.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                try:
                    audio = batch['audio'].to(self.device)
                    text = batch['text']
                    
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = self.model(audio, text)
                            loss = self.criterion(output, audio)
                    else:
                        output = self.model(audio, text)
                        loss = self.criterion(output, audio)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}", exc_info=True)
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "./checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        if self.optimizer is None or self.criterion is None:
            raise RuntimeError("Setup training first with setup_training()")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
            
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
                torch.save(checkpoint, save_dir / 'best_model.pt')
                logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history


def create_train_val_test_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset, test_dataset













