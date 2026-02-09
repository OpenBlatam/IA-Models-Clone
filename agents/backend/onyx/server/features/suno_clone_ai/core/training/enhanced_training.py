"""
Enhanced Training Pipeline with Best Practices

Implements:
- Efficient data loading with PyTorch DataLoader
- Proper train/validation/test splits and cross-validation
- Early stopping and learning rate scheduling
- Appropriate evaluation metrics
- Gradient clipping and NaN/Inf handling
- Experiment tracking (wandb, tensorboard)
"""

import logging
import os
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim import Optimizer, AdamW, Adam
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
    ExponentialLR
)
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.model_selection import KFold
import warnings

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


class EvaluationMetrics:
    """
    Collection of evaluation metrics for music generation models.
    """
    
    @staticmethod
    def compute_loss_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean"
    ) -> Dict[str, float]:
        """
        Compute various loss metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Mean Squared Error
        mse = nn.functional.mse_loss(predictions, targets, reduction=reduction)
        metrics['mse'] = mse.item() if isinstance(mse, torch.Tensor) else mse
        
        # Mean Absolute Error
        mae = nn.functional.l1_loss(predictions, targets, reduction=reduction)
        metrics['mae'] = mae.item() if isinstance(mae, torch.Tensor) else mae
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    @staticmethod
    def compute_audio_metrics(
        generated: np.ndarray,
        reference: np.ndarray,
        sample_rate: int = 32000
    ) -> Dict[str, float]:
        """
        Compute audio-specific metrics.
        
        Args:
            generated: Generated audio array
            reference: Reference audio array
            sample_rate: Sample rate
            
        Returns:
            Dictionary of audio metrics
        """
        metrics = {}
        
        # Ensure same length
        min_len = min(len(generated), len(reference))
        generated = generated[:min_len]
        reference = reference[:min_len]
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean((generated - reference) ** 2)
        if noise_power > 0:
            metrics['snr'] = 10 * np.log10(signal_power / noise_power)
        else:
            metrics['snr'] = float('inf')
        
        # Spectral Distance (simplified)
        try:
            import librosa
            gen_spec = np.abs(librosa.stft(generated, sr=sample_rate))
            ref_spec = np.abs(librosa.stft(reference, sr=sample_rate))
            
            # Spectral convergence
            num = np.linalg.norm(ref_spec - gen_spec, ord='fro')
            den = np.linalg.norm(ref_spec, ord='fro')
            metrics['spectral_convergence'] = num / den if den > 0 else 0.0
            
        except ImportError:
            logger.warning("librosa not available for spectral metrics")
        
        return metrics


class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline with best practices.
    
    Features:
    - Efficient data loading
    - Multiple evaluation metrics
    - Cross-validation support
    - Comprehensive experiment tracking
    - Proper error handling
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
        Initialize enhanced training pipeline.
        
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
            persistent_workers=num_workers > 0,
            drop_last=True  # Ensure consistent batch sizes
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
        self.metrics = EvaluationMetrics()
    
    def setup_training(
        self,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        log_dir: str = "./logs",
        wandb_project: str = "music-generation",
        wandb_config: Optional[Dict] = None
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
            wandb_project: W&B project name
            wandb_config: W&B config dictionary
        """
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        
        # Setup experiment tracking
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=wandb_project,
                config=wandb_config or {}
            )
            self.use_wandb = True
            logger.info("Weights & Biases tracking enabled")
        
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            log_path = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(str(log_path))
            logger.info(f"TensorBoard logging enabled: {log_path}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
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
                
                # Store predictions and targets for metrics
                with torch.no_grad():
                    all_predictions.append(output.detach().cpu())
                    all_targets.append(audio.detach().cpu())
                
                pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}", exc_info=True)
                continue
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        metrics = {'loss': avg_loss}
        
        if all_predictions:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            loss_metrics = self.metrics.compute_loss_metrics(predictions, targets)
            metrics.update(loss_metrics)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
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
                    
                    all_predictions.append(output.cpu())
                    all_targets.append(audio.cpu())
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}", exc_info=True)
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        metrics = {'loss': avg_loss}
        
        if all_predictions:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            loss_metrics = self.metrics.compute_loss_metrics(predictions, targets)
            metrics.update(loss_metrics)
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "./checkpoints",
        save_best: bool = True,
        save_every_n_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            save_best: Save best model based on validation loss
            save_every_n_epochs: Save checkpoint every N epochs (optional)
            
        Returns:
            Dictionary with training history
        """
        if self.optimizer is None or self.criterion is None:
            raise RuntimeError("Setup training first with setup_training()")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            if 'mse' in train_metrics:
                history['train_mse'].append(train_metrics['mse'])
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                history['val_loss'].append(val_metrics['loss'])
                if 'mse' in val_metrics:
                    history['val_mse'].append(val_metrics['mse'])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics.get('loss', 0.0):.4f}"
            )
            
            # Experiment tracking
            log_dict = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            
            if self.use_wandb:
                wandb.log(log_dict)
            
            if self.writer:
                for key, value in log_dict.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(key, value, epoch)
            
            # Save checkpoint
            if save_best and val_metrics and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'train_loss': train_metrics['loss'],
                    'history': history
                }
                if self.scheduler:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
                torch.save(checkpoint, save_dir / 'best_model.pt')
                logger.info(f"Saved best model (val_loss: {val_metrics['loss']:.4f})")
            
            # Save periodic checkpoint
            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': history
                }
                torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
            
            # Early stopping
            if self.early_stopping:
                stop_score = val_metrics.get('loss', train_metrics['loss'])
                if self.early_stopping(stop_score, self.model):
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


def cross_validate(
    model_factory: Callable[[], nn.Module],
    dataset: Dataset,
    n_splits: int = 5,
    batch_size: int = 4,
    num_epochs: int = 10,
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_factory: Function that creates a new model instance
        dataset: Full dataset
        n_splits: Number of folds
        batch_size: Batch size
        num_epochs: Number of epochs per fold
        seed: Random seed
        
    Returns:
        Dictionary with cross-validation results
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    cv_results = {
        'train_loss': [],
        'val_loss': []
    }
    
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        logger.info(f"Fold {fold+1}/{n_splits}")
        
        # Create datasets for this fold
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # Create model
        model = model_factory()
        
        # Create training pipeline
        pipeline = EnhancedTrainingPipeline(
            model=model,
            train_dataset=train_subset,
            val_dataset=val_subset,
            batch_size=batch_size
        )
        
        # Setup training
        optimizer = AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        pipeline.setup_training(optimizer, criterion)
        
        # Train
        history = pipeline.train(num_epochs=num_epochs)
        
        # Store results
        cv_results['train_loss'].append(history['train_loss'][-1])
        cv_results['val_loss'].append(history['val_loss'][-1])
    
    # Compute statistics
    cv_results['mean_train_loss'] = np.mean(cv_results['train_loss'])
    cv_results['std_train_loss'] = np.std(cv_results['train_loss'])
    cv_results['mean_val_loss'] = np.mean(cv_results['val_loss'])
    cv_results['std_val_loss'] = np.std(cv_results['val_loss'])
    
    logger.info(
        f"Cross-validation results: "
        f"Train Loss: {cv_results['mean_train_loss']:.4f} ± {cv_results['std_train_loss']:.4f}, "
        f"Val Loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}"
    )
    
    return cv_results



