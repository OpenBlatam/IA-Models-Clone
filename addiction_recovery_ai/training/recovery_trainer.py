"""
Training Pipeline for Recovery Models
Enhanced with learning rate scheduling, early stopping, gradient clipping, and comprehensive metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from typing import Optional, Dict, List, Callable
import logging
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class RecoveryModelTrainer:
    """
    Enhanced trainer for recovery prediction models
    Features: mixed precision, learning rate scheduling, early stopping, gradient clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1
    ):
        """
        Initialize trainer with enhanced features
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            use_mixed_precision: Use mixed precision
            gradient_clip_val: Gradient clipping value
            accumulate_grad_batches: Number of batches to accumulate gradients
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        
        self.model = self.model.to(self.device)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training history
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        
        logger.info(
            f"RecoveryModelTrainer initialized on {self.device} "
            f"(mixed_precision={self.use_mixed_precision}, "
            f"grad_clip={gradient_clip_val}, grad_accum={accumulate_grad_batches})"
        )
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation and clipping
        
        Args:
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            
            # Forward pass
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulate_grad_batches
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / self.accumulate_grad_batches
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate gradients
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.use_mixed_precision:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip_val
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip_val
                    )
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update learning rate if scheduler is step-based
                if scheduler is not None and isinstance(scheduler, (CosineAnnealingLR, OneCycleLR)):
                    scheduler.step()
            
            # Collect metrics
            total_loss += loss.item() * self.accumulate_grad_batches
            
            # For classification metrics
            if outputs.dim() > 1 and outputs.size(1) > 1:
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = (outputs > 0.5).cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy().flatten())
            
            num_batches += 1
            pbar.set_postfix({
                "loss": loss.item() * self.accumulate_grad_batches,
                "lr": optimizer.param_groups[0]['lr']
            })
        
        # Calculate metrics
        metrics = {"loss": total_loss / num_batches}
        
        # Classification metrics
        if len(np.unique(all_targets)) > 1:  # Binary or multi-class
            try:
                metrics["accuracy"] = accuracy_score(all_targets, all_predictions)
                metrics["precision"] = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
                metrics["recall"] = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
                metrics["f1"] = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
            except Exception as e:
                logger.warning(f"Could not calculate classification metrics: {e}")
        
        return metrics
    
    def validate(self, criterion: nn.Module) -> Dict[str, float]:
        """
        Validate model with comprehensive metrics
        
        Args:
            criterion: Loss function
            
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
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                else:
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    predictions = (probs > 0.5).astype(int)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy().flatten())
                all_probs.extend(probs)
        
        metrics = {"loss": total_loss / num_batches}
        
        # Classification metrics
        if len(np.unique(all_targets)) > 1:
            try:
                metrics["accuracy"] = accuracy_score(all_targets, all_predictions)
                metrics["precision"] = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
                metrics["recall"] = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
                metrics["f1"] = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
                
                # ROC AUC for binary classification
                if len(np.unique(all_targets)) == 2:
                    try:
                        if isinstance(all_probs[0], np.ndarray):
                            probs_flat = [p[1] if len(p) > 1 else p[0] for p in all_probs]
                        else:
                            probs_flat = all_probs
                        metrics["roc_auc"] = roc_auc_score(all_targets, probs_flat)
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC: {e}")
            except Exception as e:
                logger.warning(f"Could not calculate validation metrics: {e}")
        
        return metrics
    
    def train(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        save_dir: Optional[str] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: Optional[int] = None,
        monitor_metric: str = "loss",
        mode: str = "min"
    ):
        """
        Full training loop with early stopping and checkpointing
        
        Args:
            optimizer: Optimizer
            criterion: Loss function
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            scheduler: Learning rate scheduler
            early_stopping_patience: Patience for early stopping
            monitor_metric: Metric to monitor for early stopping
            mode: 'min' or 'max' for monitoring
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        best_metric = float('inf') if mode == 'min' else float('-inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(optimizer, criterion, epoch, scheduler)
            val_metrics = self.validate(criterion)
            
            # Update learning rate scheduler (for ReduceLROnPlateau)
            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                monitor_val = val_metrics.get(monitor_metric, val_metrics.get('loss', 0))
                scheduler.step(monitor_val)
            
            # Store history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # Log metrics
            log_msg = f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}"
            if 'accuracy' in train_metrics:
                log_msg += f", Train Acc={train_metrics['accuracy']:.4f}"
            log_msg += f", Val Loss={val_metrics.get('loss', 0):.4f}"
            if 'accuracy' in val_metrics:
                log_msg += f", Val Acc={val_metrics['accuracy']:.4f}"
            logger.info(log_msg)
            
            # Checkpointing
            if save_dir:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "train_history": self.train_history,
                    "val_history": self.val_history
                }
                
                # Save latest checkpoint
                torch.save(checkpoint, os.path.join(save_dir, "checkpoint_latest.pth"))
                
                # Save best checkpoint
                monitor_val = val_metrics.get(monitor_metric, val_metrics.get('loss', 0))
                is_better = (mode == 'min' and monitor_val < best_metric) or \
                           (mode == 'max' and monitor_val > best_metric)
                
                if is_better:
                    best_metric = monitor_val
                    patience_counter = 0
                    torch.save(checkpoint, os.path.join(save_dir, "checkpoint_best.pth"))
                    logger.info(f"New best {monitor_metric}: {best_metric:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        logger.info("Training completed")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None
) -> RecoveryModelTrainer:
    """Factory function for trainer"""
    return RecoveryModelTrainer(model, train_loader, val_loader, device)

