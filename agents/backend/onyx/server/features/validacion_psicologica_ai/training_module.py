"""
Training Module
===============
Refactored training module with best practices
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import structlog
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .config_loader import config_loader
from .experiment_tracking import experiment_tracker
from .distributed_training import DistributedTrainer, GradientAccumulator, MixedPrecisionTrainer
from .callbacks import CallbackList, EarlyStoppingCallback, ModelCheckpointCallback, LearningRateSchedulerCallback
from .loss_functions import create_loss_function
from .optimizers import create_optimizer
from .model_utils import initialize_weights, count_parameters

logger = structlog.get_logger()


class TrainingLoop:
    """
    Training loop with proper error handling, logging, and optimization
    Refactored with callbacks and better structure
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        callbacks: Optional[List] = None
    ):
        """
        Initialize training loop
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (optional, will create if not provided)
            loss_fn: Loss function (optional)
            device: Device (optional, will auto-detect)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Get device
        from .deep_learning_models import get_device
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        
        # Training config
        training_config = config_loader.get_training_config()
        self.learning_rate = training_config.get("learning_rate", 2e-5)
        self.num_epochs = training_config.get("num_epochs", 3)
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        
        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=training_config.get("weight_decay", 0.01)
            )
        else:
            self.optimizer = optimizer
        
        # Initialize loss function
        self.loss_fn = loss_fn or nn.MSELoss()
        
        # Mixed precision
        self.use_amp = training_config.get("fp16", True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulator
        self.gradient_accumulator = GradientAccumulator(self.gradient_accumulation_steps)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = training_config.get("warmup_steps", 100)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Callbacks
        if callbacks is None:
            callbacks = []
        
        # Add default callbacks if early stopping is enabled
        early_stopping_config = training_config.get("early_stopping", {})
        if early_stopping_config.get("enabled", False):
            callbacks.append(EarlyStoppingCallback(
                monitor="val_loss",
                patience=early_stopping_config.get("patience", 3),
                min_delta=early_stopping_config.get("min_delta", 0.001)
            ))
        
        self.callbacks = CallbackList(callbacks) if callbacks else None
        
        # Initialize model weights
        initialize_weights(self.model, init_type="xavier_uniform")
        
        logger.info(
            "Training loop initialized",
            device=str(self.device),
            use_amp=self.use_amp,
            num_epochs=self.num_epochs,
            total_params=count_parameters(self.model)
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        outputs = self._forward_pass(batch)
                        loss = self._compute_loss(outputs, batch)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = self._forward_pass(batch)
                    loss = self._compute_loss(outputs, batch)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights if accumulation step
                if self.gradient_accumulator.should_update():
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
                
                # Log metrics
                if batch_idx % config_loader.get("training.logging_steps", 10) == 0:
                    experiment_tracker.log_metrics({
                        "train/loss": loss.item() * self.gradient_accumulation_steps,
                        "train/learning_rate": self.scheduler.get_last_lr()[0]
                    }, step=epoch * len(self.train_loader) + batch_idx)
                
            except Exception as e:
                logger.error("Error in training step", epoch=epoch, batch=batch_idx, error=str(e))
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error("NaN/Inf loss detected, skipping batch")
                    continue
                raise
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                try:
                    batch = self._move_batch_to_device(batch)
                    
                    if self.use_amp:
                        with autocast():
                            outputs = self._forward_pass(batch)
                            loss = self._compute_loss(outputs, batch)
                    else:
                        outputs = self._forward_pass(batch)
                        loss = self._compute_loss(outputs, batch)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error("Error in validation step", error=str(e))
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "val_loss": avg_loss,
            "num_batches": num_batches
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop with callbacks
        
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        # Callback: on_train_begin
        if self.callbacks:
            self.callbacks.on_train_begin()
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            
            # Callback: on_epoch_begin
            if self.callbacks:
                self.callbacks.on_epoch_begin(epoch)
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
                history["val_loss"].append(val_metrics.get("val_loss", 0.0))
            
            # Prepare logs for callbacks
            epoch_logs = {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics.get("val_loss", 0.0)
            }
            
            # Callback: on_epoch_end
            if self.callbacks:
                should_stop = self.callbacks.on_epoch_end(epoch, epoch_logs, self.model)
                if should_stop:
                    logger.info("Early stopping triggered")
                    break
            
            # Log epoch metrics
            experiment_tracker.log_metrics({
                "epoch/train_loss": train_metrics["loss"],
                "epoch/val_loss": epoch_logs["val_loss"]
            }, step=epoch)
        
        # Callback: on_train_end
        if self.callbacks:
            self.callbacks.on_train_end(history)
        
        return history
    
    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _compute_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    


class PersonalityTrainingLoop(TrainingLoop):
    """Training loop for personality classification"""
    
    def _forward_pass(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass for personality model"""
        # This would call the actual model forward
        # Simplified for example
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for personality prediction"""
        labels = batch.get("personality_labels")
        if labels is None:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        for trait, pred in outputs.items():
            if trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
                label = labels[:, ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"].index(trait)]
                loss = self.loss_fn(pred.squeeze(), label)
                total_loss += loss
        
        return total_loss / len(outputs)

