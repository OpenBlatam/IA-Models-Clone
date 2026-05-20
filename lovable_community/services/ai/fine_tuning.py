"""
Fine-tuning Utilities with LoRA and P-tuning

Implements efficient fine-tuning techniques for transformer models:
- LoRA (Low-Rank Adaptation)
- P-tuning
- Full fine-tuning with proper training loops
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm import tqdm

from .base_service import BaseAIService
from .data_loader import TextDataset, BatchProcessor
from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class LoRAFineTuner:
    """
    Fine-tune transformer models using LoRA (Low-Rank Adaptation)
    
    LoRA is an efficient fine-tuning method that adds trainable low-rank
    matrices to the model instead of training all parameters.
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "SEQ_CLASSIFICATION",
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize LoRA fine-tuner
        
        Args:
            model_name: Name/path of the base model
            task_type: Type of task (SEQ_CLASSIFICATION, TOKEN_CLS, etc.)
            r: Rank of LoRA matrices
            alpha: LoRA alpha parameter
            dropout: Dropout rate for LoRA layers
            target_modules: Modules to apply LoRA to (default: ["q_proj", "v_proj"])
            device: Device to use
        """
        self.model_name = model_name
        self.task_type = getattr(TaskType, task_type, TaskType.SEQ_CLASSIFICATION)
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def load_model(self, num_labels: int = 2) -> None:
        """
        Load model and apply LoRA configuration
        
        Args:
            num_labels: Number of classification labels
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load base model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=self.task_type,
                r=self.r,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                target_modules=self.target_modules,
                bias="none"
            )
            
            # Apply LoRA
            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.to(self.device)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            logger.info(
                f"LoRA applied. Trainable params: {trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )
            
        except Exception as e:
            logger.error(f"Error loading model for LoRA: {e}", exc_info=True)
            raise
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        tracker: Optional[ExperimentTracker] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model with LoRA
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            gradient_accumulation_steps: Gradient accumulation steps
            tracker: Experiment tracker for logging
            
        Returns:
            Dictionary with training history
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.peft_model.parameters(),
            lr=learning_rate
        )
        
        # Calculate total steps
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Training loop
        self.peft_model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.peft_model(**batch)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.peft_model.parameters(),
                        max_grad_norm
                    )
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Log metrics
                    if tracker:
                        tracker.log_metric("train_loss", loss.item() * gradient_accumulation_steps, global_step)
                        tracker.log_metric("learning_rate", scheduler.get_last_lr()[0], global_step)
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
            
            avg_loss = epoch_loss / len(train_dataloader)
            history["train_loss"].append(avg_loss)
            
            # Validation
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                
                if tracker:
                    tracker.log_metric("val_loss", val_metrics["loss"], epoch)
                    tracker.log_metric("val_accuracy", val_metrics["accuracy"], epoch)
                
                logger.info(
                    f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Accuracy: {val_metrics['accuracy']:.4f}"
                )
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.peft_model is None:
            raise RuntimeError("Model not loaded")
        
        self.peft_model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.peft_model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                labels = batch.get("labels")
                
                if labels is not None:
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)
        
        metrics = {
            "loss": total_loss / len(dataloader),
            "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0.0
        }
        
        self.peft_model.train()
        return metrics
    
    def save_model(self, save_path: str) -> None:
        """Save the fine-tuned model"""
        if self.peft_model is None:
            raise RuntimeError("Model not loaded")
        
        self.peft_model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_finetuned_model(self, model_path: str) -> None:
        """Load a fine-tuned model"""
        from peft import PeftModel
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.peft_model = PeftModel.from_pretrained(self.model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.peft_model.to(self.device)
        logger.info(f"Fine-tuned model loaded from {model_path}")


class FullFineTuner:
    """
    Full fine-tuning of transformer models
    
    Trains all model parameters (not just LoRA adapters).
    Use this when you have enough data and compute resources.
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        device: Optional[torch.device] = None
    ):
        """
        Initialize full fine-tuner
        
        Args:
            model_name: Name/path of the model
            num_labels: Number of classification labels
            device: Device to use
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> None:
        """Load the model"""
        try:
            logger.info(f"Loading model for full fine-tuning: {self.model_name}")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            self.model.to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        tracker: Optional[ExperimentTracker] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model (full fine-tuning)
        
        Similar to LoRA training but trains all parameters.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if tracker:
                    tracker.log_metric("train_loss", loss.item(), epoch * len(train_dataloader) + len(train_dataloader))
            
            avg_loss = epoch_loss / len(train_dataloader)
            history["train_loss"].append(avg_loss)
            
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                labels = batch.get("labels")
                
                if labels is not None:
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total if total > 0 else 0.0
        }
    
    def save_model(self, save_path: str) -> None:
        """Save the model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")















