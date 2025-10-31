"""
Advanced Training Pipeline for Export IA
========================================

Comprehensive training pipeline for AI models with advanced techniques including
LoRA fine-tuning, mixed precision training, gradient accumulation, and experiment tracking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import yaml

# Transformers and AI libraries
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset as HFDataset, load_dataset

# Import our models
from ..ai_enhanced.ai_export_engine import (
    ContentOptimizationModel, QualityAssessmentModel, DocumentDataset
)

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for advanced training pipeline."""
    # Model configuration
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    num_classes: int = 5
    
    # Training configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    num_epochs: int = 10
    max_grad_norm: float = 1.0
    
    # Advanced training options
    use_mixed_precision: bool = True
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Data configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Experiment tracking
    use_wandb: bool = True
    use_tensorboard: bool = True
    project_name: str = "export-ia-training"
    experiment_name: str = "default"
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 3
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

class AdvancedDocumentDataset(Dataset):
    """Enhanced dataset for document processing with advanced features."""
    
    def __init__(
        self,
        documents: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        augmentation: bool = False
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation = augmentation
        
        # Preprocess documents
        self.processed_docs = self._preprocess_documents()
    
    def _preprocess_documents(self) -> List[Dict[str, Any]]:
        """Preprocess documents for training."""
        processed = []
        
        for doc in self.documents:
            # Extract text content
            text = self._extract_text(doc)
            
            # Create training examples
            if self.augmentation:
                # Create multiple augmented versions
                augmented_texts = self._augment_text(text)
                for aug_text in augmented_texts:
                    processed.append({
                        "text": aug_text,
                        "quality_score": doc.get("quality_score", 0.5),
                        "document_type": doc.get("type", "unknown"),
                        "metadata": doc.get("metadata", {})
                    })
            else:
                processed.append({
                    "text": text,
                    "quality_score": doc.get("quality_score", 0.5),
                    "document_type": doc.get("type", "unknown"),
                    "metadata": doc.get("metadata", {})
                })
        
        return processed
    
    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """Extract text content from document."""
        if "content" in doc:
            return doc["content"]
        elif "sections" in doc:
            return "\n".join([section.get("content", "") for section in doc["sections"]])
        else:
            return str(doc)
    
    def _augment_text(self, text: str) -> List[str]:
        """Apply text augmentation techniques."""
        augmented = [text]  # Original text
        
        # Simple augmentation techniques
        # 1. Sentence shuffling
        sentences = text.split('. ')
        if len(sentences) > 2:
            shuffled = sentences.copy()
            np.random.shuffle(shuffled)
            augmented.append('. '.join(shuffled))
        
        # 2. Synonym replacement (simplified)
        # This would use a proper synonym replacement model
        augmented.append(text.replace("good", "excellent").replace("bad", "poor"))
        
        return augmented
    
    def __len__(self):
        return len(self.processed_docs)
    
    def __getitem__(self, idx):
        doc = self.processed_docs[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            doc["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "quality_score": torch.tensor(doc["quality_score"], dtype=torch.float),
            "document_type": doc["document_type"],
            "metadata": doc["metadata"]
        }

class AdvancedTrainingPipeline:
    """Advanced training pipeline with modern techniques."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.writer = None
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
        
        logger.info(f"Advanced Training Pipeline initialized on {self.device}")
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking with wandb and tensorboard."""
        # Setup wandb
        if self.config.use_wandb:
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
        
        # Setup tensorboard
        if self.config.use_tensorboard:
            log_dir = f"runs/{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir)
    
    def _load_tokenizer_and_model(self):
        """Load tokenizer and model with LoRA configuration."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModel.from_pretrained(self.config.model_name)
            
            # Apply LoRA if enabled
            if self.config.use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )
                self.model = get_peft_model(base_model, lora_config)
                logger.info("LoRA configuration applied")
            else:
                self.model = base_model
            
            # Add custom heads for our tasks
            self.model = self._add_custom_heads(self.model)
            
            self.model = self.model.to(self.device)
            
            # Initialize mixed precision scaler
            if self.config.use_mixed_precision:
                self.scaler = GradScaler()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _add_custom_heads(self, model):
        """Add custom classification and regression heads."""
        # This would add custom heads for quality assessment
        # For now, we'll use the existing model structure
        return model
    
    def _setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info("Optimizer and scheduler setup complete")
    
    def _create_data_loaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create data loaders with proper configuration."""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        
        return train_loader, val_loader, test_loader
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float, float]:
        """Train for one epoch with advanced techniques."""
        
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        total_grad_norm = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            quality_scores = batch["quality_score"].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self._compute_loss(outputs, quality_scores)
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self._compute_loss(outputs, quality_scores)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Track metrics
                total_grad_norm += grad_norm.item()
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_accuracy += self._compute_accuracy(outputs, quality_scores)
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb and tensorboard
            if batch_idx % 100 == 0:
                self._log_training_step(epoch, batch_idx, loss.item(), grad_norm.item())
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_grad_norm = total_grad_norm / (num_batches // self.config.gradient_accumulation_steps)
        
        return avg_loss, avg_accuracy, avg_grad_norm
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                quality_scores = batch["quality_score"].to(self.device)
                
                # Forward pass
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = self._compute_loss(outputs, quality_scores)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self._compute_loss(outputs, quality_scores)
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += self._compute_accuracy(outputs, quality_scores)
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def _compute_loss(self, outputs, targets):
        """Compute loss for the model outputs."""
        # This would implement the actual loss computation
        # For now, we'll use a simple MSE loss
        if hasattr(outputs, 'logits'):
            predictions = outputs.logits
        else:
            predictions = outputs.last_hidden_state.mean(dim=1)
        
        # Simple regression loss
        loss = F.mse_loss(predictions.squeeze(), targets)
        return loss
    
    def _compute_accuracy(self, outputs, targets):
        """Compute accuracy for the model outputs."""
        # This would implement the actual accuracy computation
        # For now, we'll return a dummy accuracy
        return 0.8
    
    def _log_training_step(self, epoch: int, batch_idx: int, loss: float, grad_norm: float):
        """Log training step to wandb and tensorboard."""
        step = epoch * len(self.train_loader) + batch_idx
        
        if self.config.use_wandb:
            wandb.log({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.scheduler.get_last_lr()[0],
                "train/step": step
            })
        
        if self.config.use_tensorboard and self.writer:
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/grad_norm", grad_norm, step)
            self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], step)
    
    def _log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float, 
                          train_acc: float, val_acc: float, epoch_time: float):
        """Log epoch metrics to wandb and tensorboard."""
        
        if self.config.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "train/epoch_accuracy": train_acc,
                "val/epoch_accuracy": val_acc,
                "train/epoch_time": epoch_time
            })
        
        if self.config.use_tensorboard and self.writer:
            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
            self.writer.add_scalar("epoch/train_accuracy", train_acc, epoch)
            self.writer.add_scalar("epoch/val_accuracy", val_acc, epoch)
            self.writer.add_scalar("epoch/time", epoch_time, epoch)
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
        if val_loss < self.best_val_loss - self.config.early_stopping_threshold:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = "checkpoints/best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
    
    async def train(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        test_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Main training function with advanced techniques."""
        
        try:
            # Load tokenizer and model
            self._load_tokenizer_and_model()
            
            # Create datasets
            train_dataset = AdvancedDocumentDataset(
                training_data, self.tokenizer, self.config.max_length, augmentation=True
            )
            
            val_dataset = None
            if validation_data:
                val_dataset = AdvancedDocumentDataset(
                    validation_data, self.tokenizer, self.config.max_length
                )
            else:
                # Split training data for validation
                val_size = int(len(train_dataset) * self.config.val_split)
                train_size = len(train_dataset) - val_size
                train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            test_dataset = None
            if test_data:
                test_dataset = AdvancedDocumentDataset(
                    test_data, self.tokenizer, self.config.max_length
                )
            
            # Create data loaders
            self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders(
                train_dataset, val_dataset, test_dataset
            )
            
            # Setup optimizer and scheduler
            num_training_steps = len(self.train_loader) * self.config.num_epochs
            self._setup_optimizer_and_scheduler(num_training_steps)
            
            # Training loop
            logger.info("Starting training...")
            start_time = time.time()
            
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                
                # Train epoch
                train_loss, train_acc, avg_grad_norm = self._train_epoch(self.train_loader, epoch)
                
                # Validate epoch
                val_loss, val_acc = self._validate_epoch(self.val_loader)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Update metrics
                self.metrics.train_loss.append(train_loss)
                self.metrics.val_loss.append(val_loss)
                self.metrics.train_accuracy.append(train_acc)
                self.metrics.val_accuracy.append(val_acc)
                self.metrics.epoch_times.append(epoch_time)
                
                # Log metrics
                self._log_epoch_metrics(epoch, train_loss, val_loss, train_acc, val_acc, epoch_time)
                
                # Check for best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_steps == 0 or is_best:
                    self._save_checkpoint(epoch, val_loss, is_best)
                
                # Early stopping check
                if self._check_early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
                # Log epoch summary
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Training completed
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f}s")
            
            # Final evaluation on test set
            test_results = {}
            if self.test_loader:
                test_loss, test_acc = self._validate_epoch(self.test_loader)
                test_results = {"test_loss": test_loss, "test_accuracy": test_acc}
                logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            
            # Close experiment tracking
            if self.config.use_wandb:
                wandb.finish()
            
            if self.writer:
                self.writer.close()
            
            return {
                "training_completed": True,
                "total_time": total_time,
                "best_val_loss": self.best_val_loss,
                "final_epoch": epoch + 1,
                "metrics": self.metrics.__dict__,
                "test_results": test_results
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.config.use_wandb:
                wandb.finish()
            if self.writer:
                self.writer.close()
            raise
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer and scheduler if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

def create_training_config_from_yaml(config_path: str) -> TrainingConfig:
    """Create training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TrainingConfig(**config_dict)

async def main():
    """Example usage of the advanced training pipeline."""
    
    # Create training configuration
    config = TrainingConfig(
        model_name="microsoft/DialoGPT-medium",
        batch_size=4,
        num_epochs=5,
        learning_rate=1e-4,
        use_lora=True,
        use_mixed_precision=True,
        experiment_name="export-ia-test"
    )
    
    # Create training pipeline
    pipeline = AdvancedTrainingPipeline(config)
    
    # Sample training data
    training_data = [
        {
            "content": "This is a sample business document with professional content.",
            "quality_score": 0.8,
            "type": "business_plan",
            "metadata": {"author": "AI", "date": "2024-01-01"}
        },
        {
            "content": "Another example of high-quality professional writing.",
            "quality_score": 0.9,
            "type": "report",
            "metadata": {"author": "AI", "date": "2024-01-01"}
        }
    ]
    
    # Train the model
    results = await pipeline.train(training_data)
    
    print("Training Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())



























