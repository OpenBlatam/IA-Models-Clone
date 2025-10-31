"""
Advanced Neural Network Training Service for Facebook Posts API
Deep learning, transformers, and neural network training capabilities
"""

import asyncio
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    TrainingArguments, Trainer,
    pipeline
)
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import logging
import os
import pickle
from pathlib import Path

from ..core.config import get_settings
from ..core.models import FacebookPost, PostStatus, ContentType, AudienceType
from ..infrastructure.cache import get_cache_manager
from ..infrastructure.monitoring import get_monitor, timed
from ..infrastructure.database import get_db_manager, PostRepository

logger = structlog.get_logger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger_pytorch = logging.getLogger("pytorch")


class TrainingStatus(Enum):
    """Training status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ModelType(Enum):
    """Model type enumeration"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    BERT = "bert"
    GPT = "gpt"
    T5 = "t5"
    RESNET = "resnet"
    VIT = "vit"
    CLIP = "clip"
    DIFFUSION = "diffusion"
    GAN = "gan"
    VAE = "vae"


class OptimizerType(Enum):
    """Optimizer type enumeration"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    LAMB = "lamb"
    LION = "lion"


class SchedulerType(Enum):
    """Learning rate scheduler type enumeration"""
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    WARMUP = "warmup"
    CONSTANT = "constant"


@dataclass
class TrainingConfig:
    """Training configuration data structure"""
    model_type: ModelType
    model_name: str
    dataset_name: str
    batch_size: int
    learning_rate: float
    epochs: int
    optimizer: OptimizerType
    scheduler: SchedulerType
    loss_function: str
    validation_split: float
    early_stopping: bool
    gradient_clipping: bool
    mixed_precision: bool
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    save_strategy: str
    eval_strategy: str
    logging_steps: int
    save_steps: int
    eval_steps: int
    seed: int
    device: str
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Training metrics data structure"""
    epoch: int
    step: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    gradient_norm: float
    memory_usage: float
    gpu_utilization: float
    training_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelCheckpoint:
    """Model checkpoint data structure"""
    id: str
    model_name: str
    epoch: int
    step: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Dict[str, Any]
    config: TrainingConfig
    metrics: List[TrainingMetrics]
    file_path: str
    file_size: int
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """Training job data structure"""
    id: str
    entity_id: str
    config: TrainingConfig
    status: TrainingStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    current_epoch: int = 0
    best_metric: float = 0.0
    metrics: List[TrainingMetrics] = field(default_factory=list)
    checkpoints: List[ModelCheckpoint] = field(default_factory=list)
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedTransformerModel(nn.Module):
    """Advanced Transformer model with multiple architectures"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        max_length: int = 512,
        model_type: ModelType = ModelType.TRANSFORMER
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.model_type = model_type
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if model_type == ModelType.TRANSFORMER:
            # Standard Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
        elif model_type == ModelType.BERT:
            # BERT-style architecture
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers
            )
            
        elif model_type == ModelType.GPT:
            # GPT-style architecture with causal attention
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers
            )
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        batch_size, seq_len = input_ids.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)
        
        # Transformer encoding
        if self.model_type == ModelType.GPT:
            # GPT uses causal attention
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(input_ids.device)
            transformer_output = self.transformer(embeddings, memory=embeddings, tgt_mask=tgt_mask)
        else:
            # BERT/Transformer use bidirectional attention
            if attention_mask is not None:
                attention_mask = attention_mask.float()
                attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
                attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
            transformer_output = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # Layer normalization and output projection
        transformer_output = self.layer_norm(transformer_output)
        logits = self.output_projection(transformer_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {"logits": logits, "loss": loss}
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for GPT-style models"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class AdvancedDataset(Dataset):
    """Advanced dataset with multiple data types support"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        task_type: str = "language_modeling"
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.task_type == "language_modeling":
            return self._get_language_modeling_item(text)
        elif self.task_type == "classification":
            return self._get_classification_item(text)
        elif self.task_type == "generation":
            return self._get_generation_item(text)
        else:
            return self._get_language_modeling_item(text)
    
    def _get_language_modeling_item(self, text):
        """Get item for language modeling task"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _get_classification_item(self, text):
        """Get item for classification task"""
        # Split text and label (assuming format: "text|label")
        if '|' in text:
            text_content, label = text.split('|', 1)
            label = int(label.strip())
        else:
            text_content = text
            label = 0  # Default label
        
        encoding = self.tokenizer(
            text_content,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _get_generation_item(self, text):
        """Get item for text generation task"""
        # Split input and target (assuming format: "input|target")
        if '|' in text:
            input_text, target_text = text.split('|', 1)
        else:
            input_text = text
            target_text = text
        
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length // 2,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length // 2,
            return_tensors='pt'
        )
        
        input_ids = input_encoding['input_ids'].squeeze()
        input_attention_mask = input_encoding['attention_mask'].squeeze()
        target_ids = target_encoding['input_ids'].squeeze()
        target_attention_mask = target_encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': input_attention_mask,
            'target_ids': target_ids,
            'target_attention_mask': target_attention_mask
        }


class AdvancedTrainer:
    """Advanced trainer with multiple training strategies"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        save_dir: str = "./checkpoints"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf')
        self.training_metrics = []
        
        # Move model to device
        self.model.to(device)
    
    def _create_optimizer(self):
        """Create optimizer based on config"""
        if self.config.optimizer == OptimizerType.ADAM:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == OptimizerType.ADAMW:
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == OptimizerType.SGD:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config"""
        if self.config.scheduler == SchedulerType.LINEAR:
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.epochs
            )
        elif self.config.scheduler == SchedulerType.COSINE:
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == SchedulerType.STEP:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        else:
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.epochs
            )
    
    def _create_loss_function(self):
        """Create loss function based on config"""
        if self.config.loss_function == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif self.config.loss_function == "MSELoss":
            return nn.MSELoss()
        elif self.config.loss_function == "BCELoss":
            return nn.BCELoss()
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.get('loss', self.criterion(outputs['logits'], batch['labels']))
            else:
                outputs = self.model(**batch)
                loss = outputs.get('loss', self.criterion(outputs['logits'], batch['labels']))
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(outputs['logits'], dim=-1)
                if 'labels' in batch:
                    accuracy = (predictions == batch['labels']).float().mean()
                    total_accuracy += accuracy.item()
            
            total_loss += loss.item()
            num_batches += 1
            self.current_step += 1
            
            # Log progress
            if self.current_step % self.config.logging_steps == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Step {self.current_step}, "
                    f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs.get('loss', self.criterion(outputs['logits'], batch['labels']))
                else:
                    outputs = self.model(**batch)
                    loss = outputs.get('loss', self.criterion(outputs['logits'], batch['labels']))
                
                # Calculate accuracy
                predictions = torch.argmax(outputs['logits'], dim=-1)
                if 'labels' in batch:
                    accuracy = (predictions == batch['labels']).float().mean()
                    total_accuracy += accuracy.item()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": avg_accuracy
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Save model checkpoint"""
        checkpoint_id = f"checkpoint_epoch_{epoch}_step_{self.current_step}"
        checkpoint_path = self.save_dir / f"{checkpoint_id}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.best_metric = checkpoint['best_metric']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> List[TrainingMetrics]:
        """Full training loop"""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_dataloader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = TrainingMetrics(
                epoch=epoch,
                step=self.current_step,
                train_loss=train_metrics['train_loss'],
                val_loss=val_metrics['val_loss'],
                train_accuracy=train_metrics['train_accuracy'],
                val_accuracy=val_metrics['val_accuracy'],
                learning_rate=train_metrics['learning_rate'],
                gradient_norm=0.0,  # TODO: Calculate gradient norm
                memory_usage=0.0,   # TODO: Calculate memory usage
                gpu_utilization=0.0 # TODO: Calculate GPU utilization
            )
            
            self.training_metrics.append(epoch_metrics)
            
            # Save checkpoint if best metric improved
            if val_metrics['val_loss'] < self.best_metric:
                self.best_metric = val_metrics['val_loss']
                self.save_checkpoint(epoch, {**train_metrics, **val_metrics})
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )
        
        logger.info("Training completed")
        return self.training_metrics


class MockNeuralTrainingEngine:
    """Mock neural training engine for testing and development"""
    
    def __init__(self):
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.model_checkpoints: List[ModelCheckpoint] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path("./checkpoints")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_training_job(
        self,
        entity_id: str,
        config: TrainingConfig
    ) -> TrainingJob:
        """Create a new training job"""
        job_id = f"training_job_{int(time.time())}"
        
        job = TrainingJob(
            id=job_id,
            entity_id=entity_id,
            config=config,
            status=TrainingStatus.PENDING
        )
        
        self.training_jobs[job_id] = job
        logger.info("Training job created", job_id=job_id, entity_id=entity_id)
        return job
    
    async def start_training(self, job_id: str, dataset: List[str]) -> TrainingJob:
        """Start training job"""
        if job_id not in self.training_jobs:
            raise ValueError(f"Training job {job_id} not found")
        
        job = self.training_jobs[job_id]
        job.status = TrainingStatus.RUNNING
        job.start_time = datetime.now()
        
        try:
            # Initialize tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            # Create dataset
            train_dataset = AdvancedDataset(
                texts=dataset,
                tokenizer=tokenizer,
                max_length=job.config.max_length if hasattr(job.config, 'max_length') else 512,
                task_type="language_modeling"
            )
            
            # Split dataset
            val_size = int(len(train_dataset) * job.config.validation_split)
            train_size = len(train_dataset) - val_size
            train_data, val_data = random_split(train_dataset, [train_size, val_size])
            
            # Create dataloaders
            train_dataloader = DataLoader(
                train_data,
                batch_size=job.config.batch_size,
                shuffle=True,
                num_workers=job.config.num_workers,
                pin_memory=job.config.pin_memory
            )
            
            val_dataloader = DataLoader(
                val_data,
                batch_size=job.config.batch_size,
                shuffle=False,
                num_workers=job.config.num_workers,
                pin_memory=job.config.pin_memory
            )
            
            # Create model
            model = AdvancedTransformerModel(
                vocab_size=tokenizer.vocab_size,
                d_model=768,
                nhead=12,
                num_layers=6,
                model_type=job.config.model_type
            )
            
            # Create trainer
            trainer = AdvancedTrainer(
                model=model,
                config=job.config,
                device=self.device,
                save_dir=str(self.save_dir / job_id)
            )
            
            # Start training
            training_metrics = trainer.train(train_dataloader, val_dataloader)
            
            # Update job status
            job.status = TrainingStatus.COMPLETED
            job.end_time = datetime.now()
            job.total_steps = len(training_metrics)
            job.completed_steps = len(training_metrics)
            job.current_epoch = job.config.epochs
            job.best_metric = min([m.val_loss for m in training_metrics]) if training_metrics else 0.0
            job.metrics = training_metrics
            
            logger.info("Training job completed", job_id=job_id)
            return job
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.end_time = datetime.now()
            job.error_message = str(e)
            logger.error("Training job failed", job_id=job_id, error=str(e))
            raise
    
    async def get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID"""
        return self.training_jobs.get(job_id)
    
    async def get_training_jobs(self, entity_id: str) -> List[TrainingJob]:
        """Get all training jobs for entity"""
        return [job for job in self.training_jobs.values() if job.entity_id == entity_id]
    
    async def cancel_training_job(self, job_id: str) -> bool:
        """Cancel training job"""
        if job_id not in self.training_jobs:
            return False
        
        job = self.training_jobs[job_id]
        if job.status == TrainingStatus.RUNNING:
            job.status = TrainingStatus.CANCELLED
            job.end_time = datetime.now()
            logger.info("Training job cancelled", job_id=job_id)
            return True
        
        return False
    
    async def get_model_checkpoints(self, job_id: str) -> List[ModelCheckpoint]:
        """Get model checkpoints for training job"""
        return [cp for cp in self.model_checkpoints if cp.model_name == job_id]


class NeuralTrainingService:
    """Main neural training service orchestrator"""
    
    def __init__(self):
        self.training_engine = MockNeuralTrainingEngine()
        self.cache_manager = get_cache_manager()
        self.monitor = get_monitor()
    
    @timed("neural_create_training_job")
    async def create_training_job(
        self,
        entity_id: str,
        config: TrainingConfig
    ) -> TrainingJob:
        """Create training job"""
        return await self.training_engine.create_training_job(entity_id, config)
    
    @timed("neural_start_training")
    async def start_training(self, job_id: str, dataset: List[str]) -> TrainingJob:
        """Start training"""
        return await self.training_engine.start_training(job_id, dataset)
    
    @timed("neural_get_job")
    async def get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job"""
        return await self.training_engine.get_training_job(job_id)
    
    @timed("neural_get_jobs")
    async def get_training_jobs(self, entity_id: str) -> List[TrainingJob]:
        """Get training jobs"""
        return await self.training_engine.get_training_jobs(entity_id)
    
    @timed("neural_cancel_job")
    async def cancel_training_job(self, job_id: str) -> bool:
        """Cancel training job"""
        return await self.training_engine.cancel_training_job(job_id)
    
    @timed("neural_get_checkpoints")
    async def get_model_checkpoints(self, job_id: str) -> List[ModelCheckpoint]:
        """Get model checkpoints"""
        return await self.training_engine.get_model_checkpoints(job_id)


# Global neural training service instance
_neural_training_service: Optional[NeuralTrainingService] = None


def get_neural_training_service() -> NeuralTrainingService:
    """Get global neural training service instance"""
    global _neural_training_service
    
    if _neural_training_service is None:
        _neural_training_service = NeuralTrainingService()
    
    return _neural_training_service


# Export all classes and functions
__all__ = [
    # Enums
    'TrainingStatus',
    'ModelType',
    'OptimizerType',
    'SchedulerType',
    
    # Data classes
    'TrainingConfig',
    'TrainingMetrics',
    'ModelCheckpoint',
    'TrainingJob',
    
    # Models and Datasets
    'AdvancedTransformerModel',
    'AdvancedDataset',
    
    # Training
    'AdvancedTrainer',
    
    # Engines and Services
    'MockNeuralTrainingEngine',
    'NeuralTrainingService',
    
    # Utility functions
    'get_neural_training_service',
]



























