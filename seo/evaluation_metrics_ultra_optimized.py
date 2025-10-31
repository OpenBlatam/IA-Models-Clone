#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED SEO Evaluation Metrics System
Production-ready with PyTorch, Transformers, LoRA, Diffusers, Multi-GPU, and advanced training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from diffusers import (
    UNet2DConditionModel, DDPMScheduler, DDIMScheduler, 
    StableDiffusionPipeline, StableDiffusionXLPipeline
)
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import asyncio

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import warnings
import math
import re
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import json
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
import torch.profiler

# Import specialized SEO evaluation metrics
from seo_evaluation_metrics import SEOMetricsConfig, SEOModelEvaluator

warnings.filterwarnings('ignore')

@dataclass
class UltraOptimizedConfig:
    """Ultra-optimized configuration for SEO evaluation system."""
    # Core settings
    use_multi_gpu: bool = True
    use_distributed: bool = False
    batch_size: int = 32768
    num_workers: int = mp.cpu_count()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "mixed"
    use_amp: bool = True
    enable_profiling: bool = True
    
    # Model settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    num_heads: int = 8
    d_model: int = 512
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Diffusion settings
    use_diffusion: bool = True
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5"
    diffusion_steps: int = 1000
    diffusion_guidance_scale: float = 7.5
    
    # Training settings
    num_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    max_steps: int = 100000
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    
    # SEO evaluation settings
    seo_evaluation_config: SEOMetricsConfig = field(default_factory=lambda: SEOMetricsConfig(
        task_type="classification",
        num_classes=2,
        average="weighted",
        use_seo_specific=True,
        seo_score_threshold=0.7,
        content_quality_threshold=0.6,
        keyword_density_threshold=0.02,
        readability_threshold=0.5
    ))

class SEOTokenizer:
    """Ultra-optimized SEO tokenizer with proper text preprocessing."""
    
    def __init__(self, config: UltraOptimizedConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # SEO-specific preprocessing patterns
        self.html_patterns = [
            r'<[^>]+>',  # HTML tags
            r'&[a-zA-Z]+;',  # HTML entities
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
        ]
        self.html_regex = re.compile('|'.join(self.html_patterns))
        
        # SEO-specific vocabulary additions
        self.seo_keywords = [
            'seo', 'search', 'optimization', 'keywords', 'meta', 'title', 'description',
            'content', 'ranking', 'google', 'backlinks', 'analytics', 'traffic'
        ]
        self._extend_vocabulary()
    
    def _extend_vocabulary(self):
        """Extend tokenizer vocabulary with SEO-specific terms."""
        for keyword in self.seo_keywords:
            if keyword not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([keyword])
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for SEO analysis."""
        text = self.html_regex.sub('', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\,\!\?]', '', text)
        return text.lower().strip()
    
    def tokenize_batch(self, texts: List[str], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts with proper SEO preprocessing."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        tokenized = self.tokenizer(
            processed_texts,
            padding="longest",
            truncation="longest_first",
            max_length=self.config.max_length,
            return_tensors=return_tensors,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        return tokenized

class UltraOptimizedSEOMetricsModule(nn.Module):
    """Ultra-optimized PyTorch nn.Module with LoRA fine-tuning and diffusion models for SEO metrics."""
    
    def __init__(self, config: UltraOptimizedConfig):
        super(UltraOptimizedSEOMetricsModule, self).__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize SEO tokenizer
        self.seo_tokenizer = SEOTokenizer(config)
        
        # Initialize SEO evaluator with specialized metrics
        self.seo_evaluator = SEOModelEvaluator(config.seo_evaluation_config)
        
        # Initialize Transformers library components
        self.transformer_config = AutoConfig.from_pretrained(config.model_name)
        self.transformer_model = AutoModel.from_pretrained(config.model_name, config=self.transformer_config)
        
        # Configure LoRA for efficient fine-tuning
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["query", "value"]
            )
            self.transformer_model = get_peft_model(self.transformer_model, lora_config)
        
        # Initialize diffusion model components
        if self.config.use_diffusion:
            self._setup_diffusion_models()
        
        # SEO-specific classification head
        self.seo_classifier = nn.Sequential(
            nn.Linear(self.transformer_config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # accuracy, precision, recall, f1
        )
        
        # Initialize weights using proper techniques
        self._initialize_weights()
        self._setup_gpu()
        self._setup_profiler()
    
    def _setup_diffusion_models(self):
        """Initialize diffusion model components for SEO content generation."""
        # UNet for diffusion process
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=self.transformer_config.hidden_size
        )
        
        # Diffusion schedulers
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.config.diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        # Move to device
        self.unet.to(self.device)
    
    def _initialize_weights(self):
        """Initialize weights using proper PyTorch techniques."""
        for module in self.seo_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize diffusion model weights
        if self.config.use_diffusion:
            for module in self.unet.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def _setup_gpu(self):
        """Initialize PyTorch GPU configuration for multi-GPU training."""
        if self.config.use_multi_gpu and torch.cuda.device_count() > 1:
            if self.config.use_distributed:
                dist.init_process_group(backend='nccl')
                self.device = torch.device(f'cuda:{dist.get_rank()}')
            else:
                self.device = torch.device('cuda:0')
    
    def _setup_profiler(self):
        """Initialize PyTorch profiler for performance monitoring."""
        if self.config.enable_profiling:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, 
                           torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                record_shapes=True,
                with_stack=True
            )
    
    def generate_seo_content(self, prompt: str, num_inference_steps: int = 50) -> torch.Tensor:
        """Generate SEO-optimized content using diffusion models."""
        if not self.config.use_diffusion:
            raise ValueError("Diffusion models not enabled in configuration")
        
        # Tokenize prompt
        tokenized_prompt = self.seo_tokenizer.tokenize_batch([prompt])
        prompt_embeddings = self.transformer_model(**tokenized_prompt).last_hidden_state
        
        # Initialize noise
        batch_size = 1
        latents = torch.randn(
            (batch_size, 4, 64, 64),
            device=self.device,
            dtype=prompt_embeddings.dtype
        )
        
        # Scale latents
        latents = latents * self.ddim_scheduler.init_noise_sigma
        
        # Set timesteps
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.ddim_scheduler.timesteps
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for batch processing
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.ddim_scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeddings
                ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.diffusion_guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            
            # Compute previous noisy sample
            latents = self.ddim_scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
    def forward(self, input_texts: List[str], y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass using LoRA fine-tuned transformer with diffusion models for SEO metrics."""
        # Tokenize input texts using SEO tokenizer
        tokenized_inputs = self.seo_tokenizer.tokenize_batch(input_texts)
        
        # Move to device
        for key, value in tokenized_inputs.items():
            tokenized_inputs[key] = value.to(self.device)
        
        # Get sequence lengths for proper handling
        sequence_lengths = self.seo_tokenizer.tokenizer.get_vocab()
        
        # Extract features using LoRA fine-tuned transformer
        transformer_outputs = self.transformer_model(**tokenized_inputs)
        pooled_output = transformer_outputs.pooler_output
        
        # Process through SEO classifier
        seo_features = self.seo_classifier(pooled_output)
        
        # Calculate metrics using PyTorch operations
        accuracy = (y_true == y_pred).float().mean()
        
        # Efficient PyTorch precision, recall, F1 calculation
        true_positives = ((y_true == 1) & (y_pred == 1)).sum().float()
        false_positives = ((y_true == 0) & (y_pred == 1)).sum().float()
        false_negatives = ((y_true == 1) & (y_pred == 0)).sum().float()
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'seo_features': seo_features,
            'transformer_outputs': transformer_outputs
        }
    
    def calculate_metrics_vectorized(self, input_texts: List[str], y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics using LoRA fine-tuned transformer with diffusion models."""
        with torch.no_grad():
            metrics_tensors = self.forward(input_texts, y_true, y_pred)
            return {key: value.item() if torch.is_tensor(value) and value.numel() == 1 else value 
                   for key, value in metrics_tensors.items()}
    
    def evaluate_with_specialized_metrics(self, input_texts: List[str], y_true: torch.Tensor, 
                                        y_pred: torch.Tensor, task_type: str = "classification") -> Dict[str, float]:
        """Evaluate using specialized SEO metrics appropriate for the specific task."""
        # Convert to numpy for sklearn metrics
        y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        
        # Get predictions for probability-based metrics
        with torch.no_grad():
            outputs = self.forward(input_texts, y_true, y_pred)
            seo_features = outputs['seo_features']
            y_prob = torch.softmax(seo_features, dim=1)[:, 1].cpu().numpy()  # Probability of positive class
        
        # Use specialized SEO evaluator
        if task_type == "classification":
            metrics = self.seo_evaluator.evaluate_classification(y_true_np, y_pred_np, y_prob)
        elif task_type == "regression":
            metrics = self.seo_evaluator.evaluate_regression(y_true_np, y_pred_np)
        elif task_type == "ranking":
            metrics = self.seo_evaluator.evaluate_ranking(y_true_np, y_pred_np)
        elif task_type == "clustering":
            # For clustering, we need features
            features = outputs['seo_features'].cpu().numpy()
            metrics = self.seo_evaluator.evaluate_clustering(features, y_pred_np)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Add SEO content evaluation
        if input_texts:
            seo_content_metrics = self.seo_evaluator.evaluate_seo_content(input_texts[0])
            metrics.update(seo_content_metrics)
        
        return metrics
    
    def generate_comprehensive_report(self, input_texts: List[str], y_true: torch.Tensor, 
                                   y_pred: torch.Tensor, task_type: str = "classification") -> str:
        """Generate comprehensive evaluation report using specialized metrics."""
        metrics = self.evaluate_with_specialized_metrics(input_texts, y_true, y_pred, task_type)
        return self.seo_evaluator.generate_evaluation_report(metrics, f"SEO {task_type.title()} Evaluation")

class SEODataset(Dataset):
    """Custom dataset for SEO evaluation with proper data handling."""
    
    def __init__(self, texts: List[str], labels: torch.Tensor, tokenizer: SEOTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        tokenized = self.tokenizer.tokenize_batch([text])
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': label
        }

class UltraOptimizedSEOTrainer:
    """Ultra-optimized trainer for SEO evaluation models with early stopping and learning rate scheduling."""
    
    def __init__(self, model: UltraOptimizedSEOMetricsModule, config: UltraOptimizedConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Optimizer for LoRA parameters and diffusion model
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        if config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.max_steps, eta_min=1e-6
            )
        elif config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1000, gamma=0.9
            )
        elif config.scheduler_type == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        elif config.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            verbose=True
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir='./runs/seo_evaluation')
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        # Setup AMP scaler
        self.scaler = GradScaler() if config.use_amp else None
    
    def _check_nan_inf(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check for NaN/Inf values in tensor."""
        if torch.isnan(tensor).any():
            logging.warning(f"NaN detected in {name}")
            return True
        if torch.isinf(tensor).any():
            logging.warning(f"Inf detected in {name}")
            return True
        return False
    
    def _handle_nan_inf(self, tensor: torch.Tensor, replacement_value: float = 0.0) -> torch.Tensor:
        """Handle NaN/Inf values by replacing them."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                               torch.tensor(replacement_value, device=tensor.device, dtype=tensor.dtype), 
                               tensor)
        return tensor
    
    def _clip_gradients(self, max_norm: float = None):
        """Clip gradients to prevent exploding gradients with comprehensive monitoring."""
        if max_norm is None:
            max_norm = self.config.max_grad_norm
        
        # Monitor gradient statistics before clipping
        grad_norm_before = self._get_gradient_norm()
        logger.info(f"Gradient norm before clipping: {grad_norm_before:.6f}")
        
        # Check for NaN/Inf in gradients before clipping
        nan_inf_detected = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if self._check_nan_inf(param.grad, f"gradient of {name}"):
                    param.grad = self._handle_nan_inf(param.grad)
                    nan_inf_detected = True
        
        if nan_inf_detected:
            logger.warning("NaN/Inf values detected and handled in gradients")
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        
        # Monitor gradient statistics after clipping
        grad_norm_after = self._get_gradient_norm()
        logger.info(f"Gradient norm after clipping: {grad_norm_after:.6f}")
        
        # Log to TensorBoard
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Training/GradNorm_Before', grad_norm_before, 
                                 len(self.train_history) if self.train_history else 0)
            self.writer.add_scalar('Training/GradNorm_After', grad_norm_after, 
                                 len(self.train_history) if self.train_history else 0)
    
    def _get_gradient_norm(self) -> float:
        """Calculate the total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def _check_training_stability(self, train_loss: float, val_loss: float):
        """Check for training instability and take corrective actions."""
        # Check for NaN/Inf in losses
        if math.isnan(train_loss) or math.isinf(train_loss):
            logger.error(f"Training loss is {train_loss}, training may be unstable")
            self._handle_training_instability("train_loss_nan_inf")
        
        if math.isnan(val_loss) or math.isinf(val_loss):
            logger.error(f"Validation loss is {val_loss}, training may be unstable")
            self._handle_training_instability("val_loss_nan_inf")
        
        # Check for loss explosion
        if len(self.train_history) > 1:
            prev_train_loss = self.train_history[-2]['loss']
            if train_loss > prev_train_loss * 10:  # Loss increased by 10x
                logger.warning(f"Training loss exploded from {prev_train_loss:.6f} to {train_loss:.6f}")
                self._handle_training_instability("loss_explosion")
        
        # Check for overfitting (validation loss increasing while training loss decreasing)
        if len(self.val_history) > 1 and len(self.train_history) > 1:
            prev_val_loss = self.val_history[-2]['loss']
            prev_train_loss = self.train_history[-2]['loss']
            
            if val_loss > prev_val_loss and train_loss < prev_train_loss:
                logger.warning("Potential overfitting detected")
                self._handle_training_instability("overfitting")
    
    def _handle_training_instability(self, issue_type: str):
        """Handle training instability issues."""
        if issue_type == "train_loss_nan_inf":
            logger.info("Reducing learning rate to handle NaN/Inf in training loss")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        elif issue_type == "val_loss_nan_inf":
            logger.info("Reducing learning rate to handle NaN/Inf in validation loss")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        elif issue_type == "loss_explosion":
            logger.info("Reducing learning rate to handle loss explosion")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        elif issue_type == "overfitting":
            logger.info("Increasing weight decay to handle overfitting")
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] *= 1.5
        
        # Log the corrective action
        current_lr = self.optimizer.param_groups[0]['lr']
        current_wd = self.optimizer.param_groups[0]['weight_decay']
        logger.info(f"Adjusted learning rate: {current_lr:.2e}, weight decay: {current_wd:.2e}")
        
        # Log to TensorBoard
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Training/CorrectiveAction', 1.0, 
                                 len(self.train_history) if self.train_history else 0)
            self.writer.add_scalar('Training/AdjustedLR', current_lr, 
                                 len(self.train_history) if self.train_history else 0)
            self.writer.add_scalar('Training/AdjustedWD', current_wd, 
                                 len(self.train_history) if self.train_history else 0)
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with gradient clipping and NaN/Inf handling."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with AMP if enabled
        if self.config.use_amp and self.scaler is not None:
            with autocast():
                outputs = self.model(
                    batch_data['input_texts'],
                    batch_data['y_true'],
                    batch_data['y_pred']
                )
                
                # Calculate loss
                loss = self.criterion(outputs['seo_features'], batch_data['labels'])
                
                # Add diffusion loss if enabled
                if self.config.use_diffusion:
                    diffusion_loss = self._calculate_diffusion_loss(batch_data, outputs)
                    loss += diffusion_loss
                
                # Check for NaN/Inf in loss
                if self._check_nan_inf(loss, "loss"):
                    logger.warning("NaN/Inf detected in loss, skipping step")
                    return {'loss': float('nan'), 'accuracy': 0.0, 'f1_score': 0.0}
            
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            
            # Handle NaN/Inf in gradients and clip
            self._clip_gradients()
            
            # Optimizer step with AMP
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                batch_data['input_texts'],
                batch_data['y_true'],
                batch_data['y_pred']
            )
            
            # Calculate loss
            loss = self.criterion(outputs['seo_features'], batch_data['labels'])
            
            # Add diffusion loss if enabled
            if self.config.use_diffusion:
                diffusion_loss = self._calculate_diffusion_loss(batch_data, outputs)
                loss += diffusion_loss
            
            # Check for NaN/Inf in loss
            if self._check_nan_inf(loss, "loss"):
                logger.warning("NaN/Inf detected in loss, skipping step")
                return {'loss': float('nan'), 'accuracy': 0.0, 'f1_score': 0.0}
            
            # Backward pass
            loss.backward()
            
            # Handle NaN/Inf in gradients and clip
            self._clip_gradients()
            
            # Optimizer step
            self.optimizer.step()
        
        # Update scheduler
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'accuracy': outputs['accuracy'].item(),
            'f1_score': outputs['f1_score'].item()
        }
    
    def _calculate_diffusion_loss(self, batch_data: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate diffusion model loss for SEO content generation."""
        # This is a simplified diffusion loss calculation
        # In practice, you would implement the full DDPM training loop
        return torch.tensor(0.0, device=self.device)
    
    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader, epoch: int):
        """Train for one epoch with validation."""
        self.model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_f1 = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Training step
            metrics = self.train_step(batch)
            
            train_loss += metrics['loss']
            train_accuracy += metrics['accuracy']
            train_f1 += metrics['f1_score']
            
            # Log to TensorBoard
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/Train', metrics['loss'], epoch * len(train_loader) + batch_idx)
                self.writer.add_scalar('Accuracy/Train', metrics['accuracy'], epoch * len(train_loader) + batch_idx)
                self.writer.add_scalar('F1/Train', metrics['f1_score'], epoch * len(train_loader) + batch_idx)
        
        # Calculate average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)
        avg_train_f1 = train_f1 / len(train_loader)
        
        # Validation
        val_metrics = self.validate(val_loader)
        
        # Log epoch metrics
        self.writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        self.writer.add_scalar('Accuracy/Train_Epoch', avg_train_accuracy, epoch)
        self.writer.add_scalar('F1/Train_Epoch', avg_train_f1, epoch)
        self.writer.add_scalar('Loss/Val_Epoch', val_metrics['loss'], epoch)
        self.writer.add_scalar('Accuracy/Val_Epoch', val_metrics['accuracy'], epoch)
        self.writer.add_scalar('F1/Val_Epoch', val_metrics['f1_score'], epoch)
        
        # Store history
        self.train_history.append({
            'epoch': epoch,
            'loss': avg_train_loss,
            'accuracy': avg_train_accuracy,
            'f1_score': avg_train_f1
        })
        
        self.val_history.append({
            'epoch': epoch,
            'loss': val_metrics['loss'],
            'accuracy': val_metrics['accuracy'],
            'f1_score': val_metrics['f1_score']
        })
        
        # Check for training instability
        self._check_training_stability(avg_train_loss, val_metrics['loss'])
        
        # Early stopping check
        self.early_stopping(val_metrics['loss'])
        
        return {
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'train_f1': avg_train_f1,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_score']
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model with NaN/Inf handling."""
        self.model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_f1 = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch['input_texts'],
                    batch['y_true'],
                    batch['y_pred']
                )
                
                # Calculate loss
                loss = self.criterion(outputs['seo_features'], batch['labels'])
                
                # Check for NaN/Inf in loss and outputs
                if self._check_nan_inf(loss, "validation loss") or \
                   self._check_nan_inf(outputs['accuracy'], "validation accuracy") or \
                   self._check_nan_inf(outputs['f1_score'], "validation f1_score"):
                    logger.warning("NaN/Inf detected in validation outputs, skipping batch")
                    continue
                
                val_loss += loss.item()
                val_accuracy += outputs['accuracy'].item()
                val_f1 += outputs['f1_score'].item()
                valid_batches += 1
        
        # Handle case where no valid batches were processed
        if valid_batches == 0:
            logger.error("No valid batches in validation, returning default values")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'f1_score': 0.0
            }
        
        return {
            'loss': val_loss / valid_batches,
            'accuracy': val_accuracy / valid_batches,
            'f1_score': val_f1 / valid_batches
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.writer.close()
    
    def save_checkpoint(self, path: str, include_optimizer: bool = True):
        """Save model checkpoint with NaN/Inf safety checks."""
        # Check model parameters for NaN/Inf before saving
        has_nan_inf = False
        for name, param in self.model.named_parameters():
            if self._check_nan_inf(param.data, f"parameter {name}"):
                logger.warning(f"NaN/Inf detected in parameter {name}, cleaning before save")
                param.data = self._handle_nan_inf(param.data)
                has_nan_inf = True
        
        if has_nan_inf:
            logger.warning("Model contained NaN/Inf values that were cleaned before saving")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint with NaN/Inf safety checks."""
        if not os.path.exists(path):
            logger.error(f"Checkpoint file {path} does not exist")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Check loaded parameters for NaN/Inf
            has_nan_inf = False
            for name, param in self.model.named_parameters():
                if self._check_nan_inf(param.data, f"loaded parameter {name}"):
                    logger.warning(f"NaN/Inf detected in loaded parameter {name}, cleaning")
                    param.data = self._handle_nan_inf(param.data)
                    has_nan_inf = True
            
            if has_nan_inf:
                logger.warning("Loaded checkpoint contained NaN/Inf values that were cleaned")
            
            # Load optimizer and scheduler if requested
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training history
            if 'train_history' in checkpoint:
                self.train_history = checkpoint['train_history']
            if 'val_history' in checkpoint:
                self.val_history = checkpoint['val_history']
            
            logger.info(f"Checkpoint loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics including NaN/Inf handling."""
        stats = {
            'training_history': self.train_history,
            'validation_history': self.val_history,
            'current_learning_rate': self.optimizer.param_groups[0]['lr'],
            'current_weight_decay': self.optimizer.param_groups[0]['weight_decay'],
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(self.device),
            'amp_enabled': self.config.use_amp and self.scaler is not None
        }
        
        # Add gradient statistics if available
        if hasattr(self, '_get_gradient_norm'):
            try:
                stats['current_gradient_norm'] = self._get_gradient_norm()
            except:
                stats['current_gradient_norm'] = None
        
        return stats
    
    def monitor_training_health(self) -> Dict[str, bool]:
        """Monitor the health of the training process."""
        health_status = {
            'model_healthy': True,
            'gradients_healthy': True,
            'losses_healthy': True,
            'parameters_healthy': True
        }
        
        # Check model parameters
        for name, param in self.model.named_parameters():
            if self._check_nan_inf(param.data, f"parameter {name}"):
                health_status['parameters_healthy'] = False
                logger.error(f"Unhealthy parameter detected: {name}")
        
        # Check gradients if they exist
        for name, param in self.model.named_parameters():
            if param.grad is not None and self._check_nan_inf(param.grad, f"gradient {name}"):
                health_status['gradients_healthy'] = False
                logger.error(f"Unhealthy gradient detected: {name}")
        
        # Check recent losses
        if self.train_history and self.val_history:
            recent_train_loss = self.train_history[-1]['loss']
            recent_val_loss = self.val_history[-1]['loss']
            
            if math.isnan(recent_train_loss) or math.isinf(recent_train_loss):
                health_status['losses_healthy'] = False
            if math.isnan(recent_val_loss) or math.isinf(recent_val_loss):
                health_status['losses_healthy'] = False
        
        # Overall health
        health_status['overall_healthy'] = all(health_status.values())
        
        return health_status

class EarlyStopping:
    """Early stopping implementation for training."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
        else:
            self.best_loss = val_loss
            self.counter = 0

def create_data_loaders(texts: List[str], labels: torch.Tensor, config: UltraOptimizedConfig, 
                       tokenizer: SEOTokenizer, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Create train/validation/test data loaders with proper splits."""
    # Split data
    train_size = int(train_ratio * len(texts))
    val_size = int(val_ratio * len(texts))
    test_size = len(texts) - train_size - val_size
    
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = train_test_split(
        texts, labels, test_size=val_size + test_size, random_state=42, stratify=labels
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        val_texts, val_labels, test_size=test_size, random_state=42, stratify=val_labels
    )
    
    # Create datasets
    train_dataset = SEODataset(train_texts, train_labels, tokenizer)
    val_dataset = SEODataset(val_texts, val_labels, tokenizer)
    test_dataset = SEODataset(test_texts, test_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Usage example
async def main():
    config = UltraOptimizedConfig(
        use_multi_gpu=True,
        use_lora=True,
        use_diffusion=True,
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
        max_length=512,
        scheduler_type="cosine",
        patience=10
    )
    
    model = UltraOptimizedSEOMetricsModule(config)
    trainer = UltraOptimizedSEOTrainer(model, config)
    
    # Example training data
    texts = [
        "<h1>SEO Optimization Guide</h1><p>Learn about search engine optimization techniques.</p>",
        "<meta name='description' content='Content quality analysis for better rankings'>",
        "How to improve your website's search engine ranking",
        "Best practices for SEO content creation"
    ]
    labels = torch.tensor([1, 0, 1, 1])
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        texts, labels, config, model.seo_tokenizer
    )
    
    # Training loop
    for epoch in range(config.num_epochs):
        metrics = trainer.train_epoch(train_loader, val_loader, epoch)
        print(f"Epoch {epoch}: {metrics}")
        
        # Check early stopping
        if trainer.early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Test diffusion content generation
    if config.use_diffusion:
        generated_content = model.generate_seo_content(
            "SEO optimization techniques for better search rankings",
            num_inference_steps=50
        )
        print(f"Generated content shape: {generated_content.shape}")
    
    # Test specialized evaluation metrics
    test_texts = texts[:2]
    test_y_true = labels[:2]
    test_y_pred = torch.randint(0, 2, (2,))
    
    # Evaluate with specialized metrics
    specialized_metrics = model.evaluate_with_specialized_metrics(
        test_texts, test_y_true, test_y_pred, "classification"
    )
    print("Specialized SEO Metrics:", specialized_metrics)
    
    # Generate comprehensive report
    report = model.generate_comprehensive_report(
        test_texts, test_y_true, test_y_pred, "classification"
    )
    print(report)
    
    # Cleanup
    trainer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
