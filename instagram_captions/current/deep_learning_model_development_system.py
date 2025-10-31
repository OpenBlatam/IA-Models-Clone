"""
Deep Learning and Model Development System
Implements comprehensive deep learning workflows and model development best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import numpy as np
from enum import Enum
import json
import yaml
import time
import os
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# DEEP LEARNING AND MODEL DEVELOPMENT SYSTEM
# =============================================================================

@dataclass
class ModelDevelopmentConfig:
    """Configuration for deep learning model development"""
    # Model architecture
    model_type: str = "transformer"  # transformer, cnn, rnn, hybrid
    architecture_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Optimization
    optimizer_type: str = "adamw"  # adam, sgd, adamw
    scheduler_type: str = "cosine"  # step, cosine, plateau
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Data configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Monitoring and logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    use_tensorboard: bool = True
    use_wandb: bool = False

class DeepLearningModelDevelopmentSystem:
    """Comprehensive system for deep learning model development"""
    
    def __init__(self, config: ModelDevelopmentConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup components
        self.device = self._setup_device()
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Logging setup
        self._setup_logging()
        
        self.logger.info("Deep Learning Model Development System initialized")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device for training"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU device")
        
        return device
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        if self.config.model_type == "transformer":
            return self._create_transformer_model()
        elif self.config.model_type == "cnn":
            return self._create_cnn_model()
        elif self.config.model_type == "rnn":
            return self._create_rnn_model()
        elif self.config.model_type == "hybrid":
            return self._create_hybrid_model()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _create_transformer_model(self) -> nn.Module:
        """Create transformer model"""
        config = self.config.architecture_config
        
        model = TransformerModel(
            vocab_size=config.get('vocab_size', 30000),
            d_model=config.get('d_model', 512),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 6),
            d_ff=config.get('d_ff', 2048),
            max_seq_len=config.get('max_seq_len', 512),
            dropout=config.get('dropout', 0.1)
        )
        
        return model
    
    def _create_cnn_model(self) -> nn.Module:
        """Create CNN model"""
        config = self.config.architecture_config
        
        model = CNNModel(
            input_channels=config.get('input_channels', 3),
            num_classes=config.get('num_classes', 1000),
            base_channels=config.get('base_channels', 64),
            num_layers=config.get('num_layers', 5)
        )
        
        return model
    
    def _create_rnn_model(self) -> nn.Module:
        """Create RNN model"""
        config = self.config.architecture_config
        
        model = RNNModel(
            input_size=config.get('input_size', 512),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 2),
            num_classes=config.get('num_classes', 1000),
            bidirectional=config.get('bidirectional', True)
        )
        
        return model
    
    def _create_hybrid_model(self) -> nn.Module:
        """Create hybrid model combining multiple architectures"""
        config = self.config.architecture_config
        
        model = HybridModel(
            transformer_config=config.get('transformer', {}),
            cnn_config=config.get('cnn', {}),
            rnn_config=config.get('rnn', {}),
            fusion_method=config.get('fusion_method', 'concatenate')
        )
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        if self.config.optimizer_type == "adam":
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif self.config.optimizer_type == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "step":
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
        elif self.config.scheduler_type == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        else:
            return None
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        if self.config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir='./runs')
        
        if self.config.use_wandb:
            import wandb
            wandb.init(project="deep-learning-model-development")
    
    def train_model(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
                   criterion: nn.Module) -> Dict[str, List[float]]:
        """Complete training loop for model development"""
        self.model.to(self.device)
        self.model.train()
        
        # Training history
        train_losses = []
        val_losses = []
        learning_rates = []
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            epoch_loss = self._train_epoch(train_dataloader, criterion)
            train_losses.append(epoch_loss)
            
            # Validate epoch
            val_loss = self._validate_epoch(val_dataloader, criterion)
            val_losses.append(val_loss)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # Log progress
            self._log_epoch_progress(epoch, epoch_loss, val_loss, current_lr)
            
            # Save best model
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self._save_checkpoint('best_model.pth')
            
            # Regular checkpointing
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates
        }
    
    def _train_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                
                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })
            
            # Log training metrics
            if self.global_step % self.config.log_interval == 0:
                self._log_training_metrics(loss.item())
        
        return total_loss / num_batches
    
    def _validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.config.mixed_precision and self.scaler is not None:
                    with autocast():
                        outputs = self.model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def _log_epoch_progress(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Log epoch progress"""
        self.logger.info(
            f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"LR: {lr:.6f}"
        )
        
        if self.config.use_tensorboard:
            self.tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch)
            self.tensorboard_writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.tensorboard_writer.add_scalar('Learning_Rate', lr, epoch)
        
        if self.config.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': lr
            })
    
    def _log_training_metrics(self, loss: float):
        """Log training metrics during training"""
        if self.config.use_tensorboard:
            self.tensorboard_writer.add_scalar('Loss/Training_Step', loss, self.global_step)
        
        if self.config.use_wandb:
            import wandb
            wandb.log({'training_step_loss': loss, 'global_step': self.global_step})
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        checkpoint_path = Path('./checkpoints') / filename
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def evaluate_model(self, test_dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_dataloader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.config.mixed_precision and self.scaler is not None:
                    with autocast():
                        outputs = self.model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions.append(outputs.cpu())
                targets_list.append(targets.cpu())
        
        # Calculate metrics
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets_list, dim=0)
        
        metrics = self._calculate_metrics(predictions, targets)
        metrics['test_loss'] = total_loss / len(test_dataloader)
        
        self.logger.info(f"Test Results: {metrics}")
        return metrics
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        if predictions.dim() > 1:
            # Classification task
            predicted_classes = torch.argmax(predictions, dim=1)
            accuracy = (predicted_classes == targets).float().mean().item()
            
            return {
                'accuracy': accuracy,
                'predictions_shape': list(predictions.shape),
                'targets_shape': list(targets.shape)
            }
        else:
            # Regression task
            mse = F.mse_loss(predictions, targets).item()
            mae = F.l1_loss(predictions, targets).item()
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(history['train_losses'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_losses'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(history['learning_rates'], color='green')
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(history['train_losses'], history['val_losses'])]
        axes[1, 0].plot(loss_diff, color='orange')
        axes[1, 0].set_title('Train-Val Loss Difference')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].grid(True)
        
        # Validation loss trend
        val_losses = history['val_losses']
        val_trend = [val_losses[i] - val_losses[i-1] if i > 0 else 0 for i in range(len(val_losses))]
        axes[1, 1].plot(val_trend, color='purple')
        axes[1, 1].set_title('Validation Loss Trend')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Change')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved: {save_path}")
        
        plt.show()
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()
        
        if self.config.use_wandb:
            import wandb
            wandb.finish()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class TransformerModel(nn.Module):
    """Transformer model for sequence processing"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x

class CNNModel(nn.Module):
    """CNN model for image processing"""
    
    def __init__(self, input_channels: int, num_classes: int, base_channels: int, num_layers: int):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate feature size
        feature_size = base_channels * (2 ** (num_layers - 1))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class RNNModel(nn.Module):
    """RNN model for sequence processing"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Use last output for classification
        if self.bidirectional:
            # Concatenate forward and backward outputs
            last_forward = lstm_out[:, -1, :self.hidden_size]
            last_backward = lstm_out[:, 0, self.hidden_size:]
            last_output = torch.cat([last_forward, last_backward], dim=1)
        else:
            last_output = lstm_out[:, -1, :]
        
        output = self.classifier(last_output)
        return output

class HybridModel(nn.Module):
    """Hybrid model combining multiple architectures"""
    
    def __init__(self, transformer_config: Dict, cnn_config: Dict, rnn_config: Dict,
                 fusion_method: str = 'concatenate'):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Initialize sub-models
        if transformer_config:
            self.transformer = TransformerModel(**transformer_config)
        else:
            self.transformer = None
        
        if cnn_config:
            self.cnn = CNNModel(**cnn_config)
        else:
            self.cnn = None
        
        if rnn_config:
            self.rnn = RNNModel(**rnn_config)
        else:
            self.rnn = None
        
        # Fusion layer
        self._setup_fusion_layer()
    
    def _setup_fusion_layer(self):
        """Setup fusion layer based on sub-models"""
        input_sizes = []
        
        if self.transformer:
            input_sizes.append(self.transformer.d_model)
        
        if self.cnn:
            # Get CNN output size
            cnn_output_size = self.cnn.classifier[-1].in_features
            input_sizes.append(cnn_output_size)
        
        if self.rnn:
            # Get RNN output size
            rnn_output_size = self.rnn.classifier[-1].in_features
            input_sizes.append(rnn_output_size)
        
        if len(input_sizes) > 1:
            if self.fusion_method == 'concatenate':
                fusion_input_size = sum(input_sizes)
            elif self.fusion_method == 'attention':
                fusion_input_size = max(input_sizes)
            else:
                fusion_input_size = max(input_sizes)
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 1000)  # Assuming 1000 classes
            )
        else:
            self.fusion_layer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        if self.transformer:
            transformer_out = self.transformer(x)
            outputs.append(transformer_out)
        
        if self.cnn:
            cnn_out = self.cnn(x)
            outputs.append(cnn_out)
        
        if self.rnn:
            rnn_out = self.rnn(x)
            outputs.append(rnn_out)
        
        if len(outputs) == 1:
            return outputs[0]
        
        # Fusion
        if self.fusion_method == 'concatenate':
            fused = torch.cat(outputs, dim=1)
        elif self.fusion_method == 'attention':
            # Simple attention fusion
            fused = torch.stack(outputs, dim=1).mean(dim=1)
        else:
            fused = torch.stack(outputs, dim=1).mean(dim=1)
        
        if self.fusion_layer:
            output = self.fusion_layer(fused)
        else:
            output = fused
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_deep_learning_example():
    """Example of using the Deep Learning Model Development System"""
    
    print("=== Deep Learning Model Development System Example ===")
    
    # Configuration
    config = ModelDevelopmentConfig(
        model_type="transformer",
        architecture_config={
            'vocab_size': 30000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 512,
            'dropout': 0.1
        },
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=5,
        mixed_precision=True,
        gradient_accumulation_steps=2
    )
    
    # Create system
    dl_system = DeepLearningModelDevelopmentSystem(config)
    
    # Create sample data
    batch_size = config.batch_size
    seq_len = 100
    vocab_size = config.architecture_config['vocab_size']
    
    sample_data = torch.randint(0, vocab_size, (batch_size, seq_len))
    sample_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(sample_data, sample_targets)
    val_dataset = torch.utils.data.TensorDataset(sample_data, sample_targets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    print("Starting training...")
    history = dl_system.train_model(train_loader, val_loader, criterion)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = dl_system.evaluate_model(val_loader, criterion)
    
    # Plot training history
    print("Plotting training history...")
    dl_system.plot_training_history(history)
    
    # Cleanup
    dl_system.cleanup()
    
    print("=== Example completed successfully ===")

def main():
    """Main function demonstrating deep learning model development"""
    create_deep_learning_example()

if __name__ == "__main__":
    main()


