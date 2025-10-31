from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import asyncio
"""
🧠 Deep Learning Models for Facebook Posts Processing
====================================================

This module implements advanced deep learning architectures with proper weight initialization,
normalization techniques, loss functions, and optimization algorithms for Facebook Posts analysis.

Key Features:
- Custom nn.Module architectures
- Advanced weight initialization techniques
- Multiple normalization layers
- Comprehensive loss functions
- Optimized training pipelines
- GPU acceleration support
- Mixed precision training
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters."""
    input_dim: int = 768
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    use_mixed_precision: bool = True


class WeightInitializer:
    """Advanced weight initialization techniques."""
    
    @staticmethod
    def xavier_uniform_init(module: nn.Module) -> None:
        """Xavier/Glorot uniform initialization."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def xavier_normal_init(module: nn.Module) -> None:
        """Xavier/Glorot normal initialization."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def kaiming_uniform_init(module: nn.Module) -> None:
        """Kaiming/He uniform initialization."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def kaiming_normal_init(module: nn.Module) -> None:
        """Kaiming/He normal initialization."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
        """Orthogonal initialization for RNNs and transformers."""
        if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @staticmethod
    def sparse_init(module: nn.Module, sparsity: float = 0.1) -> None:
        """Sparse initialization for regularization."""
        if isinstance(module, nn.Linear):
            nn.init.sparse_(module.weight, sparsity=sparsity)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class NormalizationLayers:
    """Advanced normalization techniques."""
    
    @staticmethod
    def layer_norm(dim: int, eps: float = 1e-5) -> nn.LayerNorm:
        """Layer normalization."""
        return nn.LayerNorm(dim, eps=eps)
    
    @staticmethod
    def batch_norm(dim: int, eps: float = 1e-5, momentum: float = 0.1) -> nn.BatchNorm1d:
        """Batch normalization."""
        return nn.BatchNorm1d(dim, eps=eps, momentum=momentum)
    
    @staticmethod
    def instance_norm(dim: int, eps: float = 1e-5, momentum: float = 0.1) -> nn.InstanceNorm1d:
        """Instance normalization."""
        return nn.InstanceNorm1d(dim, eps=eps, momentum=momentum)
    
    @staticmethod
    def group_norm(num_groups: int, num_channels: int, eps: float = 1e-5) -> nn.GroupNorm:
        """Group normalization."""
        return nn.GroupNorm(num_groups, num_channels, eps=eps)
    
    @staticmethod
    def adaptive_layer_norm(dim: int, eps: float = 1e-5) -> nn.LayerNorm:
        """Adaptive layer normalization with learnable parameters."""
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True)


class LossFunctions:
    """Comprehensive loss functions for different tasks."""
    
    @staticmethod
    def cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                          weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-entropy loss with optional class weights."""
        return F.cross_entropy(predictions, targets, weight=weight)
    
    @staticmethod
    def focal_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                   alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def dice_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                  smooth: float = 1e-6) -> torch.Tensor:
        """Dice loss for segmentation tasks."""
        predictions = torch.sigmoid(predictions)
        intersection = (predictions * targets).sum()
        dice_coeff = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
        return 1 - dice_coeff
    
    @staticmethod
    def huber_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                   delta: float = 1.0) -> torch.Tensor:
        """Huber loss for regression tasks."""
        return F.huber_loss(predictions, targets, delta=delta)
    
    @staticmethod
    def cosine_embedding_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                             margin: float = 0.0) -> torch.Tensor:
        """Cosine embedding loss for similarity learning."""
        return F.cosine_embedding_loss(predictions, targets, 
                                      torch.ones(predictions.size(0)).to(predictions.device), margin=margin)
    
    @staticmethod
    def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, 
                     margin: float = 1.0) -> torch.Tensor:
        """Triplet loss for metric learning."""
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        return loss.mean()


class OptimizerFactory:
    """Factory for creating optimizers with different algorithms."""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: ModelConfig) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if config.optimizer_type == "adam":
            return optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif config.optimizer_type == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif config.optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
                nesterov=True
            )
        elif config.optimizer_type == "rmsprop":
            return optim.RMSprop(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9,
                alpha=0.99
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer, config: ModelConfig) -> optim.lr_scheduler._LRScheduler:
        """Create scheduler based on configuration."""
        if config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        elif config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.max_epochs,
                eta_min=config.min_lr
            )
        elif config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.factor,
                patience=config.patience,
                verbose=True
            )
        elif config.scheduler_type == "warmup_cosine":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.learning_rate,
                epochs=config.max_epochs,
                steps_per_epoch=config.steps_per_epoch
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with proper initialization."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = NormalizationLayers.layer_norm(d_model)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize attention weights properly."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
        nn.init.zeros_(self.w_q.bias)
        nn.init.zeros_(self.w_k.bias)
        nn.init.zeros_(self.w_v.bias)
        nn.init.zeros_(self.w_o.bias)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        output = self.layer_norm(output + query)  # Residual connection
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with proper initialization and normalization."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.layer_norm1 = NormalizationLayers.layer_norm(d_model)
        self.layer_norm2 = NormalizationLayers.layer_norm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize transformer block weights."""
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class FacebookPostsTransformer(nn.Module):
    """Advanced transformer model for Facebook Posts processing."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        self.position_encoding = nn.Parameter(
            torch.randn(config.max_seq_length, config.input_dim)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.input_dim,
                config.num_heads,
                config.hidden_dim,
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(config.input_dim, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize all weights with proper techniques."""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Position encoding initialization
        nn.init.normal_(self.position_encoding, mean=0, std=0.02)
        
        # Output projection initialization
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        
        # Embeddings
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.position_encoding[:seq_length].unsqueeze(0)
        embeddings = self.dropout(embeddings)
        
        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)
        
        # Apply transformer blocks
        hidden_states = embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, attention_mask)
        
        # Global average pooling
        pooled_output = hidden_states.mean(dim=1)
        
        # Classification head
        logits = self.output_projection(pooled_output)
        
        return logits


class FacebookPostsLSTM(nn.Module):
    """LSTM model for Facebook Posts processing with proper initialization."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layers
        self.output_projection = nn.Linear(config.hidden_dim * 2, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize LSTM weights with orthogonal initialization."""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # LSTM initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Output projection initialization
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeddings
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(embeddings)
        
        # Use last hidden state from both directions
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Classification head
        logits = self.output_projection(last_hidden)
        
        return logits


class FacebookPostsCNN(nn.Module):
    """CNN model for Facebook Posts processing with proper initialization."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.input_dim)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(config.input_dim, config.hidden_dim, kernel_size=k)
            for k in [3, 4, 5]
        ])
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            NormalizationLayers.batch_norm(config.hidden_dim)
            for _ in range(3)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(config.hidden_dim * 3, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize CNN weights with proper techniques."""
        # Embedding initialization
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Convolutional layers initialization
        for conv_layer in self.conv_layers:
            nn.init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            nn.init.zeros_(conv_layer.bias)
        
        # Output projection initialization
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeddings
        embeddings = self.embedding(input_ids)  # [batch, seq, embed_dim]
        embeddings = embeddings.transpose(1, 2)  # [batch, embed_dim, seq]
        
        # Convolutional layers
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = conv(embeddings)  # [batch, hidden_dim, seq-k+1]
            conv_out = F.relu(bn(conv_out))
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))  # [batch, hidden_dim, 1]
            conv_out = conv_out.squeeze(2)  # [batch, hidden_dim]
            conv_outputs.append(conv_out)
        
        # Concatenate outputs
        concatenated = torch.cat(conv_outputs, dim=1)
        concatenated = self.dropout(concatenated)
        
        # Classification head
        logits = self.output_projection(concatenated)
        
        return logits


class FacebookPostsDataset(Dataset):
    """Custom dataset for Facebook Posts."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        
    """__init__ function."""
self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FacebookPostsTrainer:
    """Advanced trainer with proper optimization and loss functions."""
    
    def __init__(self, model: nn.Module, config: ModelConfig):
        
    """__init__ function."""
self.model = model.to(DEVICE)
        self.config = config
        
        # Initialize optimizer
        self.optimizer = OptimizerFactory.create_optimizer(model, config)
        
        # Initialize scheduler
        self.scheduler = SchedulerFactory.create_scheduler(self.optimizer, config)
        
        # Initialize loss function
        self.criterion = self._get_loss_function()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _get_loss_function(self) -> Optional[Dict[str, Any]]:
        """Get appropriate loss function based on task."""
        if self.config.task_type == "classification":
            if self.config.use_focal_loss:
                return lambda pred, target: LossFunctions.focal_loss(pred, target)
            else:
                return lambda pred, target: LossFunctions.cross_entropy_loss(pred, target)
        elif self.config.task_type == "regression":
            return lambda pred, target: LossFunctions.huber_loss(pred, target)
        else:
            raise ValueError(f"Unknown task type: {self.config.task_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Complete training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.config.max_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered")
                    break
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies


def create_facebook_posts_model(model_type: str, config: ModelConfig) -> nn.Module:
    """Factory function to create different model architectures."""
    if model_type == "transformer":
        return FacebookPostsTransformer(config)
    elif model_type == "lstm":
        return FacebookPostsLSTM(config)
    elif model_type == "cnn":
        return FacebookPostsCNN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage and demonstration
if __name__ == "__main__":
    # Configuration
    config = ModelConfig(
        input_dim=768,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=32,
        max_epochs=100,
        patience=10,
        gradient_clip=1.0,
        use_mixed_precision=True
    )
    
    # Create model
    model = create_facebook_posts_model("transformer", config)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created successfully!")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model device: {next(model.parameters()).device}") 