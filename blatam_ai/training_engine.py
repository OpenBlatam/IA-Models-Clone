"""
Blatam AI - Advanced Training Engine v6.0.0
Ultra-optimized PyTorch-based training system with custom nn.Module architectures
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import gc
import psutil
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM NN.MODULE ARCHITECTURES
# ============================================================================

class AdvancedTransformerBlock(nn.Module):
    """Advanced transformer block with multi-head attention and feed-forward layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class AdvancedTransformer(nn.Module):
    """Advanced transformer model with multiple transformer blocks."""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 512, 
                 n_layers: int = 6, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AdvancedTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get sequence length
        seq_len = x.size(1)
        
        # Token embeddings
        x = self.token_embedding(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
            
        # Output projection
        logits = self.output_projection(x)
        
        return logits

class AdvancedCNN(nn.Module):
    """Advanced CNN architecture with residual connections and batch normalization."""
    
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 1000,
                 base_channels: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, blocks=2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_channels * 8, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for CNN architectures."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out

class AdvancedRNN(nn.Module):
    """Advanced RNN architecture with LSTM and attention."""
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int = 2,
                 num_classes: int = 10,
                 dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        if bidirectional:
            attention_input_size = hidden_size * 2
        else:
            attention_input_size = hidden_size
            
        self.attention = nn.Sequential(
            nn.Linear(attention_input_size, attention_input_size // 2),
            nn.Tanh(),
            nn.Linear(attention_input_size // 2, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attention_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(attended_output)
        
        return output

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Advanced configuration for model training."""
    
    # Model configuration
    model_type: str = "transformer"  # transformer, cnn, rnn
    model_params: Dict[str, Any] = None
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd, lion
    scheduler: str = "cosine"  # cosine, linear, step, exponential
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Data configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    eval_strategy: str = "steps"  # steps, epoch
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    early_stopping_mode: str = "min"  # min, max
    
    # Monitoring
    use_tensorboard: bool = True
    use_wandb: bool = False
    use_mlflow: bool = False
    log_to_console: bool = True
    
    # Cross-validation
    cross_validation_folds: int = 5
    cross_validation_strategy: str = "stratified"  # stratified, kfold
    
    # Performance optimization
    enable_amp: bool = True
    enable_xformers: bool = True
    enable_compile: bool = True
    enable_ddp: bool = False
    
    # Output configuration
    output_dir: str = "./outputs"
    save_best_model: bool = True
    save_last_model: bool = True
    save_checkpoints: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.model_params is None:
            self.model_params = {}
            
        # Validate split ratios
        total_ratio = self.train_split + self.val_split + self.test_split
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

# ============================================================================
# ADVANCED TRAINING MANAGER
# ============================================================================

class AdvancedTrainingManager:
    """Advanced training manager with comprehensive features."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.training_history = defaultdict(list)
        
        # Monitoring
        self.writer = None
        self.early_stopping = None
        self.metrics_tracker = MetricsTracker()
        
        # Performance tracking
        self.start_time = None
        self.performance_metrics = {}
        
        self._setup_logging()
        self._setup_monitoring()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=f"{self.config.output_dir}/tensorboard")
                self.logger.info("TensorBoard logging enabled")
            except ImportError:
                self.logger.warning("TensorBoard not available, disabling")
                
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(project="blatam-ai", config=vars(self.config))
                self.logger.info("Weights & Biases logging enabled")
            except ImportError:
                self.logger.warning("Weights & Biases not available, disabling")
                
    def _setup_monitoring(self):
        """Setup monitoring and early stopping."""
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            threshold=self.config.early_stopping_threshold,
            mode=self.config.early_stopping_mode
        )
        
    def setup_training(self, model: nn.Module, train_dataset: Dataset, 
                      val_dataset: Dataset, test_dataset: Dataset):
        """Setup complete training pipeline."""
        self.model = model.to(self.device)
        
        # Setup data loaders
        self._setup_data_loaders(train_dataset, val_dataset, test_dataset)
        
        # Setup training components
        self._setup_training_components()
        
        # Setup performance optimizations
        self._setup_performance_optimizations()
        
        self.logger.info("Training setup completed")
        
    def _setup_data_loaders(self, train_dataset: Dataset, val_dataset: Dataset, 
                           test_dataset: Dataset):
        """Setup data loaders with proper configuration."""
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
            
        # Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        if self.config.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        elif self.config.scheduler == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
            )
        elif self.config.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=total_steps // 10, gamma=0.1
            )
            
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        if self.config.enable_amp:
            self.scaler = GradScaler()
            
    def _setup_performance_optimizations(self):
        """Setup performance optimizations."""
        if self.config.enable_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            self.logger.info("Model compilation enabled")
            
        if self.config.enable_ddp and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.logger.info("DataParallel enabled")

# ============================================================================
# UTILITY CLASSES
# ============================================================================

class EarlyStopping:
    """Early stopping mechanism for training."""
    
    def __init__(self, patience: int = 10, threshold: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.counter = 0
        self.best_score = None
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_score is None:
            self.best_score = val_loss
        elif self.mode == "min":
            if val_loss < self.best_score - self.threshold:
                self.best_score = val_loss
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if val_loss > self.best_score + self.threshold:
                self.best_score = val_loss
                self.counter = 0
            else:
                self.counter += 1
                
        return self.counter >= self.patience

class MetricsTracker:
    """Track and manage training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def update(self, metric_name: str, value: float):
        """Update a metric."""
        self.metrics[metric_name].append(value)
        
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
        
    def get_average(self, metric_name: str, window: int = 10) -> Optional[float]:
        """Get the average value for a metric over a window."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            values = self.metrics[metric_name][-window:]
            return sum(values) / len(values)
        return None

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_model(config: TrainingConfig) -> nn.Module:
    """Create model based on configuration."""
    if config.model_type == "transformer":
        model_params = {
            "vocab_size": 10000,
            "d_model": 512,
            "n_layers": 6,
            "n_heads": 8,
            "d_ff": 2048,
            "max_seq_len": 512,
            "dropout": 0.1
        }
        model_params.update(config.model_params)
        return AdvancedTransformer(**model_params)
        
    elif config.model_type == "cnn":
        model_params = {
            "in_channels": 3,
            "num_classes": 1000,
            "base_channels": 64
        }
        model_params.update(config.model_params)
        return AdvancedCNN(**model_params)
        
    elif config.model_type == "rnn":
        model_params = {
            "input_size": 100,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": 10,
            "dropout": 0.1,
            "bidirectional": True
        }
        model_params.update(config.model_params)
        return AdvancedRNN(**model_params)
        
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

def create_training_manager(config: TrainingConfig) -> AdvancedTrainingManager:
    """Create training manager with configuration."""
    return AdvancedTrainingManager(config)

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

class AdvancedTrainingLoop:
    """Advanced training loop with comprehensive features."""
    
    def __init__(self, training_manager: AdvancedTrainingManager):
        self.manager = training_manager
        self.config = training_manager.config
        self.logger = training_manager.logger
        
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.manager.start_time = time.time()
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        best_model_state = None
        training_summary = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        for epoch in range(self.config.num_epochs):
            self.manager.current_epoch = epoch
            
            # Training epoch
            train_loss, train_metrics = self._train_epoch()
            
            # Validation epoch
            val_loss, val_metrics = self._validate_epoch()
            
            # Update learning rate
            if self.manager.scheduler:
                self.manager.scheduler.step()
                
            # Log metrics
            self._log_epoch_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Update training history
            training_summary['train_losses'].append(train_loss)
            training_summary['val_losses'].append(val_loss)
            training_summary['train_metrics'].append(train_metrics)
            training_summary['val_metrics'].append(val_metrics)
            training_summary['learning_rates'].append(
                self.manager.optimizer.param_groups[0]['lr']
            )
            
            # Save checkpoint
            if self.config.save_checkpoints and (epoch + 1) % self.config.save_steps == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)
                
            # Early stopping check
            if self.manager.early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
                
            # Save best model
            if val_loss < self.manager.best_metric:
                self.manager.best_metric = val_loss
                best_model_state = self.manager.model.state_dict().copy()
                if self.config.save_best_model:
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                    
        # Restore best model
        if best_model_state:
            self.manager.model.load_state_dict(best_model_state)
            
        # Final evaluation
        test_metrics = self._evaluate_model()
        training_summary['test_metrics'] = test_metrics
        training_summary['total_training_time'] = time.time() - self.manager.start_time
        
        return training_summary
        
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.manager.model.train()
        total_loss = 0.0
        epoch_metrics = defaultdict(list)
        
        for batch_idx, (data, targets) in enumerate(self.manager.train_loader):
            data, targets = data.to(self.manager.device), targets.to(self.manager.device)
            
            # Forward pass with mixed precision
            if self.config.enable_amp:
                with autocast():
                    outputs = self.manager.model(data)
                    loss = self.manager.criterion(outputs, targets)
            else:
                outputs = self.manager.model(data)
                loss = self.manager.criterion(outputs, targets)
                
            # Backward pass
            if self.config.enable_amp:
                self.manager.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    if self.config.enable_amp:
                        self.manager.scaler.unscale_(self.manager.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.manager.model.parameters(),
                        self.config.gradient_clip
                    )
                    
                # Optimizer step
                if self.config.enable_amp:
                    self.manager.scaler.step(self.manager.optimizer)
                    self.manager.scaler.update()
                else:
                    self.manager.optimizer.step()
                self.manager.optimizer.zero_grad()
                
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == targets).float().mean().item()
            
            epoch_metrics['accuracy'].append(accuracy)
            epoch_metrics['loss'].append(loss.item())
            
            # Update progress
            if batch_idx % self.config.logging_steps == 0:
                self.logger.info(
                    f"Epoch {self.manager.current_epoch + 1}, "
                    f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                    f"Accuracy: {accuracy:.4f}"
                )
                
            self.manager.global_step += 1
            
        avg_loss = total_loss / len(self.manager.train_loader)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
        
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.manager.model.eval()
        total_loss = 0.0
        epoch_metrics = defaultdict(list)
        
        with torch.no_grad():
            for data, targets in self.manager.val_loader:
                data, targets = data.to(self.manager.device), targets.to(self.manager.device)
                
                # Forward pass
                if self.config.enable_amp:
                    with autocast():
                        outputs = self.manager.model(data)
                        loss = self.manager.criterion(outputs, targets)
                else:
                    outputs = self.manager.model(data)
                    loss = self.manager.criterion(outputs, targets)
                    
                # Update metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == targets).float().mean().item()
                
                epoch_metrics['accuracy'].append(accuracy)
                epoch_metrics['loss'].append(loss.item())
                
        avg_loss = total_loss / len(self.manager.val_loader)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
        
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.manager.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, targets in self.manager.test_loader:
                data, targets = data.to(self.manager.device), targets.to(self.manager.device)
                
                outputs = self.manager.model(data)
                loss = self.manager.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            all_predictions, all_targets, 
            total_loss / len(self.manager.test_loader)
        )
        
        return metrics
        
    def _calculate_comprehensive_metrics(self, predictions: List[int], 
                                       targets: List[int], loss: float) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, roc_auc_score, average_precision_score
        )
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        metrics = {
            'test_loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
        
    def _log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float,
                           train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics for current epoch."""
        if self.manager.writer:
            self.manager.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.manager.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.manager.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.manager.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            
        if self.config.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_metrics['accuracy'],
                'val_accuracy': val_metrics['accuracy']
            })
            
        self.logger.info(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
        
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.manager.model.state_dict(),
            'optimizer_state_dict': self.manager.optimizer.state_dict(),
            'scheduler_state_dict': self.manager.scheduler.state_dict() if self.manager.scheduler else None,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = f"{self.config.output_dir}/checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = f"{self.config.output_dir}/best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
            
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Main training example."""
    # Configuration
    config = TrainingConfig(
        model_type="transformer",
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-4,
        mixed_precision=True,
        enable_compile=True
    )
    
    # Create model
    model = create_model(config)
    
    # Create training manager
    training_manager = create_training_manager(config)
    
    # Create training loop
    training_loop = AdvancedTrainingLoop(training_manager)
    
    # Setup training (assuming datasets are available)
    # training_manager.setup_training(model, train_dataset, val_dataset, test_dataset)
    
    # Start training
    # training_summary = training_loop.train()
    
    print("Training engine ready!")

if __name__ == "__main__":
    main()

