from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json
from enum import Enum
    from transformers import (
    import sentence_transformers
    from sentence_transformers import SentenceTransformer, losses
                    import bitsandbytes as bnb
                    from transformers import BitsAndBytesConfig
            import torch
        from torch.cuda.amp import autocast
            import torch
        from torch.cuda.amp import autocast
from typing import Any, List, Dict, Optional
import asyncio
"""
Deep Learning Models for Blog Analysis
=====================================

Advanced deep learning architectures and training pipelines for blog content analysis.
Implements state-of-the-art models with proper GPU utilization and mixed precision training.
"""



# Transformers and NLP libraries
try:
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoConfig, Trainer, TrainingArguments,
        DataCollatorWithPadding, EarlyStoppingCallback
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelArchitecture(Enum):
    """Available model architectures."""
    BERT = "bert"
    DISTILBERT = "distilbert"
    ROBERTA = "roberta"
    ELECTRA = "electra"
    DEBERTA = "deberta"
    CUSTOM_TRANSFORMER = "custom_transformer"
    LSTM_ATTENTION = "lstm_attention"
    CNN_LSTM = "cnn_lstm"
    HYBRID = "hybrid"


class TaskType(Enum):
    """Available task types."""
    SENTIMENT_CLASSIFICATION = "sentiment_classification"
    QUALITY_REGRESSION = "quality_regression"
    READABILITY_REGRESSION = "readability_regression"
    KEYWORD_EXTRACTION = "keyword_extraction"
    MULTI_TASK = "multi_task"


@dataclass
class ModelConfig:
    """Configuration for deep learning models."""
    architecture: ModelArchitecture
    task_type: TaskType
    model_name: str = "distilbert-base-uncased"
    num_classes: int = 2
    hidden_size: int = 768
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_length: int = 512
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    
    # Advanced training settings
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    use_early_stopping: bool = True
    early_stopping_patience: int = 5
    use_lr_scheduling: bool = True
    scheduler_type: str = "cosine"  # "cosine", "plateau", "linear"
    
    # Multi-GPU settings
    use_distributed_training: bool = False
    num_gpus: int = 1
    distributed_backend: str = "nccl"
    
    # Model-specific settings
    use_attention_mechanism: bool = True
    attention_heads: int = 8
    num_layers: int = 6
    bidirectional: bool = True
    lstm_hidden_size: int = 256
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    
    def __post_init__(self) -> Any:
        """Validate and set default values."""
        if self.num_gpus > 1 and torch.cuda.is_available():
            self.use_distributed_training = True
        
        if self.use_mixed_precision and not torch.cuda.is_available():
            self.use_mixed_precision = False
            logger.warning("Mixed precision disabled - CUDA not available")


class BlogDataset(Dataset):
    """Custom dataset for blog content analysis."""
    
    def __init__(self, texts: List[str], labels: Optional[List[Union[int, float]]] = None,
                 tokenizer=None, max_length: int = 512, task_type: TaskType = TaskType.SENTIMENT_CLASSIFICATION):
        
    """__init__ function."""
self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        
        # Validate inputs
        if labels is not None and len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        # Tokenize text
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Remove batch dimension
            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }
            
            if 'token_type_ids' in encoding:
                item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        else:
            # Fallback for non-tokenizer models
            item = {'text': text}
        
        # Add labels if available
        if self.labels is not None:
            label = self.labels[idx]
            if self.task_type == TaskType.SENTIMENT_CLASSIFICATION:
                item['labels'] = torch.tensor(label, dtype=torch.long)
            else:
                item['labels'] = torch.tensor(label, dtype=torch.float)
        
        return item


class AttentionMechanism(nn.Module):
    """Multi-head attention mechanism for custom models."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_size)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Output projection and residual connection
        output = self.output(context)
        output = self.layer_norm(output + x)
        
        return output


class CustomTransformerModel(nn.Module):
    """Custom transformer model for blog analysis, now optimized to use HuggingFace Transformers if available and bitsandbytes for quantization if available."""
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.use_hf = False
        self.use_bnb = False
        # Si HuggingFace Transformers está disponible, usa modelo preentrenado
        if TRANSFORMERS_AVAILABLE:
            try:
                # bitsandbytes para cuantización 8-bit
                try:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.transformer = AutoModel.from_pretrained(config.model_name, quantization_config=bnb_config)
                    self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                    self.use_bnb = True
                except Exception:
                    self.transformer = AutoModel.from_pretrained(config.model_name)
                    self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self.use_hf = True
            except Exception as e:
                logger.warning(f"Fallo al cargar modelo HuggingFace: {e}. Usando implementación custom.")
                self.use_hf = False
        if not self.use_hf:
            self.embedding = nn.Embedding(30000, config.hidden_size)
            self.position_encoding = nn.Embedding(config.max_length, config.hidden_size)
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.attention_heads,
                    dim_feedforward=config.hidden_size * 4,
                    dropout=config.dropout_rate,
                    batch_first=True
                ) for _ in range(config.num_layers)
            ])
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.task_type == TaskType.SENTIMENT_CLASSIFICATION:
            self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        elif config.task_type == TaskType.QUALITY_REGRESSION:
            self.regressor = nn.Linear(config.hidden_size, 1)
        elif config.task_type == TaskType.MULTI_TASK:
            self.sentiment_classifier = nn.Linear(config.hidden_size, 2)
            self.quality_regressor = nn.Linear(config.hidden_size, 1)
            self.readability_regressor = nn.Linear(config.hidden_size, 1)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.use_hf:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        else:
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            embeddings = self.embedding(input_ids) + self.position_encoding(positions)
            hidden_states = embeddings
            for layer in self.transformer_layers:
                hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask == 0 if attention_mask is not None else None)
            if attention_mask is not None:
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                pooled_output = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled_output = hidden_states.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        outputs = {}
        if self.config.task_type == TaskType.SENTIMENT_CLASSIFICATION:
            outputs['logits'] = self.classifier(pooled_output)
        elif self.config.task_type == TaskType.QUALITY_REGRESSION:
            outputs['predictions'] = self.regressor(pooled_output).squeeze(-1)
        elif self.config.task_type == TaskType.MULTI_TASK:
            outputs['sentiment_logits'] = self.sentiment_classifier(pooled_output)
            outputs['quality_predictions'] = self.quality_regressor(pooled_output).squeeze(-1)
            outputs['readability_predictions'] = self.readability_regressor(pooled_output).squeeze(-1)
        return outputs


class LSTMAttentionModel(nn.Module):
    """LSTM with attention mechanism for blog analysis. Optimized con torch.compile y autocast."""
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.embedding = nn.Embedding(30000, config.hidden_size)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=2,
            bidirectional=config.bidirectional,
            dropout=config.dropout_rate if 2 > 1 else 0,
            batch_first=True
        )
        if config.use_attention_mechanism:
            lstm_output_size = config.lstm_hidden_size * 2 if config.bidirectional else config.lstm_hidden_size
            self.attention = AttentionMechanism(lstm_output_size, config.attention_heads, config.dropout_rate)
        self.dropout = nn.Dropout(config.dropout_rate)
        lstm_output_size = config.lstm_hidden_size * 2 if config.bidirectional else config.lstm_hidden_size
        if config.task_type == TaskType.SENTIMENT_CLASSIFICATION:
            self.classifier = nn.Linear(lstm_output_size, config.num_classes)
        elif config.task_type == TaskType.QUALITY_REGRESSION:
            self.regressor = nn.Linear(lstm_output_size, 1)
        elif config.task_type == TaskType.MULTI_TASK:
            self.sentiment_classifier = nn.Linear(lstm_output_size, 2)
            self.quality_regressor = nn.Linear(lstm_output_size, 1)
            self.readability_regressor = nn.Linear(lstm_output_size, 1)
        # torch.compile para acelerar si está disponible
        try:
            if hasattr(torch, 'compile'):
                self.forward = torch.compile(self.forward)
        except Exception:
            pass

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        with autocast(enabled=torch.cuda.is_available()):
            embeddings = self.embedding(input_ids)
            lstm_output, (hidden, cell) = self.lstm(embeddings)
            if self.config.use_attention_mechanism:
                lstm_output = self.attention(lstm_output, attention_mask)
            if attention_mask is not None:
                masked_output = lstm_output * attention_mask.unsqueeze(-1)
                pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled_output = lstm_output.mean(dim=1)
            pooled_output = self.dropout(pooled_output)
            outputs = {}
            if self.config.task_type == TaskType.SENTIMENT_CLASSIFICATION:
                outputs['logits'] = self.classifier(pooled_output)
            elif self.config.task_type == TaskType.QUALITY_REGRESSION:
                outputs['predictions'] = self.regressor(pooled_output).squeeze(-1)
            elif self.config.task_type == TaskType.MULTI_TASK:
                outputs['sentiment_logits'] = self.sentiment_classifier(pooled_output)
                outputs['quality_predictions'] = self.quality_regressor(pooled_output).squeeze(-1)
                outputs['readability_predictions'] = self.readability_regressor(pooled_output).squeeze(-1)
            return outputs


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for blog analysis. Optimized con torch.compile y autocast."""
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.embedding = nn.Embedding(30000, config.hidden_size)
        self.conv_layers = nn.ModuleList()
        for i, (filters, kernel_size) in enumerate(zip(config.cnn_filters, config.cnn_kernel_sizes)):
            in_channels = config.hidden_size match i:
    case 0 else config.cnn_filters[i-1]
            conv = nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv)
        lstm_input_size = config.cnn_filters[-1]
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=2,
            bidirectional=config.bidirectional,
            dropout=config.dropout_rate if 2 > 1 else 0,
            batch_first=True
        )
        if config.use_attention_mechanism:
            lstm_output_size = config.lstm_hidden_size * 2 if config.bidirectional else config.lstm_hidden_size
            self.attention = AttentionMechanism(lstm_output_size, config.attention_heads, config.dropout_rate)
        self.dropout = nn.Dropout(config.dropout_rate)
        lstm_output_size = config.lstm_hidden_size * 2 if config.bidirectional else config.lstm_hidden_size
        if config.task_type == TaskType.SENTIMENT_CLASSIFICATION:
            self.classifier = nn.Linear(lstm_output_size, config.num_classes)
        elif config.task_type == TaskType.QUALITY_REGRESSION:
            self.regressor = nn.Linear(lstm_output_size, 1)
        elif config.task_type == TaskType.MULTI_TASK:
            self.sentiment_classifier = nn.Linear(lstm_output_size, 2)
            self.quality_regressor = nn.Linear(lstm_output_size, 1)
            self.readability_regressor = nn.Linear(lstm_output_size, 1)
        # torch.compile para acelerar si está disponible
        try:
            if hasattr(torch, 'compile'):
                self.forward = torch.compile(self.forward)
        except Exception:
            pass

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        with autocast(enabled=torch.cuda.is_available()):
            embeddings = self.embedding(input_ids)
            # Conv1d espera (batch, channels, seq_len)
            x = embeddings.transpose(1, 2)
            for conv in self.conv_layers:
                x = conv(x)
                x = F.relu(x)
            x = x.transpose(1, 2)  # Regresa a (batch, seq_len, features)
            lstm_output, (hidden, cell) = self.lstm(x)
            if self.config.use_attention_mechanism:
                lstm_output = self.attention(lstm_output, attention_mask)
            if attention_mask is not None:
                masked_output = lstm_output * attention_mask.unsqueeze(-1)
                pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled_output = lstm_output.mean(dim=1)
            pooled_output = self.dropout(pooled_output)
            outputs = {}
            if self.config.task_type == TaskType.SENTIMENT_CLASSIFICATION:
                outputs['logits'] = self.classifier(pooled_output)
            elif self.config.task_type == TaskType.QUALITY_REGRESSION:
                outputs['predictions'] = self.regressor(pooled_output).squeeze(-1)
            elif self.config.task_type == TaskType.MULTI_TASK:
                outputs['sentiment_logits'] = self.sentiment_classifier(pooled_output)
                outputs['quality_predictions'] = self.quality_regressor(pooled_output).squeeze(-1)
                outputs['readability_predictions'] = self.readability_regressor(pooled_output).squeeze(-1)
            return outputs


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_model(config: ModelConfig) -> nn.Module:
        """Create model based on configuration."""
        if config.architecture == ModelArchitecture.CUSTOM_TRANSFORMER:
            return CustomTransformerModel(config)
        elif config.architecture == ModelArchitecture.LSTM_ATTENTION:
            return LSTMAttentionModel(config)
        elif config.architecture == ModelArchitecture.CNN_LSTM:
            return CNNLSTMModel(config)
        elif config.architecture in [ModelArchitecture.BERT, ModelArchitecture.DISTILBERT, 
                                   ModelArchitecture.ROBERTA, ModelArchitecture.ELECTRA, 
                                   ModelArchitecture.DEBERTA]:
            return ModelFactory._create_pretrained_model(config)
        else:
            raise ValueError(f"Unsupported architecture: {config.architecture}")
    
    @staticmethod
    def _create_pretrained_model(config: ModelConfig) -> nn.Module:
        """Create pretrained transformer model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        if config.task_type == TaskType.SENTIMENT_CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_classes,
                hidden_dropout_prob=config.dropout_rate,
                attention_probs_dropout_prob=config.dropout_rate
            )
        elif config.task_type == TaskType.QUALITY_REGRESSION:
            # Use sequence classification for regression (single output)
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=1,
                hidden_dropout_prob=config.dropout_rate,
                attention_probs_dropout_prob=config.dropout_rate
            )
        elif config.task_type == TaskType.MULTI_TASK:
            # Create custom multi-task wrapper
            base_model = AutoModel.from_pretrained(config.model_name)
            return MultiTaskTransformerModel(base_model, config)
        else:
            raise ValueError(f"Unsupported task type: {config.task_type}")
        
        return model


class MultiTaskTransformerModel(nn.Module):
    """Multi-task transformer model wrapper."""
    
    def __init__(self, base_model: nn.Module, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Task-specific heads
        self.sentiment_classifier = nn.Linear(config.hidden_size, 2)
        self.quality_regressor = nn.Linear(config.hidden_size, 1)
        self.readability_regressor = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Task-specific predictions
        return {
            'sentiment_logits': self.sentiment_classifier(pooled_output),
            'quality_predictions': self.quality_regressor(pooled_output).squeeze(-1),
            'readability_predictions': self.readability_regressor(pooled_output).squeeze(-1)
        }


class DeepLearningTrainer:
    """Advanced trainer for deep learning models."""
    
    def __init__(self, config: ModelConfig, model: nn.Module, tokenizer=None):
        
    """__init__ function."""
self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup distributed training if needed
        if config.use_distributed_training and torch.cuda.device_count() > 1:
            self._setup_distributed_training()
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision
        if config.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _setup_distributed_training(self) -> Any:
        """Setup distributed training."""
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    def _setup_optimizer_and_scheduler(self) -> Any:
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        
        # Scheduler
        if self.config.use_lr_scheduling:
            if self.config.scheduler_type == "cosine":
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
            elif self.config.scheduler_type == "plateau":
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3)
            else:
                self.scheduler = None
        else:
            self.scheduler = None
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss based on task type."""
        if self.config.task_type == TaskType.SENTIMENT_CLASSIFICATION:
            return F.cross_entropy(outputs['logits'], labels['labels'])
        elif self.config.task_type == TaskType.QUALITY_REGRESSION:
            return F.mse_loss(outputs['predictions'], labels['labels'])
        elif self.config.task_type == TaskType.MULTI_TASK:
            sentiment_loss = F.cross_entropy(outputs['sentiment_logits'], labels['sentiment_labels'])
            quality_loss = F.mse_loss(outputs['quality_predictions'], labels['quality_labels'])
            readability_loss = F.mse_loss(outputs['readability_predictions'], labels['readability_labels'])
            return sentiment_loss + quality_loss + readability_loss
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                    loss = self._compute_loss(outputs, {'labels': batch['labels']})
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                loss = self._compute_loss(outputs, {'labels': batch['labels']})
                
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                        loss = self._compute_loss(outputs, {'labels': batch['labels']})
                else:
                    outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                    loss = self._compute_loss(outputs, {'labels': batch['labels']})
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, List[float]]:
        """Complete training loop."""
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.config.use_early_stopping:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {path}")


# Example usage and testing
def main():
    """Example usage of the deep learning system."""
    print("🚀 Deep Learning Models for Blog Analysis")
    print("=" * 50)
    
    # Configuration
    config = ModelConfig(
        architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
        task_type=TaskType.SENTIMENT_CLASSIFICATION,
        num_classes=2,
        hidden_size=256,
        num_layers=4,
        attention_heads=8,
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=5,
        use_mixed_precision=True,
        use_gradient_accumulation=True
    )
    
    # Create model
    model = ModelFactory.create_model(config)
    print(f"Model created: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample data
    sample_texts = [
        "This is an excellent blog post about technology.",
        "I didn't like this article at all.",
        "This is a neutral article with facts.",
        "Amazing content with great insights!",
        "Terrible writing and poor structure."
    ]
    
    sample_labels = [1, 0, 1, 1, 0]  # 1 for positive, 0 for negative
    
    # Create dataset
    dataset = BlogDataset(sample_texts, sample_labels, max_length=128, task_type=config.task_type)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create trainer
    trainer = DeepLearningTrainer(config, model)
    
    # Train model
    print("\n📊 Starting training...")
    training_history = trainer.train(train_dataset, val_dataset)
    
    print(f"Training completed!")
    print(f"Final train loss: {training_history['train_losses'][-1]:.4f}")
    print(f"Final val loss: {training_history['val_losses'][-1]:.4f}")
    
    # Save model
    trainer.save_model("blog_analysis_model.pth")
    
    print("\n✅ Deep learning model training completed!")


match __name__:
    case "__main__":
    main() 