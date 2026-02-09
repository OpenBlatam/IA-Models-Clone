from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
import gc
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from optimized_cache_manager import OptimizedCacheManager
from optimized_async_processor import OptimizedAsyncProcessor
from optimized_performance_monitor import OptimizedPerformanceMonitor
                from sklearn.preprocessing import LabelEncoder
                    from sklearn.utils.class_weight import compute_class_weight
        import pandas as pd
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Optimized Training Pipeline
Advanced deep learning training with production-ready optimizations.
"""




    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)


# Import existing optimized components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class OptimizedTrainingConfig:
    """Advanced training configuration with optimizations"""
    
    def __init__(self, **kwargs) -> Any:
        # Model parameters
        self.model_name = kwargs.get('model_name', 'bert-base-uncased')
        self.model_type = kwargs.get('model_type', 'transformer')
        self.task_type = kwargs.get('task_type', 'classification')
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 16)
        self.learning_rate = kwargs.get('learning_rate', 2e-5)
        self.num_epochs = kwargs.get('num_epochs', 3)
        self.warmup_steps = kwargs.get('warmup_steps', 500)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        
        # Advanced optimizations
        self.use_amp = kwargs.get('use_amp', True)
        self.use_gradient_accumulation = kwargs.get('use_gradient_accumulation', True)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 4)
        self.use_mixed_precision = kwargs.get('use_mixed_precision', True)
        self.use_gradient_checkpointing = kwargs.get('use_gradient_checkpointing', False)
        self.use_torch_compile = kwargs.get('use_torch_compile', True)
        self.torch_compile_mode = kwargs.get('torch_compile_mode', None)  # None | 'default' | 'reduce-overhead' | 'max-autotune'
        
        # Memory optimizations
        self.use_gradient_clipping = kwargs.get('use_gradient_clipping', True)
        self.use_memory_efficient_attention = kwargs.get('use_memory_efficient_attention', True)
        self.use_activation_checkpointing = kwargs.get('use_activation_checkpointing', False)
        
        # Regularization
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        self.label_smoothing = kwargs.get('label_smoothing', 0.0)
        self.use_weight_decay = kwargs.get('use_weight_decay', True)
        
        # Monitoring
        self.eval_steps = kwargs.get('eval_steps', 500)
        self.save_steps = kwargs.get('save_steps', 1000)
        self.logging_steps = kwargs.get('logging_steps', 100)
        
        # Hardware
        self.device = kwargs.get('device', 'auto')
        self.num_workers = kwargs.get('num_workers', 4)
        self.pin_memory = kwargs.get('pin_memory', True)
        
        # Distributed training
        self.use_ddp = kwargs.get('use_ddp', False)
        self.local_rank = kwargs.get('local_rank', -1)
        self.world_size = kwargs.get('world_size', 1)
        
        # Early stopping
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 3)
        self.early_stopping_threshold = kwargs.get('early_stopping_threshold', 0.001)
        
        # Checkpointing
        self.save_total_limit = kwargs.get('save_total_limit', 3)
        self.load_best_model_at_end = kwargs.get('load_best_model_at_end', True)
        
        # Experiment tracking
        self.use_wandb = kwargs.get('use_wandb', True)
        self.use_tensorboard = kwargs.get('use_tensorboard', True)
        self.project_name = kwargs.get('project_name', 'optimized-training')
        self.run_name = kwargs.get('run_name', None)
        
        # Performance monitoring
        self.enable_performance_monitoring = kwargs.get('enable_performance_monitoring', True)
        self.monitor_gpu_memory = kwargs.get('monitor_gpu_memory', True)
        self.monitor_training_speed = kwargs.get('monitor_training_speed', True)
        
        # Caching
        self.enable_model_caching = kwargs.get('enable_model_caching', True)
        self.enable_data_caching = kwargs.get('enable_data_caching', True)
        
        # Auto device selection
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        
        # Generate run name if not provided
        if self.run_name is None:
            self.run_name = f"{self.model_name}_{self.task_type}_{int(time.time())}"

class OptimizedDataset(Dataset):
    """Optimized dataset with caching and memory management"""
    
    def __init__(self, texts: List[str], labels: Optional[List] = None, 
                 tokenizer=None, max_length: int = 512, cache_manager=None):
        
    """__init__ function."""
self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_manager = cache_manager
        self.cache_key_prefix = f"dataset_{hash(str(texts[:10]))}"
        
        # Pre-tokenize if cache manager is available
        if self.cache_manager and self.tokenizer:
            self._pre_tokenize()
    
    def _pre_tokenize(self) -> Any:
        """Pre-tokenize all texts and cache them"""
        logger.info("Pre-tokenizing dataset for caching...")
        
        for i, text in enumerate(tqdm(self.texts, desc="Pre-tokenizing")):
            cache_key = f"{self.cache_key_prefix}_{i}"
            
            # Check if already cached
            cached_item = self.cache_manager.get(cache_key)
            if cached_item is None:
                # Tokenize and cache
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Cache the tokenized data
                self.cache_manager.set(cache_key, encoding, ttl=3600)  # 1 hour TTL
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        text = str(self.texts[idx])
        
        # Try to get from cache first
        if self.cache_manager and self.tokenizer:
            cache_key = f"{self.cache_key_prefix}_{idx}"
            cached_item = self.cache_manager.get(cache_key)
            
            if cached_item is not None:
                item = {k: v.clone() for k, v in cached_item.items()}
            else:
                # Fallback to tokenization
                item = self._tokenize_text(text)
        else:
            item = self._tokenize_text(text)
        
        # Add labels if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item
    
    def _tokenize_text(self, text: str) -> Dict:
        """Tokenize text"""
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
            
            if 'token_type_ids' in encoding:
                item['token_type_ids'] = encoding['token_type_ids'].flatten()
        else:
            item = {'text': text}
        
        return item

class OptimizedDataManager:
    """Optimized data manager with caching and performance monitoring"""
    
    def __init__(self, config: OptimizedTrainingConfig):
        
    """__init__ function."""
self.config = config
        self.tokenizer = None
        self.label_encoder = None
        self.class_weights = None
        
        # Initialize cache manager
        self.cache_manager = OptimizedCacheManager() if config.enable_data_caching else None
        
        # Initialize performance monitor
        self.performance_monitor = OptimizedPerformanceMonitor() if config.enable_performance_monitoring else None
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare optimized data loaders"""
        logger.info("Loading and preparing optimized data...")
        
        start_time = time.time()
        
        # Load raw data
        train_data = self._load_file(self.config.train_file)
        
        if self.config.val_file:
            val_data = self._load_file(self.config.val_file)
            test_data = self._load_file(self.config.test_file) if self.config.test_file else None
        else:
            # Split data if validation file not provided
            train_data, val_data, test_data = self._split_data(train_data)
        
        # Prepare tokenizer
        self._prepare_tokenizer()
        
        # Prepare datasets with caching
        train_dataset = self._prepare_dataset(train_data, is_train=True)
        val_dataset = self._prepare_dataset(val_data, is_train=False)
        test_dataset = self._prepare_dataset(test_data, is_train=False) if test_data else None
        
        # Create optimized data loaders
        train_loader = self._create_optimized_loader(train_dataset, shuffle=True)
        val_loader = self._create_optimized_loader(val_dataset, shuffle=False)
        test_loader = self._create_optimized_loader(test_dataset, shuffle=False) if test_dataset else None
        
        load_time = time.time() - start_time
        logger.info(f"Data loaded in {load_time:.2f}s: {len(train_dataset)} train, "
                   f"{len(val_dataset)} val, {len(test_dataset) if test_dataset else 0} test samples")
        
        # Log performance metrics
        if self.performance_monitor:
            self.performance_monitor.record_metric("data_loading_time", load_time)
            self.performance_monitor.record_metric("total_samples", len(train_dataset) + len(val_dataset))
        
        return train_loader, val_loader, test_loader
    
    def _create_optimized_loader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create optimized data loader with performance settings"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=2 if self.config.num_workers > 0 else None
        )
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load data file with caching"""
        file_path = Path(file_path)
        
        # Try to get from cache
        if self.cache_manager:
            cache_key = f"data_file_{file_path.name}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Load from file
        if file_path.suffix == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            data = pd.read_json(file_path)
        elif file_path.suffix == '.parquet':
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Cache the loaded data
        if self.cache_manager:
            cache_key = f"data_file_{file_path.name}"
            self.cache_manager.set(cache_key, data, ttl=1800)  # 30 minutes TTL
        
        return data
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test with stratification"""
        train_data, temp_data = train_test_split(
            data, 
            train_size=self.config.train_size,
            random_state=42,
            stratify=data[self.config.label_column] if self.config.label_column in data.columns else None
        )
        
        val_size_adjusted = self.config.val_size / (self.config.val_size + self.config.test_size)
        val_data, test_data = train_test_split(
            temp_data,
            train_size=val_size_adjusted,
            random_state=42,
            stratify=temp_data[self.config.label_column] if self.config.label_column in temp_data.columns else None
        )
        
        return train_data, val_data, test_data
    
    def _prepare_tokenizer(self) -> Any:
        """Prepare tokenizer with optimizations"""
        if self.config.model_type == 'transformer':
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Enable fast tokenization if available
            if hasattr(self.tokenizer, 'is_fast') and not self.tokenizer.is_fast:
                logger.warning("Consider using a fast tokenizer for better performance")
    
    def _prepare_dataset(self, data: pd.DataFrame, is_train: bool) -> OptimizedDataset:
        """Prepare optimized dataset"""
        texts = data[self.config.text_column].tolist()
        labels = None
        
        if self.config.label_column in data.columns:
            labels = data[self.config.label_column].tolist()
            
            # Encode labels if needed
            if is_train and self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                labels = self.label_encoder.fit_transform(labels)
                
                # Compute class weights for imbalanced datasets
                if self.config.compute_class_weights:
                    unique_labels = np.unique(labels)
                    class_weights = compute_class_weight(
                        'balanced',
                        classes=unique_labels,
                        y=labels
                    )
                    self.class_weights = torch.FloatTensor(class_weights)
            elif self.label_encoder is not None:
                labels = self.label_encoder.transform(labels)
        
        return OptimizedDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            cache_manager=self.cache_manager
        )

class OptimizedModelManager:
    """Optimized model manager with advanced features"""
    
    def __init__(self, config: OptimizedTrainingConfig):
        
    """__init__ function."""
self.config = config
        self.model = None
        self.device = torch.device(config.device)
        self.cache_manager = OptimizedCacheManager() if config.enable_model_caching else None
    
    def create_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """Create optimized model"""
        logger.info(f"Creating optimized {self.config.model_type} model: {self.config.model_name}")
        
        # Try to load from cache first
        if self.cache_manager:
            cache_key = f"model_{self.config.model_name}_{self.config.model_type}"
            cached_model = self.cache_manager.get(cache_key)
            if cached_model is not None:
                self.model = cached_model
                logger.info("Model loaded from cache")
                return self.model
        
        # Create new model
        if self.config.model_type == 'transformer':
            self.model = self._create_transformer_model(num_classes)
        elif self.config.model_type == 'cnn':
            self.model = self._create_cnn_model(num_classes)
        elif self.config.model_type == 'rnn':
            self.model = self._create_rnn_model(num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Apply optimizations
        self._apply_model_optimizations()
        
        # Move to device
        self.model.to(self.device)
        
        # Wrap with DDP if using distributed training
        if self.config.use_ddp:
            self.model = DDP(self.model, device_ids=[self.config.local_rank])
        
        # Cache the model
        if self.cache_manager:
            cache_key = f"model_{self.config.model_name}_{self.config.model_type}"
            self.cache_manager.set(cache_key, self.model, ttl=3600)  # 1 hour TTL
        
        return self.model
    
    def _apply_model_optimizations(self) -> Any:
        """Apply various model optimizations"""
        # Prefer TF32 where supported for speedups on Ampere+ GPUs
        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if self.config.use_memory_efficient_attention:
            # Enable memory efficient attention if available
            if hasattr(self.model.config, 'attention_mode'):
                self.model.config.attention_mode = 'flash_attention_2'
        
        if self.config.use_activation_checkpointing:
            # Enable activation checkpointing for memory efficiency
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True

        # torch.compile (PyTorch 2.x) for graph capture and kernel fusion
        if getattr(torch, 'compile', None) is not None and self.config.use_torch_compile:
            try:
                compile_mode = self.config.torch_compile_mode
                if compile_mode is None:
                    # Prefer 'max-autotune' on GPU, 'reduce-overhead' on CPU
                    compile_mode = 'max-autotune' if torch.cuda.is_available() else 'reduce-overhead'
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.info(f"Model compiled with torch.compile (mode={compile_mode})")
            except Exception as e:
                logger.warning(f"torch.compile failed or unavailable: {e}")
    
    def _create_transformer_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """Create optimized transformer model"""
        if self.config.task_type == 'classification':
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=num_classes,
                dropout=self.config.dropout_rate,
                torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
            )
        elif self.config.task_type == 'regression':
            model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
            )
            model.classifier = nn.Linear(model.config.hidden_size, 1)
        else:
            model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
            )
        
        return model
    
    def _create_cnn_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """Create optimized CNN model"""
        class OptimizedTextCNN(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int = 128, 
                         num_filters: int = 100, filter_sizes: List[int] = [3, 4, 5],
                         num_classes: int = 2, dropout: float = 0.1):
                
    """__init__ function."""
super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.convs = nn.ModuleList([
                    nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
                ])
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
                
                # Initialize weights
                self._init_weights()
            
            def _init_weights(self) -> Any:
                """Initialize weights for better training"""
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                    elif isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            
            def forward(self, x) -> Any:
                x = self.embedding(x).unsqueeze(1)
                x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
                x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
                x = torch.cat(x, 1)
                x = self.dropout(x)
                return self.fc(x)
        
        return OptimizedTextCNN(
            vocab_size=30000,
            num_classes=num_classes or 2,
            dropout=self.config.dropout_rate
        )
    
    def _create_rnn_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """Create optimized RNN model"""
        class OptimizedTextRNN(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int = 128, 
                         hidden_dim: int = 256, num_layers: int = 2,
                         num_classes: int = 2, dropout: float = 0.1):
                
    """__init__ function."""
super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(
                    embed_dim, hidden_dim, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
                
                # Initialize weights
                self._init_weights()
            
            def _init_weights(self) -> Any:
                """Initialize weights for better training"""
                for name, param in self.named_parameters():
                    if 'weight' in name:
                        if 'lstm' in name:
                            nn.init.orthogonal_(param)
                        else:
                            nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            
            def forward(self, x) -> Any:
                embedded = self.embedding(x)
                lstm_out, (hidden, cell) = self.lstm(embedded)
                # Use last hidden state from both directions
                output = torch.cat((hidden[-2], hidden[-1]), dim=1)
                output = self.dropout(output)
                return self.fc(output)
        
        return OptimizedTextRNN(
            vocab_size=30000,
            num_classes=num_classes or 2,
            dropout=self.config.dropout_rate
        )

class OptimizedTrainingManager:
    """Optimized training manager with advanced features"""
    
    def __init__(self, config: OptimizedTrainingConfig, model: nn.Module, 
                 data_manager: OptimizedDataManager):
        
    """__init__ function."""
self.config = config
        self.model = model
        self.data_manager = data_manager
        self.device = torch.device(config.device)
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.use_amp else None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Performance monitoring
        self.performance_monitor = OptimizedPerformanceMonitor() if config.enable_performance_monitoring else None
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
    
    def _setup_experiment_tracking(self) -> Any:
        """Setup experiment tracking (wandb, tensorboard)"""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.__dict__
            )
        
        if self.config.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(f"runs/{self.config.run_name}")
    
    def setup_training(self, num_classes: int):
        """Setup optimized training components"""
        logger.info("Setting up optimized training components...")
        
        # Setup optimizer with advanced features
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        total_steps = self._calculate_total_steps()
        self.scheduler = self._create_scheduler(total_steps)
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Setup performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimized optimizer"""
        # Use different optimizers based on model type
        if self.config.model_type == 'transformer':
            # For transformers, use AdamW with weight decay
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config.weight_decay,
                },
                {
                    'params': [p for n, p in self.model.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                }
            ]
            return optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            # For other models, use standard AdamW
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self, total_steps: int) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.model_type == 'transformer':
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function"""
        if self.config.task_type == 'classification':
            if self.data_manager.class_weights is not None:
                return nn.CrossEntropyLoss(
                    weight=self.data_manager.class_weights.to(self.device),
                    label_smoothing=self.config.label_smoothing
                )
            else:
                return nn.CrossEntropyLoss(
                    label_smoothing=self.config.label_smoothing
                )
        elif self.config.task_type == 'regression':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
    
    def _calculate_total_steps(self) -> int:
        """Calculate total training steps"""
        # This is a simplified calculation - adjust based on your dataset size
        return self.config.num_epochs * 1000  # Approximate
    
    async def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Optimized training loop"""
        logger.info("Starting optimized training...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = await self._train_epoch(train_loader)
            self.train_metrics.append(train_metrics)
            
            # Validation phase
            val_metrics = await self._validate_epoch(val_loader)
            self.val_metrics.append(val_metrics)
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Check early stopping
            if self._should_stop_early(val_metrics):
                logger.info("Early stopping triggered")
                break
            
            # Save checkpoint
            if self._should_save_checkpoint(val_metrics):
                self._save_checkpoint(val_metrics)
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Final evaluation
        final_metrics = await self._final_evaluation(val_loader)
        
        logger.info("Optimized training completed!")
        return final_metrics
    
    async def _train_epoch(self, train_loader: DataLoader) -> Dict:
        """Optimized training for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(**batch)
                loss = self.criterion(outputs.logits, batch['labels'])
                
                # Scale loss for gradient accumulation
                if self.config.use_gradient_accumulation:
                    loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_gradient_clipping:
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item()
            if self.config.task_type == 'classification':
                predictions = torch.argmax(outputs.logits, dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_training_step(loss.item())
            
            # Performance monitoring
            if self.performance_monitor:
                self._monitor_training_performance()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _monitor_training_performance(self) -> Any:
        """Monitor training performance"""
        if self.performance_monitor:
            # Monitor GPU memory
            if self.config.monitor_gpu_memory and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                self.performance_monitor.record_metric("gpu_memory_gb", gpu_memory)
            
            # Monitor training speed
            if self.config.monitor_training_speed:
                self.performance_monitor.record_metric("training_steps", self.global_step)
    
    async def _validate_epoch(self, val_loader: DataLoader) -> Dict:
        """Optimized validation for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs.logits, batch['labels'])
                
                # Update metrics
                total_loss += loss.item()
                
                if self.config.task_type == 'classification':
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        metrics = {'loss': avg_loss}
        
        if self.config.task_type == 'classification':
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            
            # Calculate ROC AUC for binary classification
            if len(np.unique(all_labels)) == 2:
                roc_auc = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
                metrics['roc_auc'] = roc_auc
            
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return metrics
    
    async def _final_evaluation(self, val_loader: DataLoader) -> Dict:
        """Final evaluation on validation set"""
        logger.info("Running final evaluation...")
        return await self._validate_epoch(val_loader)
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to all tracking systems"""
        combined_metrics = {
            'epoch': self.current_epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss']
        }
        
        if 'accuracy' in train_metrics:
            combined_metrics['train_accuracy'] = train_metrics['accuracy']
        if 'accuracy' in val_metrics:
            combined_metrics['val_accuracy'] = val_metrics['accuracy']
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log(combined_metrics)
        
        # Log to tensorboard
        if self.config.use_tensorboard:
            for key, value in combined_metrics.items():
                self.tensorboard_writer.add_scalar(key, value, self.current_epoch)
        
        # Log to console
        logger.info(f"Epoch {self.current_epoch + 1}: "
                   f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}")
    
    def _log_training_step(self, loss: float):
        """Log training step metrics"""
        step_metrics = {
            'step': self.global_step,
            'train_step_loss': loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        if self.config.use_wandb:
            wandb.log(step_metrics)
        
        if self.config.use_tensorboard:
            for key, value in step_metrics.items():
                self.tensorboard_writer.add_scalar(key, value, self.global_step)
    
    def _should_stop_early(self, val_metrics: Dict) -> bool:
        """Check if early stopping should be triggered"""
        current_metric = val_metrics['loss']
        
        if current_metric < self.best_metric - self.config.early_stopping_threshold:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _should_save_checkpoint(self, val_metrics: Dict) -> bool:
        """Check if checkpoint should be saved"""
        current_metric = val_metrics['loss']
        return current_metric < self.best_metric
    
    def _save_checkpoint(self, val_metrics: Dict):
        """Save model checkpoint"""
        checkpoint_path = f"checkpoints/best_model_epoch_{self.current_epoch}"
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), f"{checkpoint_path}/model.pt")
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'val_metrics': val_metrics
        }, f"{checkpoint_path}/training_state.pt")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")

class OptimizedTrainingWorkflow:
    """Optimized training workflow orchestrator"""
    
    def __init__(self, config: OptimizedTrainingConfig):
        
    """__init__ function."""
self.config = config
        
        # Initialize components
        self.data_manager = OptimizedDataManager(config)
        self.model_manager = OptimizedModelManager(config)
        self.training_manager = None
    
    async def run_workflow(self) -> Dict:
        """Run the complete optimized training workflow"""
        logger.info("Starting Optimized Training Workflow")
        
        try:
            # Step 1: Load and prepare data
            train_loader, val_loader, test_loader = self.data_manager.load_data()
            
            # Step 2: Create model
            num_classes = len(self.data_manager.label_encoder.classes_) if self.data_manager.label_encoder else None
            model = self.model_manager.create_model(num_classes)
            
            # Step 3: Setup training
            self.training_manager = OptimizedTrainingManager(
                self.config, model, self.data_manager
            )
            self.training_manager.setup_training(num_classes)
            
            # Step 4: Train model
            final_metrics = await self.training_manager.train(train_loader, val_loader)
            
            # Step 5: Evaluate on test set
            if test_loader:
                test_metrics = await self.training_manager._validate_epoch(test_loader)
                final_metrics['test_metrics'] = test_metrics
            
            # Step 6: Save final model
            self.model_manager.save_model("models/final_model", model.state_dict())
            
            # Step 7: Generate training report
            report = self._generate_report(final_metrics)
            
            logger.info("Optimized Training Workflow completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            raise
    
    def _generate_report(self, final_metrics: Dict) -> Dict:
        """Generate comprehensive training report"""
        report = {
            'training_config': self.config.__dict__,
            'final_metrics': final_metrics,
            'training_history': {
                'train_metrics': self.training_manager.train_metrics,
                'val_metrics': self.training_manager.val_metrics
            },
            'model_info': {
                'model_type': self.config.model_type,
                'task_type': self.config.task_type,
                'model_name': self.config.model_name,
                'total_parameters': sum(p.numel() for p in self.model_manager.model.parameters())
            },
            'performance_info': {
                'gpu_memory_peak': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                'training_time': time.time() - getattr(self, '_start_time', time.time())
            }
        }
        
        # Save report
        with open("optimized_training_report.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        return report

# Utility functions for common workflows
async def run_optimized_text_classification(
    train_file: str,
    model_name: str = "bert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    **kwargs
) -> Dict:
    """Run optimized text classification workflow"""
    
    # Configuration
    config = OptimizedTrainingConfig(
        model_name=model_name,
        model_type='transformer',
        task_type='classification',
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_file=train_file,
        text_column="text",
        label_column="label",
        **kwargs
    )
    
    # Run workflow
    workflow = OptimizedTrainingWorkflow(config)
    return await workflow.run_workflow()

async def run_optimized_text_regression(
    train_file: str,
    model_name: str = "bert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    **kwargs
) -> Dict:
    """Run optimized text regression workflow"""
    
    # Configuration
    config = OptimizedTrainingConfig(
        model_name=model_name,
        model_type='transformer',
        task_type='regression',
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_file=train_file,
        text_column="text",
        label_column="target",
        **kwargs
    )
    
    # Run workflow
    workflow = OptimizedTrainingWorkflow(config)
    return await workflow.run_workflow()

# Example usage
if __name__ == "__main__":
    async def main():
        
    """main function."""
# Create sample data
        
        # Sample data
        data = {
            'text': [
                "This is a positive review",
                "This is a negative review",
                "Amazing product!",
                "Terrible service",
                "Great experience",
                "Poor quality"
            ],
            'label': [1, 0, 1, 0, 1, 0]
        }
        
        df = pd.DataFrame(data)
        df.to_csv('sample_data.csv', index=False)
        
        # Run optimized workflow
        result = await run_optimized_text_classification(
            train_file='sample_data.csv',
            model_name='bert-base-uncased',
            num_epochs=2,
            batch_size=2,
            use_amp=True,
            enable_performance_monitoring=True
        )
        
        print("Optimized training completed!")
        print(f"Final metrics: {result['final_metrics']}")
    
    asyncio.run(main()) 