from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
                from sklearn.preprocessing import LabelEncoder
                    from sklearn.utils.class_weight import compute_class_weight
        import pandas as pd
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Deep Learning Workflow System
Prioritizes clarity, efficiency, and best practices in deep learning workflows.
"""




    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class ModelType(Enum):
    """Supported model types"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    CUSTOM = "custom"

class TaskType(Enum):
    """Supported task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    EMBEDDING = "embedding"

@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Model parameters
    model_name: str = "bert-base-uncased"
    model_type: ModelType = ModelType.TRANSFORMER
    task_type: TaskType = TaskType.CLASSIFICATION
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    use_amp: bool = True  # Automatic Mixed Precision
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # Monitoring
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4
    
    # Distributed training
    use_ddp: bool = False
    local_rank: int = -1
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Experiment tracking
    use_wandb: bool = True
    project_name: str = "deep-learning-workflow"
    run_name: Optional[str] = None
    
    def __post_init__(self) -> Any:
        """Validate and set default values"""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.run_name is None:
            self.run_name = f"{self.model_name}_{self.task_type.value}_{int(time.time())}"

@dataclass
class DatasetConfig:
    """Configuration for dataset"""
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Data preprocessing
    max_length: int = 512
    text_column: str = "text"
    label_column: str = "label"
    
    # Augmentation
    use_augmentation: bool = False
    augmentation_config: Dict = field(default_factory=dict)
    
    # Sampling
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    
    # Class weights for imbalanced datasets
    compute_class_weights: bool = True

class CustomDataset(Dataset):
    """Custom dataset for deep learning tasks"""
    
    def __init__(self, texts: List[str], labels: Optional[List] = None, 
                 tokenizer=None, max_length: int = 512):
        
    """__init__ function."""
self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
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
            
            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
            
            if 'token_type_ids' in encoding:
                item['token_type_ids'] = encoding['token_type_ids'].flatten()
        else:
            item = {'text': text}
        
        # Add labels if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class DataManager:
    """Manages data loading, preprocessing, and augmentation"""
    
    def __init__(self, config: DatasetConfig):
        
    """__init__ function."""
self.config = config
        self.tokenizer = None
        self.label_encoder = None
        self.class_weights = None
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data loaders"""
        logger.info("Loading and preparing data...")
        
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
        
        # Prepare datasets
        train_dataset = self._prepare_dataset(train_data, is_train=True)
        val_dataset = self._prepare_dataset(val_data, is_train=False)
        test_dataset = self._prepare_dataset(test_data, is_train=False) if test_data else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        
        logger.info(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, "
                   f"{len(test_dataset) if test_dataset else 0} test samples")
        
        return train_loader, val_loader, test_loader
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load data file"""
        file_path = Path(file_path)
        
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test"""
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
        """Prepare tokenizer for the model"""
        if self.config.model_type == ModelType.TRANSFORMER:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _prepare_dataset(self, data: pd.DataFrame, is_train: bool) -> CustomDataset:
        """Prepare dataset from DataFrame"""
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
        
        return CustomDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )

class ModelManager:
    """Manages model creation, loading, and saving"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.model = None
        self.device = torch.device(config.device)
    
    def create_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """Create model based on configuration"""
        logger.info(f"Creating {self.config.model_type.value} model: {self.config.model_name}")
        
        if self.config.model_type == ModelType.TRANSFORMER:
            self.model = self._create_transformer_model(num_classes)
        elif self.config.model_type == ModelType.CNN:
            self.model = self._create_cnn_model(num_classes)
        elif self.config.model_type == ModelType.RNN:
            self.model = self._create_rnn_model(num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Move to device
        self.model.to(self.device)
        
        # Wrap with DDP if using distributed training
        if self.config.use_ddp:
            self.model = DDP(self.model, device_ids=[self.config.local_rank])
        
        return self.model
    
    def _create_transformer_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """Create transformer model"""
        if self.config.task_type == TaskType.CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=num_classes,
                dropout=self.config.dropout_rate
            )
        elif self.config.task_type == TaskType.REGRESSION:
            model = AutoModel.from_pretrained(self.config.model_name)
            model.classifier = nn.Linear(model.config.hidden_size, 1)
        else:
            model = AutoModel.from_pretrained(self.config.model_name)
        
        return model
    
    def _create_cnn_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """Create CNN model"""
        # Simple CNN for text classification
        class TextCNN(nn.Module):
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
            
            def forward(self, x) -> Any:
                x = self.embedding(x).unsqueeze(1)  # Add channel dimension
                x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
                x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
                x = torch.cat(x, 1)
                x = self.dropout(x)
                return self.fc(x)
        
        return TextCNN(
            vocab_size=30000,  # Default vocab size
            num_classes=num_classes or 2,
            dropout=self.config.dropout_rate
        )
    
    def _create_rnn_model(self, num_classes: Optional[int] = None) -> nn.Module:
        """Create RNN model"""
        class TextRNN(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int = 128, 
                         hidden_dim: int = 256, num_layers: int = 2,
                         num_classes: int = 2, dropout: float = 0.1):
                
    """__init__ function."""
super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(
                    embed_dim, hidden_dim, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, num_classes)
            
            def forward(self, x) -> Any:
                embedded = self.embedding(x)
                lstm_out, (hidden, cell) = self.lstm(embedded)
                # Use last hidden state
                output = self.dropout(hidden[-1])
                return self.fc(output)
        
        return TextRNN(
            vocab_size=30000,  # Default vocab size
            num_classes=num_classes or 2,
            dropout=self.config.dropout_rate
        )
    
    def save_model(self, path: str, model_state: Optional[Dict] = None):
        """Save model and configuration"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if model_state is None:
            model_state = self.model.state_dict()
        
        torch.save(model_state, save_path / "model.pt")
        
        # Save configuration
        config_dict = {
            'model_config': self.config.__dict__,
            'model_type': self.config.model_type.value,
            'task_type': self.config.task_type.value
        }
        
        with open(save_path / "config.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str) -> nn.Module:
        """Load model from path"""
        load_path = Path(path)
        
        # Load configuration
        with open(load_path / "config.json", 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        
        # Update config
        for key, value in config_dict['model_config'].items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Create model
        num_classes = config_dict.get('num_classes', None)
        self.model = self.create_model(num_classes)
        
        # Load model state
        model_state = torch.load(load_path / "model.pt", map_location=self.device)
        self.model.load_state_dict(model_state)
        
        logger.info(f"Model loaded from {load_path}")
        return self.model

class TrainingManager:
    """Manages the training process with best practices"""
    
    def __init__(self, config: TrainingConfig, model: nn.Module, 
                 data_manager: DataManager):
        
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
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.__dict__
            )
    
    def setup_training(self, num_classes: int):
        """Setup training components"""
        logger.info("Setting up training components...")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        total_steps = self._calculate_total_steps()
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup loss function
        if self.config.task_type == TaskType.CLASSIFICATION:
            if self.data_manager.class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(
                    weight=self.data_manager.class_weights.to(self.device),
                    label_smoothing=self.config.label_smoothing
                )
            else:
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.config.label_smoothing
                )
        elif self.config.task_type == TaskType.REGRESSION:
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
    
    def _calculate_total_steps(self) -> int:
        """Calculate total training steps"""
        # This is a simplified calculation - adjust based on your dataset size
        return self.config.num_epochs * 1000  # Approximate
    
    async def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Main training loop"""
        logger.info("Starting training...")
        
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
        
        # Final evaluation
        final_metrics = await self._final_evaluation(val_loader)
        
        logger.info("Training completed!")
        return final_metrics
    
    async def _train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
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
            if self.config.task_type == TaskType.CLASSIFICATION:
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
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    async def _validate_epoch(self, val_loader: DataLoader) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
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
                
                if self.config.task_type == TaskType.CLASSIFICATION:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        metrics = {'loss': avg_loss}
        
        if self.config.task_type == TaskType.CLASSIFICATION:
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            
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
        """Log metrics to wandb and console"""
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
        
        # Log to console
        logger.info(f"Epoch {self.current_epoch + 1}: "
                   f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}")
    
    def _log_training_step(self, loss: float):
        """Log training step metrics"""
        if self.config.use_wandb:
            wandb.log({
                'step': self.global_step,
                'train_step_loss': loss,
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
    
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

class DeepLearningWorkflow:
    """Main workflow orchestrator"""
    
    def __init__(self, training_config: TrainingConfig, dataset_config: DatasetConfig):
        
    """__init__ function."""
self.training_config = training_config
        self.dataset_config = dataset_config
        
        # Initialize components
        self.data_manager = DataManager(dataset_config)
        self.model_manager = ModelManager(training_config)
        self.training_manager = None
    
    async def run_workflow(self) -> Dict:
        """Run the complete deep learning workflow"""
        logger.info("Starting Deep Learning Workflow")
        
        try:
            # Step 1: Load and prepare data
            train_loader, val_loader, test_loader = self.data_manager.load_data()
            
            # Step 2: Create model
            num_classes = len(self.data_manager.label_encoder.classes_) if self.data_manager.label_encoder else None
            model = self.model_manager.create_model(num_classes)
            
            # Step 3: Setup training
            self.training_manager = TrainingManager(
                self.training_config, model, self.data_manager
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
            
            logger.info("Deep Learning Workflow completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            raise
    
    def _generate_report(self, final_metrics: Dict) -> Dict:
        """Generate comprehensive training report"""
        report = {
            'training_config': self.training_config.__dict__,
            'dataset_config': self.dataset_config.__dict__,
            'final_metrics': final_metrics,
            'training_history': {
                'train_metrics': self.training_manager.train_metrics,
                'val_metrics': self.training_manager.val_metrics
            },
            'model_info': {
                'model_type': self.training_config.model_type.value,
                'task_type': self.training_config.task_type.value,
                'model_name': self.training_config.model_name,
                'total_parameters': sum(p.numel() for p in self.model_manager.model.parameters())
            }
        }
        
        # Save report
        with open("training_report.json", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        return report

# Utility functions for common workflows
async def run_text_classification(
    train_file: str,
    model_name: str = "bert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> Dict:
    """Run text classification workflow"""
    
    # Configuration
    training_config = TrainingConfig(
        model_name=model_name,
        model_type=ModelType.TRANSFORMER,
        task_type=TaskType.CLASSIFICATION,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    dataset_config = DatasetConfig(
        train_file=train_file,
        text_column="text",
        label_column="label"
    )
    
    # Run workflow
    workflow = DeepLearningWorkflow(training_config, dataset_config)
    return await workflow.run_workflow()

async def run_text_regression(
    train_file: str,
    model_name: str = "bert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
) -> Dict:
    """Run text regression workflow"""
    
    # Configuration
    training_config = TrainingConfig(
        model_name=model_name,
        model_type=ModelType.TRANSFORMER,
        task_type=TaskType.REGRESSION,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    dataset_config = DatasetConfig(
        train_file=train_file,
        text_column="text",
        label_column="target"
    )
    
    # Run workflow
    workflow = DeepLearningWorkflow(training_config, dataset_config)
    return await workflow.run_workflow()

# Example usage
if __name__ == "__main__":
    # Example: Text classification
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
        
        # Run workflow
        result = await run_text_classification(
            train_file='sample_data.csv',
            model_name='bert-base-uncased',
            num_epochs=2,
            batch_size=2
        )
        
        print("Training completed!")
        print(f"Final metrics: {result['final_metrics']}")
    
    asyncio.run(main()) 