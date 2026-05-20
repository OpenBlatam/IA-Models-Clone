"""
Modular Code Structure System
Follows key convention: Create modular code structures with separate files for models, data loading, training, and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODELS MODULE
# ============================================================================

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._build_model()
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        pass
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu'):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class TransformerModel(BaseModel):
    """Transformer-based model for NLP tasks"""
    
    def _build_model(self):
        """Build transformer architecture"""
        self.embedding = nn.Embedding(
            self.config['vocab_size'],
            self.config['hidden_size']
        )
        
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.config['max_length'], self.config['hidden_size'])
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config['hidden_size'],
            nhead=self.config['num_heads'],
            dim_feedforward=self.config['ff_dim'],
            dropout=self.config['dropout'],
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config['num_layers']
        )
        
        self.output_projection = nn.Linear(
            self.config['hidden_size'],
            self.config['output_size']
        )
        
        self.dropout = nn.Dropout(self.config['dropout'])
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through transformer"""
        # Embedding + positional encoding
        x = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        x = self.dropout(x)
        
        # Transformer encoding
        if attention_mask is not None:
            # Convert attention mask to transformer format
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output projection
        output = self.output_projection(x)
        
        return output

class DiffusionModel(BaseModel):
    """Diffusion model for text generation"""
    
    def _build_model(self):
        """Build diffusion model architecture"""
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.config['input_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(self.config['hidden_dim'], self.config['input_dim'])
        )
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.config['time_dim']),
            nn.ReLU(),
            nn.Linear(self.config['time_dim'], self.config['hidden_dim'])
        )
    
    def forward(self, x, t):
        """Forward pass through diffusion model"""
        # Time embedding
        t_emb = self.time_embedding(t.unsqueeze(-1).float())
        
        # Noise prediction
        noise_pred = self.noise_predictor(x)
        
        # Combine with time information
        output = noise_pred + t_emb.mean(dim=1, keepdim=True)
        
        return output

class UNetModel(BaseModel):
    """UNet architecture for diffusion models"""
    
    def _build_model(self):
        """Build UNet architecture"""
        self.input_projection = nn.Linear(self.config['input_dim'], self.config['hidden_dim'])
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            self._create_encoder_block(self.config['hidden_dim'], self.config['hidden_dim'])
            for _ in range(self.config['num_encoder_blocks'])
        ])
        
        # Bottleneck
        self.bottleneck = self._create_encoder_block(
            self.config['hidden_dim'], 
            self.config['hidden_dim']
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            self._create_decoder_block(self.config['hidden_dim'], self.config['hidden_dim'])
            for _ in range(self.config['num_decoder_blocks'])
        ])
        
        self.output_projection = nn.Linear(self.config['hidden_dim'], self.config['output_dim'])
    
    def _create_encoder_block(self, in_dim, out_dim):
        """Create encoder block"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(self.config['dropout'])
        )
    
    def _create_decoder_block(self, in_dim, out_dim):
        """Create decoder block"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(self.config['dropout']),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(self.config['dropout'])
        )
    
    def forward(self, x, t=None):
        """Forward pass through UNet"""
        x = self.input_projection(x)
        
        # Encoder path
        encoder_outputs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i < len(encoder_outputs):
                x = x + encoder_outputs[-(i+1)]  # Skip connection
            x = decoder_block(x)
        
        output = self.output_projection(x)
        return output

# ============================================================================
# DATA LOADING MODULE
# ============================================================================

class BaseDataset(Dataset, ABC):
    """Abstract base class for datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = None
        self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """Load the dataset"""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return dataset length"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        """Get item by index"""
        pass

class TextDataset(BaseDataset):
    """Text dataset for NLP tasks"""
    
    def _load_data(self):
        """Load text data from file"""
        data_path = self.config['data_path']
        
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            self.data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded dataset with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get text sample by index"""
        sample = self.data.iloc[idx]
        
        # Extract text and target
        text = sample[self.config['text_column']]
        target = sample[self.config['target_column']] if self.config.get('target_column') else None
        
        # Tokenize text
        if hasattr(self, 'tokenizer'):
            tokens = self.tokenizer(
                text,
                max_length=self.config['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            # Simple character-level tokenization
            tokens = torch.tensor([ord(c) for c in text[:self.config['max_length']]])
            if len(tokens) < self.config['max_length']:
                tokens = F.pad(tokens, (0, self.config['max_length'] - len(tokens)))
        
        if target is not None:
            return {
                'input_ids': tokens,
                'target': torch.tensor(target, dtype=torch.long)
            }
        else:
            return {
                'input_ids': tokens
            }

class DiffusionDataset(BaseDataset):
    """Dataset for diffusion models"""
    
    def _load_data(self):
        """Load data for diffusion training"""
        data_path = self.config['data_path']
        
        if data_path.endswith('.npy'):
            self.data = np.load(data_path)
        elif data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path).values
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded diffusion dataset with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get diffusion sample by index"""
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # Add noise for diffusion training
        noise = torch.randn_like(sample)
        timestep = torch.randint(0, self.config['num_timesteps'], (1,))
        
        return {
            'clean_data': sample,
            'noisy_data': sample + noise * timestep.float() / self.config['num_timesteps'],
            'timestep': timestep,
            'noise': noise
        }

class DataLoaderFactory:
    """Factory for creating data loaders"""
    
    @staticmethod
    def create_dataloader(
        dataset: BaseDataset,
        config: Dict[str, Any]
    ) -> DataLoader:
        """Create data loader with specified configuration"""
        
        return DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config.get('shuffle', True),
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', True),
            drop_last=config.get('drop_last', False)
        )

# ============================================================================
# TRAINING MODULE
# ============================================================================

class BaseTrainer(ABC):
    """Abstract base class for trainers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = self._setup_device()
        self._setup_logging()
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if torch.cuda.is_available() and self.config.get('use_gpu', True):
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def _setup_logging(self):
        """Setup training logging"""
        self.log_dir = Path(self.config.get('log_dir', './logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = self.log_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Setup console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train the model"""
        pass
    
    @abstractmethod
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        pass
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch'], checkpoint['metrics']

class StandardTrainer(BaseTrainer):
    """Standard trainer for supervised learning"""
    
    def __init__(self, config: Dict[str, Any], model: BaseModel):
        super().__init__(config)
        self.model = model.to(self.device)
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.01)
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'none').lower()
        
        if scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 100)
            )
        elif scheduler_name == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('factor', 0.1),
                patience=self.config.get('patience', 10)
            )
        else:
            self.scheduler = None
    
    def _setup_loss_function(self):
        """Setup loss function"""
        loss_name = self.config.get('loss_function', 'cross_entropy').lower()
        
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_name == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train the model"""
        num_epochs = self.config['num_epochs']
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            if val_loader:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        self.log_dir / 'best_model.pth',
                        epoch,
                        val_metrics
                    )
                
                # Update scheduler
                if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f}")
            
            # Update scheduler
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(batch, dict) and 'target' in batch:
                output = self.model(batch['input_ids'])
                loss = self.criterion(output, batch['target'])
            else:
                output = self.model(batch)
                # For unsupervised tasks, use reconstruction loss
                loss = self.criterion(output, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if isinstance(batch, dict) and 'target' in batch:
                    output = self.model(batch['input_ids'])
                    loss = self.criterion(output, batch['target'])
                else:
                    output = self.model(batch)
                    loss = self.criterion(output, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

class DiffusionTrainer(BaseTrainer):
    """Trainer for diffusion models"""
    
    def __init__(self, config: Dict[str, Any], model: BaseModel):
        super().__init__(config)
        self.model = model.to(self.device)
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_diffusion_params()
    
    def _setup_diffusion_params(self):
        """Setup diffusion parameters"""
        self.num_timesteps = self.config['num_timesteps']
        self.beta_start = self.config.get('beta_start', 0.0001)
        self.beta_end = self.config.get('beta_end', 0.02)
        
        # Create noise schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _setup_optimizer(self):
        """Setup optimizer for diffusion"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs']
        )
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Train the diffusion model"""
        num_epochs = self.config['num_epochs']
        
        for epoch in range(num_epochs):
            train_loss = self._train_epoch(train_loader, epoch)
            
            if val_loader:
                val_metrics = self.validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_metrics['loss']:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f}")
            
            self.scheduler.step()
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train diffusion model for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            clean_data = batch['clean_data'].to(self.device)
            noisy_data = batch['noisy_data'].to(self.device)
            timestep = batch['timestep'].to(self.device)
            noise = batch['noise'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predicted_noise = self.model(noisy_data, timestep)
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the diffusion model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                clean_data = batch['clean_data'].to(self.device)
                noisy_data = batch['noisy_data'].to(self.device)
                timestep = batch['timestep'].to(self.device)
                noise = batch['noise'].to(self.device)
                
                predicted_noise = self.model(noisy_data, timestep)
                loss = F.mse_loss(predicted_noise, noise)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

# ============================================================================
# EVALUATION MODULE
# ============================================================================

class BaseEvaluator(ABC):
    """Abstract base class for evaluators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {}
    
    @abstractmethod
    def evaluate(self, model: BaseModel, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        pass
    
    def save_results(self, filepath: str):
        """Save evaluation results"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Evaluation results saved to {filepath}")

class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks"""
    
    def evaluate(self, model: BaseModel, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate classification model"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(next(model.parameters()).device)
                    targets = batch['target'].to(next(model.parameters()).device)
                else:
                    input_ids = batch[0].to(next(model.parameters()).device)
                    targets = batch[1].to(next(model.parameters()).device)
                
                outputs = model(input_ids)
                predictions = torch.argmax(outputs, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        self.metrics = self._calculate_classification_metrics(
            all_targets, all_predictions
        )
        
        return self.metrics
    
    def _calculate_classification_metrics(self, targets, predictions):
        """Calculate classification metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }

class GenerationEvaluator(BaseEvaluator):
    """Evaluator for text generation tasks"""
    
    def evaluate(self, model: BaseModel, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate generation model"""
        model.eval()
        generated_texts = []
        reference_texts = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(next(model.parameters()).device)
                    if 'target' in batch:
                        targets = batch['target'].to(next(model.parameters()).device)
                    else:
                        targets = None
                else:
                    input_ids = batch[0].to(next(model.parameters()).device)
                    targets = batch[1].to(next(model.parameters()).device) if len(batch) > 1 else None
                
                # Generate text
                generated = self._generate_text(model, input_ids)
                generated_texts.extend(generated)
                
                if targets is not None:
                    reference_texts.extend(self._decode_targets(targets))
        
        # Calculate metrics
        self.metrics = self._calculate_generation_metrics(
            generated_texts, reference_texts
        )
        
        return self.metrics
    
    def _generate_text(self, model: BaseModel, input_ids: torch.Tensor) -> List[str]:
        """Generate text from model"""
        # Simple greedy decoding
        outputs = model(input_ids)
        predictions = torch.argmax(outputs, dim=-1)
        
        # Convert to text (simplified)
        generated_texts = []
        for pred in predictions:
            text = ''.join([chr(int(token)) for token in pred if token > 0])
            generated_texts.append(text)
        
        return generated_texts
    
    def _decode_targets(self, targets: torch.Tensor) -> List[str]:
        """Decode target tokens to text"""
        decoded_texts = []
        for target in targets:
            text = ''.join([chr(int(token)) for token in target if token > 0])
            decoded_texts.append(text)
        
        return decoded_texts
    
    def _calculate_generation_metrics(self, generated_texts, reference_texts):
        """Calculate generation metrics"""
        # Simple metrics for demonstration
        avg_length = np.mean([len(text) for text in generated_texts])
        unique_ratio = len(set(generated_texts)) / len(generated_texts)
        
        metrics = {
            'average_length': avg_length,
            'unique_ratio': unique_ratio,
            'num_generated': len(generated_texts)
        }
        
        # Add BLEU score if references are available
        if reference_texts:
            try:
                from nltk.translate.bleu_score import sentence_bleu
                bleu_scores = []
                for gen, ref in zip(generated_texts, reference_texts):
                    bleu = sentence_bleu([ref.split()], gen.split())
                    bleu_scores.append(bleu)
                
                metrics['bleu_score'] = np.mean(bleu_scores)
            except ImportError:
                logger.warning("NLTK not available, skipping BLEU score")
        
        return metrics

class DiffusionEvaluator(BaseEvaluator):
    """Evaluator for diffusion models"""
    
    def evaluate(self, model: BaseModel, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate diffusion model"""
        model.eval()
        reconstruction_errors = []
        sample_qualities = []
        
        with torch.no_grad():
            for batch in test_loader:
                clean_data = batch['clean_data'].to(next(model.parameters()).device)
                noisy_data = batch['noisy_data'].to(next(model.parameters()).device)
                timestep = batch['timestep'].to(next(model.parameters()).device)
                
                # Denoise the data
                predicted_noise = model(noisy_data, timestep)
                denoised_data = noisy_data - predicted_noise
                
                # Calculate reconstruction error
                error = F.mse_loss(denoised_data, clean_data)
                reconstruction_errors.append(error.item())
                
                # Calculate sample quality (variance)
                quality = torch.var(denoised_data).item()
                sample_qualities.append(quality)
        
        # Calculate metrics
        self.metrics = {
            'reconstruction_error': np.mean(reconstruction_errors),
            'sample_quality': np.mean(sample_qualities),
            'num_samples': len(reconstruction_errors)
        }
        
        return self.metrics

# ============================================================================
# CONFIGURATION MODULE
# ============================================================================

class ConfigManager:
    """Manager for configuration files"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to file"""
        config_path = Path(config_path)
        
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to {config_path}")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """Main training pipeline that orchestrates all components"""
    
    def __init__(self, config_path: str):
        self.config = ConfigManager.load_config(config_path)
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self):
        """Setup the training pipeline"""
        logger.info("Setting up training pipeline...")
        
        # Create model
        self.model = self._create_model()
        
        # Create trainer
        self.trainer = self._create_trainer()
        
        # Create evaluator
        self.evaluator = self._create_evaluator()
        
        # Create data loaders
        self._create_data_loaders()
        
        logger.info("Training pipeline setup complete!")
    
    def _create_model(self) -> BaseModel:
        """Create model based on configuration"""
        model_type = self.config['model']['type']
        
        if model_type == 'transformer':
            return TransformerModel(self.config['model'])
        elif model_type == 'diffusion':
            return DiffusionModel(self.config['model'])
        elif model_type == 'unet':
            return UNetModel(self.config['model'])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_trainer(self) -> BaseTrainer:
        """Create trainer based on configuration"""
        trainer_type = self.config['training']['type']
        
        if trainer_type == 'standard':
            return StandardTrainer(self.config['training'], self.model)
        elif trainer_type == 'diffusion':
            return DiffusionTrainer(self.config['training'], self.model)
        else:
            raise ValueError(f"Unsupported trainer type: {trainer_type}")
    
    def _create_evaluator(self) -> BaseEvaluator:
        """Create evaluator based on configuration"""
        task_type = self.config['task']['type']
        
        if task_type == 'classification':
            return ClassificationEvaluator(self.config['evaluation'])
        elif task_type == 'generation':
            return GenerationEvaluator(self.config['evaluation'])
        elif task_type == 'diffusion':
            return DiffusionEvaluator(self.config['evaluation'])
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _create_data_loaders(self):
        """Create data loaders"""
        # Training dataset
        train_dataset = self._create_dataset('train')
        self.train_loader = DataLoaderFactory.create_dataloader(
            train_dataset, self.config['data']['train']
        )
        
        # Validation dataset
        if 'val' in self.config['data']:
            val_dataset = self._create_dataset('val')
            self.val_loader = DataLoaderFactory.create_dataloader(
                val_dataset, self.config['data']['val']
            )
        
        # Test dataset
        if 'test' in self.config['data']:
            test_dataset = self._create_dataset('test')
            self.test_loader = DataLoaderFactory.create_dataloader(
                test_dataset, self.config['data']['test']
            )
    
    def _create_dataset(self, split: str) -> BaseDataset:
        """Create dataset for specific split"""
        dataset_type = self.config['data'][split]['type']
        
        if dataset_type == 'text':
            return TextDataset(self.config['data'][split])
        elif dataset_type == 'diffusion':
            return DiffusionDataset(self.config['data'][split])
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    def train(self):
        """Run training"""
        logger.info("Starting training...")
        self.trainer.train(self.train_loader, self.val_loader)
        logger.info("Training complete!")
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation"""
        if self.test_loader is None:
            logger.warning("No test loader available for evaluation")
            return {}
        
        logger.info("Starting evaluation...")
        metrics = self.evaluator.evaluate(self.model, self.test_loader)
        logger.info("Evaluation complete!")
        
        # Save results
        results_path = Path(self.config.get('output_dir', './outputs')) / 'evaluation_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluator.save_results(str(results_path))
        
        return metrics
    
    def run(self):
        """Run complete training and evaluation pipeline"""
        self.setup()
        self.train()
        self.evaluate()
        logger.info("Pipeline execution complete!")

def main():
    """Example usage of the modular code structure system"""
    
    # Example configuration
    config = {
        'model': {
            'type': 'transformer',
            'vocab_size': 10000,
            'hidden_size': 512,
            'num_heads': 8,
            'ff_dim': 2048,
            'num_layers': 6,
            'max_length': 512,
            'dropout': 0.1,
            'output_size': 1000
        },
        'data': {
            'train': {
                'type': 'text',
                'data_path': './data/train.csv',
                'text_column': 'text',
                'target_column': 'label',
                'max_length': 512,
                'batch_size': 32,
                'shuffle': True
            },
            'val': {
                'type': 'text',
                'data_path': './data/val.csv',
                'text_column': 'text',
                'target_column': 'label',
                'max_length': 512,
                'batch_size': 32,
                'shuffle': False
            }
        },
        'training': {
            'type': 'standard',
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'scheduler': 'cosine',
            'grad_clip': 1.0,
            'log_interval': 10
        },
        'task': {
            'type': 'classification'
        },
        'evaluation': {
            'metrics': ['accuracy', 'f1', 'precision', 'recall']
        },
        'output_dir': './outputs'
    }
    
    # Save configuration
    config_path = './config.yaml'
    ConfigManager.save_config(config, config_path)
    
    # Create and run pipeline
    pipeline = TrainingPipeline(config_path)
    pipeline.run()

if __name__ == "__main__":
    main()


