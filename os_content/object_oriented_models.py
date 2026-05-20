from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, TypeVar, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from transformers import (
        from transformers import get_linear_schedule_with_warmup
        from sklearn.metrics import (
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Object-Oriented Model Architectures
Clean, modular model architectures using object-oriented programming
that complement functional data processing pipelines.
"""



    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModelForQuestionAnswering
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Type variables
T = TypeVar('T')
ModelInput = TypeVar('ModelInput')
ModelOutput = TypeVar('ModelOutput')

class ModelType(Enum):
    """Supported model types"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    CUSTOM = "custom"

class TaskType(Enum):
    """Supported task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TOKEN_CLASSIFICATION = "token_classification"
    QUESTION_ANSWERING = "question_answering"
    GENERATION = "generation"
    EMBEDDING = "embedding"

@dataclass
class ModelConfig:
    """Configuration for model architectures"""
    model_type: ModelType
    task_type: TaskType
    model_name: str = "bert-base-uncased"
    
    # Architecture parameters
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    
    # Task-specific parameters
    num_classes: int = 2
    max_length: int = 512
    
    # Optimization parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    
    # Advanced parameters
    use_attention: bool = True
    use_positional_encoding: bool = True
    use_layer_norm: bool = True
    activation_function: str = "gelu"

class ModelInput(Protocol):
    """Protocol for model input"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None

class ModelOutput(Protocol):
    """Protocol for model output"""
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: ModelOutput, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for the model"""
        pass
    
    def to_device(self, device: torch.device) -> 'BaseModel':
        """Move model to device"""
        self.device = device
        if self.model:
            self.model.to(device)
        return self
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'tokenizer': self.tokenizer
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint.get('tokenizer')
        logger.info(f"Model loaded from {path}")

class TransformerModel(BaseModel):
    """Transformer-based model architecture"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        self.model = self.build_model()
        self.tokenizer = self._load_tokenizer()
    
    def build_model(self) -> nn.Module:
        """Build transformer model"""
        if self.config.task_type == TaskType.CLASSIFICATION:
            return AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_classes,
                dropout=self.config.dropout_rate
            )
        elif self.config.task_type == TaskType.REGRESSION:
            base_model = AutoModel.from_pretrained(self.config.model_name)
            return TransformerRegressionModel(base_model, self.config)
        elif self.config.task_type == TaskType.TOKEN_CLASSIFICATION:
            return AutoModelForTokenClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_classes
            )
        elif self.config.task_type == TaskType.QUESTION_ANSWERING:
            return AutoModelForQuestionAnswering.from_pretrained(self.config.model_name)
        else:
            return AutoModel.from_pretrained(self.config.model_name)
    
    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Forward pass"""
        return self.model(**inputs)
    
    def compute_loss(self, outputs: ModelOutput, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        if self.config.task_type == TaskType.CLASSIFICATION:
            return F.cross_entropy(outputs.logits, targets)
        elif self.config.task_type == TaskType.REGRESSION:
            return F.mse_loss(outputs.logits.squeeze(), targets)
        else:
            raise NotImplementedError(f"Loss computation not implemented for {self.config.task_type}")
    
    def _load_tokenizer(self) -> Any:
        """Load tokenizer"""
        return AutoTokenizer.from_pretrained(self.config.model_name)

class TransformerRegressionModel(nn.Module):
    """Custom transformer model for regression tasks"""
    
    def __init__(self, base_model: nn.Module, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(base_model.config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1)
        )
    
    def forward(self, **inputs) -> ModelOutput:
        """Forward pass"""
        base_outputs = self.base_model(**inputs)
        logits = self.regression_head(base_outputs.last_hidden_state[:, 0, :])
        
        return type('ModelOutput', (), {
            'logits': logits,
            'hidden_states': base_outputs.hidden_states,
            'attentions': base_outputs.attentions
        })()

class CNNModel(BaseModel):
    """CNN-based model architecture"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        self.model = self.build_model()
        self.tokenizer = self._create_tokenizer()
    
    def build_model(self) -> nn.Module:
        """Build CNN model"""
        return TextCNN(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_filters=self.config.num_filters,
            filter_sizes=self.config.filter_sizes,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate
        )
    
    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Forward pass"""
        return self.model(inputs)
    
    def compute_loss(self, outputs: ModelOutput, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        return F.cross_entropy(outputs.logits, targets)
    
    def _create_tokenizer(self) -> Any:
        """Create simple tokenizer for CNN"""
        # This would be a custom tokenizer implementation
        return None

class TextCNN(nn.Module):
    """Text CNN architecture"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 num_filters: int = 100, filter_sizes: List[int] = [3, 4, 5],
                 num_classes: int = 2, dropout_rate: float = 0.1):
        
    """__init__ function."""
super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Forward pass"""
        # Assuming inputs.input_ids is the tokenized text
        x = self.embedding(inputs.input_ids).unsqueeze(1)  # Add channel dimension
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x)).squeeze(3)  # Remove last dimension
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(conv_out)
        
        # Concatenate and classify
        x = torch.cat(conv_outputs, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        
        return type('ModelOutput', (), {'logits': logits})()

class RNNModel(BaseModel):
    """RNN-based model architecture"""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        self.model = self.build_model()
        self.tokenizer = self._create_tokenizer()
    
    def build_model(self) -> nn.Module:
        """Build RNN model"""
        return TextRNN(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            rnn_type=self.config.model_type.value
        )
    
    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Forward pass"""
        return self.model(inputs)
    
    def compute_loss(self, outputs: ModelOutput, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        return F.cross_entropy(outputs.logits, targets)
    
    def _create_tokenizer(self) -> Any:
        """Create simple tokenizer for RNN"""
        return None

class TextRNN(nn.Module):
    """Text RNN architecture"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 2, dropout_rate: float = 0.1,
                 rnn_type: str = "lstm"):
        
    """__init__ function."""
super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Choose RNN type
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=True
            )
            hidden_dim *= 2  # Bidirectional
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=True
            )
            hidden_dim *= 2  # Bidirectional
        else:
            self.rnn = nn.RNN(
                embed_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=True
            )
            hidden_dim *= 2  # Bidirectional
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Forward pass"""
        embedded = self.embedding(inputs.input_ids)
        embedded = self.dropout(embedded)
        
        # RNN forward pass
        rnn_out, (hidden, cell) = self.rnn(embedded)
        
        # Use last hidden state from both directions
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[-2:]  # Last layer, both directions
            hidden = torch.cat(hidden, dim=1)
        else:  # GRU or RNN
            hidden = hidden[-2:]  # Last layer, both directions
            hidden = torch.cat(hidden, dim=1)
        
        # Classification
        output = self.dropout(hidden)
        logits = self.fc(output)
        
        return type('ModelOutput', (), {'logits': logits})()

class ModelFactory:
    """Factory for creating model instances"""
    
    @staticmethod
    def create_model(config: ModelConfig) -> BaseModel:
        """Create model based on configuration"""
        if config.model_type == ModelType.TRANSFORMER:
            return TransformerModel(config)
        elif config.model_type == ModelType.CNN:
            return CNNModel(config)
        elif config.model_type == ModelType.RNN:
            return RNNModel(config)
        elif config.model_type == ModelType.LSTM:
            config.model_type = ModelType.RNN  # LSTM is implemented in RNN
            return RNNModel(config)
        elif config.model_type == ModelType.GRU:
            config.model_type = ModelType.RNN  # GRU is implemented in RNN
            return RNNModel(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

class ModelTrainer:
    """Object-oriented model trainer"""
    
    def __init__(self, model: BaseModel, config: ModelConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if torch.cuda.is_available() else None
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        return optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=500,
            num_training_steps=1000  # This should be calculated based on dataset size
        )
    
    async def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                outputs = self.model.forward(batch)
                loss = self.model.compute_loss(outputs, batch['labels'])
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            if self.config.task_type == TaskType.CLASSIFICATION:
                predictions = torch.argmax(outputs.logits, dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch['labels'].size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples if total_samples > 0 else 0
        }
    
    async def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model.forward(batch)
                loss = self.model.compute_loss(outputs, batch['labels'])
                
                # Update metrics
                total_loss += loss.item()
                if self.config.task_type == TaskType.CLASSIFICATION:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    total_correct += (predictions == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples if total_samples > 0 else 0
        }

class ModelEvaluator:
    """Object-oriented model evaluator"""
    
    def __init__(self, model: BaseModel):
        
    """__init__ function."""
self.model = model
    
    async def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set"""
        self.model.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model.forward(batch)
                
                # Get predictions
                if self.model.config.task_type == TaskType.CLASSIFICATION:
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        return metrics
    
    def _calculate_metrics(self, labels: List, predictions: List, 
                          probabilities: List) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'total_samples': len(labels)
        }
        
        if self.model.config.task_type == TaskType.CLASSIFICATION:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
                'classification_report': classification_report(labels, predictions, output_dict=True)
            })
        
        return metrics

# Example usage
def create_text_classification_model(model_name: str = "bert-base-uncased", 
                                   num_classes: int = 2) -> BaseModel:
    """Create a text classification model"""
    config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        task_type=TaskType.CLASSIFICATION,
        model_name=model_name,
        num_classes=num_classes
    )
    
    return ModelFactory.create_model(config)

def create_text_regression_model(model_name: str = "bert-base-uncased") -> BaseModel:
    """Create a text regression model"""
    config = ModelConfig(
        model_type=ModelType.TRANSFORMER,
        task_type=TaskType.REGRESSION,
        model_name=model_name
    )
    
    return ModelFactory.create_model(config)

def create_cnn_model(vocab_size: int = 30000, num_classes: int = 2) -> BaseModel:
    """Create a CNN model"""
    config = ModelConfig(
        model_type=ModelType.CNN,
        task_type=TaskType.CLASSIFICATION,
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=128,
        num_filters=100,
        filter_sizes=[3, 4, 5]
    )
    
    return ModelFactory.create_model(config)

# Example usage
if __name__ == "__main__":
    # Create a text classification model
    model = create_text_classification_model("bert-base-uncased", num_classes=2)
    
    print(f"Model type: {model.config.model_type}")
    print(f"Task type: {model.config.task_type}")
    print(f"Model name: {model.config.model_name}")
    print(f"Number of classes: {model.config.num_classes}")
    
    # Create trainer
    trainer = ModelTrainer(model, model.config)
    
    # Create evaluator
    evaluator = ModelEvaluator(model)
    
    print("Object-oriented model architecture created successfully!") 