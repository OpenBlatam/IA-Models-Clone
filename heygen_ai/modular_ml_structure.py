from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf, DictConfig
from typing import Any, List, Dict, Optional
import asyncio
"""
Modular ML/DL Project Structure Implementation

This module demonstrates a comprehensive modular code structure following key conventions:
- Separate files for models, data loading, training, and evaluation
- Clear separation of concerns
- Modular and extensible design
- Consistent naming conventions
- Proper error handling and logging
"""


# Data processing and numerical computing

# Deep learning frameworks

# Progress tracking

# Visualization

# Configuration management

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    # Model architecture
    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    output_size: int = 10
    dropout_rate: float = 0.2
    activation: str = "relu"
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Data parameters
    data_path: str = "./data"
    model_save_path: str = "./models"
    logs_path: str = "./logs"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Experiment tracking
    use_tensorboard: bool = True
    use_wandb: bool = False
    experiment_name: str = "default_experiment"


# ============================================================================
# DATA LOADING MODULE
# ============================================================================

class DataProcessor:
    """Handles data preprocessing and preparation."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from various sources.
        
        Args:
            data_path: Path to the data file or directory
            
        Returns:
            Tuple of features and labels
        """
        logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.csv'):
            return self._load_csv(data_path)
        elif data_path.endswith('.npy'):
            return self._load_numpy(data_path)
        elif os.path.isdir(data_path):
            return self._load_from_directory(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
    
    def _load_csv(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            # Assume last column is target, rest are features
            features = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values
            logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
            return features, labels
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def _load_numpy(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from NumPy file."""
        try:
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray):
                # Assume first dimension is samples, last is target
                features = data[:, :-1]
                labels = data[:, -1]
            else:
                features = data['features']
                labels = data['labels']
            logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
            return features, labels
        except Exception as e:
            logger.error(f"Error loading NumPy file: {e}")
            raise
    
    def _load_from_directory(self, dir_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from directory structure."""
        # This is a placeholder for more complex data loading
        # (e.g., image folders, text files, etc.)
        raise NotImplementedError("Directory loading not implemented")
    
    def preprocess_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data (scaling, encoding, etc.).
        
        Args:
            features: Input features
            labels: Target labels
            
        Returns:
            Tuple of processed features and labels
        """
        logger.info("Preprocessing data...")
        
        # Scale features
        if not self.is_fitted:
            features_scaled = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            features_scaled = self.scaler.transform(features)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Preprocessed data: {features_scaled.shape}, labels: {labels_encoded.shape}")
        return features_scaled, labels_encoded
    
    def split_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and validation sets.
        
        Args:
            features: Input features
            labels: Target labels
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=self.config.validation_split,
            random_state=42,
            stratify=labels
        )
        
        logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
        return X_train, X_val, y_train, y_val


class CustomDataset(Dataset):
    """Custom PyTorch dataset."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        
    """__init__ function."""
self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> Any:
        return len(self.features)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return self.features[idx], self.labels[idx]


class DataLoaderManager:
    """Manages data loading and creates DataLoaders."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
self.config = config
        self.processor = DataProcessor(config)
    
    def create_dataloaders(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders.
        
        Args:
            data_path: Path to the data
            
        Returns:
            Tuple of train and validation DataLoaders
        """
        # Load and preprocess data
        features, labels = self.processor.load_data(data_path)
        features_processed, labels_processed = self.processor.preprocess_data(features, labels)
        
        # Split data
        X_train, X_val, y_train, y_val = self.processor.split_data(
            features_processed, labels_processed
        )
        
        # Create datasets
        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        logger.info(f"Created DataLoaders: Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        return train_loader, val_loader


# ============================================================================
# MODELS MODULE
# ============================================================================

class BaseModel(ABC, nn.Module):
    """Abstract base class for all models."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.to(self.device)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    def save_model(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


class MLPModel(BaseModel):
    """Multi-layer perceptron model."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        layers = []
        input_size = config.input_size
        
        # Build hidden layers
        for hidden_size in config.hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
                nn.Dropout(config.dropout_rate)
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, config.output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class CNNModel(BaseModel):
    """Convolutional Neural Network model."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
super().__init__(config)
        
        # Adjust input size for CNN (assuming square images)
        input_channels = 1  # Grayscale
        image_size = int(np.sqrt(config.input_size))
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, config.output_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Reshape input for CNN (batch_size, channels, height, width)
        batch_size = x.size(0)
        image_size = int(np.sqrt(self.config.input_size))
        x = x.view(batch_size, 1, image_size, image_size)
        
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


class ModelFactory:
    """Factory for creating different model types."""
    
    @staticmethod
    def create_model(model_type: str, config: ModelConfig) -> BaseModel:
        """
        Create a model based on the specified type.
        
        Args:
            model_type: Type of model to create ('mlp', 'cnn')
            config: Model configuration
            
        Returns:
            Model instance
        """
        if model_type.lower() == "mlp":
            return MLPModel(config)
        elif model_type.lower() == "cnn":
            return CNNModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# TRAINING MODULE
# ============================================================================

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0


class EarlyStopping:
    """Early stopping mechanism."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        
    """__init__ function."""
self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class Trainer:
    """Handles model training."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        
        # Initialize logging
        if config.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(config.logs_path, config.experiment_name))
        else:
            self.writer = None
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(config.early_stopping_patience)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_model(self, model: BaseModel, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing training history
        """
        logger.info("Starting model training...")
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_metrics.val_loss)
            
            # Record metrics
            metrics = TrainingMetrics(
                train_loss=train_metrics.train_loss,
                train_acc=train_metrics.train_acc,
                val_loss=val_metrics.val_loss,
                val_acc=val_metrics.val_acc,
                learning_rate=optimizer.param_groups[0]['lr'],
                epoch=epoch + 1
            )
            
            self._log_metrics(metrics, history)
            
            # Save best model
            if val_metrics.val_loss < best_val_loss:
                best_val_loss = val_metrics.val_loss
                model.save_model(os.path.join(self.config.model_save_path, 'best_model.pth'))
            
            # Check early stopping
            if self.early_stopping(val_metrics.val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        if self.writer:
            self.writer.close()
        
        logger.info("Training completed!")
        return history
    
    def _train_epoch(self, model: BaseModel, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module) -> TrainingMetrics:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        return TrainingMetrics(
            train_loss=total_loss / len(train_loader),
            train_acc=correct / total
        )
    
    def _validate_epoch(self, model: BaseModel, val_loader: DataLoader, 
                       criterion: nn.Module) -> TrainingMetrics:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return TrainingMetrics(
            val_loss=total_loss / len(val_loader),
            val_acc=correct / total
        )
    
    def _log_metrics(self, metrics: TrainingMetrics, history: Dict[str, List[float]]):
        """Log metrics to history and TensorBoard."""
        # Update history
        history['train_loss'].append(metrics.train_loss)
        history['train_acc'].append(metrics.train_acc)
        history['val_loss'].append(metrics.val_loss)
        history['val_acc'].append(metrics.val_acc)
        history['learning_rate'].append(metrics.learning_rate)
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/Train', metrics.train_loss, metrics.epoch)
            self.writer.add_scalar('Loss/Validation', metrics.val_loss, metrics.epoch)
            self.writer.add_scalar('Accuracy/Train', metrics.train_acc, metrics.epoch)
            self.writer.add_scalar('Accuracy/Validation', metrics.val_acc, metrics.epoch)
            self.writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch)
        
        # Log to console
        logger.info(
            f"Epoch {metrics.epoch}: "
            f"Train Loss: {metrics.train_loss:.4f}, Train Acc: {metrics.train_acc:.4f}, "
            f"Val Loss: {metrics.val_loss:.4f}, Val Acc: {metrics.val_acc:.4f}, "
            f"LR: {metrics.learning_rate:.6f}"
        )


# ============================================================================
# EVALUATION MODULE
# ============================================================================

class ModelEvaluator:
    """Handles model evaluation and analysis."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
    
    def evaluate_model(self, model: BaseModel, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        
        # Generate classification report
        class_report = classification_report(all_targets, all_predictions, output_dict=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        return results
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """
        Plot evaluation results.
        
        Args:
            results: Evaluation results
            save_path: Path to save plots
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot confusion matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Plot accuracy metrics
        metrics = results['classification_report']
        precision = [metrics[str(i)]['precision'] for i in range(len(metrics) - 3)]
        recall = [metrics[str(i)]['recall'] for i in range(len(metrics) - 3)]
        f1 = [metrics[str(i)]['f1-score'] for i in range(len(metrics) - 3)]
        
        x = range(len(precision))
        width = 0.25
        
        axes[1].bar([i - width for i in x], precision, width, label='Precision')
        axes[1].bar(x, recall, width, label='Recall')
        axes[1].bar([i + width for i in x], f1, width, label='F1-Score')
        axes[1].set_title('Per-Class Metrics')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].set_xticks(x)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            save_path: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class MLPipeline:
    """Main pipeline that orchestrates the entire ML workflow."""
    
    def __init__(self, config: ModelConfig):
        
    """__init__ function."""
self.config = config
        
        # Create necessary directories
        os.makedirs(self.config.data_path, exist_ok=True)
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(self.config.logs_path, exist_ok=True)
        
        # Initialize components
        self.data_manager = DataLoaderManager(config)
        self.trainer = Trainer(config)
        self.evaluator = ModelEvaluator(config)
        
        logger.info("ML Pipeline initialized")
    
    def run_experiment(self, data_path: str, model_type: str = "mlp") -> Dict[str, Any]:
        """
        Run a complete ML experiment.
        
        Args:
            data_path: Path to the data
            model_type: Type of model to use
            
        Returns:
            Dictionary containing experiment results
        """
        logger.info(f"Starting experiment with {model_type} model")
        
        try:
            # 1. Data loading and preprocessing
            train_loader, val_loader = self.data_manager.create_dataloaders(data_path)
            
            # 2. Model creation
            model = ModelFactory.create_model(model_type, self.config)
            logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
            
            # 3. Training
            history = self.trainer.train_model(model, train_loader, val_loader)
            
            # 4. Evaluation (using validation set as test set for this example)
            results = self.evaluator.evaluate_model(model, val_loader)
            
            # 5. Save results
            results_path = os.path.join(self.config.logs_path, f"{self.config.experiment_name}_results.json")
            self.evaluator.save_results(results, results_path)
            
            # 6. Plot results
            plots_path = os.path.join(self.config.logs_path, f"{self.config.experiment_name}_plots.png")
            self.evaluator.plot_results(results, plots_path)
            
            experiment_results = {
                'model_type': model_type,
                'config': self.config,
                'history': history,
                'evaluation_results': results,
                'model_path': os.path.join(self.config.model_save_path, 'best_model.pth')
            }
            
            logger.info("Experiment completed successfully!")
            return experiment_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_data(n_samples: int = 1000, n_features: int = 784, n_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        Tuple of features and labels
    """
    # Generate random features
    features = np.random.randn(n_samples, n_features)
    
    # Generate random labels
    labels = np.random.randint(0, n_classes, n_samples)
    
    return features, labels


def save_sample_data(features: np.ndarray, labels: np.ndarray, file_path: str):
    """Save sample data to file."""
    data = {
        'features': features,
        'labels': labels
    }
    np.save(file_path, data)
    logger.info(f"Sample data saved to {file_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the modular ML pipeline."""
    
    # Create configuration
    config = ModelConfig(
        input_size=784,
        hidden_sizes=[512, 256, 128],
        output_size=10,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=10,
        experiment_name="sample_experiment"
    )
    
    # Create sample data
    features, labels = create_sample_data()
    data_path = "./data/sample_data.npy"
    save_sample_data(features, labels, data_path)
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    # Run experiment
    results = pipeline.run_experiment(data_path, model_type="mlp")
    
    print("Experiment completed!")
    print(f"Best accuracy: {results['evaluation_results']['accuracy']:.4f}")


match __name__:
    case "__main__":
    main() 