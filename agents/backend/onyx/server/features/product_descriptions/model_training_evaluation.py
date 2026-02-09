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
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import uuid4
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import (
from sklearn.model_selection import (
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import mlflow
import mlflow.pytorch
from transformers import (
                import shutil
                import shutil
from typing import Any, List, Dict, Optional
"""
Model Training and Evaluation System for Cybersecurity Applications

This module provides a comprehensive framework for training, evaluating, and deploying
machine learning models for cybersecurity applications including:
- Threat detection models
- Anomaly detection
- Malware classification
- Network traffic analysis
- Security log analysis

Features:
- Automated training pipelines
- Comprehensive evaluation metrics
- Model versioning and tracking
- Hyperparameter optimization
- Cross-validation strategies
- Model interpretability
- Production deployment
- A/B testing capabilities
"""


    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
    Trainer, TrainingArguments, AutoModelForSequenceClassification,
    AutoTokenizer, DataCollatorWithPadding
)

# Configure structured logging
logger = structlog.get_logger(__name__)


class ModelType(Enum):
    """Supported model types for cybersecurity applications."""
    THREAT_DETECTION = "threat_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    MALWARE_CLASSIFICATION = "malware_classification"
    NETWORK_TRAFFIC_ANALYSIS = "network_traffic_analysis"
    LOG_ANALYSIS = "log_analysis"
    PHISHING_DETECTION = "phishing_detection"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"


class TrainingStatus(Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: ModelType
    model_name: str
    dataset_path: str
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    early_stopping_patience: int = 5
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    fp16: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    output_dir: str = "./models"
    cache_dir: str = "./cache"
    logging_dir: str = "./logs"
    seed: int = 42


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for cybersecurity models."""
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    
    # Regression metrics (for anomaly detection)
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Security-specific metrics
    false_positive_rate: float
    false_negative_rate: float
    true_positive_rate: float
    true_negative_rate: float
    
    # Performance metrics
    inference_time: float
    training_time: float
    model_size_mb: float
    
    # Additional metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    model_id: str
    model_type: ModelType
    model_name: str
    version: str
    created_at: datetime
    training_config: TrainingConfig
    evaluation_metrics: EvaluationMetrics
    dataset_info: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    dependencies: Dict[str, str]
    model_path: str
    artifacts_path: str
    is_production: bool = False
    tags: List[str] = field(default_factory=list)


class BaseDataset(Dataset, ABC):
    """Abstract base class for cybersecurity datasets."""
    
    def __init__(self, data_path: str, tokenizer=None, max_length: int = 512):
        
    """__init__ function."""
self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        self._load_data()
    
    @abstractmethod
    def _load_data(self) -> Any:
        """Load and preprocess the dataset."""
        pass
    
    def __len__(self) -> Any:
        return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return self.data[idx], self.labels[idx]


class ThreatDetectionDataset(BaseDataset):
    """Dataset for threat detection models."""
    
    def _load_data(self) -> Any:
        """Load threat detection dataset."""
        try:
            df = pd.read_csv(self.data_path)
            
            # Assuming columns: 'text', 'label' (0: benign, 1: threat)
            texts = df['text'].tolist()
            labels = df['label'].tolist()
            
            if self.tokenizer:
                # Tokenize texts
                encodings = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                self.data = [
                    {
                        'input_ids': encodings['input_ids'][i],
                        'attention_mask': encodings['attention_mask'][i]
                    }
                    for i in range(len(texts))
                ]
            else:
                self.data = texts
            
            self.labels = labels
            
        except Exception as e:
            logger.error("Failed to load threat detection dataset", error=str(e))
            raise


class AnomalyDetectionDataset(BaseDataset):
    """Dataset for anomaly detection models."""
    
    def _load_data(self) -> Any:
        """Load anomaly detection dataset."""
        try:
            df = pd.read_csv(self.data_path)
            
            # Assuming columns: 'features' (JSON), 'label' (0: normal, 1: anomaly)
            features = df['features'].apply(json.loads).tolist()
            labels = df['label'].tolist()
            
            self.data = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
            
        except Exception as e:
            logger.error("Failed to load anomaly detection dataset", error=str(e))
            raise


class ModelTrainer:
    """Comprehensive model trainer for cybersecurity applications."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_history = []
        self.best_metrics = None
        
        # Setup directories
        self._setup_directories()
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=self.config.logging_dir)
        
        logger.info("Model trainer initialized", config=config)
    
    def _setup_directories(self) -> Any:
        """Create necessary directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
    
    def _setup_mlflow(self) -> Any:
        """Setup MLflow for experiment tracking."""
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(f"cybersecurity_{self.config.model_type.value}")
    
    def _load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split the dataset."""
        if self.config.model_type == ModelType.THREAT_DETECTION:
            dataset_class = ThreatDetectionDataset
        elif self.config.model_type == ModelType.ANOMALY_DETECTION:
            dataset_class = AnomalyDetectionDataset
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Load full dataset
        full_dataset = dataset_class(
            self.config.dataset_path,
            tokenizer=self.tokenizer,
            max_length=512
        )
        
        # Split dataset
        train_size = int((1 - self.config.validation_split - self.config.test_split) * len(full_dataset))
        val_size = int(self.config.validation_split * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.random_state)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def _setup_model(self) -> Any:
        """Setup the model based on type."""
        if self.config.model_type == ModelType.THREAT_DETECTION:
            self._setup_transformer_model()
        elif self.config.model_type == ModelType.ANOMALY_DETECTION:
            self._setup_anomaly_detection_model()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _setup_transformer_model(self) -> Any:
        """Setup transformer-based model for threat detection."""
        model_name = self.config.model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.config.cache_dir
        )
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Binary classification
            cache_dir=self.config.cache_dir
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _setup_anomaly_detection_model(self) -> Any:
        """Setup anomaly detection model."""
        # Simple autoencoder for anomaly detection
        class AnomalyDetector(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int = 64):
                
    """__init__ function."""
super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim // 4, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, x) -> Any:
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        # Determine input dimension from dataset
        sample_dataset = AnomalyDetectionDataset(self.config.dataset_path)
        input_dim = sample_dataset.data.shape[1]
        
        self.model = AnomalyDetector(input_dim=input_dim)
    
    def _setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            seed=self.config.seed
        )
    
    async def train(self) -> ModelMetadata:
        """Train the model."""
        start_time = time.time()
        
        try:
            logger.info("Starting model training", model_type=self.config.model_type.value)
            
            # Setup model
            self._setup_model()
            
            # Load datasets
            train_dataset, val_dataset, test_dataset = self._load_dataset()
            
            # Setup training arguments
            training_args = self._setup_training_arguments()
            
            # Setup trainer
            if self.config.model_type == ModelType.THREAT_DETECTION:
                self.trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=self.tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
                )
            else:
                # Custom training loop for non-transformer models
                await self._custom_training_loop(train_dataset, val_dataset)
            
            # Start training
            with mlflow.start_run():
                mlflow.log_params(self.config.__dict__)
                
                if self.trainer:
                    train_result = self.trainer.train()
                    self.training_history = train_result.history
                    
                    # Evaluate on test set
                    test_results = self.trainer.evaluate(test_dataset)
                    self.best_metrics = test_results
                    
                    # Log metrics
                    mlflow.log_metrics(test_results)
                
                # Save model
                model_path = os.path.join(self.config.output_dir, "final_model")
                if self.trainer:
                    self.trainer.save_model(model_path)
                else:
                    torch.save(self.model.state_dict(), os.path.join(model_path, "model.pt"))
                
                # Create model metadata
                metadata = self._create_model_metadata(
                    model_path=model_path,
                    training_time=time.time() - start_time
                )
                
                # Log model
                mlflow.pytorch.log_model(self.model, "model")
                
                logger.info("Training completed successfully", metadata=metadata)
                return metadata
                
        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise
    
    async def _custom_training_loop(self, train_dataset: Dataset, val_dataset: Dataset):
        """Custom training loop for non-transformer models."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Setup data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory
        )
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        criterion = nn.MSELoss()  # For anomaly detection (reconstruction loss)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, data)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                optimizer.step()
                train_loss += loss.item()
                
                if batch_idx % self.config.logging_steps == 0:
                    logger.info(
                        "Training progress",
                        epoch=epoch,
                        batch=batch_idx,
                        loss=loss.item()
                    )
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(device)
                    output = self.model(data)
                    loss = criterion(output, data)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            
            logger.info(
                "Epoch completed",
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss
            )
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(self.config.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
    
    def _create_model_metadata(self, model_path: str, training_time: float) -> ModelMetadata:
        """Create model metadata."""
        model_id = str(uuid4())
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate model size
        model_size_mb = self._calculate_model_size(model_path)
        
        # Create evaluation metrics
        metrics = EvaluationMetrics(
            accuracy=0.0,  # Will be calculated during evaluation
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            roc_auc=0.0,
            confusion_matrix=np.array([]),
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            true_positive_rate=0.0,
            true_negative_rate=0.0,
            inference_time=0.0,
            training_time=training_time,
            model_size_mb=model_size_mb
        )
        
        # Get dataset info
        dataset_info = self._get_dataset_info()
        
        return ModelMetadata(
            model_id=model_id,
            model_type=self.config.model_type,
            model_name=self.config.model_name,
            version=version,
            created_at=datetime.now(),
            training_config=self.config,
            evaluation_metrics=metrics,
            dataset_info=dataset_info,
            hyperparameters=self.config.__dict__,
            dependencies=self._get_dependencies(),
            model_path=model_path,
            artifacts_path=os.path.join(self.config.output_dir, "artifacts")
        )
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate model size in MB."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def _get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        try:
            df = pd.read_csv(self.config.dataset_path)
            return {
                "total_samples": len(df),
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
        except Exception:
            return {}
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current package dependencies."""
        return {
            "torch": torch.__version__,
            "transformers": "4.x.x",  # Update with actual version
            "sklearn": "1.x.x",  # Update with actual version
            "numpy": np.__version__,
            "pandas": pd.__version__
        }


class ModelEvaluator:
    """Comprehensive model evaluator for cybersecurity applications."""
    
    def __init__(self, model_path: str, model_type: ModelType):
        
    """__init__ function."""
self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self) -> Any:
        """Load the trained model."""
        try:
            if self.model_type == ModelType.THREAT_DETECTION:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                # Load custom model
                self.model = torch.load(os.path.join(self.model_path, "model.pt"))
            
            self.model.eval()
            logger.info("Model loaded successfully", model_path=self.model_path)
            
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise
    
    async def evaluate(self, test_data: Union[str, Dataset]) -> EvaluationMetrics:
        """Evaluate the model on test data."""
        start_time = time.time()
        
        try:
            # Load test data
            if isinstance(test_data, str):
                test_dataset = self._load_test_dataset(test_data)
            else:
                test_dataset = test_data
            
            # Make predictions
            predictions, true_labels, inference_time = await self._make_predictions(test_dataset)
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, true_labels, inference_time)
            
            logger.info("Model evaluation completed", metrics=metrics)
            return metrics
            
        except Exception as e:
            logger.error("Model evaluation failed", error=str(e))
            raise
    
    def _load_test_dataset(self, test_data_path: str) -> Dataset:
        """Load test dataset."""
        if self.model_type == ModelType.THREAT_DETECTION:
            return ThreatDetectionDataset(test_data_path, self.tokenizer)
        elif self.model_type == ModelType.ANOMALY_DETECTION:
            return AnomalyDetectionDataset(test_data_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    async def _make_predictions(self, test_dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, float]:
        """Make predictions on test dataset."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        predictions = []
        true_labels = []
        inference_start = time.time()
        
        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in test_loader:
                if self.model_type == ModelType.THREAT_DETECTION:
                    # Handle transformer inputs
                    inputs = {k: v.to(device) for k, v in batch[0].items()}
                    outputs = self.model(**inputs)
                    batch_predictions = torch.softmax(outputs.logits, dim=1)
                    batch_labels = batch[1]
                else:
                    # Handle tensor inputs
                    data = batch[0].to(device)
                    outputs = self.model(data)
                    batch_predictions = outputs
                    batch_labels = batch[1]
                
                predictions.extend(batch_predictions.cpu().numpy())
                true_labels.extend(batch_labels.cpu().numpy())
        
        inference_time = time.time() - inference_start
        
        return np.array(predictions), np.array(true_labels), inference_time
    
    def _calculate_metrics(self, predictions: np.ndarray, true_labels: np.ndarray, inference_time: float) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""
        if self.model_type == ModelType.THREAT_DETECTION:
            # Classification metrics
            pred_labels = np.argmax(predictions, axis=1)
            pred_probs = predictions[:, 1]  # Probability of positive class
            
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, average='weighted')
            recall = recall_score(true_labels, pred_labels, average='weighted')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            roc_auc = roc_auc_score(true_labels, pred_probs)
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            
            # Security-specific metrics
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            
        else:
            # Anomaly detection metrics (using reconstruction error)
            reconstruction_errors = np.mean((predictions - true_labels) ** 2, axis=1)
            
            # Use threshold-based classification
            threshold = np.percentile(reconstruction_errors, 95)  # 95th percentile as threshold
            pred_labels = (reconstruction_errors > threshold).astype(int)
            
            # Assuming we have ground truth labels for evaluation
            if len(true_labels.shape) > 1:
                true_labels = true_labels[:, 0]  # Take first column as labels
            
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, average='weighted')
            recall = recall_score(true_labels, pred_labels, average='weighted')
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            roc_auc = roc_auc_score(true_labels, reconstruction_errors)
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            
            # Security-specific metrics
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            true_positive_rate=tpr,
            true_negative_rate=tnr,
            inference_time=inference_time,
            training_time=0.0,  # Will be set by trainer
            model_size_mb=0.0,  # Will be set by trainer
            custom_metrics={
                "threshold": threshold if self.model_type == ModelType.ANOMALY_DETECTION else None
            }
        )


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, model_type: ModelType, dataset_path: str):
        
    """__init__ function."""
self.model_type = model_type
        self.dataset_path = dataset_path
        self.study = None
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        study = optuna.create_study(
            direction="minimize",  # Minimize validation loss
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            lambda trial: self._objective(trial),
            n_trials=n_trials,
            timeout=3600  # 1 hour timeout
        )
        
        self.study = study
        
        logger.info(
            "Hyperparameter optimization completed",
            best_params=study.best_params,
            best_value=study.best_value
        )
        
        return study.best_params
    
    def _objective(self, trial) -> float:
        """Objective function for optimization."""
        # Define hyperparameter search space
        config = TrainingConfig(
            model_type=self.model_type,
            model_name=self.model_type.value,
            dataset_path=self.dataset_path,
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            num_epochs=trial.suggest_int("num_epochs", 5, 20),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            warmup_steps=trial.suggest_int("warmup_steps", 100, 1000)
        )
        
        try:
            # Train model with current hyperparameters
            trainer = ModelTrainer(config)
            metadata = await trainer.train()
            
            # Return validation loss
            return metadata.evaluation_metrics.f1_score  # Use F1 score as optimization target
            
        except Exception as e:
            logger.error("Trial failed", error=str(e))
            return float('inf')  # Return high loss for failed trials


class ModelVersionManager:
    """Model versioning and management system."""
    
    def __init__(self, models_dir: str = "./models"):
        
    """__init__ function."""
self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> Any:
        """Load existing model metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> Any:
        """Save model metadata."""
        with open(self.metadata_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.metadata, f, indent=2, default=str)
    
    def register_model(self, metadata: ModelMetadata):
        """Register a new model version."""
        self.metadata[metadata.model_id] = {
            "model_id": metadata.model_id,
            "model_type": metadata.model_type.value,
            "model_name": metadata.model_name,
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "is_production": metadata.is_production,
            "model_path": metadata.model_path,
            "artifacts_path": metadata.artifacts_path,
            "evaluation_metrics": {
                "accuracy": metadata.evaluation_metrics.accuracy,
                "f1_score": metadata.evaluation_metrics.f1_score,
                "precision": metadata.evaluation_metrics.precision,
                "recall": metadata.evaluation_metrics.recall
            },
            "tags": metadata.tags
        }
        
        self._save_metadata()
        logger.info("Model registered", model_id=metadata.model_id)
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by ID."""
        return self.metadata.get(model_id)
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """List all models, optionally filtered by type."""
        models = list(self.metadata.values())
        
        if model_type:
            models = [m for m in models if m["model_type"] == model_type.value]
        
        return sorted(models, key=lambda x: x["created_at"], reverse=True)
    
    def set_production_model(self, model_id: str, model_type: ModelType):
        """Set a model as production for a specific type."""
        # Remove production flag from other models of the same type
        for mid, model_data in self.metadata.items():
            if model_data["model_type"] == model_type.value:
                model_data["is_production"] = False
        
        # Set the specified model as production
        if model_id in self.metadata:
            self.metadata[model_id]["is_production"] = True
            self._save_metadata()
            logger.info("Production model set", model_id=model_id, model_type=model_type.value)
        else:
            raise ValueError(f"Model {model_id} not found")
    
    def get_production_model(self, model_type: ModelType) -> Optional[Dict[str, Any]]:
        """Get the current production model for a type."""
        for model_data in self.metadata.values():
            if (model_data["model_type"] == model_type.value and 
                model_data["is_production"]):
                return model_data
        return None
    
    def delete_model(self, model_id: str):
        """Delete a model and its artifacts."""
        if model_id in self.metadata:
            model_data = self.metadata[model_id]
            
            # Delete model files
            model_path = Path(model_data["model_path"])
            if model_path.exists():
                shutil.rmtree(model_path)
            
            # Delete artifacts
            artifacts_path = Path(model_data["artifacts_path"])
            if artifacts_path.exists():
                shutil.rmtree(artifacts_path)
            
            # Remove from metadata
            del self.metadata[model_id]
            self._save_metadata()
            
            logger.info("Model deleted", model_id=model_id)
        else:
            raise ValueError(f"Model {model_id} not found")


class ModelDeploymentManager:
    """Model deployment and serving manager."""
    
    def __init__(self, models_dir: str = "./models"):
        
    """__init__ function."""
self.models_dir = Path(models_dir)
        self.deployed_models = {}
        self.version_manager = ModelVersionManager(models_dir)
    
    async def deploy_model(self, model_type: ModelType, model_id: Optional[str] = None) -> str:
        """Deploy a model for serving."""
        if model_id is None:
            # Get production model
            model_data = self.version_manager.get_production_model(model_type)
            if not model_data:
                raise ValueError(f"No production model found for type {model_type.value}")
            model_id = model_data["model_id"]
        
        # Load model
        model_data = self.version_manager.get_model(model_id)
        if not model_data:
            raise ValueError(f"Model {model_id} not found")
        
        # Create model instance
        if model_type == ModelType.THREAT_DETECTION:
            model = AutoModelForSequenceClassification.from_pretrained(model_data["model_path"])
            tokenizer = AutoTokenizer.from_pretrained(model_data["model_path"])
            deployed_model = {
                "model": model,
                "tokenizer": tokenizer,
                "model_type": model_type,
                "model_id": model_id
            }
        else:
            # Load custom model
            model = torch.load(Path(model_data["model_path"]) / "model.pt")
            deployed_model = {
                "model": model,
                "model_type": model_type,
                "model_id": model_id
            }
        
        # Store deployed model
        deployment_id = f"{model_type.value}_{model_id}"
        self.deployed_models[deployment_id] = deployed_model
        
        logger.info("Model deployed", deployment_id=deployment_id, model_id=model_id)
        return deployment_id
    
    async def predict(self, deployment_id: str, input_data: Any) -> Dict[str, Any]:
        """Make prediction using deployed model."""
        if deployment_id not in self.deployed_models:
            raise ValueError(f"Model {deployment_id} not deployed")
        
        deployed_model = self.deployed_models[deployment_id]
        model = deployed_model["model"]
        model_type = deployed_model["model_type"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        start_time = time.time()
        
        with torch.no_grad():
            if model_type == ModelType.THREAT_DETECTION:
                # Tokenize input
                tokenizer = deployed_model["tokenizer"]
                inputs = tokenizer(
                    input_data,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Make prediction
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
                
            else:
                # Handle tensor input for anomaly detection
                if isinstance(input_data, list):
                    input_tensor = torch.tensor(input_data, dtype=torch.float32)
                else:
                    input_tensor = input_data
                
                input_tensor = input_tensor.to(device)
                output = model(input_tensor)
                
                # Calculate reconstruction error
                reconstruction_error = torch.mean((output - input_tensor) ** 2).item()
                prediction = 1 if reconstruction_error > 0.1 else 0  # Threshold-based
                confidence = 1.0 - reconstruction_error
        
        inference_time = time.time() - start_time
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "inference_time": inference_time,
            "model_id": deployed_model["model_id"],
            "model_type": model_type.value
        }
    
    def list_deployed_models(self) -> List[str]:
        """List all deployed models."""
        return list(self.deployed_models.keys())
    
    def undeploy_model(self, deployment_id: str):
        """Undeploy a model."""
        if deployment_id in self.deployed_models:
            del self.deployed_models[deployment_id]
            logger.info("Model undeployed", deployment_id=deployment_id)
        else:
            raise ValueError(f"Model {deployment_id} not deployed")


# Factory function for creating model trainers
def create_model_trainer(model_type: ModelType, **kwargs) -> ModelTrainer:
    """Factory function to create model trainers."""
    config = TrainingConfig(model_type=model_type, **kwargs)
    return ModelTrainer(config)


# Utility functions
def calculate_model_complexity(model: nn.Module) -> Dict[str, int]:
    """Calculate model complexity metrics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def validate_model_performance(metrics: EvaluationMetrics, thresholds: Dict[str, float]) -> bool:
    """Validate if model meets performance thresholds."""
    checks = [
        metrics.accuracy >= thresholds.get("min_accuracy", 0.7),
        metrics.f1_score >= thresholds.get("min_f1", 0.7),
        metrics.false_positive_rate <= thresholds.get("max_fpr", 0.1),
        metrics.inference_time <= thresholds.get("max_inference_time", 1.0)
    ]
    
    return all(checks)


async def run_ab_test(model_a_id: str, model_b_id: str, test_data: str, 
                     traffic_split: float = 0.5) -> Dict[str, Any]:
    """Run A/B test between two models."""
    deployment_manager = ModelDeploymentManager()
    
    # Deploy both models
    deployment_a = await deployment_manager.deploy_model(
        ModelType.THREAT_DETECTION, model_a_id
    )
    deployment_b = await deployment_manager.deploy_model(
        ModelType.THREAT_DETECTION, model_b_id
    )
    
    # Load test data
    test_dataset = ThreatDetectionDataset(test_data)
    
    results_a = []
    results_b = []
    
    # Run predictions with traffic split
    for i, (data, label) in enumerate(test_dataset):
        if i % 2 == 0:  # Route to model A
            result = await deployment_manager.predict(deployment_a, data)
            results_a.append((result, label))
        else:  # Route to model B
            result = await deployment_manager.predict(deployment_b, data)
            results_b.append((result, label))
    
    # Calculate metrics for each model
    evaluator_a = ModelEvaluator(f"models/{model_a_id}", ModelType.THREAT_DETECTION)
    evaluator_b = ModelEvaluator(f"models/{model_b_id}", ModelType.THREAT_DETECTION)
    
    metrics_a = await evaluator_a.evaluate(test_data)
    metrics_b = await evaluator_b.evaluate(test_data)
    
    # Cleanup
    deployment_manager.undeploy_model(deployment_a)
    deployment_manager.undeploy_model(deployment_b)
    
    return {
        "model_a": {
            "model_id": model_a_id,
            "metrics": metrics_a.__dict__
        },
        "model_b": {
            "model_id": model_b_id,
            "metrics": metrics_b.__dict__
        },
        "winner": "model_a" if metrics_a.f1_score > metrics_b.f1_score else "model_b"
    }


if __name__ == "__main__":
    # Example usage
    async def main():
        
    """main function."""
# Create training config
        config = TrainingConfig(
            model_type=ModelType.THREAT_DETECTION,
            model_name="distilbert-base-uncased",
            dataset_path="data/threat_detection.csv",
            num_epochs=3,
            batch_size=16
        )
        
        # Train model
        trainer = ModelTrainer(config)
        metadata = await trainer.train()
        
        # Evaluate model
        evaluator = ModelEvaluator(metadata.model_path, config.model_type)
        metrics = await evaluator.evaluate("data/test.csv")
        
        # Register model
        version_manager = ModelVersionManager()
        version_manager.register_model(metadata)
        
        # Deploy model
        deployment_manager = ModelDeploymentManager()
        deployment_id = await deployment_manager.deploy_model(config.model_type)
        
        # Make prediction
        prediction = await deployment_manager.predict(
            deployment_id, 
            "This is a suspicious email with malicious content"
        )
        
        print(f"Prediction: {prediction}")
    
    asyncio.run(main()) 