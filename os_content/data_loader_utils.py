from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from pydantic import BaseModel, Field, validator
    import time
from typing import Any, List, Dict, Optional
"""
Efficient Data Loading and Cross-Validation Utilities
Implements PyTorch DataLoader with proper train/validation/test splits
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityDataPoint(BaseModel):
    """Security data point for analysis."""
    timestamp: str
    source_ip: str
    destination_ip: str
    port: int
    protocol: str
    payload_size: int
    is_malicious: bool
    threat_type: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)

class DataLoaderConfig(BaseModel):
    """Configuration for data loading."""
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)
    train_split: float = Field(default=0.7, gt=0.0, lt=1.0)
    validation_split: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.15, gt=0.0, lt=1.0)
    random_seed: int = Field(default=42)
    shuffle: bool = Field(default=True)
    pin_memory: bool = Field(default=True)
    persistent_workers: bool = Field(default=True)
    
    @validator('train_split', 'validation_split', 'test_split')
    def validate_splits(cls, v, values) -> bool:
        if 'train_split' in values and 'validation_split' in values:
            total = values['train_split'] + values['validation_split'] + v
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError("Splits must sum to 1.0")
        return v

class CrossValidationConfig(BaseModel):
    """Configuration for cross-validation."""
    n_splits: int = Field(default=5, gt=1)
    shuffle: bool = Field(default=True)
    random_state: int = Field(default=42)
    stratified: bool = Field(default=True)

class SecurityDataset(Dataset):
    """PyTorch Dataset for security data analysis."""
    
    def __init__(self, data: List[SecurityDataPoint], transform=None):
        
    """__init__ function."""
self.data = data
        self.transform = transform
        self._prepare_features()
    
    def _prepare_features(self) -> Any:
        """Prepare features for machine learning."""
        self.features = []
        self.labels = []
        
        for point in self.data:
            feature_vector = [
                hash(point.source_ip) % 10000,
                hash(point.destination_ip) % 10000,
                point.port,
                hash(point.protocol) % 100,
                point.payload_size,
                point.confidence_score
            ]
            
            self.features.append(feature_vector)
            self.labels.append(1 if point.is_malicious else 0)
        
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self) -> Any:
        return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, label

async def load_security_data(data_path: str) -> List[SecurityDataPoint]:
    """Load security data from various sources."""
    data_path = Path(data_path)
    
    if data_path.suffix == '.json':
        return await _load_json_data(data_path)
    elif data_path.suffix == '.csv':
        return await _load_csv_data(data_path)
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")

async def _load_json_data(file_path: Path) -> List[SecurityDataPoint]:
    """Load data from JSON file."""
    try:
        with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            raw_data = json.load(f)
        
        data_points = []
        for item in raw_data:
            try:
                data_point = SecurityDataPoint(**item)
                data_points.append(data_point)
            except Exception as e:
                logger.warning(f"Skipping invalid data point: {e}")
                continue
        
        logger.info(f"Loaded {len(data_points)} valid data points from JSON")
        return data_points
    
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        raise

async def _load_csv_data(file_path: Path) -> List[SecurityDataPoint]:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        data_points = []
        
        for _, row in df.iterrows():
            try:
                data_point = SecurityDataPoint(
                    timestamp=str(row['timestamp']),
                    source_ip=str(row['source_ip']),
                    destination_ip=str(row['destination_ip']),
                    port=int(row['port']),
                    protocol=str(row['protocol']),
                    payload_size=int(row['payload_size']),
                    is_malicious=bool(row['is_malicious']),
                    threat_type=row.get('threat_type'),
                    confidence_score=float(row.get('confidence_score', 0.5))
                )
                data_points.append(data_point)
            except Exception as e:
                logger.warning(f"Skipping invalid CSV row: {e}")
                continue
        
        logger.info(f"Loaded {len(data_points)} valid data points from CSV")
        return data_points
    
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def create_data_splits(dataset: SecurityDataset, config: DataLoaderConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test splits with DataLoaders."""
    total_size = len(dataset)
    train_size = int(config.train_split * total_size)
    validation_size = int(config.validation_split * total_size)
    test_size = total_size - train_size - validation_size
    
    torch.manual_seed(config.random_seed)
    
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )
    
    logger.info(f"Created splits: Train={len(train_dataset)}, "
                f"Validation={len(validation_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, validation_loader, test_loader

def create_cross_validation_splits(dataset: SecurityDataset, config: CrossValidationConfig) -> List[Tuple[DataLoader, DataLoader]]:
    """Create cross-validation splits with DataLoaders."""
    total_size = len(dataset)
    indices = list(range(total_size))
    labels = [dataset.labels[i].item() for i in indices]
    
    if config.stratified:
        kfold = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=config.random_state
        )
        splits = kfold.split(indices, labels)
    else:
        kfold = KFold(
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=config.random_state
        )
        splits = kfold.split(indices)
    
    cv_loaders = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        cv_loaders.append((train_loader, val_loader))
        logger.info(f"Created CV fold {fold + 1}: Train={len(train_subset)}, Validation={len(val_subset)}")
    
    return cv_loaders

class SecurityAnalysisModel(nn.Module):
    """Neural network for security threat analysis."""
    
    def __init__(self, input_size: int = 6, hidden_size: int = 128, num_classes: int = 2):
        
    """__init__ function."""
super(SecurityAnalysisModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x) -> Any:
        return self.layers(x)

async def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 10, learning_rate: float = 0.001, device: str = 'cpu') -> Dict[str, List[float]]:
    """Train the security analysis model."""
    device = torch.device(device)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    logger.info(f"Starting training for {epochs} epochs on {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch + 1}/{epochs}: "
                   f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

async def cross_validate_model(dataset: SecurityDataset, config: CrossValidationConfig,
                             epochs: int = 10, learning_rate: float = 0.001,
                             device: str = 'cpu') -> Dict[str, List[float]]:
    """Perform cross-validation on the security model."""
    cv_loaders = create_cross_validation_splits(dataset, config)
    
    cv_results = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
    
    logger.info(f"Starting {config.n_splits}-fold cross-validation")
    
    for fold, (train_loader, val_loader) in enumerate(cv_loaders):
        logger.info(f"Training fold {fold + 1}/{config.n_splits}")
        
        model = SecurityAnalysisModel()
        fold_results = await train_model(
            model, train_loader, val_loader, epochs, learning_rate, device
        )
        
        for key in cv_results:
            cv_results[key].extend(fold_results[key])
        
        logger.info(f"Completed fold {fold + 1}")
    
    avg_metrics = {}
    for key in cv_results:
        values = cv_results[key]
        avg_metrics[f'avg_{key}'] = np.mean(values)
        avg_metrics[f'std_{key}'] = np.std(values)
    
    logger.info(f"Cross-validation completed. Average validation accuracy: {avg_metrics['avg_val_accuracies']:.2f}% ± {avg_metrics['std_val_accuracies']:.2f}%")
    
    return cv_results

async def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Dict[str, Any]:
    """Evaluate the trained model on test data."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    
    class_report = classification_report(all_labels, all_predictions, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': all_predictions,
        'true_labels': all_labels
    }
    
    logger.info(f"Test Results - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    return results

async def run_security_analysis_pipeline(data_path: str, config: DataLoaderConfig,
                                       cv_config: CrossValidationConfig,
                                       use_cross_validation: bool = False) -> Dict[str, Any]:
    """Run complete security analysis pipeline with efficient data loading."""
    start_time = time.time()
    
    try:
        # Load data
        logger.info(f"Loading security data from {data_path}")
        data_points = await load_security_data(data_path)
        
        if not data_points:
            raise ValueError("No valid data points loaded")
        
        # Create dataset
        dataset = SecurityDataset(data_points)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        if use_cross_validation:
            # Perform cross-validation
            logger.info("Running cross-validation")
            cv_results = await cross_validate_model(dataset, cv_config, device=device)
            
            # Train final model on full dataset
            train_loader, val_loader, test_loader = create_data_splits(dataset, config)
            final_model = SecurityAnalysisModel()
            training_results = await train_model(final_model, train_loader, val_loader, device=device)
            
            # Evaluate final model
            evaluation_results = await evaluate_model(final_model, test_loader, device=device)
            
            results = {
                'cross_validation_results': cv_results,
                'final_training_results': training_results,
                'evaluation_results': evaluation_results,
                'pipeline_duration': time.time() - start_time
            }
        else:
            # Standard train/validation/test split
            train_loader, val_loader, test_loader = create_data_splits(dataset, config)
            
            # Train model
            model = SecurityAnalysisModel()
            training_results = await train_model(model, train_loader, val_loader, device=device)
            
            # Evaluate model
            evaluation_results = await evaluate_model(model, test_loader, device=device)
            
            results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'pipeline_duration': time.time() - start_time
            }
        
        logger.info(f"Security analysis pipeline completed in {results['pipeline_duration']:.2f} seconds")
        return results
        
    except Exception as e:
        logger.error(f"Error in security analysis pipeline: {e}")
        raise

# Example usage
async def main():
    """Example usage of the data loader utilities."""
    
    config = DataLoaderConfig(
        batch_size=64,
        num_workers=4,
        train_split=0.7,
        validation_split=0.15,
        test_split=0.15
    )
    
    cv_config = CrossValidationConfig(
        n_splits=5,
        stratified=True,
        shuffle=True
    )
    
    data_path = "security_data.json"
    
    try:
        results = await run_security_analysis_pipeline(
            data_path, config, cv_config, use_cross_validation=True
        )
        
        print("Security Analysis Results:")
        print(f"Pipeline Duration: {results['pipeline_duration']:.2f} seconds")
        
        if 'cross_validation_results' in results:
            cv_results = results['cross_validation_results']
            print(f"Cross-validation average accuracy: {np.mean(cv_results['val_accuracies']):.2f}%")
        
        eval_results = results['evaluation_results']
        print(f"Final Test Accuracy: {eval_results['test_accuracy']:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

match __name__:
    case "__main__":
    asyncio.run(main()) 