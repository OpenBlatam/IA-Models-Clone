from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from model_training_evaluation import (
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Test script for the Model Training and Evaluation Framework
"""


# Import the training framework
    TrainingConfig, ModelTrainer, ModelEvaluator, 
    EfficientDataLoader, TrainingMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTestDataset(Dataset):
    """Simple test dataset"""
    
    def __init__(self, num_samples=100, num_features=10, num_classes=2) -> Any:
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self._generate_data()
    
    def _generate_data(self) -> Any:
        """Generate synthetic data"""
        np.random.seed(42)
        self.features = torch.randn(self.num_samples, self.num_features)
        self.labels = torch.randint(0, self.num_classes, (self.num_samples,))
    
    def __len__(self) -> Any:
        return self.num_samples
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }

class SimpleTestModel(nn.Module):
    """Simple test model"""
    
    def __init__(self, input_size=10, hidden_size=32, num_classes=2) -> Any:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x) -> Any:
        if isinstance(x, dict):
            x = x['features']
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def test_training_config():
    """Test TrainingConfig creation"""
    logger.info("Testing TrainingConfig...")
    
    model = SimpleTestModel()
    train_dataset = SimpleTestDataset(100)
    val_dataset = SimpleTestDataset(20)
    
    config = TrainingConfig(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        optimizer="adamw",
        scheduler="cosine"
    )
    
    logger.info(f"TrainingConfig created successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Device: {config.device}")
    
    return config

def test_efficient_data_loader():
    """Test EfficientDataLoader"""
    logger.info("Testing EfficientDataLoader...")
    
    config = test_training_config()
    data_loader = EfficientDataLoader(config)
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    
    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Val loader batches: {len(val_loader)}")
    
    # Test batch iteration
    for batch in train_loader:
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Features shape: {batch['features'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        break
    
    return data_loader

def test_model_trainer():
    """Test ModelTrainer"""
    logger.info("Testing ModelTrainer...")
    
    config = test_training_config()
    trainer = ModelTrainer(config)
    
    logger.info(f"ModelTrainer created successfully")
    logger.info(f"Device: {trainer.device}")
    logger.info(f"Optimizer: {type(trainer.optimizer).__name__}")
    logger.info(f"Scheduler: {type(trainer.scheduler).__name__}")
    
    return trainer

def test_model_evaluator():
    """Test ModelEvaluator"""
    logger.info("Testing ModelEvaluator...")
    
    model = SimpleTestModel()
    evaluator = ModelEvaluator(model)
    
    logger.info(f"ModelEvaluator created successfully")
    logger.info(f"Device: {evaluator.device}")
    
    return evaluator

def test_training_loop():
    """Test complete training loop"""
    logger.info("Testing complete training loop...")
    
    config = test_training_config()
    trainer = ModelTrainer(config)
    
    # Train for a few epochs
    logger.info("Starting training...")
    metrics = trainer.train()
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {metrics.best_val_loss:.4f}")
    logger.info(f"Best validation accuracy: {metrics.best_val_accuracy:.4f}")
    logger.info(f"Total epochs: {metrics.current_epoch}")
    
    return metrics

def test_evaluation():
    """Test model evaluation"""
    logger.info("Testing model evaluation...")
    
    # Create model and test dataset
    model = SimpleTestModel()
    test_dataset = SimpleTestDataset(50)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create evaluator
    evaluator = ModelEvaluator(model)
    
    # Evaluate
    test_metrics = evaluator.evaluate(test_loader, task_type="classification")
    
    logger.info(f"Evaluation completed!")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1_score']:.4f}")
    
    return test_metrics

def main():
    """Run all tests"""
    logger.info("Starting training framework tests...")
    
    try:
        # Test individual components
        test_training_config()
        test_efficient_data_loader()
        test_model_trainer()
        test_model_evaluator()
        
        # Test complete training
        training_metrics = test_training_loop()
        
        # Test evaluation
        evaluation_metrics = test_evaluation()
        
        logger.info("All tests completed successfully!")
        
        # Print summary
        logger.info("\n=== Test Summary ===")
        logger.info(f"Training - Best val loss: {training_metrics.best_val_loss:.4f}")
        logger.info(f"Training - Best val accuracy: {training_metrics.best_val_accuracy:.4f}")
        logger.info(f"Evaluation - Test accuracy: {evaluation_metrics['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

match __name__:
    case "__main__":
    main() 